from __future__ import annotations

"""Build the authored QPSK phase-tracking tutorial snapshot.

The checked-in scalar-metric CSV is a deterministic projection of one
completed five-method UI sweep.  ``--import-runs`` is the only mode that reads
the local run store; normal generation and ``--check`` use the committed CSV
so the documentation remains reproducible without a developer's ``.noema``
directory.
"""

import argparse
import csv
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noema_lab.core.structured_input import decode_strict_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / ".noema" / "runs"
DEFAULT_DATA_DIR = ROOT / "docs" / "demo" / "data" / "qpsk_phase_tracking"
DEFAULT_ASSET_DIR = ROOT / "docs" / "demo" / "assets" / "qpsk_phase_tracking"
DEFAULT_SOURCE = DEFAULT_DATA_DIR / "snapshot_metrics.csv"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "snapshot_manifest.json"

SOURCE_FIELDS = ("Method", "Run", "SNR (dB)", "Step", "Metric", "Value")
SUMMARY_FIELDS = (
    "method",
    "run_id",
    "snr_db",
    "bit_errors",
    "compared_bits",
    "ber",
    "block_errors",
    "compared_blocks",
    "bler",
    "authored_recipe_sha256",
    "effective_recipe_sha256",
    "data_seed",
    "wireless_seed",
    "carrier_impairment_seed",
    "trained_artifact_manifest_sha256",
)

METHOD_ORDER = (
    "uncompensated",
    "pilot_smoothing",
    "decision_directed_pll",
    "oracle",
    "learned_artifact",
)
METHOD_LABELS = {
    "uncompensated": "Uncompensated QPSK",
    "pilot_smoothing": "Pilot smoothing",
    "decision_directed_pll": "Decision-directed PLL",
    "oracle": "True-phase oracle",
    "learned_artifact": "Learned receiver",
}
METHOD_STYLES = {
    "uncompensated": {"color": "#6b7280", "marker": "s", "linestyle": "--"},
    "pilot_smoothing": {"color": "#2563eb", "marker": "o", "linestyle": "-"},
    "decision_directed_pll": {"color": "#7c3aed", "marker": "D", "linestyle": "-"},
    "oracle": {"color": "#059669", "marker": "^", "linestyle": "-"},
    "learned_artifact": {"color": "#dc2626", "marker": "x", "linestyle": "-"},
}
EXPECTED_SNRS = (-2.0, 2.0, 6.0, 10.0)
BITS_PER_PACKET = 1024
PACKETS_PER_RUN = 2
BITS_PER_RUN = BITS_PER_PACKET * PACKETS_PER_RUN


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    method: str
    snr_db: float


RUN_SPECS = (
    RunSpec("20260723T222123Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_-2__receiver.mode_uncompensated", "uncompensated", -2),
    RunSpec("20260723T222124Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_-2__receiver.mode_pilot_smoothing", "pilot_smoothing", -2),
    RunSpec("20260723T222125Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_-2__receiver.mode_decision_directed_pll", "decision_directed_pll", -2),
    RunSpec("20260723T222126Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_-2__receiver.mode_oracle", "oracle", -2),
    RunSpec("20260723T222128Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_-2__receiver.mode_learned_artifact", "learned_artifact", -2),
    RunSpec("20260723T222129Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_2__receiver.mode_uncompensated", "uncompensated", 2),
    RunSpec("20260723T222130Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_2__receiver.mode_pilot_smoothing", "pilot_smoothing", 2),
    RunSpec("20260723T222131Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_2__receiver.mode_decision_directed_pll", "decision_directed_pll", 2),
    RunSpec("20260723T222132Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_2__receiver.mode_oracle", "oracle", 2),
    RunSpec("20260723T222133Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_2__receiver.mode_learned_artifact", "learned_artifact", 2),
    RunSpec("20260723T222134Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_6__receiver.mode_uncompensated", "uncompensated", 6),
    RunSpec("20260723T222136Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_6__receiver.mode_pilot_smoothing", "pilot_smoothing", 6),
    RunSpec("20260723T222137Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_6__receiver.mode_decision_directed_pll", "decision_directed_pll", 6),
    RunSpec("20260723T222138Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_6__receiver.mode_oracle", "oracle", 6),
    RunSpec("20260723T222140Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_6__receiver.mode_learned_artifact", "learned_artifact", 6),
    RunSpec("20260723T222141Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_10__receiver.mode_uncompensated", "uncompensated", 10),
    RunSpec("20260723T222143Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_10__receiver.mode_pilot_smoothing", "pilot_smoothing", 10),
    RunSpec("20260723T222144Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_10__receiver.mode_decision_directed_pll", "decision_directed_pll", 10),
    RunSpec("20260723T222145Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_10__receiver.mode_oracle", "oracle", 10),
    RunSpec("20260723T222147Z_qpsk_pilot_phase_tracking_2048_snapshot__channel.snr_db_10__receiver.mode_learned_artifact", "learned_artifact", 10),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the QPSK phase-tracking tutorial snapshot."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument(
        "--import-runs",
        type=Path,
        metavar="RUNS_DIR",
        help="rebuild the source CSV and provenance manifest from the named run store",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when committed derived assets differ from a fresh generation",
    )
    args = parser.parse_args(argv)
    if args.check and args.import_runs is not None:
        parser.error("--check and --import-runs cannot be combined")

    if args.import_runs is not None:
        import_snapshot(args.import_runs, args.source, args.manifest)

    records = load_records(args.source)
    validate_manifest(args.manifest, args.source, records)
    if args.check:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = generate(
                records, root / "data", root / "assets", manifest_path=args.manifest
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
                "derived assets are stale: %s; rerun without --check"
                % ", ".join(sorted(stale))
            )
        return 0

    generate(records, args.data_dir, args.asset_dir, manifest_path=args.manifest)
    return 0


def import_snapshot(runs_root: Path, source: Path, manifest_path: Path) -> None:
    source.parent.mkdir(parents=True, exist_ok=True)
    run_records: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    learned_hashes: set[str] = set()
    learned_packages: set[str] = set()

    for spec in RUN_SPECS:
        run_dir = runs_root / spec.run_id
        summary = _load_json(run_dir / "summary.json", "run summary")
        recipe = _load_json(run_dir / "recipe.json", "effective recipe")
        if summary.get("run_id") != spec.run_id or summary.get("status") != "completed":
            raise ValueError("run is missing, incomplete, or mismatched: %s" % spec.run_id)

        steps = {str(step["id"]): step for step in summary.get("steps", [])}
        recipe_steps = {str(step["id"]): step for step in recipe.get("steps", [])}
        demodulator = steps["demodulator"]
        mode = str(demodulator.get("metadata", {}).get("receiver_mode") or "")
        if mode != spec.method:
            raise ValueError(
                "%s declares receiver mode %s, expected %s"
                % (spec.run_id, mode, spec.method)
            )
        measured_snr = float(steps["wireless_channel"]["metrics"]["channel.snr_db"])
        if not math.isclose(measured_snr, spec.snr_db, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("%s has unexpected SNR" % spec.run_id)

        for step in summary.get("steps", []):
            for metric, value in sorted(step.get("metrics", {}).items()):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        "%s contains nonnumeric scalar metric %s" % (spec.run_id, metric)
                    )
                scalar_rows.append(
                    {
                        "Method": spec.method,
                        "Run": spec.run_id,
                        "SNR (dB)": _number(spec.snr_db),
                        "Step": str(step["id"]),
                        "Metric": str(metric),
                        "Value": _number(value),
                    }
                )

        coded_ber = steps["coded_ber"]["metrics"]
        coded_bler = steps["coded_bler"]["metrics"]
        compared_bits = int(coded_ber["channel.coded.compare_bit_count"])
        if compared_bits != BITS_PER_RUN:
            raise ValueError(
                "%s compares %d bits, expected %d"
                % (spec.run_id, compared_bits, BITS_PER_RUN)
            )

        data_seed = int(recipe_steps["data"]["params"]["seed"])
        packet_bits = int(recipe_steps["data"]["params"]["bit_count"])
        packet_count = int(recipe_steps["data"]["params"]["batch_size"])
        block_size = int(recipe_steps["coded_bler"]["params"]["block_size"])
        compared_blocks = int(coded_bler["channel.coded.block_count"])
        if (
            packet_bits != BITS_PER_PACKET
            or packet_count != PACKETS_PER_RUN
            or block_size != BITS_PER_PACKET
            or compared_blocks != PACKETS_PER_RUN
        ):
            raise ValueError(
                "%s does not preserve the %d packets x %d bits snapshot contract"
                % (spec.run_id, PACKETS_PER_RUN, BITS_PER_PACKET)
            )
        wireless_seed = int(recipe_steps["wireless_channel"]["params"]["seed"])
        carrier_seed = int(recipe_steps["carrier_impairment"]["params"]["seed"])
        pilot_seed = int(recipe_steps["modulator"]["params"]["pilot_seed"])
        artifact_manifest_sha = ""
        artifact_package_sha = ""
        if spec.method == "learned_artifact":
            bits_metadata = demodulator["outputs"]["bits"]["metadata"]
            artifact_manifest_sha = str(
                bits_metadata["trained_artifact_manifest_sha256"]
            )
            artifact_package_sha = str(
                recipe_steps["demodulator"]["params"]["artifact_package_sha256"]
            )
            learned_hashes.add(artifact_manifest_sha)
            learned_packages.add(artifact_package_sha)

        run_records.append(
            {
                "run_id": spec.run_id,
                "method": spec.method,
                "snr_db": spec.snr_db,
                "status": "completed",
                "completed_at_utc": summary.get("completed_at_utc"),
                "authored_recipe_sha256": summary.get("authored_recipe_sha256"),
                "effective_recipe_sha256": summary.get("effective_recipe_sha256"),
                "seeds": {
                    "data": data_seed,
                    "wireless_channel": wireless_seed,
                    "carrier_impairment": carrier_seed,
                    "pilot_sequence": pilot_seed,
                },
                "bit_errors": int(coded_ber["channel.coded.error_count"]),
                "compared_bits": compared_bits,
                "ber": float(coded_ber["channel.coded.ber"]),
                "block_errors": int(coded_bler["channel.coded.block_error_count"]),
                "compared_blocks": compared_blocks,
                "bler": float(coded_bler["channel.coded.bler"]),
                "upstream_artifact_sha256": {
                    "payload_bits": str(steps["data"]["outputs"]["bits"]["sha256"]),
                    "transmitted_symbols": str(
                        steps["modulator"]["outputs"]["symbols"]["sha256"]
                    ),
                    "pilot_context": str(
                        steps["modulator"]["outputs"]["pilot_context"]["sha256"]
                    ),
                    "awgn_rx_symbols": str(
                        steps["wireless_channel"]["outputs"]["rx_symbols"]["sha256"]
                    ),
                    "carrier_impaired_rx_symbols": str(
                        steps["carrier_impairment"]["outputs"]["rx_symbols"]["sha256"]
                    ),
                    "carrier_phase_truth": str(
                        steps["carrier_impairment"]["outputs"]["phase_truth"]["sha256"]
                    ),
                },
                **(
                    {
                        "trained_artifact_manifest_sha256": artifact_manifest_sha,
                        "trained_artifact_package_sha256": artifact_package_sha,
                    }
                    if artifact_manifest_sha
                    else {}
                ),
            }
        )

    if len(learned_hashes) != 1 or len(learned_packages) != 1:
        raise ValueError("learned runs do not share one artifact identity")
    _validate_pairing(run_records)

    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(scalar_rows)

    manifest = {
        "schema_version": 1,
        "kind": "noema.authored_demo_result_snapshot",
        "source": {
            "path": source.name,
            "sha256": _sha256(source),
            "data_row_count": len(scalar_rows),
        },
        "evidence_level": "illustrative_paired_snapshot",
        "scope": {
            "methods": list(METHOD_ORDER),
            "snr_db": list(EXPECTED_SNRS),
            "bits_per_run": BITS_PER_RUN,
            "bits_per_packet": BITS_PER_PACKET,
            "packets_per_run": PACKETS_PER_RUN,
            "pairing": "identical payload, AWGN, carrier-impairment, and pilot seeds at each SNR",
        },
        "limitations": [
            "ordinary recipe-matrix outputs, not a noema benchmark result bundle",
            "one paired seed realization and two 1,024-bit packets per method/SNR point",
            "too few bits for tight confidence intervals or a publication-strength method ranking",
            "pilot interpolation was not included in this five-method snapshot",
        ],
        "learned_artifact": {
            "manifest_sha256": next(iter(learned_hashes)),
            "package_sha256": next(iter(learned_packages)),
            "phase_truth_forwarded_to_runtime": False,
        },
        "runs": run_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_records(source: Path) -> list[dict[str, Any]]:
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_FIELDS:
            raise ValueError("unexpected snapshot CSV columns: %r" % (reader.fieldnames,))
        rows = list(reader)

    by_run: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = str(row["Run"])
        method = str(row["Method"])
        snr_db = float(row["SNR (dB)"])
        record = by_run.setdefault(
            run_id,
            {"method": method, "run_id": run_id, "snr_db": snr_db, "metrics": {}},
        )
        if record["method"] != method or record["snr_db"] != snr_db:
            raise ValueError("run %s has inconsistent coordinates" % run_id)
        selector = (str(row["Step"]), str(row["Metric"]))
        if selector in {
            ("coded_ber", "channel.coded.error_count"),
            ("coded_ber", "channel.coded.compare_bit_count"),
            ("coded_ber", "channel.coded.ber"),
            ("coded_bler", "channel.coded.block_error_count"),
            ("coded_bler", "channel.coded.block_count"),
            ("coded_bler", "channel.coded.bler"),
        }:
            record["metrics"][selector] = float(row["Value"])

    records: list[dict[str, Any]] = []
    for record in by_run.values():
        metrics = record.pop("metrics")
        required = {
            ("coded_ber", "channel.coded.error_count"),
            ("coded_ber", "channel.coded.compare_bit_count"),
            ("coded_ber", "channel.coded.ber"),
            ("coded_bler", "channel.coded.block_error_count"),
            ("coded_bler", "channel.coded.block_count"),
            ("coded_bler", "channel.coded.bler"),
        }
        if set(metrics) != required:
            raise ValueError("run %s is missing BER/BLER evidence" % record["run_id"])
        records.append(
            {
                **record,
                "bit_errors": int(metrics[("coded_ber", "channel.coded.error_count")]),
                "compared_bits": int(metrics[("coded_ber", "channel.coded.compare_bit_count")]),
                "ber": metrics[("coded_ber", "channel.coded.ber")],
                "block_errors": int(metrics[("coded_bler", "channel.coded.block_error_count")]),
                "compared_blocks": int(metrics[("coded_bler", "channel.coded.block_count")]),
                "bler": metrics[("coded_bler", "channel.coded.bler")],
            }
        )
    records.sort(key=lambda row: (METHOD_ORDER.index(row["method"]), row["snr_db"]))
    _validate_pairing(records)
    return records


def validate_manifest(
    manifest_path: Path, source: Path, records: list[dict[str, Any]]
) -> None:
    manifest = _load_json(manifest_path, "snapshot manifest")
    source_record = manifest.get("source", {})
    if source_record.get("sha256") != _sha256(source):
        raise ValueError("snapshot manifest does not bind the source CSV")
    with source.open("r", encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    if source_record.get("data_row_count") != row_count:
        raise ValueError("snapshot manifest has the wrong source row count")
    manifest_runs = {run["run_id"]: run for run in manifest.get("runs", [])}
    if set(manifest_runs) != {record["run_id"] for record in records}:
        raise ValueError("snapshot manifest and source CSV list different runs")
    for record in records:
        run = manifest_runs[record["run_id"]]
        for field in (
            "method",
            "snr_db",
            "bit_errors",
            "compared_bits",
            "ber",
            "block_errors",
            "compared_blocks",
            "bler",
        ):
            if run[field] != record[field]:
                raise ValueError("snapshot manifest disagrees on %s" % field)
    _validate_pairing(list(manifest_runs.values()))


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
    manifest = _load_json(manifest_path, "snapshot manifest")
    run_manifest = {run["run_id"]: run for run in manifest["runs"]}
    _write_summary(paths["summary_table"], records, run_manifest)
    _write_ber_table(paths["ber_table"], records)
    _write_ber_plot(paths["ber_plot"], records)
    return paths


def output_paths(data_dir: Path, asset_dir: Path) -> dict[str, Path]:
    return {
        "summary_table": data_dir / "summary_table.csv",
        "ber_table": data_dir / "ber_table.csv",
        "ber_plot": asset_dir / "ber_vs_snr.svg",
    }


def _write_summary(
    path: Path,
    records: Iterable[Mapping[str, Any]],
    manifest_runs: Mapping[str, Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            provenance = manifest_runs[str(record["run_id"])]
            seeds = provenance["seeds"]
            writer.writerow(
                {
                    **{field: record[field] for field in SUMMARY_FIELDS if field in record},
                    "authored_recipe_sha256": provenance["authored_recipe_sha256"],
                    "effective_recipe_sha256": provenance["effective_recipe_sha256"],
                    "data_seed": seeds["data"],
                    "wireless_seed": seeds["wireless_channel"],
                    "carrier_impairment_seed": seeds["carrier_impairment"],
                    "trained_artifact_manifest_sha256": provenance.get(
                        "trained_artifact_manifest_sha256", ""
                    ),
                }
            )


def _write_ber_table(path: Path, records: list[dict[str, Any]]) -> None:
    grouped = _by_snr(records)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "SNR (dB)",
                "Uncompensated BER",
                "Pilot smoothing BER",
                "Decision-directed PLL BER",
                "True-phase oracle BER",
                "Learned receiver BER",
            )
        )
        for snr, methods in grouped.items():
            writer.writerow(
                (_number(snr),)
                + tuple("%.6f" % methods[method]["ber"] for method in METHOD_ORDER)
            )


def _write_ber_plot(path: Path, records: list[dict[str, Any]]) -> None:
    matplotlib.rcParams["svg.hashsalt"] = "noema-qpsk-phase-tracking-snapshot-v2"
    display_floor = 0.5 / BITS_PER_RUN
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
            points = [record for record in records if record["method"] == method]
            style = METHOD_STYLES[method]
            plotted_ber = [max(float(record["ber"]), display_floor) for record in points]
            axis.plot(
                [record["snr_db"] for record in points],
                plotted_ber,
                label=METHOD_LABELS[method],
                linewidth=1.9,
                markersize=6.2,
                markeredgewidth=1.4,
                **style,
            )
            for record, y_value in zip(points, plotted_ber):
                if float(record["ber"]) == 0.0:
                    axis.annotate(
                        "0/%d" % BITS_PER_RUN,
                        (record["snr_db"], y_value),
                        xytext=(-6, 8),
                        textcoords="offset points",
                        ha="right",
                        fontsize=8,
                        color=style["color"],
                    )
        axis.set_yscale("log")
        axis.set_ylim(min(3e-4, 0.75 * display_floor), 0.8)
        axis.set_xticks(EXPECTED_SNRS)
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel("Data-bit error rate")
        axis.grid(True, which="both", color="#d1d5db", linewidth=0.7, alpha=0.72)
        axis.set_axisbelow(True)
        axis.margins(x=0.045)
        axis.legend(frameon=False, ncols=2, loc="lower left")
        figure.savefig(
            path,
            format="svg",
            metadata={
                "Date": None,
                "Creator": "Noema QPSK phase-tracking demo asset generator",
            },
            facecolor="white",
        )
        plt.close(figure)


def _validate_pairing(records: list[Mapping[str, Any]]) -> None:
    if len(records) != len(METHOD_ORDER) * len(EXPECTED_SNRS):
        raise ValueError("expected exactly 20 method/SNR runs")
    grouped = _by_snr(records)
    if tuple(grouped) != EXPECTED_SNRS:
        raise ValueError("snapshot does not contain the expected SNR coordinates")
    for snr, methods in grouped.items():
        if set(methods) != set(METHOD_ORDER):
            raise ValueError("SNR %g does not contain all five methods" % snr)
        for record in methods.values():
            if int(record["compared_bits"]) != BITS_PER_RUN:
                raise ValueError("snapshot contains an unexpected BER denominator")
            expected_ber = int(record["bit_errors"]) / BITS_PER_RUN
            if not math.isclose(float(record["ber"]), expected_ber, abs_tol=1e-15):
                raise ValueError("stored BER does not match error count")
        seed_records = [record.get("seeds") for record in methods.values()]
        if all(isinstance(seeds, Mapping) for seeds in seed_records):
            paired_seed_sets = {
                tuple(sorted((str(key), int(value)) for key, value in seeds.items()))
                for seeds in seed_records
            }
            if len(paired_seed_sets) != 1:
                raise ValueError("SNR %g does not use one paired seed set" % snr)
        artifact_records = [
            record.get("upstream_artifact_sha256") for record in methods.values()
        ]
        if all(isinstance(identity, Mapping) for identity in artifact_records):
            paired_artifacts = {
                json.dumps(dict(identity), sort_keys=True, separators=(",", ":"))
                for identity in artifact_records
            }
            if len(paired_artifacts) != 1:
                raise ValueError(
                    "SNR %g does not use byte-identical upstream artifacts" % snr
                )


def _by_snr(
    records: Iterable[Mapping[str, Any]],
) -> dict[float, dict[str, Mapping[str, Any]]]:
    grouped: dict[float, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(float(record["snr_db"]), {})[str(record["method"])] = record
    return dict(sorted(grouped.items()))


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("%s does not exist: %s" % (label, path))
    payload = decode_strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("%s must contain a JSON object" % label)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: int | float) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return format(float(value), ".17g")


if __name__ == "__main__":
    raise SystemExit(main())
