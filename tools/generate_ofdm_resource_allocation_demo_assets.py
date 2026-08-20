from __future__ import annotations

"""Generate the authored OFDM allocation tutorial assets from a UI CSV export.

The input file is kept unchanged as a source snapshot.  This script extracts a
small, documented projection for the tutorial so figures can be restyled later
without losing or manually transcribing the original metrics.
"""

import argparse
import csv
import math
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "docs"
    / "demo"
    / "data"
    / "ofdm_resource_allocation"
    / "current_results_all_metrics.csv"
)
DEFAULT_DATA_DIR = DEFAULT_INPUT.parent
DEFAULT_ASSET_DIR = (
    ROOT / "docs" / "demo" / "assets" / "ofdm_resource_allocation"
)

EXPECTED_COLUMNS = ("Recipe", "Run", "Variant", "Source", "Step", "Metric", "Value")
POLICY_ORDER = ("equal_power", "learned_allocator", "water_filling")
POLICY_LABELS = {
    "equal_power": "Equal power",
    "learned_allocator": "Learned allocator",
    "water_filling": "Water filling (Shannon oracle)",
}
POLICY_STYLES = {
    "equal_power": {"color": "#2563eb", "marker": "s", "zorder": 2},
    "learned_allocator": {"color": "#dc2626", "marker": "x", "zorder": 4},
    # A wide translucent line keeps the oracle visible when the learned curve
    # overlaps it to plotting precision.  It remains a measured solid series.
    "water_filling": {
        "color": "#059669",
        "marker": "o",
        "zorder": 3,
        "linewidth": 3.4,
        "alpha": 0.52,
        "markerfacecolor": "white",
    },
}

METRIC_FIELDS = {
    "average_transmit_power_budget": (
        "allocation_evaluation",
        "resource.average_transmit_power_budget",
    ),
    "noise_variance": ("allocation_evaluation", "channel.noise_variance"),
    "channel_gain_average": ("channel_state", "channel.gain.average"),
    "spectral_efficiency_bps_hz": (
        "allocation_evaluation",
        "resource.theoretical_shannon_spectral_efficiency_bps_hz",
    ),
    "payload_goodput_bits_per_resource_element": (
        "allocation_evaluation",
        "channel.achieved_payload_goodput_bits_per_resource_element",
    ),
    "scheduled_resource_element_count": (
        "allocation_evaluation",
        "channel.scheduled_resource_element_count",
    ),
    "data_bearing_resource_element_count": (
        "allocation_evaluation",
        "channel.data_bearing_resource_element_count",
    ),
    "resource_element_utilization": (
        "allocation_evaluation",
        "channel.resource_element_utilization",
    ),
    "relative_gap_to_water_filling": (
        "allocation_evaluation",
        "resource.water_filling_relative_optimality_gap",
    ),
    "max_power_budget_error": (
        "allocation_evaluation",
        "resource.power_constraint.max_abs_error",
    ),
    "negative_power_violation": (
        "allocation_evaluation",
        "resource.power_constraint.max_negative_violation",
    ),
    "payload_bler": ("payload_bler", "channel.payload.bler"),
    "allocator_wall_time_s": ("tx_power", "timing.step.wall_time_s"),
}

SUMMARY_FIELDS = (
    "policy",
    "run_id",
    "average_transmit_power_budget",
    "nominal_transmit_snr_db",
    "noise_variance",
    "channel_gain_average",
    "spectral_efficiency_bps_hz",
    "shannon_optimum_achievement_percent",
    "payload_bler",
    "payload_goodput_bits_per_resource_element",
    "scheduled_resource_element_count",
    "data_bearing_resource_element_count",
    "resource_element_utilization",
    "relative_gap_to_water_filling",
    "max_power_budget_error",
    "negative_power_violation",
    "allocator_wall_time_s",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the OFDM allocation tutorial tables and figures."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when committed derived assets differ from a fresh generation.",
    )
    args = parser.parse_args(argv)

    records = load_records(args.input)
    if args.check:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = generate(records, root / "data", root / "assets")
            expected = _output_paths(args.data_dir, args.asset_dir)
            mismatches = [
                name
                for name, path in generated.items()
                if not expected[name].is_file()
                or expected[name].read_bytes() != path.read_bytes()
            ]
        if mismatches:
            parser.error(
                "derived assets are stale: %s; rerun this command without --check"
                % ", ".join(sorted(mismatches))
            )
        return 0

    generate(records, args.data_dir, args.asset_dir)
    return 0


def load_records(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(
                "unexpected all-metrics CSV columns: %r" % (reader.fieldnames,)
            )
        rows = list(reader)

    by_run: dict[str, dict[str, object]] = {}
    selectors = {selector: field for field, selector in METRIC_FIELDS.items()}
    for row in rows:
        run_id = str(row["Run"])
        policy = _policy_id(str(row["Recipe"]))
        run = by_run.setdefault(
            run_id,
            {"policy": policy, "run_id": run_id, "metrics": {}},
        )
        if run["policy"] != policy:
            raise ValueError("run %s is assigned to multiple policies" % run_id)
        selector = (str(row["Step"]), str(row["Metric"]))
        field = selectors.get(selector)
        if field is None:
            continue
        metrics = run["metrics"]
        assert isinstance(metrics, dict)
        if field in metrics:
            raise ValueError("run %s repeats %s" % (run_id, selector[1]))
        try:
            value = float(row["Value"])
        except ValueError as exc:
            raise ValueError(
                "run %s has a nonnumeric value for %s" % (run_id, selector[1])
            ) from exc
        if not math.isfinite(value):
            raise ValueError("run %s has a non-finite %s" % (run_id, selector[1]))
        metrics[field] = value

    records: list[dict[str, float | str]] = []
    expected_fields = set(METRIC_FIELDS)
    for run_id, run in by_run.items():
        metrics = run["metrics"]
        assert isinstance(metrics, dict)
        missing = expected_fields - set(metrics)
        if missing:
            raise ValueError(
                "run %s is missing required metrics: %s"
                % (run_id, ", ".join(sorted(missing)))
            )
        power = float(metrics["average_transmit_power_budget"])
        noise = float(metrics["noise_variance"])
        if power <= 0 or noise <= 0:
            raise ValueError("power and noise variance must be positive")
        record: dict[str, float | str] = {
            "policy": str(run["policy"]),
            "run_id": run_id,
            **{field: float(metrics[field]) for field in expected_fields},
            "nominal_transmit_snr_db": 10.0 * math.log10(power / noise),
        }
        records.append(record)

    records.sort(
        key=lambda row: (
            float(row["average_transmit_power_budget"]),
            POLICY_ORDER.index(str(row["policy"])),
        )
    )
    _validate_campaign(records)
    for methods in _records_by_power(records).values():
        oracle = float(
            methods["water_filling"]["spectral_efficiency_bps_hz"]
        )
        if oracle <= 0.0:
            raise ValueError(
                "water-filling Shannon spectral efficiency must be positive"
            )
        for record in methods.values():
            achievement = (
                100.0
                * float(record["spectral_efficiency_bps_hz"])
                / oracle
            )
            if achievement > 100.0 + 1e-6:
                raise ValueError(
                    "policy %s exceeds the water-filling Shannon upper bound"
                    % record["policy"]
                )
            record["shannon_optimum_achievement_percent"] = min(
                achievement,
                100.0,
            )
    return records


def generate(
    records: list[dict[str, float | str]], data_dir: Path, asset_dir: Path
) -> dict[str, Path]:
    paths = _output_paths(data_dir, asset_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(paths["summary_table"], records)
    _write_spectral_efficiency_table(
        paths["spectral_efficiency_table"], records
    )
    _write_line_plot(
        paths["spectral_efficiency_plot"],
        records,
        y_field="spectral_efficiency_bps_hz",
        y_label="Shannon spectral efficiency (bit/s/Hz)",
        y_min_zero=False,
    )
    _write_line_plot(
        paths["shannon_optimum_achievement_plot"],
        records,
        y_field="shannon_optimum_achievement_percent",
        y_label="Water-filling Shannon optimum achieved (%)",
        y_min_zero=False,
    )
    return paths


def _output_paths(data_dir: Path, asset_dir: Path) -> dict[str, Path]:
    return {
        "summary_table": data_dir / "summary_table.csv",
        "spectral_efficiency_table": data_dir / "spectral_efficiency_table.csv",
        "spectral_efficiency_plot": (
            asset_dir / "spectral_efficiency_vs_nominal_snr.svg"
        ),
        "shannon_optimum_achievement_plot": (
            asset_dir / "shannon_optimum_achievement_vs_nominal_snr.svg"
        ),
    }


def _write_summary(
    path: Path, records: Iterable[Mapping[str, float | str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in SUMMARY_FIELDS})


def _write_spectral_efficiency_table(
    path: Path, records: list[dict[str, float | str]]
) -> None:
    by_power = _records_by_power(records)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "Nominal TX SNR (dB)",
                "Power budget",
                "Equal power (bit/s/Hz)",
                "Learned (bit/s/Hz)",
                "Water filling (bit/s/Hz)",
                "Learned gap (%)",
            )
        )
        for power, methods in by_power.items():
            learned = methods["learned_allocator"]
            writer.writerow(
                (
                    "%.2f" % float(learned["nominal_transmit_snr_db"]),
                    "%g" % power,
                    "%.6f"
                    % float(methods["equal_power"]["spectral_efficiency_bps_hz"]),
                    "%.6f" % float(learned["spectral_efficiency_bps_hz"]),
                    "%.6f"
                    % float(methods["water_filling"]["spectral_efficiency_bps_hz"]),
                    "%.8f"
                    % (100.0 * float(learned["relative_gap_to_water_filling"])),
                )
            )


def _write_line_plot(
    path: Path,
    records: list[dict[str, float | str]],
    *,
    y_field: str,
    y_label: str,
    y_min_zero: bool,
) -> None:
    matplotlib.rcParams["svg.hashsalt"] = "noema-ofdm-resource-allocation-v1"
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
        figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        figure.patch.set_facecolor("white")
        axis.set_facecolor("white")
        for policy in POLICY_ORDER:
            points = [row for row in records if row["policy"] == policy]
            style = dict(POLICY_STYLES[policy])
            axis.plot(
                [float(row["nominal_transmit_snr_db"]) for row in points],
                [float(row[y_field]) for row in points],
                label=POLICY_LABELS[policy],
                linestyle="-",
                linewidth=style.pop("linewidth", 1.9),
                markersize=6.2,
                markeredgewidth=1.4,
                **style,
            )
        axis.set_xlabel("Nominal transmit-SNR budget (dB)")
        axis.set_ylabel(y_label)
        axis.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.72)
        axis.set_axisbelow(True)
        axis.margins(x=0.045, y=0.10)
        if y_min_zero:
            _, upper = axis.get_ylim()
            axis.set_ylim(bottom=0.0, top=upper)
        axis.legend(frameon=False, ncols=3, loc="best")
        figure.savefig(
            path,
            format="svg",
            metadata={
                "Date": None,
                "Creator": "Noema OFDM demo asset generator",
            },
            facecolor="white",
        )
        plt.close(figure)


def _records_by_power(
    records: list[dict[str, float | str]],
) -> dict[float, dict[str, dict[str, float | str]]]:
    grouped: dict[float, dict[str, dict[str, float | str]]] = {}
    for record in records:
        grouped.setdefault(
            float(record["average_transmit_power_budget"]), {}
        )[str(record["policy"])] = record
    return dict(sorted(grouped.items()))


def _validate_campaign(records: list[dict[str, float | str]]) -> None:
    grouped = _records_by_power(records)
    if len(records) != 15 or len(grouped) != 5:
        raise ValueError(
            "expected 15 runs across five power budgets; found %d runs across %d budgets"
            % (len(records), len(grouped))
        )
    expected = set(POLICY_ORDER)
    all_noise_values = {
        float(record["noise_variance"]) for record in records
    }
    all_gain_averages = {
        float(record["channel_gain_average"]) for record in records
    }
    if len(all_noise_values) != 1 or not math.isclose(
        next(iter(all_noise_values)), 0.2, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("expected one globally fixed noise variance of 0.2")
    if len(all_gain_averages) != 1 or not math.isclose(
        next(iter(all_gain_averages)), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("expected channel gains normalized to unit average power")
    for power, methods in grouped.items():
        if set(methods) != expected:
            raise ValueError(
                "power budget %g has policies %s"
                % (power, ", ".join(sorted(methods)))
            )
        noise_values = {
            float(record["noise_variance"]) for record in methods.values()
        }
        if len(noise_values) != 1:
            raise ValueError("power budget %g does not share one noise variance" % power)


def _policy_id(label: str) -> str:
    normalized = label.strip().lower()
    if normalized == "equal power":
        return "equal_power"
    if normalized == "water filling":
        return "water_filling"
    if normalized.startswith("policy learned artifact"):
        return "learned_allocator"
    raise ValueError("unrecognized recipe/policy label: %s" % label)


if __name__ == "__main__":
    raise SystemExit(main())
