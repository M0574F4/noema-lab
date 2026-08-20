#!/usr/bin/env python3
"""Development-only paired SNR sweep for the delayed-CSI NR-PUSCH pilot.

This script does not alter or reuse the pilot's existing outcome files. Each
SNR/unit combination runs in a fresh process, and the same two unit seeds are
reused across SNR values to make the waterfall comparison paired.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = Path(__file__).with_name("pilot.py")
OUTPUT = ROOT / ".noema" / "demos" / "delayed_csi_nr_pusch_link_snr_sweep_v2"

SNR_VALUES = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0)
UNIT_SEEDS = (282000001, 282000002)
TRAJECTORIES_PER_UNIT = 16
DESIGN = "broad_v2"
MAXIMUM_PEAK_RSS_BYTES = 3 * 1024**3
MAXIMUM_UNIT_SECONDS = 180.0
METHODS = (
    "equal_power",
    "stale_csi_box_water_filling",
    "causal_ar_box_water_filling",
    "learned_transfer_seed35023",
    "current_csi_box_water_filling_diagnostic",
)


class SweepError(RuntimeError):
    pass


def _activate_design(name: str) -> None:
    global DESIGN, OUTPUT, SNR_VALUES, UNIT_SEEDS, TRAJECTORIES_PER_UNIT
    if name == "broad_v2":
        DESIGN = name
        OUTPUT = ROOT / ".noema" / "demos" / "delayed_csi_nr_pusch_link_snr_sweep_v2"
        SNR_VALUES = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0)
        UNIT_SEEDS = (282000001, 282000002)
        TRAJECTORIES_PER_UNIT = 16
    elif name == "focused_v1":
        DESIGN = name
        OUTPUT = ROOT / ".noema" / "demos" / "delayed_csi_nr_pusch_link_snr_refinement_v1"
        SNR_VALUES = (6.25, 6.5, 6.75)
        UNIT_SEEDS = tuple(range(283000001, 283000007))
        TRAJECTORIES_PER_UNIT = 16
    else:
        raise SweepError("unknown sweep design")


def _snr_token(snr_db: float) -> str:
    return ("%.1f" % snr_db).replace("-", "m").replace(".", "p")


def _unit_path(snr_db: float, unit_index: int) -> Path:
    return OUTPUT / "units" / (
        "snr_%s_unit_%02d.json" % (_snr_token(snr_db), unit_index + 1)
    )


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pilot() -> Any:
    spec = importlib.util.spec_from_file_location("noema_nr_pusch_pilot_for_sweep", PILOT_PATH)
    if spec is None or spec.loader is None:
        raise SweepError("could not load pilot implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _noise_variance_at_snr(snr_db: float) -> float:
    energy_per_information_bit = 1800.0 / 1608.0
    return energy_per_information_bit / (10.0 ** (float(snr_db) / 10.0))


def self_test() -> None:
    if not SNR_VALUES or tuple(sorted(SNR_VALUES)) != SNR_VALUES:
        raise SweepError("unexpected SNR grid")
    if len(set(UNIT_SEEDS)) != len(UNIT_SEEDS) or min(UNIT_SEEDS) <= 281000005:
        raise SweepError("sweep seeds overlap or are not fresh")
    if DESIGN == "broad_v2" and (
        len(SNR_VALUES) != 9 or SNR_VALUES[0] != 3.0 or SNR_VALUES[-1] != 7.0
    ):
        raise SweepError("broad SNR design changed")
    if DESIGN == "focused_v1" and (
        SNR_VALUES != (6.25, 6.5, 6.75) or len(UNIT_SEEDS) != 6
    ):
        raise SweepError("focused SNR design changed")
    pilot = _load_pilot()
    if float(pilot.CONFIG["pusch"]["ebno_db"]) != 5.0:
        raise SweepError("base pilot coordinate changed")
    if not math.isclose(pilot._noise_variance(), _noise_variance_at_snr(5.0)):
        raise SweepError("base pilot 5 dB noise mapping changed")
    if tuple(pilot.CONFIG["methods"]) != METHODS:
        raise SweepError("pilot method roster changed")
    print("self-test passed design=%s" % DESIGN)


def worker(snr_db: float, unit_index: int) -> None:
    if snr_db not in SNR_VALUES:
        raise SweepError("SNR is outside the fixed development grid")
    if unit_index < 0 or unit_index >= len(UNIT_SEEDS):
        raise SweepError("invalid unit index")
    path = _unit_path(snr_db, unit_index)
    if path.exists():
        print("existing unit retained: %s" % path)
        return

    pilot = _load_pilot()
    pilot.CONFIG["pusch"]["ebno_db"] = float(snr_db)
    # The fixed-coordinate pilot intentionally encodes 5 dB in its helper.
    # Override that helper only inside this fresh development-sweep worker.
    pilot._noise_variance = lambda: _noise_variance_at_snr(snr_db)
    if not math.isclose(pilot._noise_variance(), _noise_variance_at_snr(snr_db)):
        raise SweepError("sweep SNR-to-noise mapping did not activate")
    result = pilot._run_batch(UNIT_SEEDS[unit_index], TRAJECTORIES_PER_UNIT)
    constraints = result["constraints"]
    guard = (
        float(result["elapsed_seconds"]) <= MAXIMUM_UNIT_SECONDS
        and int(result["peak_rss_bytes"]) <= MAXIMUM_PEAK_RSS_BYTES
        and float(constraints["maximum_RB_sum_power_error"]) <= 1e-9
        and float(constraints["maximum_lower_power_violation"]) <= 1e-9
        and float(constraints["maximum_upper_power_violation"]) <= 1e-9
        and float(constraints["maximum_paired_waveform_energy_relative_error"]) <= 1e-5
    )
    result.update(
        {
            "kind": "delayed_csi_nr_pusch_link_snr_sweep_unit",
            "scope": "development_only_not_publication_evidence",
            "supersedes_invalid_sweep": (
                "delayed_csi_nr_pusch_link_snr_sweep_v1, whose worker changed the "
                "configuration label but not the fixed 5 dB noise helper"
            ),
            "snr_db": float(snr_db),
            "unit_index": unit_index + 1,
            "unit_seed": UNIT_SEEDS[unit_index],
            "resource_guard_passed": bool(guard),
        }
    )
    _write(path, result)
    print(
        "snr=%.1f unit=%d n=%d wall=%.2fs peak_mib=%.1f corr=%.3f guard=%s"
        % (
            snr_db,
            unit_index + 1,
            TRAJECTORIES_PER_UNIT,
            float(result["elapsed_seconds"]),
            int(result["peak_rss_bytes"]) / 1024**2,
            float(result["correlations"]["delayed_current_gain_correlation"]),
            guard,
        )
    )
    if not guard:
        raise SweepError("resource or power guard failed")


def run() -> None:
    self_test()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for snr_db in SNR_VALUES:
        for unit_index in range(len(UNIT_SEEDS)):
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(Path(__file__).resolve()),
                    "worker",
                    "--snr-db",
                    str(snr_db),
                    "--unit-index",
                    str(unit_index),
                    "--design",
                    DESIGN,
                ],
                cwd=ROOT,
                check=True,
            )
    analyze()


def _method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    values = [row["methods"][method] for row in rows]
    attempts = len(values)
    deliveries = sum(bool(value["crc_pass"]) for value in values)
    bit_errors = [int(value["bit_errors"]) for value in values]
    return {
        "attempts": attempts,
        "crc_deliveries": deliveries,
        "bler": 1.0 - deliveries / attempts,
        "mean_decoded_bit_errors": sum(bit_errors) / attempts,
        "total_decoded_bit_errors": sum(bit_errors),
    }


def _contrast(rows: list[dict[str, Any]], comparator: str) -> dict[str, Any]:
    learned = [row["methods"]["learned_transfer_seed35023"] for row in rows]
    reference = [row["methods"][comparator] for row in rows]
    delivery_deltas = [
        int(bool(left["crc_pass"])) - int(bool(right["crc_pass"]))
        for left, right in zip(learned, reference, strict=True)
    ]
    bit_error_deltas = [
        int(left["bit_errors"]) - int(right["bit_errors"])
        for left, right in zip(learned, reference, strict=True)
    ]
    return {
        "delivery_count_difference": sum(delivery_deltas),
        "delivery_rate_difference": sum(delivery_deltas) / len(delivery_deltas),
        "paired_delivery_wins": sum(delta > 0 for delta in delivery_deltas),
        "paired_delivery_ties": sum(delta == 0 for delta in delivery_deltas),
        "paired_delivery_losses": sum(delta < 0 for delta in delivery_deltas),
        "mean_decoded_bit_error_difference": sum(bit_error_deltas) / len(bit_error_deltas),
    }


def analyze() -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for snr_db in SNR_VALUES:
        paths = [_unit_path(snr_db, index) for index in range(len(UNIT_SEEDS))]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise SweepError("missing unit(s): %s" % ", ".join(missing))
        units = [_read(path) for path in paths]
        rows = [row for unit in units for row in unit["rows"]]
        methods = {method: _method_summary(rows, method) for method in METHODS}
        contrasts = {
            "learned_minus_equal_power": _contrast(rows, "equal_power"),
            "learned_minus_stale_csi_box_water_filling": _contrast(
                rows, "stale_csi_box_water_filling"
            ),
            "learned_minus_causal_ar_box_water_filling": _contrast(
                rows, "causal_ar_box_water_filling"
            ),
            "learned_minus_current_csi_box_water_filling_diagnostic": _contrast(
                rows, "current_csi_box_water_filling_diagnostic"
            ),
        }
        points.append(
            {
                "snr_db": snr_db,
                "trajectory_count": len(rows),
                "mean_delayed_current_gain_correlation": sum(
                    float(unit["correlations"]["delayed_current_gain_correlation"])
                    for unit in units
                )
                / len(units),
                "methods": methods,
                "contrasts": contrasts,
                "resource_guards_passed": all(
                    bool(unit["resource_guard_passed"]) for unit in units
                ),
            }
        )

    candidates = []
    for point in points:
        equal_bler = float(point["methods"]["equal_power"]["bler"])
        learned_equal = point["contrasts"]["learned_minus_equal_power"]
        oracle_equal = (
            int(point["methods"]["current_csi_box_water_filling_diagnostic"]["crc_deliveries"])
            - int(point["methods"]["equal_power"]["crc_deliveries"])
        )
        stale = point["contrasts"]["learned_minus_stale_csi_box_water_filling"]
        causal = point["contrasts"]["learned_minus_causal_ar_box_water_filling"]
        if DESIGN == "focused_v1":
            promising = (
                0.15 <= equal_bler <= 0.85
                and int(learned_equal["delivery_count_difference"]) >= 2
                and int(stale["delivery_count_difference"]) > 0
                and int(causal["delivery_count_difference"]) > 0
                and int(learned_equal["paired_delivery_wins"])
                > int(learned_equal["paired_delivery_losses"])
                and int(stale["paired_delivery_wins"]) > int(stale["paired_delivery_losses"])
                and int(causal["paired_delivery_wins"]) > int(causal["paired_delivery_losses"])
            )
        else:
            promising = (
                0.15 <= equal_bler <= 0.85
                and int(learned_equal["delivery_count_difference"]) > 0
                and int(learned_equal["paired_delivery_wins"])
                > int(learned_equal["paired_delivery_losses"])
                and oracle_equal > 0
            )
        if promising:
            candidates.append(float(point["snr_db"]))

    report = {
        "schema_version": 1,
        "kind": "delayed_csi_nr_pusch_link_snr_sweep_analysis",
        "design": DESIGN,
        "scope": "post-pilot development-only SNR waterfall; not publication evidence",
        "supersedes_invalid_sweep": (
            "delayed_csi_nr_pusch_link_snr_sweep_v1; all of its nominal SNR points "
            "executed the pilot's fixed 5 dB noise variance"
        ),
        "snr_values_db": list(SNR_VALUES),
        "unit_seeds": list(UNIT_SEEDS),
        "trajectories_per_snr": len(UNIT_SEEDS) * TRAJECTORIES_PER_UNIT,
        "paired_across_snr": True,
        "points": points,
        "promising_snr_candidates": candidates,
        "decision": (
            "candidate_coordinate_found_for_new_prospective_design"
            if candidates
            else "no_candidate_coordinate_in_sweep"
        ),
        "interpretation_rule": (
            (
                "The focused development gate requires at least two more learned deliveries than "
                "equal power, positive gains over both water-filling methods, and more paired wins "
                "than losses for all three contrasts."
                if DESIGN == "focused_v1"
                else "The broad development gate requires non-extreme equal-power BLER, positive "
                "learned-minus-equal deliveries with more paired wins than losses, and a positive "
                "current-CSI diagnostic gain."
            )
            + " Any confirmatory study must use a new frozen protocol and fresh seeds."
        ),
        "selection_disclosure": (
            "The focused 6.25--6.75 dB grid was selected after inspecting the broad v2 sweep."
            if DESIGN == "focused_v1"
            else "The broad 3--7 dB grid was fixed before its corrected v2 outcomes were accessed."
        ),
    }
    _write(OUTPUT / "analysis.json", report)

    denominator = len(UNIT_SEEDS) * TRAJECTORIES_PER_UNIT
    print("SNR   equal  stale  causal  learned  current  L-E  L-S  L-C  mean(L-E bit errors)")
    for point in points:
        methods = point["methods"]
        contrasts = point["contrasts"]
        print(
            "%4.2f  %2d/%d  %2d/%d  %2d/%d  %2d/%d   %2d/%d   %+3d  %+3d  %+3d  %+7.2f"
            % (
                point["snr_db"],
                methods["equal_power"]["crc_deliveries"],
                denominator,
                methods["stale_csi_box_water_filling"]["crc_deliveries"],
                denominator,
                methods["causal_ar_box_water_filling"]["crc_deliveries"],
                denominator,
                methods["learned_transfer_seed35023"]["crc_deliveries"],
                denominator,
                methods["current_csi_box_water_filling_diagnostic"]["crc_deliveries"],
                denominator,
                contrasts["learned_minus_equal_power"]["delivery_count_difference"],
                contrasts["learned_minus_stale_csi_box_water_filling"][
                    "delivery_count_difference"
                ],
                contrasts["learned_minus_causal_ar_box_water_filling"][
                    "delivery_count_difference"
                ],
                contrasts["learned_minus_equal_power"][
                    "mean_decoded_bit_error_difference"
                ],
            )
        )
    print("decision=%s candidates=%s" % (report["decision"], candidates))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("self-test", "worker", "run", "analyze"))
    parser.add_argument("--snr-db", type=float)
    parser.add_argument("--unit-index", type=int)
    parser.add_argument("--design", choices=("broad_v2", "focused_v1"), default="broad_v2")
    args = parser.parse_args(argv)
    _activate_design(args.design)
    if args.command == "self-test":
        self_test()
    elif args.command == "worker":
        if args.snr_db is None or args.unit_index is None:
            raise SweepError("worker requires --snr-db and --unit-index")
        worker(args.snr_db, args.unit_index)
    elif args.command == "run":
        run()
    else:
        analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
