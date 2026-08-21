#!/usr/bin/env python3
"""Generate and verify Noema's single launch-evidence projection.

The projection is intentionally narrower than the paper.  It derives one
public, self-contained demonstration from the retained QPSK I/Q-calibration
benchmark projection and keeps scientific status separate from distribution
clearance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import statistics
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from noema_lab.core.study_io import (
    content_bound_document,
    file_sha256,
    verify_content_bound_document,
    write_study_json,
)
from noema_lab.core.structured_input import load_strict_yaml_or_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RELATIVE = Path("launch_evidence.json")
SCHEMA_RELATIVE = Path("schemas/launch_evidence.schema.json")
MANIFEST_RELATIVE = Path("docs/demo/data/qpsk_iq_calibration/benchmark_manifest.json")
PROJECTION_RELATIVE = Path(
    "docs/demo/data/qpsk_iq_calibration/benchmark_projection.csv"
)
RELEASE_IDENTITY_RELATIVE = Path("release_identity.yaml")
PUBLICATION_HANDOFF_RELATIVE = Path("publication_handoff.yaml")
PUBLICATION_HANDOFF_SCHEMA_RELATIVE = Path("schemas/publication_handoff.schema.json")
GENERATOR_RELATIVE = Path("tools/generate_launch_evidence.py")
UPSTREAM_GENERATOR_RELATIVE = Path("tools/generate_qpsk_iq_calibration_demo_assets.py")

RESULT_ID = "20260726T155403Z_neural_receiver_ai_phy.learned_qpsk_iq_calibration_v1"
BENCHMARK_ID = "neural_receiver_ai_phy.learned_qpsk_iq_calibration_v1"
BENCHMARK_VERSION = "1.0.0"
COMPLETED_AT_UTC = "2026-07-26T15:56:42.211Z"
EXPERIMENT_ID = "learned_qpsk_demapper_demo"
EXPERIMENT_TITLE = "Learned QPSK Receiver Calibration"
METHODS = (
    {
        "id": "uncompensated_qpsk",
        "label": "Uncompensated QPSK",
        "role": "baseline",
    },
    {
        "id": "calibrated_iq_oracle",
        "label": "Calibrated I/Q oracle",
        "role": "reference",
    },
    {
        "id": "learned_receiver",
        "label": "Learned I/Q receiver",
        "role": "candidate",
    },
)
METHOD_BY_ID = {item["id"]: item for item in METHODS}
METHOD_ORDER = tuple(METHOD_BY_ID)
EXPECTED_SNRS = (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
EXPECTED_SEEDS = (71001, 72001, 73001)
BITS_PER_RUN = 1_048_576
BLOCKS_PER_RUN = 1_024
RUN_COUNT = len(METHODS) * len(EXPECTED_SNRS) * len(EXPECTED_SEEDS)
EXPECTED_PROJECTION_SHA256 = (
    "1a1211d8aa2f45293111c135d4d669f8496610dfc41ded446ebbb3ed4ed6985a"
)
EXPECTED_PROJECTION_SIZE_BYTES = 30_342
EXPECTED_MANIFEST_SHA256 = (
    "3961df773f103ea2b126d29edaa8711148939a2181f309ce4c76f8538c28152a"
)
EXPECTED_MANIFEST_SIZE_BYTES = 2_328
EXPECTED_UPSTREAM_GENERATOR_SHA256 = (
    "13210569905a0ad567cdc6b422fe3d0c129ee311e2d3e3ba0638b535587d1f94"
)
EXPECTED_UPSTREAM_GENERATOR_SIZE_BYTES = 19_328
EXPECTED_OBSERVATION_GRID_SHA256 = (
    "b3744246c697c9c452caccd6423ba979c58f680a2e241c780293a4b041bc3bba"
)
EXPECTED_ORIGIN_DIGESTS = {
    "result_json_sha256": "aa051f9e893ace504f0b5748dd5319c1cf2897f0e3fa97f4ff926281460ed781",
    "benchmark_json_sha256": "e98630aaaf83bacfad82a7366e2f8556e0985ed01aa148bc2cae0aa615f077ca",
    "metrics_csv_sha256": "ad33417be0cf1a8a58dcecde305d4e501199cb798ff3ef495fc6724156784451",
    "benchmark_semantic_sha256": "8dbfe6298ccee25b5490781ed8fa312026de1b598914055a396b60e816a960eb",
    "trained_artifact_runtime_identity_sha256": "b6fc5cf1fdf6cd06771169a8ed8f22e176247e3f4340071bc418a2ed6a3af762",
}

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
EXPECTED_WARNING = (
    "This completed experimental demonstration is not yet a publication-ready "
    "canonical benchmark."
)
EXPECTED_LIMITATIONS = (
    "The committed CSV is a deterministic projection of the local completed "
    "result, not a replacement for its full run-evidence snapshot.",
    "Three paired held-out seeds support a demonstration comparison but are not "
    "a universal publication-strength sample size.",
)
DISCLOSURE = (
    "Completed experimental demonstration—not a publication-ready canonical "
    "benchmark."
)
FINAL_RELEASE_CLEARANCE_SCOPE = "stable_v0.2.0_and_paper_bundle"
PUBLIC_REPOSITORY_STATUS = "public_development_preview"
REFERENCE_ROLE = (
    "The calibrated I/Q oracle is a diagnostic reference with calibration "
    "knowledge, not a deployable competitor using the same information."
)
CANONICAL_SURFACES = (
    "readme",
    "documentation",
    "webpage",
    "experiment_figure",
    "result_table",
    "video_overlay",
)
CONSUMER_RULES = (
    "Derive every displayed number from series, comparisons, or headline.",
    "Display scientific_status.disclosure beside any quantitative claim.",
    "Describe point bands as observed minima and maxima, never confidence intervals.",
    "Never use distribution clearance to upgrade the experiment's scientific tier.",
)
STATISTICAL_UNIT = "paired held-out seed within a predeclared SNR cell"
PRESENTATION_STATUS = "canonical_asset_generated"
LAUNCH_VIDEO_ID = "bKNXS_vHLHc"
LAUNCH_VIDEO_WATCH_URL = f"https://www.youtube.com/watch?v={LAUNCH_VIDEO_ID}"
LAUNCH_VIDEO_EMBED_URL = (
    f"https://www.youtube-nocookie.com/embed/{LAUNCH_VIDEO_ID}"
)
LAUNCH_VIDEO_TITLE = (
    "Learned QPSK I/Q calibration receiver — complete Noema workflow"
)
LAUNCH_ASSET_GENERATOR_RELATIVE = Path("tools/generate_launch_assets.py")
CANONICAL_FIGURE_RELATIVE = Path(
    "docs/_static/launch/f3-receiver-ber-vs-snr.svg"
)
CANONICAL_TABLE_RELATIVES = (
    Path("docs/_static/launch/tables/t0-result-summary.csv"),
    Path("docs/_static/launch/tables/t0-result-summary.md"),
)


class LaunchEvidenceError(ValueError):
    """Raised when the launch projection or one of its bindings is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchEvidenceError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    _require(
        actual == expected,
        f"{label} fields differ: missing={sorted(expected - actual)}, "
        f"unexpected={sorted(actual - expected)}",
    )


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        payload = load_strict_yaml_or_json(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise LaunchEvidenceError(f"cannot load {label}: {exc}") from exc
    return _mapping(payload, label)


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value or "")
    _require(
        len(digest) == 64
        and digest == digest.lower()
        and all(char in "0123456789abcdef" for char in digest),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return digest


def _safe_relative_path(value: Any, label: str) -> str:
    path = str(value or "")
    candidate = PurePosixPath(path)
    _require(
        bool(path)
        and "\\" not in path
        and not candidate.is_absolute()
        and bool(candidate.parts)
        and path == candidate.as_posix()
        and all(part not in {"", ".", ".."} for part in candidate.parts),
        f"{label} must be a safe POSIX-relative path",
    )
    _require(
        candidate.parts[0] != "paper",
        f"{label} must not depend on the paper tree",
    )
    return path


def _validate_binding(value: Any, label: str) -> dict[str, Any]:
    binding = _mapping(value, label)
    _exact_keys(binding, {"path", "sha256", "size_bytes"}, label)
    binding["path"] = _safe_relative_path(binding["path"], f"{label}.path")
    binding["sha256"] = _require_sha256(binding["sha256"], f"{label}.sha256")
    _require(
        isinstance(binding["size_bytes"], int)
        and not isinstance(binding["size_bytes"], bool)
        and binding["size_bytes"] >= 0,
        f"{label}.size_bytes must be a non-negative integer",
    )
    return binding


def _binding(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    _require(
        path.is_file() and not path.is_symlink(),
        f"bound file is missing or unsafe: {relative}",
    )
    return {
        "path": _safe_relative_path(relative.as_posix(), "bound path"),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _frozen_binding(relative: Path, *, sha256: str, size_bytes: int) -> dict[str, Any]:
    return {
        "path": relative.as_posix(),
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _validate_manifest(
    manifest: Mapping[str, Any], manifest_path: Path, projection_path: Path
) -> dict[str, Any]:
    _exact_keys(
        manifest,
        {
            "schema_version",
            "kind",
            "evidence_level",
            "result",
            "projection",
            "scope",
            "verification",
            "limitations",
        },
        "benchmark manifest",
    )
    _require(manifest.get("schema_version") == 1, "manifest schema version changed")
    _require(
        manifest.get("kind") == "noema.completed_benchmark_projection",
        "manifest kind changed",
    )
    _require(
        manifest.get("evidence_level") == "completed_experimental_benchmark",
        "manifest evidence level was removed or upgraded",
    )

    result = _mapping(manifest.get("result"), "manifest result")
    _exact_keys(
        result,
        {
            "id",
            "benchmark_id",
            "benchmark_version",
            "status",
            "created_at_utc",
            "completed_at_utc",
            "result_json_sha256",
            "benchmark_json_sha256",
            "metrics_csv_sha256",
            "benchmark_semantic_sha256",
        },
        "manifest result",
    )
    _require(result.get("id") == RESULT_ID, "manifest result ID changed")
    _require(result.get("benchmark_id") == BENCHMARK_ID, "benchmark ID changed")
    _require(
        result.get("benchmark_version") == BENCHMARK_VERSION,
        "benchmark version changed",
    )
    _require(result.get("status") == "completed", "benchmark is not completed")
    for key in (
        "result_json_sha256",
        "benchmark_json_sha256",
        "metrics_csv_sha256",
        "benchmark_semantic_sha256",
    ):
        _require_sha256(result.get(key), f"manifest result.{key}")
    for key in ("created_at_utc", "completed_at_utc"):
        _require(
            isinstance(result.get(key), str) and result[key],
            f"manifest result.{key} is missing",
        )
    _require(
        result.get("completed_at_utc") == COMPLETED_AT_UTC,
        "benchmark completion identity changed",
    )

    projection = _mapping(manifest.get("projection"), "manifest projection")
    _exact_keys(
        projection,
        {"path", "sha256", "row_count", "fields"},
        "manifest projection",
    )
    _require(projection.get("path") == projection_path.name, "projection path changed")
    _require(projection.get("row_count") == RUN_COUNT, "projection row count changed")
    _require(
        tuple(projection.get("fields") or ()) == SOURCE_FIELDS,
        "projection fields changed",
    )
    declared_projection_sha = _require_sha256(
        projection.get("sha256"), "manifest projection.sha256"
    )
    _require(
        declared_projection_sha == file_sha256(projection_path),
        "manifest does not bind the projection bytes",
    )
    _require(
        declared_projection_sha == EXPECTED_PROJECTION_SHA256
        and projection_path.stat().st_size == EXPECTED_PROJECTION_SIZE_BYTES,
        "frozen launch projection bytes changed",
    )

    scope = _mapping(manifest.get("scope"), "manifest scope")
    _exact_keys(
        scope,
        {
            "methods",
            "snr_db",
            "paired_seeds",
            "bits_per_run",
            "runs_per_method_and_snr",
            "trained_artifact_runtime_identity_sha256",
        },
        "manifest scope",
    )
    _require(
        tuple(scope.get("methods") or ()) == METHOD_ORDER, "manifest methods changed"
    )
    _require(
        tuple(scope.get("snr_db") or ()) == EXPECTED_SNRS, "manifest SNR grid changed"
    )
    _require(
        tuple(scope.get("paired_seeds") or ()) == EXPECTED_SEEDS,
        "manifest seed grid changed",
    )
    _require(scope.get("bits_per_run") == BITS_PER_RUN, "bits-per-run changed")
    _require(
        scope.get("runs_per_method_and_snr") == len(EXPECTED_SEEDS),
        "runs-per-cell changed",
    )
    _require_sha256(
        scope.get("trained_artifact_runtime_identity_sha256"),
        "manifest trained-artifact runtime identity",
    )
    _require(
        _origin_digests(result, scope) == EXPECTED_ORIGIN_DIGESTS,
        "manifest-recorded origin identities changed",
    )

    verification = _mapping(manifest.get("verification"), "manifest verification")
    _exact_keys(
        verification,
        {"status", "integrity_errors", "warnings"},
        "manifest verification",
    )
    _require(
        verification.get("status") == "warning", "verification warning status changed"
    )
    _require(
        verification.get("integrity_errors") == [], "manifest has integrity errors"
    )
    _require(
        verification.get("warnings") == [EXPECTED_WARNING],
        "mandatory experimental warning changed",
    )
    _require(
        tuple(manifest.get("limitations") or ()) == EXPECTED_LIMITATIONS,
        "mandatory experimental limitations changed",
    )
    _require(
        file_sha256(manifest_path) == EXPECTED_MANIFEST_SHA256
        and manifest_path.stat().st_size == EXPECTED_MANIFEST_SIZE_BYTES,
        "selected benchmark manifest bytes changed",
    )
    return {"result": result, "scope": scope, "verification": verification}


def _parse_int(value: Any, label: str) -> int:
    text = str(value)
    _require(text and text.lstrip("-").isdigit(), f"{label} must be an integer")
    return int(text)


def _parse_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LaunchEvidenceError(f"{label} must be numeric") from exc
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _snr_token(snr_db: float) -> str:
    value = _number(snr_db)
    return f"snrm{value[1:]}" if value.startswith("-") else f"snr{value}"


def _expected_recipe_id(method: str, snr_db: float, seed: int) -> str:
    return f"{method}_{_snr_token(snr_db)}_seed{seed}"


def _read_records(path: Path) -> list[dict[str, Any]]:
    _require(
        path.is_file() and not path.is_symlink(),
        "benchmark projection is missing or unsafe",
    )
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(
                tuple(reader.fieldnames or ()) == SOURCE_FIELDS,
                "projection CSV fields changed",
            )
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise LaunchEvidenceError(f"cannot read projection CSV: {exc}") from exc
    _require(len(raw_rows) == RUN_COUNT, f"projection must contain {RUN_COUNT} rows")

    records: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows, start=2):
        label = f"projection row {index}"
        _require(set(row) == set(SOURCE_FIELDS), f"{label} fields changed")
        method = str(row["method"])
        _require(method in METHOD_BY_ID, f"{label} has an unexpected method")
        expected_role = METHOD_BY_ID[method]["role"]
        _require(row["role"] == expected_role, f"{label} method role changed")
        seed = _parse_int(row["paired_seed"], f"{label}.paired_seed")
        snr_db = _parse_float(row["snr_db"], f"{label}.snr_db")
        bit_errors = _parse_int(row["bit_errors"], f"{label}.bit_errors")
        compared_bits = _parse_int(row["compared_bits"], f"{label}.compared_bits")
        block_errors = _parse_int(row["block_errors"], f"{label}.block_errors")
        compared_blocks = _parse_int(row["compared_blocks"], f"{label}.compared_blocks")
        ber = _parse_float(row["ber"], f"{label}.ber")
        bler = _parse_float(row["bler"], f"{label}.bler")
        _require(row["result_id"] == RESULT_ID, f"{label} result ID changed")
        _require(seed in EXPECTED_SEEDS, f"{label} seed is outside the frozen grid")
        _require(snr_db in EXPECTED_SNRS, f"{label} SNR is outside the frozen grid")
        _require(compared_bits == BITS_PER_RUN, f"{label} compared-bit count changed")
        _require(
            compared_blocks == BLOCKS_PER_RUN, f"{label} compared-block count changed"
        )
        _require(0 <= bit_errors <= compared_bits, f"{label} bit errors are invalid")
        _require(
            0 <= block_errors <= compared_blocks, f"{label} block errors are invalid"
        )
        expected_ber = bit_errors / compared_bits
        expected_bler = block_errors / compared_blocks
        _require(
            math.isclose(ber, expected_ber, rel_tol=0.0, abs_tol=1e-15),
            f"{label} BER does not match integer counts",
        )
        _require(
            math.isclose(bler, expected_bler, rel_tol=0.0, abs_tol=1e-15),
            f"{label} BLER does not match integer counts",
        )
        recipe_id = str(row["recipe_id"])
        run_id = str(row["run_id"])
        expected_recipe_id = _expected_recipe_id(method, snr_db, seed)
        _require(
            recipe_id == expected_recipe_id,
            f"{label} recipe ID does not match its method/SNR/seed coordinate",
        )
        _require(
            row["recipe_label"]
            == f"{METHOD_BY_ID[method]['label']} · {_number(snr_db)} dB · seed {seed}",
            f"{label} recipe label does not match its coordinate",
        )
        _require(
            run_id.endswith(f"__{expected_recipe_id}")
            and len(run_id) > len(expected_recipe_id) + 2,
            f"{label} run ID does not match its recipe coordinate",
        )
        records.append(
            {
                "method": method,
                "role": expected_role,
                "paired_seed": seed,
                "snr_db": snr_db,
                "recipe_id": recipe_id,
                "run_id": run_id,
                "bit_errors": bit_errors,
                "compared_bits": compared_bits,
                "ber": expected_ber,
                "block_errors": block_errors,
                "compared_blocks": compared_blocks,
                "bler": expected_bler,
                "recipe_sha256": _require_sha256(
                    row["recipe_sha256"], f"{label}.recipe_sha256"
                ),
                "semantic_recipe_sha256": _require_sha256(
                    row["semantic_recipe_sha256"],
                    f"{label}.semantic_recipe_sha256",
                ),
            }
        )

    method_index = {method: index for index, method in enumerate(METHOD_ORDER)}
    ordered = sorted(
        records,
        key=lambda row: (
            row["snr_db"],
            row["paired_seed"],
            method_index[row["method"]],
        ),
    )
    _require(records == ordered, "projection row order differs from the frozen order")
    coordinates = {
        (row["method"], row["snr_db"], row["paired_seed"]) for row in records
    }
    expected_coordinates = {
        (method, snr, seed)
        for method in METHOD_ORDER
        for snr in EXPECTED_SNRS
        for seed in EXPECTED_SEEDS
    }
    _require(
        coordinates == expected_coordinates, "projection coordinate grid is incomplete"
    )
    _require(
        len({row["recipe_id"] for row in records}) == RUN_COUNT,
        "projection recipe IDs are not unique",
    )
    _require(
        len({row["run_id"] for row in records}) == RUN_COUNT,
        "projection run IDs are not unique",
    )
    return records


def _series(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = list(records)
    result: list[dict[str, Any]] = []
    for method in METHODS:
        points: list[dict[str, Any]] = []
        for snr_db in EXPECTED_SNRS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method["id"] and row["snr_db"] == snr_db
                ),
                key=lambda row: row["paired_seed"],
            )
            _require(len(selected) == len(EXPECTED_SEEDS), "series cell is incomplete")
            observations = [
                {
                    key: row[key]
                    for key in (
                        "paired_seed",
                        "recipe_id",
                        "run_id",
                        "bit_errors",
                        "compared_bits",
                        "ber",
                        "block_errors",
                        "compared_blocks",
                        "bler",
                        "recipe_sha256",
                        "semantic_recipe_sha256",
                    )
                }
                for row in selected
            ]
            values = [row["ber"] for row in observations]
            points.append(
                {
                    "snr_db": snr_db,
                    "observations": observations,
                    "summary": {
                        "mean_ber": statistics.fmean(values),
                        "minimum_ber": min(values),
                        "maximum_ber": max(values),
                    },
                }
            )
        result.append({**method, "points": points})
    return result


def _observation_grid_sha256(series: Iterable[Mapping[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for item in series:
        method = str(item["id"])
        for point in item["points"]:
            for observation in point["observations"]:
                rows.append(
                    {
                        "method": method,
                        "snr_db": point["snr_db"],
                        **dict(observation),
                    }
                )
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _comparisons(series: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_method = {str(item["id"]): item for item in series}
    rows: list[dict[str, Any]] = []
    for point_index, snr_db in enumerate(EXPECTED_SNRS):
        baseline = by_method["uncompensated_qpsk"]["points"][point_index]["summary"]
        reference = by_method["calibrated_iq_oracle"]["points"][point_index]["summary"]
        learned = by_method["learned_receiver"]["points"][point_index]["summary"]
        baseline_mean = float(baseline["mean_ber"])
        reference_mean = float(reference["mean_ber"])
        learned_mean = float(learned["mean_ber"])
        _require(
            baseline_mean > 0 and reference_mean > 0,
            "relative comparison denominator is zero",
        )
        rows.append(
            {
                "snr_db": snr_db,
                "uncompensated_mean_ber": baseline_mean,
                "calibrated_reference_mean_ber": reference_mean,
                "learned_mean_ber": learned_mean,
                "learned_minimum_ber": float(learned["minimum_ber"]),
                "learned_maximum_ber": float(learned["maximum_ber"]),
                "learned_reduction_percent": 100.0
                * (baseline_mean - learned_mean)
                / baseline_mean,
                "learned_relative_gap_to_reference_percent": 100.0
                * (learned_mean - reference_mean)
                / reference_mean,
            }
        )
    return rows


def _headline(comparisons: list[Mapping[str, Any]]) -> dict[str, Any]:
    primary = max(comparisons, key=lambda row: float(row["snr_db"]))
    reductions = [float(row["learned_reduction_percent"]) for row in comparisons]
    max_gap = max(
        abs(float(row["learned_relative_gap_to_reference_percent"]))
        for row in comparisons
    )
    primary_reduction = float(primary["learned_reduction_percent"])
    if primary_reduction >= 0:
        effect = "reduced"
        magnitude = primary_reduction
    else:
        effect = "increased"
        magnitude = abs(primary_reduction)
    text = (
        f"At {float(primary['snr_db']):g} dB, the learned receiver {effect} mean "
        f"pre-decoder BER by {magnitude:.2f}% relative to uncompensated QPSK."
    )
    accessible = (
        f"Across seven predeclared SNR cells, the observed learned-versus-"
        f"uncompensated BER change ranged from {min(reductions):.2f}% to "
        f"{max(reductions):.2f}%; the largest absolute relative gap between the "
        f"learned receiver and calibrated diagnostic reference was {max_gap:.2f}%."
    )
    return {
        "text": text,
        "accessible_summary": accessible,
        "selection_rules": {
            "primary_cell": "highest_predeclared_snr",
            "reference_gap": "maximum_absolute_relative_gap_across_all_predeclared_cells",
        },
        "primary_snr_db": float(primary["snr_db"]),
        "primary_learned_reduction_percent": primary_reduction,
        "observed_reduction_range_percent": {
            "minimum": min(reductions),
            "maximum": max(reductions),
        },
        "maximum_absolute_relative_gap_to_reference_percent": max_gap,
    }


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _presentation_contract() -> dict[str, Any]:
    return {
        "figure": {
            "status": PRESENTATION_STATUS,
            "path": CANONICAL_FIGURE_RELATIVE.as_posix(),
            "generator_path": LAUNCH_ASSET_GENERATOR_RELATIVE.as_posix(),
            "id": "qpsk_iq_calibration_ber",
            "title": "Receiver BER vs SNR",
            "type": "line_with_observed_range",
            "x_axis": "SNR (dB)",
            "y_axis": "Pre-decoder bit error rate (log scale)",
            "data_path": "series",
        },
        "table": {
            "status": PRESENTATION_STATUS,
            "paths": [item.as_posix() for item in CANONICAL_TABLE_RELATIVES],
            "generator_path": LAUNCH_ASSET_GENERATOR_RELATIVE.as_posix(),
            "id": "qpsk_iq_calibration_ber_table",
            "title": "Paired QPSK I/Q-calibration BER summary",
            "data_path": "comparisons",
        },
        "video_overlay": {
            "status": "recorded_external_video",
            "provider": "youtube",
            "video_id": LAUNCH_VIDEO_ID,
            "watch_url": LAUNCH_VIDEO_WATCH_URL,
            "embed_url": LAUNCH_VIDEO_EMBED_URL,
            "title": LAUNCH_VIDEO_TITLE,
            "data_paths": [
                "headline",
                "series",
                "scientific_status.disclosure",
            ],
        },
    }


def _release_state(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    release = _load_mapping(root / RELEASE_IDENTITY_RELATIVE, "release identity")
    _require(
        release.get("kind") == "noema.release_identity", "release identity kind changed"
    )
    _require(
        release.get("identity_status") == "frozen", "release identity is not frozen"
    )
    software = _mapping(release.get("software"), "release software identity")
    for key in ("display_name", "current_version", "target_version", "release_stage"):
        _require(
            isinstance(software.get(key), str) and software[key],
            f"release software.{key} is missing",
        )
    stability = _mapping(release.get("stability"), "release stability")
    _require(
        "locally verified static result pages"
        in (stability.get("experimental_surfaces") or []),
        "release identity no longer classifies static result pages as experimental",
    )

    handoff = _load_mapping(root / PUBLICATION_HANDOFF_RELATIVE, "publication handoff")
    handoff_schema = _load_mapping(
        root / PUBLICATION_HANDOFF_SCHEMA_RELATIVE,
        "publication-handoff schema",
    )
    handoff_schema_errors = sorted(
        Draft202012Validator(handoff_schema).iter_errors(handoff),
        key=lambda item: list(item.absolute_path),
    )
    _require(
        not handoff_schema_errors,
        "publication-handoff schema errors: "
        + "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '$'}: "
            f"{error.message}"
            for error in handoff_schema_errors
        ),
    )
    _require(handoff.get("kind") == "noema.publication_handoff", "handoff kind changed")
    handoff_status = str(handoff.get("handoff_status") or "")
    _require(
        handoff_status in {"review_in_progress", "ready_for_release"},
        "handoff status is unsupported",
    )
    rights = _mapping(handoff.get("rights"), "handoff rights")
    overall_rights = str(rights.get("review_status") or "")
    _require(
        overall_rights in {"required", "cleared"},
        "overall rights status is unsupported",
    )
    rights_items = rights.get("items")
    _require(isinstance(rights_items, list), "handoff rights items must be a list")
    rights_item_ids = [str(item.get("id") or "") for item in rights_items]
    _require(
        len(rights_item_ids) == len(set(rights_item_ids)),
        "handoff rights item IDs are not unique",
    )
    matching = [
        _mapping(item, "launch-evidence rights item")
        for item in rights_items
        if isinstance(item, Mapping) and item.get("id") == "launch_evidence_snapshot"
    ]
    _require(
        len(matching) == 1,
        "handoff must contain one launch_evidence_snapshot rights item",
    )
    launch_rights = matching[0]
    _require(
        launch_rights.get("paths") == [OUTPUT_RELATIVE.as_posix()]
        and launch_rights.get("channels")
        == ["public_repository", "documentation_site", "software_archive"]
        and launch_rights.get("disposition") == "include",
        "launch-evidence rights scope changed",
    )
    launch_rights_status = str(launch_rights.get("status") or "")
    _require(
        launch_rights_status in {"review_required", "cleared"},
        "launch-evidence rights status is unsupported",
    )
    gate = _mapping(handoff.get("release_gate"), "handoff release gate")
    _require(
        gate.get("scope") == FINAL_RELEASE_CLEARANCE_SCOPE,
        "handoff release-gate scope changed",
    )
    gate_blocked = gate.get("blocked")
    _require(isinstance(gate_blocked, bool), "handoff release gate is malformed")
    paper = _mapping(handoff.get("paper"), "handoff paper")
    paper_review_status = str(paper.get("review_status") or "")
    paper_binding_status = (
        "declared" if isinstance(paper.get("final_binding"), Mapping) else "missing"
    )
    final_review = rights.get("final_review")
    final_review_status = (
        "approved"
        if isinstance(final_review, Mapping)
        and final_review.get("decision") == "approved"
        else "missing"
    )
    included_items = [
        item for item in rights_items if item.get("disposition") == "include"
    ]
    included_rights_status = (
        "all_cleared"
        if included_items
        and all(item.get("status") == "cleared" for item in included_items)
        else "pending"
    )
    unresolved_review = any(
        item.get("status") == "review_required" for item in rights_items
    )
    cleared = (
        handoff_status == "ready_for_release"
        and paper_review_status == "finalized"
        and paper_binding_status == "declared"
        and overall_rights == "cleared"
        and final_review_status == "approved"
        and included_rights_status == "all_cleared"
        and not unresolved_review
        and launch_rights_status == "cleared"
        and gate_blocked is False
    )
    clearance = {
        "status": (
            "cleared_for_stable_release" if cleared else "pending_final_review"
        ),
        "clearance_scope": FINAL_RELEASE_CLEARANCE_SCOPE,
        "public_repository_status": PUBLIC_REPOSITORY_STATUS,
        "handoff_status": handoff_status,
        "paper_review_status": paper_review_status,
        "paper_final_binding_status": paper_binding_status,
        "overall_rights_status": overall_rights,
        "rights_final_review_status": final_review_status,
        "included_rights_status": included_rights_status,
        "launch_asset_rights_status": launch_rights_status,
        "stable_release_gate_blocked": gate_blocked,
    }
    return software, clearance


def _origin_digests(
    result: Mapping[str, Any], scope: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "result_json_sha256": str(result["result_json_sha256"]),
        "benchmark_json_sha256": str(result["benchmark_json_sha256"]),
        "metrics_csv_sha256": str(result["metrics_csv_sha256"]),
        "benchmark_semantic_sha256": str(result["benchmark_semantic_sha256"]),
        "trained_artifact_runtime_identity_sha256": str(
            scope["trained_artifact_runtime_identity_sha256"]
        ),
    }


def build_projection(root: Path = ROOT) -> dict[str, Any]:
    """Build the deterministic launch projection from checked source inputs."""

    root = Path(root).resolve()
    manifest_path = root / MANIFEST_RELATIVE
    projection_path = root / PROJECTION_RELATIVE
    manifest = _load_mapping(manifest_path, "QPSK benchmark manifest")
    manifest_parts = _validate_manifest(manifest, manifest_path, projection_path)
    records = _read_records(projection_path)
    series = _series(records)
    _require(
        _observation_grid_sha256(series) == EXPECTED_OBSERVATION_GRID_SHA256,
        "frozen launch observation grid changed",
    )
    comparisons = _comparisons(series)
    software, clearance = _release_state(root)
    result = manifest_parts["result"]
    scope = manifest_parts["scope"]
    verification = manifest_parts["verification"]
    payload = {
        "kind": "noema.launch_evidence_projection",
        "schema_version": 1,
        "projection_id": "noema.qpsk_iq_calibration.launch.v1",
        "role": "launch_demonstration",
        "scientific_status": {
            "evidence_level": "completed_experimental_benchmark",
            "benchmark_tier": "experimental",
            "verification_status": "warning",
            "publication_ready": False,
            "disclosure": DISCLOSURE,
            "warnings": list(verification["warnings"]),
            "limitations": list(manifest["limitations"]),
            "aggregation": "arithmetic_mean_across_paired_seeds",
            "uncertainty_display": "observed_min_max_not_confidence_interval",
            "reference_role": REFERENCE_ROLE,
        },
        "distribution_clearance": clearance,
        "identity": {
            "project": str(software["display_name"]),
            "current_version": str(software["current_version"]),
            "target_version": str(software["target_version"]),
            "experiment_id": EXPERIMENT_ID,
            "title": EXPERIMENT_TITLE,
            "tutorial_path": "docs/tutorials/learned_qpsk_demapper_demo.md",
            "result_id": str(result["id"]),
            "benchmark_id": str(result["benchmark_id"]),
            "benchmark_version": str(result["benchmark_version"]),
            "result_status": str(result["status"]),
            "completed_at_utc": str(result["completed_at_utc"]),
            "manifest_recorded_origin_digests": _origin_digests(result, scope),
        },
        "design": {
            "methods": [dict(item) for item in METHODS],
            "snr_db": list(EXPECTED_SNRS),
            "paired_seeds": list(EXPECTED_SEEDS),
            "bits_per_run": BITS_PER_RUN,
            "blocks_per_run": BLOCKS_PER_RUN,
            "runs_per_method_and_snr": len(EXPECTED_SEEDS),
            "run_count": RUN_COUNT,
            "statistical_unit": STATISTICAL_UNIT,
            "frozen_observation_grid_sha256": EXPECTED_OBSERVATION_GRID_SHA256,
        },
        "series": series,
        "comparisons": comparisons,
        "headline": _headline(comparisons),
        "presentation": _presentation_contract(),
        "consumer_contract": {
            "canonical_for": list(CANONICAL_SURFACES),
            "rules": list(CONSUMER_RULES),
        },
        "provenance": {
            "verified_inputs": [
                _binding(root, relative)
                for relative in (
                    PROJECTION_RELATIVE,
                    MANIFEST_RELATIVE,
                    PUBLICATION_HANDOFF_RELATIVE,
                    PUBLICATION_HANDOFF_SCHEMA_RELATIVE,
                    RELEASE_IDENTITY_RELATIVE,
                )
            ],
            "generator": _binding(root, GENERATOR_RELATIVE),
            "upstream_generator": _binding(root, UPSTREAM_GENERATOR_RELATIVE),
            "schema": _binding(root, SCHEMA_RELATIVE),
        },
    }
    projection = content_bound_document(payload)
    verify_projection(projection, root=root)
    return projection


def _schema_errors(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> list[str]:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda item: list(item.absolute_path),
    )
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '$'}: {error.message}"
        for error in errors
    ]


def _close(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def verify_projection(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    """Verify schema, self hash, coordinate closure, and all derived arithmetic."""

    root = Path(root).resolve()
    schema = _load_mapping(root / SCHEMA_RELATIVE, "launch-evidence schema")
    errors = _schema_errors(payload, schema)
    _require(not errors, "launch-evidence schema errors: " + "; ".join(errors))
    try:
        verified = verify_content_bound_document(payload, "launch evidence")
    except ValueError as exc:
        raise LaunchEvidenceError(str(exc)) from exc

    scientific = _mapping(verified["scientific_status"], "scientific status")
    _require(scientific["warnings"] == [EXPECTED_WARNING], "launch warning changed")
    _require(
        tuple(scientific["limitations"]) == EXPECTED_LIMITATIONS,
        "launch limitations changed",
    )
    _require(scientific["disclosure"] == DISCLOSURE, "launch disclosure changed")
    _require(
        scientific["reference_role"] == REFERENCE_ROLE,
        "reference-role disclosure changed",
    )

    clearance = _mapping(verified["distribution_clearance"], "distribution clearance")
    should_be_cleared = (
        clearance["clearance_scope"] == FINAL_RELEASE_CLEARANCE_SCOPE
        and clearance["public_repository_status"] == PUBLIC_REPOSITORY_STATUS
        and clearance["handoff_status"] == "ready_for_release"
        and clearance["paper_review_status"] == "finalized"
        and clearance["paper_final_binding_status"] == "declared"
        and clearance["overall_rights_status"] == "cleared"
        and clearance["rights_final_review_status"] == "approved"
        and clearance["included_rights_status"] == "all_cleared"
        and clearance["launch_asset_rights_status"] == "cleared"
        and clearance["stable_release_gate_blocked"] is False
    )
    _require(
        (clearance["status"] == "cleared_for_stable_release") is should_be_cleared,
        "distribution-clearance state is inconsistent",
    )

    design = _mapping(verified["design"], "launch design")
    _require(
        design["methods"] == [dict(item) for item in METHODS], "method design changed"
    )
    _require(tuple(design["snr_db"]) == EXPECTED_SNRS, "SNR design changed")
    _require(tuple(design["paired_seeds"]) == EXPECTED_SEEDS, "seed design changed")
    _require(design["bits_per_run"] == BITS_PER_RUN, "bit denominator changed")
    _require(design["blocks_per_run"] == BLOCKS_PER_RUN, "block denominator changed")
    _require(
        design["runs_per_method_and_snr"] == len(EXPECTED_SEEDS),
        "runs-per-cell denominator changed",
    )
    _require(design["run_count"] == RUN_COUNT, "run denominator changed")
    _require(design["statistical_unit"] == STATISTICAL_UNIT, "statistical unit changed")
    _require(
        design["frozen_observation_grid_sha256"] == EXPECTED_OBSERVATION_GRID_SHA256,
        "frozen observation-grid identity changed",
    )

    series = verified["series"]
    _require(isinstance(series, list), "series must be a list")
    _require(
        [item.get("id") for item in series] == list(METHOD_ORDER),
        "series order changed",
    )
    recipe_ids: set[str] = set()
    run_ids: set[str] = set()
    recomputed_series: list[dict[str, Any]] = []
    for method, raw_series in zip(METHODS, series):
        item = _mapping(raw_series, f"series {method['id']}")
        _require(
            {key: item[key] for key in ("id", "label", "role")} == method,
            f"series identity changed for {method['id']}",
        )
        points = item["points"]
        _require(
            [point.get("snr_db") for point in points] == list(EXPECTED_SNRS),
            f"{method['id']} point order changed",
        )
        for point in points:
            observations = point["observations"]
            _require(
                [row.get("paired_seed") for row in observations]
                == list(EXPECTED_SEEDS),
                f"{method['id']} seed order changed",
            )
            values: list[float] = []
            for observation in observations:
                row = _mapping(observation, "launch observation")
                expected_recipe_id = _expected_recipe_id(
                    str(method["id"]),
                    float(point["snr_db"]),
                    int(row["paired_seed"]),
                )
                _require(
                    row["recipe_id"] == expected_recipe_id,
                    "observation recipe ID does not match its coordinate",
                )
                _require(
                    str(row["run_id"]).endswith(f"__{expected_recipe_id}"),
                    "observation run ID does not match its coordinate",
                )
                _require(
                    row["compared_bits"] == BITS_PER_RUN,
                    "observation bit denominator changed",
                )
                _require(
                    row["compared_blocks"] == BLOCKS_PER_RUN,
                    "observation block denominator changed",
                )
                _require(
                    0 <= row["bit_errors"] <= BITS_PER_RUN,
                    "observation bit errors are invalid",
                )
                _require(
                    0 <= row["block_errors"] <= BLOCKS_PER_RUN,
                    "observation block errors are invalid",
                )
                expected_ber = row["bit_errors"] / BITS_PER_RUN
                expected_bler = row["block_errors"] / BLOCKS_PER_RUN
                _require(
                    _close(row["ber"], expected_ber),
                    "observation BER arithmetic changed",
                )
                _require(
                    _close(row["bler"], expected_bler),
                    "observation BLER arithmetic changed",
                )
                recipe_id = str(row["recipe_id"])
                run_id = str(row["run_id"])
                _require(
                    recipe_id not in recipe_ids, "launch recipe IDs are not unique"
                )
                _require(run_id not in run_ids, "launch run IDs are not unique")
                recipe_ids.add(recipe_id)
                run_ids.add(run_id)
                values.append(expected_ber)
            summary = point["summary"]
            _require(
                _close(summary["mean_ber"], statistics.fmean(values)),
                "series mean changed",
            )
            _require(
                _close(summary["minimum_ber"], min(values)), "series minimum changed"
            )
            _require(
                _close(summary["maximum_ber"], max(values)), "series maximum changed"
            )
        recomputed_series.append(item)
    _require(
        len(recipe_ids) == len(run_ids) == RUN_COUNT, "launch observation count changed"
    )
    _require(
        _observation_grid_sha256(recomputed_series) == EXPECTED_OBSERVATION_GRID_SHA256,
        "embedded observations do not match the frozen launch grid",
    )

    expected_comparisons = _comparisons(recomputed_series)
    actual_comparisons = verified["comparisons"]
    _require(
        len(actual_comparisons) == len(expected_comparisons), "comparison count changed"
    )
    for actual, expected in zip(actual_comparisons, expected_comparisons):
        _require(set(actual) == set(expected), "comparison fields changed")
        _require(
            all(_close(actual[key], expected[key]) for key in expected),
            "comparison arithmetic changed",
        )
    expected_headline = _headline(expected_comparisons)
    _require(
        verified["headline"] == expected_headline,
        "headline selection or arithmetic changed",
    )
    _require(
        verified["consumer_contract"]
        == {"canonical_for": list(CANONICAL_SURFACES), "rules": list(CONSUMER_RULES)},
        "consumer contract changed",
    )
    _require(
        verified["presentation"] == _presentation_contract(),
        "presentation data contract changed",
    )
    identity = _mapping(verified["identity"], "launch identity")
    _require(identity["result_id"] == RESULT_ID, "launch result ID changed")
    _require(identity["benchmark_id"] == BENCHMARK_ID, "launch benchmark ID changed")
    _require(
        identity["benchmark_version"] == BENCHMARK_VERSION,
        "launch benchmark version changed",
    )
    _require(
        identity["completed_at_utc"] == COMPLETED_AT_UTC,
        "launch completion identity changed",
    )
    _require(
        identity["manifest_recorded_origin_digests"] == EXPECTED_ORIGIN_DIGESTS,
        "manifest-recorded origin identities changed",
    )

    provenance = _mapping(verified["provenance"], "launch provenance")
    all_bindings = [
        *provenance["verified_inputs"],
        provenance["generator"],
        provenance["upstream_generator"],
        provenance["schema"],
    ]
    all_bindings = [
        _validate_binding(binding, f"provenance binding {index}")
        for index, binding in enumerate(all_bindings)
    ]
    paths = [binding["path"] for binding in all_bindings]
    _require(len(paths) == len(set(paths)), "provenance paths are not unique")
    _require(
        [binding["path"] for binding in provenance["verified_inputs"]]
        == [
            PROJECTION_RELATIVE.as_posix(),
            MANIFEST_RELATIVE.as_posix(),
            PUBLICATION_HANDOFF_RELATIVE.as_posix(),
            PUBLICATION_HANDOFF_SCHEMA_RELATIVE.as_posix(),
            RELEASE_IDENTITY_RELATIVE.as_posix(),
        ],
        "verified-input provenance changed",
    )
    _require(
        next(
            binding
            for binding in provenance["verified_inputs"]
            if binding["path"] == PROJECTION_RELATIVE.as_posix()
        )
        == _frozen_binding(
            PROJECTION_RELATIVE,
            sha256=EXPECTED_PROJECTION_SHA256,
            size_bytes=EXPECTED_PROJECTION_SIZE_BYTES,
        ),
        "frozen projection provenance changed",
    )
    _require(
        next(
            binding
            for binding in provenance["verified_inputs"]
            if binding["path"] == MANIFEST_RELATIVE.as_posix()
        )
        == _frozen_binding(
            MANIFEST_RELATIVE,
            sha256=EXPECTED_MANIFEST_SHA256,
            size_bytes=EXPECTED_MANIFEST_SIZE_BYTES,
        ),
        "frozen manifest provenance changed",
    )
    _require(
        provenance["upstream_generator"]
        == _frozen_binding(
            UPSTREAM_GENERATOR_RELATIVE,
            sha256=EXPECTED_UPSTREAM_GENERATOR_SHA256,
            size_bytes=EXPECTED_UPSTREAM_GENERATOR_SIZE_BYTES,
        ),
        "upstream generator provenance changed",
    )
    private_control_paths = (
        PUBLICATION_HANDOFF_RELATIVE,
        PUBLICATION_HANDOFF_SCHEMA_RELATIVE,
        RELEASE_IDENTITY_RELATIVE,
    )
    private_control_presence = [
        (root / relative).exists() for relative in private_control_paths
    ]
    _require(
        all(private_control_presence) or not any(private_control_presence),
        "private release-control inputs are only partially present",
    )
    if all(private_control_presence):
        software, expected_clearance = _release_state(root)
        _require(clearance == expected_clearance, "distribution clearance is stale")
        _require(
            identity["project"] == software["display_name"],
            "project identity is stale",
        )
        _require(
            identity["current_version"] == software["current_version"],
            "current version is stale",
        )
        _require(
            identity["target_version"] == software["target_version"],
            "target version is stale",
        )
        for relative in private_control_paths:
            declared = next(
                binding
                for binding in provenance["verified_inputs"]
                if binding["path"] == relative.as_posix()
            )
            _require(
                declared == _binding(root, relative),
                f"{relative} binding is stale",
            )
        _require(
            provenance["generator"] == _binding(root, GENERATOR_RELATIVE),
            "generator binding is stale",
        )
    else:
        _require(identity["project"] == "Noema", "project identity is stale")
        _require(
            identity["current_version"] == "0.2.0.dev0",
            "current version is stale",
        )
        _require(identity["target_version"] == "0.2.0", "target version is stale")
    _require(
        provenance["schema"] == _binding(root, SCHEMA_RELATIVE),
        "schema binding is stale",
    )
    for relative in (
        PROJECTION_RELATIVE,
        MANIFEST_RELATIVE,
        UPSTREAM_GENERATOR_RELATIVE,
    ):
        path = root / relative
        if path.exists():
            _require(
                path.is_file() and not path.is_symlink(),
                f"optional source binding is unsafe: {relative}",
            )
            declared = next(
                binding
                for binding in all_bindings
                if binding["path"] == relative.as_posix()
            )
            _require(
                declared == _binding(root, relative), f"{relative} binding is stale"
            )
    return verified


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_projection(path: Path) -> dict[str, Any]:
    return _load_mapping(path, "launch-evidence projection")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and require byte equality with the checked-in projection",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="verify the standalone projection without reading excluded demo inputs",
    )
    parser.add_argument(
        "--require-cleared",
        action="store_true",
        help="also require final distribution clearance (for stable release only)",
    )
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT_RELATIVE)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    try:
        if args.verify:
            projection = verify_projection(_load_projection(output), root=ROOT)
        else:
            projection = build_projection(ROOT)
            if args.check:
                checked = _load_projection(output)
                verify_projection(checked, root=ROOT)
                _require(
                    output.read_bytes() == _canonical_bytes(projection),
                    f"launch evidence is stale; run {GENERATOR_RELATIVE.as_posix()}",
                )
            else:
                write_study_json(output, projection)
        if args.require_cleared:
            _require(
                projection["distribution_clearance"]["status"]
                == "cleared_for_stable_release",
                "launch evidence is not cleared for final distribution",
            )
    except (LaunchEvidenceError, OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    status = projection["distribution_clearance"]["status"]
    action = "verified" if args.verify else "current" if args.check else "generated"
    print(f"launch evidence: {action} ({status})\t{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
