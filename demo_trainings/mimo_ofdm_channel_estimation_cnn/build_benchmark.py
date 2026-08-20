from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from noema_lab.core.trained_artifacts import inspect_trained_artifact
from noema_lab.ops import build_registry

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


DEFAULT_SNR_DB = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
DEFAULT_SEEDS = (91001, 92001, 93001)
DEFAULT_TDL_PROFILES = ("A", "C", "E")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the paired mixed-profile LS/fixed-prior-LMMSE/learned "
            "MIMO-OFDM estimator benchmark."
        )
    )
    parser.add_argument("--artifact")
    parser.add_argument("--output", default="benchmark_pack.yaml")
    parser.add_argument("--snr-db", default=",".join(str(value) for value in DEFAULT_SNR_DB))
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument(
        "--tdl-profiles",
        default=",".join(DEFAULT_TDL_PROFILES),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = _mapping(root / "train_config.yaml")
    training = dict(config.get("training") or {})
    project_root = Path(str(config["project_root"])).resolve()
    artifact_path = (
        Path(args.artifact).expanduser().resolve()
        if args.artifact
        else _resolve(root, training["artifact_manifest_path"])
    )
    artifact_runtime = _validate_artifact(
        artifact_path,
        project_root=project_root,
    )
    history_path = root / "training_history.json"
    evaluation_path = root / "evaluation_metrics.json"
    for path, label in (
        (history_path, "training history"),
        (evaluation_path, "held-out evaluation"),
    ):
        if not path.is_file():
            raise FileNotFoundError("%s does not exist: %s" % (label, path))
    source_recipe = _mapping(root / "noema_recipe.yaml")
    source_recipe.pop("dataset_capture", None)
    metadata = dict(source_recipe.get("metadata") or {})
    metadata.pop("matrix", None)
    source_recipe["metadata"] = metadata
    benchmark_recipe_path = root / "benchmark_recipe.yaml"
    benchmark_recipe_path.write_text(
        yaml.safe_dump(source_recipe, sort_keys=False),
        encoding="utf-8",
    )
    output_path = (root / args.output).resolve()
    snr_values = _floats(args.snr_db)
    seeds = _integers(args.seeds)
    tdl_profiles = _tdl_profiles(args.tdl_profiles)
    pack = _pack(
        recipe_path=benchmark_recipe_path.name,
        artifact_path=_project_path(artifact_path, project_root),
        artifact_package_sha256=artifact_runtime["runtime_identity_sha256"],
        snr_values=snr_values,
        seeds=seeds,
        tdl_profiles=tdl_profiles,
        history=_evidence(history_path, root),
        evaluation=_evidence(evaluation_path, root),
        artifact=_evidence(artifact_path, root),
    )
    output_path.write_text(yaml.safe_dump(pack, sort_keys=False), encoding="utf-8")
    print("wrote benchmark recipe: %s" % benchmark_recipe_path)
    print("wrote benchmark pack: %s" % output_path)
    print("from the Noema project root, validate and run:")
    print("  noema benchmark validate %s" % _project_path(output_path, project_root))
    print("  noema benchmark run %s" % _project_path(output_path, project_root))
    return 0


def _pack(
    *,
    recipe_path: str,
    artifact_path: str,
    artifact_package_sha256: str,
    snr_values: list[float],
    seeds: list[int],
    tdl_profiles: list[str],
    history: dict[str, str],
    evaluation: dict[str, str],
    artifact: dict[str, str],
) -> dict[str, Any]:
    methods = (
        ("least_squares", "Least squares", "baseline", {"mode": "least_squares"}),
        (
            "fixed_prior_lmmse",
            "Fixed-prior LMMSE",
            "baseline",
            {
                "mode": "linear_mmse_reference",
                "lmmse_assumed_tap_count": 4,
            },
        ),
        (
            "learned_estimator",
            "Learned dual-domain estimator",
            "candidate",
            {
                "mode": "learned_artifact",
                "artifact_manifest_path": artifact_path,
                "artifact_entrypoint": "channel_estimator",
                "artifact_package_sha256": artifact_package_sha256,
            },
        ),
    )
    recipes = []
    scenario_units = [
        (tdl_profiles[index % len(tdl_profiles)], seed)
        for index, seed in enumerate(seeds)
    ]
    for snr_db in snr_values:
        for tdl_profile, seed in scenario_units:
            pairing_id = "%s:%d" % (tdl_profile, seed)
            for method_id, label, role, estimator_params in methods:
                recipes.append(
                    {
                        "id": "%s_tdl%s_snr%s_seed%d"
                        % (
                            method_id,
                            tdl_profile.lower(),
                            _number_id(snr_db),
                            seed,
                        ),
                        "label": "%s · TDL-%s · SNR=%g dB · seed %d"
                        % (label, tdl_profile, snr_db, seed),
                        "role": role,
                        "path": recipe_path,
                        "params": {
                            "method_id": method_id,
                            "metadata": {
                                "seed": seed + 500_000,
                                "benchmark_method": method_id,
                                "benchmark_method_label": label,
                                "benchmark_paired_seed": seed,
                                "benchmark_tdl_profile": tdl_profile,
                                "pairing_id": pairing_id,
                                "aggregation_cell_id": "snr_db=%s"
                                % _number_id(snr_db),
                                "statistical_unit": (
                                    "paired held-out TDL profile/channel seed"
                                ),
                                "benchmark_held_out": True,
                            },
                            "step_params": {
                                "data": {
                                    "tdl_model": tdl_profile,
                                    "seed": seed,
                                },
                                "pilots": {"seed": 1701},
                                "pilot_observation": {
                                    "snr_db": snr_db,
                                    "seed": seed + 100_000,
                                },
                                "estimator": dict(estimator_params),
                            },
                        },
                    }
                )
    method_order = [item[0] for item in methods]
    metrics = [
        _metric("mimo.channel_estimation.nmse_db", "quality", "dB", "lower_is_better"),
        _metric(
            "mimo.channel_estimation.zf_spectral_efficiency_bps_hz",
            "system",
            "bit/s/Hz",
            "higher_is_better",
        ),
        _metric(
            "mimo.channel_estimation.perfect_csi_zf_spectral_efficiency_bps_hz",
            "diagnostic",
            "bit/s/Hz",
            "higher_is_better",
        ),
        _metric(
            "mimo.channel_estimation.zf_rate_retention",
            "system",
            "ratio",
            "higher_is_better",
        ),
        _metric(
            "channel_estimation.complex_correlation",
            "quality",
            "ratio",
            "higher_is_better",
        ),
        _metric("channel.snr_db", "channel", "dB", "neutral"),
    ]
    return {
        "schema_version": 1,
        "id": "mimo_ofdm.learned_channel_estimation_post_training_v2",
        "version": "2.0.0",
        "name": "Learned 2×2 MIMO-OFDM Estimation under Unknown TDL Profile",
        "description": (
            "LS, a fixed four-tap exponential-PDP LMMSE, and a portable "
            "learned estimator compared across paired held-out Sionna 3GPP "
            "TDL-A/C/E channels."
        ),
        "suite": {
            "id": "mimo_ofdm",
            "name": "MIMO-OFDM",
            "status": "experimental",
            "version": "v1-draft",
        },
        "dataset": {
            "id": "sionna_3gpp_mixed_tdl_mimo_ofdm_pilots",
            "modality": "wireless",
            "version": "sionna-tdl-ace-2x2-64sc-v2",
            "split": "held_out_profile_and_seeded_channel",
        },
        "task": {
            "id": "mimo_ofdm_channel_estimation",
            "kind": "estimation",
            "modality": "wireless",
        },
        "metrics": metrics,
        "baselines": ["least_squares", "fixed_prior_lmmse"],
        "recipes": recipes,
        "metadata": {
            "benchmark_tier": "experimental",
            "protocol_revision": "mimo-ofdm-channel-estimation-mixed-tdl-v2",
            "paired_held_out_seeds": seeds,
            "paired_held_out_scenario_units": [
                {"tdl_profile": profile, "seed": seed}
                for profile, seed in scenario_units
            ],
            "snr_db": snr_values,
            "tdl_profiles": tdl_profiles,
            "statistical_unit": "paired held-out TDL profile/channel seed",
            "protocol": {
                "paired": True,
                "pairing_keys": [
                    "benchmark_tdl_profile",
                    "benchmark_paired_seed",
                    "channel.snr_db",
                ],
                "held_constant": [
                    "TDL profile within each paired comparison",
                    "mobility, delay spread, and pilot pattern",
                    "paired channel and observation-noise seeds",
                    "2×2 antenna dimensions and 64-subcarrier grid",
                ],
                "changed": [
                    "estimator method",
                    "channel SNR",
                    "TDL profile across statistical units",
                ],
            },
            "demo": {
                "schema_version": 1,
                "slug": "learned-mimo-ofdm-channel-estimation",
                "title": "Learned 2×2 MIMO-OFDM channel estimation",
                "summary": (
                    "A compact dual-domain estimator uses sparse pilots and an "
                    "LS fallback when the active TDL profile is not known to "
                    "the receiver."
                ),
                "question": (
                    "Can one learned prior across TDL-A/C/E beat interpolation "
                    "and a fixed mismatched LMMSE prior?"
                ),
                "tutorial": "../../../tutorials/learned_mimo_ofdm_channel_estimation_demo.html",
                "primary_metric": "mimo.channel_estimation.nmse_db",
                "comparison_axis": "channel.snr_db",
                "series": [
                    {"id": item[0], "label": item[1], "role": item[2]}
                    for item in methods
                ],
                "plots": [
                    {
                        "id": "nmse-vs-snr",
                        "title": "Channel-estimation NMSE vs SNR",
                        "kind": "line",
                        "x": "channel.snr_db",
                        "y": "mimo.channel_estimation.nmse_db",
                        "group": "benchmark_method",
                        "method_order": method_order,
                        "style": {"aggregation": "mean_ci", "y_scale": "linear"},
                    },
                    {
                        "id": "zf-rate-vs-snr",
                        "title": "Post-ZF spectral efficiency vs SNR",
                        "kind": "line",
                        "x": "channel.snr_db",
                        "y": "mimo.channel_estimation.zf_spectral_efficiency_bps_hz",
                        "group": "benchmark_method",
                        "method_order": method_order,
                        "style": {"aggregation": "mean_ci", "y_scale": "linear"},
                    },
                    {
                        "id": "zf-retention-vs-snr",
                        "title": "Perfect-CSI ZF rate retained",
                        "kind": "line",
                        "x": "channel.snr_db",
                        "y": "mimo.channel_estimation.zf_rate_retention",
                        "group": "benchmark_method",
                        "method_order": method_order,
                        "style": {"aggregation": "mean_ci", "y_scale": "linear"},
                    },
                ],
                "table_metrics": [item["id"] for item in metrics],
                "training_evidence": [
                    {
                        "series": "learned_estimator",
                        "trained_artifact_manifest": artifact,
                        "training_history": history,
                        "evaluation_metrics": evaluation,
                    }
                ],
            },
        },
    }


def _metric(metric_id: str, family: str, unit: str, direction: str) -> dict[str, Any]:
    return {
        "id": metric_id,
        "definition_version": 1,
        "source_step": "evaluation",
        "source_operation": "metrics.channel_estimation",
        "family": family,
        "unit": unit,
        "direction": direction,
    }


def _tdl_profiles(value: str) -> list[str]:
    profiles = [item.strip().upper() for item in str(value).split(",") if item.strip()]
    if not profiles or any(item not in {"A", "B", "C", "D", "E"} for item in profiles):
        raise ValueError("--tdl-profiles must contain one or more of A,B,C,D,E")
    if len(set(profiles)) != len(profiles):
        raise ValueError("--tdl-profiles must not contain duplicates")
    return profiles


def _validate_artifact(path: Path, *, project_root: Path) -> dict[str, str]:
    payload = _mapping(path)
    if int(payload.get("schema_version") or 0) != 2:
        raise ValueError("trained artifact must use schema_version: 2")
    if str((payload.get("runtime") or {}).get("backend") or "") != "onnxruntime":
        raise ValueError("trained artifact runtime must be onnxruntime")
    components = list(payload.get("components") or [])
    if len(components) != 1:
        raise ValueError("trained artifact must contain one ONNX component")
    component = dict(components[0])
    component_path = path.parent / str(component.get("path") or "")
    component_sha256 = _sha256(component_path) if component_path.is_file() else ""
    if component_sha256 != str(component.get("sha256") or ""):
        raise ValueError("trained artifact ONNX component is missing or fails SHA-256 verification")
    inspected = inspect_trained_artifact(
        path,
        project_root=project_root,
        registry=build_registry(),
    )
    if inspected.get("issues") or not inspected.get("ready"):
        raise ValueError(
            "trained artifact is not runtime-ready: %s"
            % ("; ".join(str(item) for item in inspected.get("issues") or []) or "runtime unavailable")
        )
    runtime_identity = str(
        inspected.get("runtime_identity_sha256")
        or inspected.get("package_sha256")
        or ""
    )
    if len(runtime_identity) != 64:
        raise ValueError("trained artifact runtime identity is missing")
    return {
        "component_sha256": component_sha256,
        "runtime_identity_sha256": runtime_identity,
    }


def _evidence(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": os.path.relpath(path.resolve(), root.resolve()),
        "sha256": _sha256(path),
    }


def _mapping(path: Path) -> dict:
    value = load_strict_yaml_or_json(path)
    if not isinstance(value, dict):
        raise ValueError("Expected a YAML/JSON object: %s" % path)
    return dict(value)


def _resolve(root: Path, value) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _project_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _integers(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _number_id(value: float) -> str:
    rendered = ("%g" % float(value)).replace("-", "m").replace(".", "p")
    return rendered or "0"


if __name__ == "__main__":
    raise SystemExit(main())
