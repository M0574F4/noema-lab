"""Build the paired post-training CSI compression/feedback benchmark."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from datamodule import materialize_data_contract_inventory
from noema_lab.core.trained_artifacts import inspect_trained_artifact
from noema_lab.ops import build_registry

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


DEFAULT_SNRS = (0.0, 5.0, 10.0, 15.0, 20.0)
DEFAULT_SEEDS = (91001, 92001, 93001)
DEFAULT_SAMPLE_COUNT = 512


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the paired learned/KLT/truncated/perfect-CSIT CSI "
            "feedback benchmark."
        )
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--training-history", type=Path)
    parser.add_argument("--evaluation-metrics", type=Path)
    parser.add_argument("--klt-artifact", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark_pack.yaml"))
    parser.add_argument(
        "--snr-db",
        default=",".join(str(value) for value in DEFAULT_SNRS),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_SEEDS),
    )
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else _find_project_root(here)
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output.is_absolute()
        else (here / args.output).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path = _selected_path(
        args.recipe,
        (
            here / "noema_recipe.yaml",
            project_root / "recipes" / "csi_feedback_sionna_train.yaml",
        ),
        "source recipe",
    )
    artifact_path = _selected_path(
        args.artifact,
        (
            here.parent / "trained_artifact.yaml",
            project_root
            / "differentiable_exports"
            / "csi_feedback_sionna_128bit_v2"
            / "trained_artifact.yaml",
        ),
        "learned artifact",
    )
    history_path = _selected_path(
        args.training_history,
        (
            here / "training_history.json",
            project_root
            / "differentiable_exports"
            / "csi_feedback_sionna_128bit_v2"
            / "reference_training"
            / "training_history.json",
        ),
        "training history",
    )
    evaluation_path = _selected_path(
        args.evaluation_metrics,
        (
            here / "evaluation_metrics.json",
            project_root
            / "differentiable_exports"
            / "csi_feedback_sionna_128bit_v2"
            / "reference_training"
            / "evaluation_metrics.json",
        ),
        "held-out evaluation",
    )
    klt_path = _selected_path(
        args.klt_artifact,
        (
            project_root
            / "trained_artifacts"
            / "references"
            / "klt-csi-feedback-sionna-128bit-v2"
            / "trained_artifact.yaml",
        ),
        "matched KLT artifact",
    )
    snrs = _floats(args.snr_db)
    seeds = _integers(args.seeds)
    if not snrs or not seeds:
        raise ValueError("SNR and seed lists must not be empty")
    if len(set(snrs)) != len(snrs) or len(set(seeds)) != len(seeds):
        raise ValueError("SNR and seed lists must not contain duplicates")
    if args.sample_count < 1:
        raise ValueError("--sample-count must be positive")

    _freeze_legacy_capture_inventory(artifact_path)
    learned = _validate_codec_artifact(
        artifact_path,
        project_root=project_root,
    )
    klt = _validate_codec_artifact(klt_path, project_root=project_root)
    _validate_training_evidence(
        artifact_path,
        history_path,
        evaluation_path,
    )
    recipe = _mapping(recipe_path)
    benchmark_recipe_path = _write_benchmark_recipe(
        recipe,
        output_path.parent / "benchmark_recipe.yaml",
    )
    pack = _build_pack(
        recipe_reference=os.path.relpath(
            benchmark_recipe_path,
            output_path.parent,
        ),
        learned_reference=_project_path(artifact_path, project_root),
        learned_package_sha256=learned["runtime_identity_sha256"],
        klt_reference=_project_path(klt_path, project_root),
        klt_package_sha256=klt["runtime_identity_sha256"],
        learned_manifest=_evidence(artifact_path, output_path.parent),
        history=_evidence(history_path, output_path.parent),
        evaluation=_evidence(evaluation_path, output_path.parent),
        klt_manifest=_evidence(klt_path, output_path.parent),
        snrs=snrs,
        seeds=seeds,
        sample_count=int(args.sample_count),
    )
    output_path.write_text(
        yaml.safe_dump(pack, sort_keys=False),
        encoding="utf-8",
    )
    print("wrote benchmark recipe: %s" % benchmark_recipe_path)
    print("wrote benchmark pack: %s" % output_path)
    print("from the Noema project root, validate and run:")
    print(
        "  uv run --extra wireless --extra onnx noema benchmark validate %s"
        % _project_path(output_path, project_root)
    )
    print(
        "  uv run --extra wireless --extra onnx noema benchmark run %s"
        % _project_path(output_path, project_root)
    )
    return 0


def _freeze_legacy_capture_inventory(artifact_path: Path) -> None:
    """Upgrade pre-inventory demo artifacts without retraining their model."""

    config_path = artifact_path.parent / "reference_training" / "train_config.yaml"
    if not config_path.is_file():
        return
    config = _mapping(config_path)
    contract_reference = str(
        ((config.get("data") or {}).get("contract_path") or "")
    ).strip()
    if not contract_reference:
        return
    contract_path = (config_path.parent / contract_reference).resolve()
    contract = _mapping(contract_path)
    splits = contract.get("splits")
    if (
        isinstance(splits, list)
        and splits
        and all(
            isinstance(split, Mapping)
            and isinstance(split.get("files"), list)
            and bool(split.get("files"))
            for split in splits
        )
    ):
        return

    current_directory = Path.cwd()
    try:
        os.chdir(config_path.parent)
        updated = materialize_data_contract_inventory(
            config,
            config_path=config_path.name,
        )
    finally:
        os.chdir(current_directory)

    artifact = _mapping(artifact_path)
    source = dict(artifact.get("source") or {})
    binding = dict(source.get("data_contract") or {})
    data = dict(updated.get("data") or {})
    binding["sha256"] = str(data.get("contract_sha256") or "")
    binding["file_sha256"] = str(data.get("contract_file_sha256") or "")
    source["data_contract"] = binding
    artifact["source"] = source
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(artifact, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(artifact_path)
    print("froze captured CSI shard inventory: %s" % contract_path)


def _build_pack(
    *,
    recipe_reference: str,
    learned_reference: str,
    learned_package_sha256: str,
    klt_reference: str,
    klt_package_sha256: str,
    learned_manifest: Mapping[str, str],
    history: Mapping[str, str],
    evaluation: Mapping[str, str],
    klt_manifest: Mapping[str, str],
    snrs: list[float],
    seeds: list[int],
    sample_count: int,
) -> dict[str, Any]:
    methods = (
        (
            "truncated_angular_delay",
            "Truncated angular-delay",
            "baseline",
            {
                "encoder": {
                    "runtime": "truncated_angular_delay",
                    "feedback_dimension": 32,
                },
                "decoder": {
                    "runtime": "truncated_angular_delay",
                    "feedback_dimension": 32,
                },
                "link": {
                    "mode": "uniform_quantized",
                    "bits_per_latent": 4,
                    "clip_value": 1.0,
                },
            },
        ),
        (
            "matched_klt",
            "Matched KLT/PCA",
            "baseline",
            _artifact_method(
                klt_reference,
                klt_package_sha256,
            ),
        ),
        (
            "learned_codec",
            "Learned CSI codec",
            "candidate",
            _artifact_method(
                learned_reference,
                learned_package_sha256,
            ),
        ),
        (
            "perfect_csit",
            "Perfect CSIT",
            "upper_bound",
            {
                "encoder": {"runtime": "identity", "feedback_dimension": 512},
                "decoder": {"runtime": "identity", "feedback_dimension": 512},
                "link": {
                    "mode": "ideal_noiseless",
                    "bits_per_latent": 4,
                    "clip_value": 1.0,
                },
            },
        ),
    )
    recipes = []
    for snr in snrs:
        for seed in seeds:
            pairing_id = "tdlA:%s:%d" % (_number_id(snr), seed)
            for method_id, label, role, settings in methods:
                recipes.append(
                    {
                        "id": "%s_snr%s_seed%d" % (method_id, _number_id(snr), seed),
                        "label": "%s · %g dB · seed %d" % (label, snr, seed),
                        "role": role,
                        "path": recipe_reference,
                        "params": {
                            "method_id": method_id,
                            "metadata": {
                                "seed": seed + 500_000,
                                "benchmark_method": method_id,
                                "benchmark_method_label": label,
                                "benchmark_paired_seed": seed,
                                "benchmark_tdl_profile": "A",
                                "pairing_id": pairing_id,
                                "aggregation_cell_id": "snr_db=%s" % _number_id(snr),
                                "statistical_unit": (
                                    "paired held-out channel realization seed"
                                ),
                                "benchmark_held_out": True,
                            },
                            "step_params": {
                                "channel_state": {
                                    "sample_count": sample_count,
                                    "tdl_model": "A",
                                    "downlink_snr_db": snr,
                                    "seed": seed,
                                },
                                "feedback_encoder": dict(settings["encoder"]),
                                "feedback_link": dict(settings["link"]),
                                "feedback_decoder": dict(settings["decoder"]),
                            },
                        },
                    }
                )
    method_order = [item[0] for item in methods]
    metrics = [
        _metric(
            "csi_feedback.achieved_spectral_efficiency_bps_hz",
            "system",
            "bit/s/Hz",
            "higher_is_better",
        ),
        _metric(
            "csi_feedback.perfect_csi_spectral_efficiency_bps_hz",
            "diagnostic",
            "bit/s/Hz",
            "higher_is_better",
        ),
        _metric(
            "csi_feedback.spectral_efficiency_retention",
            "system",
            "ratio",
            "higher_is_better",
        ),
        _metric(
            "csi_feedback.spectral_efficiency_loss_bps_hz",
            "system",
            "bit/s/Hz",
            "lower_is_better",
        ),
        _metric(
            "csi_feedback.nmse_db",
            "quality",
            "dB",
            "lower_is_better",
        ),
        _metric(
            "csi_feedback.phase_invariant_cosine",
            "quality",
            "ratio",
            "higher_is_better",
        ),
        _metric(
            "csi_feedback.feedback_bits_per_sample",
            "rate",
            "bit/sample",
            "neutral",
            source_step="feedback_link",
            source_operation="channel.csi_feedback_link",
            applicable_roles=["baseline", "candidate"],
        ),
        _metric(
            "csi_feedback.quantization_mse",
            "diagnostic",
            "MSE",
            "lower_is_better",
            source_step="feedback_link",
            source_operation="channel.csi_feedback_link",
            applicable_roles=["baseline", "candidate"],
        ),
        _metric(
            "csi_feedback.quantization_clipped_fraction",
            "diagnostic",
            "ratio",
            "lower_is_better",
            source_step="feedback_link",
            source_operation="channel.csi_feedback_link",
            applicable_roles=["baseline", "candidate"],
        ),
        _metric(
            "channel.snr_db",
            "channel",
            "dB",
            "neutral",
            source_step="channel_state",
            source_operation="wireless.miso_ofdm_csi",
        ),
    ]
    return {
        "schema_version": 1,
        "id": "channel_estimation.learned_csi_feedback_post_training_v1",
        "version": "1.0.0",
        "name": "Learned 128-bit CSI Compression and Feedback",
        "description": (
            "A learned paired CSI codec, matched KLT/PCA, truncated "
            "angular-delay feedback, and perfect CSIT compared on paired "
            "held-out Sionna TDL-A MISO-OFDM channels."
        ),
        "suite": {
            "id": "channel_estimation",
            "name": "Channel Estimation",
            "status": "experimental",
            "version": "v1-draft",
        },
        "dataset": {
            "id": "sionna_correlated_miso_ofdm_csi",
            "modality": "wireless",
            "version": "sionna-tdl-a-8x32-v1",
            "split": "held_out_seeded_channel",
        },
        "task": {
            "id": "csi_compression_feedback",
            "kind": "feedback_control",
            "modality": "wireless",
        },
        "metrics": metrics,
        "baselines": ["truncated_angular_delay", "matched_klt"],
        "recipes": recipes,
        "metadata": {
            "benchmark_tier": "experimental",
            "protocol_revision": "csi-feedback-128bit-tdla-v1",
            "paired_held_out_seeds": seeds,
            "snr_db": snrs,
            "statistical_unit": ("paired held-out channel realization seed"),
            "protocol": {
                "paired": True,
                "pairing_keys": [
                    "benchmark_tdl_profile",
                    "benchmark_paired_seed",
                ],
                "channel_model": "Sionna 3GPP TDL-A",
                "feedback_budget_bits": 128,
                "feedback_dimension": 32,
                "bits_per_latent": 4,
                "sample_count_per_run": sample_count,
                "training_data_excluded": True,
                "perfect_csit_role": ("non-deployable full-dimensional upper bound"),
            },
            "reference_artifacts": [
                {
                    "series": "matched_klt",
                    "trained_artifact_manifest": dict(klt_manifest),
                }
            ],
            "demo": {
                "schema_version": 1,
                "slug": "learned-csi-compression-feedback",
                "title": "Learning 128-bit CSI compression and feedback",
                "summary": (
                    "The learned codec and two classical 128-bit codecs are "
                    "evaluated through the downstream MRT rate they retain."
                ),
                "question": (
                    "Can a learned paired encoder/decoder preserve more "
                    "downlink utility than matched classical 128-bit feedback?"
                ),
                "changed": [
                    "CSI feedback encoder/decoder implementation",
                    "downlink SNR",
                    "held-out paired channel seed",
                ],
                "held_constant": [
                    "Sionna TDL-A channel distribution",
                    "8 transmit antennas and 32 subcarriers",
                    "128-bit matched feedback budget",
                    "paired channel realization seed across methods",
                ],
                "primary_metric": ("csi_feedback.achieved_spectral_efficiency_bps_hz"),
                "comparison_axis": "channel.snr_db",
                "series": [
                    {"id": item[0], "label": item[1], "role": item[2]}
                    for item in methods
                ],
                "plots": [
                    {
                        "id": "rate-vs-snr",
                        "title": "Achieved downlink spectral efficiency",
                        "kind": "line",
                        "x": "channel.snr_db",
                        "y": ("csi_feedback.achieved_spectral_efficiency_bps_hz"),
                        "group": "benchmark_method",
                        "method_order": method_order,
                        "style": {
                            "aggregation": "mean_ci",
                            "y_scale": "linear",
                        },
                    },
                    {
                        "id": "retention-vs-snr",
                        "title": "Perfect-CSIT rate retained",
                        "kind": "line",
                        "x": "channel.snr_db",
                        "y": ("csi_feedback.spectral_efficiency_retention"),
                        "group": "benchmark_method",
                        "method_order": method_order,
                        "style": {
                            "aggregation": "mean_ci",
                            "y_scale": "linear",
                        },
                    },
                    {
                        "id": "nmse-vs-snr",
                        "title": "CSI reconstruction NMSE",
                        "kind": "line",
                        "x": "channel.snr_db",
                        "y": "csi_feedback.nmse_db",
                        "group": "benchmark_method",
                        "method_order": method_order,
                        "style": {
                            "aggregation": "mean_ci",
                            "y_scale": "linear",
                        },
                    },
                ],
                "table_metrics": [item["id"] for item in metrics],
                "training_evidence": [
                    {
                        "series": "learned_codec",
                        "trained_artifact_manifest": dict(learned_manifest),
                        "training_history": dict(history),
                        "evaluation_metrics": dict(evaluation),
                    }
                ],
                "tutorial": "../../../tutorials/learned_csi_feedback.html",
            },
        },
    }


def _artifact_method(
    manifest_path: str,
    package_sha256: str,
) -> dict[str, dict[str, Any]]:
    shared = {
        "runtime": "learned_artifact",
        "feedback_dimension": 32,
        "artifact_manifest_path": manifest_path,
        "artifact_package_sha256": package_sha256,
    }
    return {
        "encoder": {**shared, "artifact_entrypoint": "encoder"},
        "decoder": {**shared, "artifact_entrypoint": "decoder"},
        "link": {
            "mode": "uniform_quantized",
            "bits_per_latent": 4,
            "clip_value": 1.0,
        },
    }


def _metric(
    metric_id: str,
    family: str,
    unit: str,
    direction: str,
    *,
    source_step: str = "evaluation",
    source_operation: str = "metrics.csi_feedback",
    applicable_roles: list[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": metric_id,
        "definition_version": 1,
        "source_step": source_step,
        "source_operation": source_operation,
        "family": family,
        "unit": unit,
        "direction": direction,
    }
    if applicable_roles:
        result["applicable_roles"] = applicable_roles
    return result


def _validate_codec_artifact(
    path: Path,
    *,
    project_root: Path,
) -> dict[str, str]:
    payload = _mapping(path)
    if int(payload.get("schema_version") or 0) != 2:
        raise ValueError("CSI codec artifact must use schema_version: 2")
    if str((payload.get("runtime") or {}).get("backend") or "") != "onnxruntime":
        raise ValueError("CSI codec artifact runtime must be onnxruntime")
    components = {
        str(item.get("id") or ""): dict(item)
        for item in payload.get("components") or []
        if isinstance(item, Mapping)
    }
    if set(components) != {"encoder", "decoder"}:
        raise ValueError(
            "CSI codec artifact must contain encoder and decoder components"
        )
    for component_id, component in components.items():
        component_path = path.parent / str(component.get("path") or "")
        if not component_path.is_file():
            raise FileNotFoundError(
                "CSI codec %s component is missing: %s" % (component_id, component_path)
            )
        if _sha256(component_path) != str(component.get("sha256") or ""):
            raise ValueError(
                "CSI codec %s component fails SHA-256 verification" % component_id
            )
    inspected = inspect_trained_artifact(
        path,
        project_root=project_root,
        registry=build_registry(),
    )
    if inspected.get("issues") or not inspected.get("ready"):
        raise ValueError(
            "CSI codec artifact is not runtime-ready: %s"
            % (
                "; ".join(str(item) for item in inspected.get("issues") or [])
                or "runtime unavailable"
            )
        )
    runtime_identity = str(
        inspected.get("runtime_identity_sha256")
        or inspected.get("package_sha256")
        or ""
    )
    if len(runtime_identity) != 64:
        raise ValueError("CSI codec artifact runtime identity is missing")
    return {"runtime_identity_sha256": runtime_identity}


def _validate_training_evidence(
    artifact_path: Path,
    history_path: Path,
    evaluation_path: Path,
) -> None:
    artifact = _mapping(artifact_path)
    history = _mapping(history_path)
    evaluation = _mapping(evaluation_path)
    training = artifact.get("training")
    if not isinstance(training, Mapping) or not training.get("best_seed"):
        raise ValueError("learned CSI artifact lacks training provenance")
    if not list(history.get("epochs") or []):
        raise ValueError("CSI training history has no epochs")
    if str(evaluation.get("split") or "") != "test":
        raise ValueError("CSI evaluation evidence must use the held-out test split")
    if (
        evaluation.get("test_capture_used_for_training_or_checkpoint_selection")
        is not False
    ):
        raise ValueError("CSI held-out test evidence was exposed to training")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping) or "nmse_db" not in metrics:
        raise ValueError("CSI held-out evaluation metrics are incomplete")


def _write_benchmark_recipe(
    recipe: Mapping[str, Any],
    destination: Path,
) -> Path:
    payload = dict(recipe)
    payload.pop("dataset_capture", None)
    metadata = dict(payload.get("metadata") or {})
    metadata.pop("matrix", None)
    metadata["research_stage"] = "post_training_paired_benchmark"
    metadata["training_performed"] = True
    payload["metadata"] = metadata
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def _mapping(path: Path) -> dict[str, Any]:
    value = load_strict_yaml_or_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("Expected a YAML/JSON object: %s" % path)
    return dict(value)


def _selected_path(
    explicit: Path | None,
    candidates: tuple[Path, ...],
    label: str,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("%s does not exist: %s" % (label, path))
        return path
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path.is_file():
            return path
    raise FileNotFoundError(
        "%s does not exist; checked %s"
        % (label, ", ".join(str(path) for path in candidates))
    )


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "noema_lab"
        ).is_dir():
            return candidate.resolve()
    raise FileNotFoundError("could not locate the Noema project root")


def _evidence(path: Path, base: Path) -> dict[str, str]:
    return {
        "path": os.path.relpath(path.resolve(), base.resolve()),
        "sha256": _sha256(path),
    }


def _project_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _integers(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _number_id(value: float) -> str:
    return ("%g" % float(value)).replace("-", "m").replace(".", "p") or "0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
