from __future__ import annotations

"""Build the paired fixed-budget digital-versus-DeepJSCC campaign."""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from noema_lab.core.trained_artifacts import inspect_trained_artifact
from noema_lab.ops import build_registry

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


DEFAULT_SNR_DB = (
    -6.0,
    -4.0,
    -2.0,
    0.0,
    4.0,
    8.0,
    12.0,
    16.0,
)
DEFAULT_HELD_OUT_SEEDS = (71001, 72001, 73001)
HELD_OUT_IMAGE_IDS = ("kodim21", "kodim22", "kodim23", "kodim24")
CHANNEL_USES_PER_PIXEL_BUDGET = 0.5
PREVIEW_SNR_DB = (-6.0, -4.0, 0.0, 16.0)
PREVIEW_SEED = DEFAULT_HELD_OUT_SEEDS[0]
STATISTICAL_UNIT = "held-out four-image Kodak evaluation at one SNR"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the post-training digital-separation/DeepJSCC benchmark."
    )
    parser.add_argument(
        "--artifact",
        help="Returned trained_artifact.yaml (defaults to train_config.yaml).",
    )
    parser.add_argument(
        "--recipe",
        default="noema_recipe.yaml",
        help="Canonical DeepJSCC scenario copied into this training bundle.",
    )
    parser.add_argument(
        "--capacity-recipe",
        help=(
            "SNR-adaptive JPEG plus ideal-capacity recipe "
            "(defaults to the project theoretical separation reference)."
        ),
    )
    parser.add_argument(
        "--output",
        default="benchmark_pack.yaml",
        help="Generated benchmark-pack path.",
    )
    parser.add_argument(
        "--project-root",
        help="Noema project root (defaults to train_config.yaml).",
    )
    parser.add_argument(
        "--snr-db",
        default=",".join(str(value) for value in DEFAULT_SNR_DB),
        help="Comma-separated held-out SNR grid in dB.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_HELD_OUT_SEEDS),
        help="Comma-separated held-out channel-noise seeds for DeepJSCC.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    project_root = _project_root(args.project_root)
    recipe_path = Path(args.recipe).expanduser().resolve()
    capacity_recipe_path = (
        Path(args.capacity_recipe).expanduser().resolve()
        if args.capacity_recipe
        else project_root
        / "recipes"
        / "jpeg_capacity_matched_kodak_awgn.yaml"
    )
    artifact_path = _artifact_path(args.artifact)
    snr_grid = _finite_floats(args.snr_db, "snr-db")
    seeds = _integer_list(args.seeds, "seeds")

    recipe = _load_mapping(recipe_path, "DeepJSCC source recipe")
    capacity_recipe = _load_mapping(
        capacity_recipe_path, "capacity-matched digital source recipe"
    )
    _validate_source_recipes(recipe, capacity_recipe)
    _validate_partition(recipe)
    _assert_fresh_seeds(recipe, seeds)

    inspected = inspect_trained_artifact(
        artifact_path,
        project_root=project_root,
        registry=build_registry(),
    )
    if not bool(inspected.get("ready")):
        raise ValueError(
            "returned paired DeepJSCC artifact is not runtime-ready: %s"
            % "; ".join(str(item) for item in inspected.get("issues") or [])
        )
    sender_params = _binding_params(
        inspected, "model.deepjscc_external_encode", "encoder"
    )
    receiver_params = _binding_params(
        inspected, "model.deepjscc_external_decode", "decoder"
    )
    runtime_identity = str(
        inspected.get("runtime_identity_sha256")
        or inspected.get("package_sha256")
        or ""
    )
    if re.fullmatch(r"[0-9a-f]{64}", runtime_identity) is None:
        raise ValueError("returned artifact runtime identity is missing")

    history_path = Path("training_history.json").resolve()
    evaluation_path = Path("evaluation_metrics.json").resolve()
    _load_json(history_path, "training history")
    evaluation = _load_json(evaluation_path, "validation metrics")
    if str(evaluation.get("split") or "") != "validation":
        raise ValueError("DeepJSCC example evaluation must remain validation-only")

    benchmark_recipe_dir = output_path.parent / "benchmark_recipes"
    deepjscc_benchmark_recipe = _write_benchmark_recipe(
        recipe,
        benchmark_recipe_dir / "deepjscc.yaml",
    )
    capacity_benchmark_recipe = _write_benchmark_recipe(
        capacity_recipe,
        benchmark_recipe_dir / "jpeg_capacity.yaml",
    )
    pack = _build_pack(
        deepjscc_recipe_reference=_relative_path(
            deepjscc_benchmark_recipe, output_path.parent
        ),
        capacity_recipe_reference=_relative_path(
            capacity_benchmark_recipe, output_path.parent
        ),
        sender_params=sender_params,
        receiver_params=receiver_params,
        runtime_identity_sha256=runtime_identity,
        artifact_evidence=_file_spec(artifact_path, output_path.parent),
        history_evidence=_file_spec(history_path, output_path.parent),
        evaluation_evidence=_file_spec(evaluation_path, output_path.parent),
        snr_grid=snr_grid,
        seeds=seeds,
    )
    output_path.write_text(yaml.safe_dump(pack, sort_keys=False), encoding="utf-8")
    _print_commands(output_path, project_root)
    return 0


def _build_pack(
    *,
    deepjscc_recipe_reference: str,
    capacity_recipe_reference: str,
    sender_params: Mapping[str, Any],
    receiver_params: Mapping[str, Any],
    runtime_identity_sha256: str,
    artifact_evidence: Mapping[str, str],
    history_evidence: Mapping[str, str],
    evaluation_evidence: Mapping[str, str],
    snr_grid: list[float],
    seeds: list[int],
) -> dict[str, Any]:
    recipes: list[dict[str, Any]] = []
    held_out_ids = ",".join(HELD_OUT_IMAGE_IDS)
    for snr_db in snr_grid:
        preview_params = (
            {"preview_count": 1, "preview_size": 160}
            if float(snr_db) in set(PREVIEW_SNR_DB)
            else {"preview_count": 0}
        )
        recipes.append(
            {
                "id": "jpeg_capacity_snr%s" % _number_id(snr_db),
                "label": (
                    "JPEG + ideal capacity, adaptive quality · %g dB"
                    % snr_db
                ),
                "role": "baseline",
                "path": capacity_recipe_reference,
                "params": {
                    "method_id": "jpeg_capacity",
                    "metadata": {
                        "benchmark_held_out": True,
                        "aggregation_cell_id": "snr_db=%s"
                        % _number_id(snr_db),
                        "benchmark_method": "jpeg_capacity",
                        "deterministic_reference": True,
                        "resource_constraint": (
                            "exact_0.5_complex_channel_use_per_source_pixel"
                        ),
                        "statistical_unit": STATISTICAL_UNIT,
                    },
                    "step_params": {
                        "data": {
                            "image_ids": held_out_ids,
                            "crop_size": 256,
                            "repeat_count": 1,
                        },
                        "wireless_channel": {
                            "snr_db": _yaml_number(snr_db),
                            "channel_uses_per_pixel": (
                                CHANNEL_USES_PER_PIXEL_BUDGET
                            ),
                            "minimum_quality": 1,
                            "maximum_quality": 95,
                            "subsampling": "420",
                            "optimize": False,
                            "progressive": False,
                            "on_outage": "gray_image",
                        },
                        "evaluation": preview_params,
                    },
                },
            }
        )
        for paired_seed in seeds:
            common_metadata = {
                "benchmark_channel_seed": int(paired_seed),
                "benchmark_held_out": True,
                "aggregation_cell_id": "snr_db=%s" % _number_id(snr_db),
                "statistical_unit": STATISTICAL_UNIT,
            }
            recipes.append(
                {
                    "id": "learned_deepjscc_snr%s_seed%d"
                    % (_number_id(snr_db), paired_seed),
                    "label": "Learned DeepJSCC · %g dB · seed %d"
                    % (snr_db, paired_seed),
                    "role": "candidate",
                    "path": deepjscc_recipe_reference,
                    "params": {
                        "method_id": "learned_deepjscc",
                        "metadata": {
                            **common_metadata,
                            "benchmark_method": "learned_deepjscc",
                            "artifact_runtime_identity_sha256": runtime_identity_sha256,
                            "resource_constraint": "exact_0.5_complex_channel_use_per_source_pixel",
                        },
                        "step_params": {
                            "data": {
                                "image_ids": held_out_ids,
                                "crop_size": 256,
                                "repeat_count": 1,
                            },
                            "sender": dict(sender_params),
                            "receiver": dict(receiver_params),
                            "wireless_channel": {
                                "channel": "awgn",
                                "snr_db": _yaml_number(snr_db),
                                "wireless_backend": "sionna",
                                "seed": int(paired_seed),
                            },
                            "evaluation": (
                                {"preview_count": 1, "preview_size": 160}
                                if paired_seed == PREVIEW_SEED
                                and float(snr_db) in set(PREVIEW_SNR_DB)
                                else {"preview_count": 0}
                            ),
                        },
                    },
                }
            )

    demo = {
        "schema_version": 1,
        "slug": "digital-versus-deepjscc",
        "title": "Digital separation versus learned joint source-channel coding",
        "summary": (
            "A jointly trained DeepJSCC endpoint pair is compared with an "
            "SNR-adaptive JPEG separation upper bound that assumes ideal "
            "capacity-achieving AWGN channel coding under the same fixed "
            "channel-use budget."
        ),
        "question": (
            "At one fixed bandwidth ratio, where does a single robust DeepJSCC "
            "model outperform or trail ideal SNR-adaptive JPEG separation?"
        ),
        "tutorial": "../../../tutorials/digital_vs_deepjscc_sionna.html",
        "held_constant": [
            "held-out Kodak images 21–24 and deterministic 256×256 center crops",
            "exactly 0.5 complex channel uses per spatial source pixel",
            "unit average transmitted-symbol power",
            "complex AWGN capacity log2(1+SNR)",
            "the same SNR coordinate",
        ],
        "changed": [
            "ideal SNR-adaptive JPEG separation versus joint learned endpoint pair",
            "channel SNR",
            "held-out learned-channel seed",
        ],
        "primary_metric": "quality.psnr_db",
        "comparison_axis": "channel.snr_db",
        "series": [
            {
                "id": "jpeg_capacity",
                "label": "JPEG + ideal capacity (adaptive quality)",
                "role": "baseline",
            },
            {
                "id": "learned_deepjscc",
                "label": "Learned DeepJSCC",
                "role": "candidate",
            },
        ],
        "plots": [
            {
                "id": "quality-vs-snr",
                "title": "Reconstruction quality vs SNR",
                "kind": "line",
                "x": "channel.snr_db",
                "y": "quality.psnr_db",
                "group": "benchmark_method",
                "method_order": [
                    "jpeg_capacity",
                    "learned_deepjscc",
                ],
                "style": {"aggregation": "mean_ci", "y_scale": "linear"},
            },
            {
                "id": "perceptual-quality-vs-snr",
                "title": "MS-SSIM vs SNR",
                "kind": "line",
                "x": "channel.snr_db",
                "y": "quality.ms_ssim",
                "group": "benchmark_method",
                "method_order": [
                    "jpeg_capacity",
                    "learned_deepjscc",
                ],
                "style": {"aggregation": "mean_ci", "y_scale": "linear"},
            },
        ],
        "table_metrics": [
            "channel.snr_db",
            "quality.psnr_db",
            "quality.ms_ssim",
            "quality.mse",
            "channel.channel_use_count",
            "channel.uses_per_pixel",
            "channel.awgn.capacity_bpp",
            "rate.native_codec_bpp",
            "codec.jpeg.selected_quality_mean",
        ],
        "training_evidence": [
            {
                "series": "learned_deepjscc",
                "trained_artifact_manifest": artifact_evidence,
                "training_history": history_evidence,
                "evaluation_metrics": evaluation_evidence,
            }
        ],
    }
    return {
        "schema_version": 1,
        "id": "semantic_comm.digital_vs_deepjscc_post_training_v3",
        "version": "3.0.0",
        "name": "Digital Separation versus DeepJSCC",
        "description": (
            "A held-out fixed-bandwidth AWGN campaign comparing a returned "
            "DeepJSCC endpoint pair with an SNR-adaptive JPEG plus ideal "
            "capacity-achieving channel-code upper bound."
        ),
        "suite": {
            "id": "semantic_comm",
            "name": "Semantic Communication",
            "status": "active",
            "version": "v1",
        },
        "dataset": {
            "id": "kodak",
            "modality": "image",
            "version": "kodak-heldout-21-24-center-crop-256-v1",
            "split": "held_out_test",
        },
        "task": {
            "id": "image_reconstruction",
            "kind": "reconstruction",
            "modality": "image",
        },
        "metrics": [
            {
                "id": "channel.snr_db",
                "definition_version": 1,
                "source_step": "wireless_channel",
                "source_operations": [
                    "channel.jpeg_capacity_oracle",
                    "wireless.channel",
                ],
                "family": "channel",
                "unit": "dB",
                "direction": "neutral",
            },
            {
                "id": "quality.psnr_db",
                "definition_version": 1,
                "source_step": "evaluation",
                "source_operation": "metrics.image_reconstruction",
                "family": "reconstruction",
                "unit": "dB",
                "direction": "higher_is_better",
            },
            {
                "id": "quality.mse",
                "definition_version": 1,
                "source_step": "evaluation",
                "source_operation": "metrics.image_reconstruction",
                "family": "reconstruction",
                "unit": "normalized_squared_error",
                "direction": "lower_is_better",
            },
            {
                "id": "quality.ms_ssim",
                "definition_version": 1,
                "source_step": "evaluation",
                "source_operation": "metrics.image_reconstruction",
                "family": "reconstruction",
                "unit": "fraction",
                "direction": "higher_is_better",
            },
            {
                "id": "channel.awgn.capacity_bpp",
                "definition_version": 1,
                "applicable_roles": ["baseline"],
                "source_step": "wireless_channel",
                "source_operation": "channel.jpeg_capacity_oracle",
                "family": "rate",
                "unit": "bit/source_pixel",
                "direction": "neutral",
            },
            {
                "id": "channel.channel_use_count",
                "definition_version": 1,
                "source_step": "wireless_channel",
                "source_operations": [
                    "channel.jpeg_capacity_oracle",
                    "wireless.channel",
                ],
                "family": "rate",
                "unit": "complex_channel_use",
                "direction": "neutral",
            },
            {
                "id": "channel.uses_per_pixel",
                "definition_version": 1,
                "source_step": "wireless_channel",
                "source_operations": [
                    "channel.jpeg_capacity_oracle",
                    "wireless.channel",
                ],
                "family": "rate",
                "unit": "complex_channel_use/source_pixel",
                "direction": "neutral",
            },
            {
                "id": "rate.native_codec_bpp",
                "definition_version": 1,
                "applicable_roles": ["baseline"],
                "source_step": "wireless_channel",
                "source_operation": "channel.jpeg_capacity_oracle",
                "family": "rate",
                "unit": "bit/source_pixel",
                "direction": "neutral",
            },
            {
                "id": "codec.jpeg.selected_quality_mean",
                "definition_version": 1,
                "applicable_roles": ["baseline"],
                "source_step": "wireless_channel",
                "source_operation": "channel.jpeg_capacity_oracle",
                "family": "codec",
                "unit": "jpeg_quality_factor",
                "direction": "neutral",
            },
            {
                "id": "codec.jpeg.selected_quality_min",
                "definition_version": 1,
                "applicable_roles": ["baseline"],
                "source_step": "wireless_channel",
                "source_operation": "channel.jpeg_capacity_oracle",
                "family": "codec",
                "unit": "jpeg_quality_factor",
                "direction": "neutral",
            },
            {
                "id": "codec.jpeg.selected_quality_max",
                "definition_version": 1,
                "applicable_roles": ["baseline"],
                "source_step": "wireless_channel",
                "source_operation": "channel.jpeg_capacity_oracle",
                "family": "codec",
                "unit": "jpeg_quality_factor",
                "direction": "neutral",
            },
        ],
        "baselines": ["jpeg_capacity"],
        "recipes": recipes,
        "metadata": {
            "benchmark_tier": "experimental",
            "protocol_revision": "digital-vs-deepjscc-capacity-matched-v3",
            "learned_held_out_channel_seeds": list(seeds),
            "snr_db_grid": list(snr_grid),
            "held_out_image_ids": list(HELD_OUT_IMAGE_IDS),
            "resource_budget": {
                "metric": (
                    "steps.wireless_channel.channel."
                    "max_source_item_uses_per_pixel"
                ),
                "maximum": CHANNEL_USES_PER_PIXEL_BUDGET,
                "tolerance": 1.0e-9,
                "policy": "reject",
                "unit": "complex_channel_use/source_pixel",
            },
            "statistical_unit": STATISTICAL_UNIT,
            "demo": demo,
        },
    }


def _binding_params(
    inspected: Mapping[str, Any], operation: str, entrypoint: str
) -> dict[str, Any]:
    bindings = [
        dict(item)
        for item in inspected.get("compatible_operations") or []
        if isinstance(item, Mapping)
        and str(item.get("operation") or "") == operation
    ]
    if len(bindings) != 1:
        raise ValueError("returned artifact must bind exactly one %s operation" % operation)
    binding = bindings[0]
    if not bool(binding.get("available")):
        raise ValueError("returned artifact binding is unavailable for %s" % operation)
    if str(binding.get("runtime_entrypoint") or "") != entrypoint:
        raise ValueError("%s must bind runtime entrypoint %s" % (operation, entrypoint))
    params = dict(binding.get("params") or {})
    if str(params.get("runtime") or "") != "learned_artifact":
        raise ValueError("%s binding must select runtime=learned_artifact" % operation)
    return params


def _validate_source_recipes(
    deepjscc_recipe: Mapping[str, Any],
    capacity_recipe: Mapping[str, Any],
) -> None:
    _require_step(deepjscc_recipe, "sender", "model.deepjscc_external_encode")
    _require_step(deepjscc_recipe, "receiver", "model.deepjscc_external_decode")
    _require_step(deepjscc_recipe, "wireless_channel", "wireless.channel")
    _require_step(
        capacity_recipe,
        "wireless_channel",
        "channel.jpeg_capacity_oracle",
    )
    _require_step(
        capacity_recipe, "evaluation", "metrics.image_reconstruction"
    )


def _write_benchmark_recipe(recipe: Mapping[str, Any], path: Path) -> Path:
    payload = dict(recipe)
    metadata = dict(payload.get("metadata") or {})
    metadata.pop("matrix", None)
    metadata.pop("sweeps", None)
    metadata.pop("ui_sweeps", None)
    metadata.pop("matrix_selection", None)
    metadata.pop("matrix_variant_id", None)
    metadata["benchmark_base_recipe"] = True
    payload["metadata"] = metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _validate_partition(recipe: Mapping[str, Any]) -> None:
    data = _require_step(recipe, "data", "source.image_dataset")
    raw_ids = str((data.get("params") or {}).get("image_ids") or "")
    train_ids = {item.strip() for item in raw_ids.split(",") if item.strip()}
    overlap = sorted(train_ids.intersection(HELD_OUT_IMAGE_IDS))
    if overlap:
        raise ValueError(
            "DeepJSCC training recipe exposes held-out benchmark image(s): %s"
            % ", ".join(overlap)
        )


def _assert_fresh_seeds(recipe: Mapping[str, Any], seeds: list[int]) -> None:
    channel = _require_step(recipe, "wireless_channel", "wireless.channel")
    used = {int((channel.get("params") or {}).get("seed") or 0), 9001, 19001}
    overlap = sorted(set(seeds).intersection(used))
    if overlap:
        raise ValueError(
            "benchmark seeds overlap training/validation seed(s): %s"
            % ", ".join(str(item) for item in overlap)
        )


def _require_step(
    recipe: Mapping[str, Any], step_id: str, operation: str
) -> dict[str, Any]:
    matches = [
        dict(item)
        for item in recipe.get("steps") or []
        if isinstance(item, Mapping) and str(item.get("id") or "") == step_id
    ]
    if len(matches) != 1:
        raise ValueError("recipe must contain exactly one step %s" % step_id)
    if str(matches[0].get("op") or "") != operation:
        raise ValueError("step %s must use operation %s" % (step_id, operation))
    return matches[0]


def _artifact_path(argument: str | None) -> Path:
    if argument:
        return Path(argument).expanduser().resolve()
    config = _load_mapping(Path("train_config.yaml"), "train_config.yaml")
    configured = str(
        (config.get("training") or {}).get("artifact_manifest_path") or ""
    ).strip()
    if not configured:
        raise ValueError("train_config.yaml does not declare an artifact manifest")
    return Path(configured).expanduser().resolve()


def _project_root(argument: str | None) -> Path:
    if argument:
        return Path(argument).expanduser().resolve()
    config = _load_mapping(Path("train_config.yaml"), "train_config.yaml")
    configured = str(config.get("project_root") or "").strip()
    if not configured:
        raise ValueError("train_config.yaml does not declare project_root")
    return Path(configured).expanduser().resolve()


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("%s does not exist: %s" % (label, resolved))
    payload = load_strict_yaml_or_json(resolved)
    if not isinstance(payload, Mapping):
        raise ValueError("%s must contain a mapping" % label)
    return dict(payload)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("%s does not exist: %s" % (label, resolved))
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("%s must contain a mapping" % label)
    return dict(payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_spec(path: Path, relative_to: Path) -> dict[str, str]:
    return {
        "path": _relative_path(path, relative_to),
        "sha256": _file_sha256(path),
    }


def _relative_path(path: Path, base: Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(base).resolve())


def _finite_floats(value: str, label: str) -> list[float]:
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or any(not math.isfinite(item) for item in values):
        raise ValueError("%s must contain finite numeric values" % label)
    if len(values) != len(set(values)):
        raise ValueError("%s values must be unique" % label)
    return values


def _integer_list(value: str, label: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("%s must contain unique integer values" % label)
    return values


def _number_id(value: float) -> str:
    text = ("%g" % float(value)).replace("-", "m").replace(".", "p")
    return text


def _yaml_number(value: float) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _print_commands(output_path: Path, project_root: Path) -> None:
    print("wrote benchmark pack: %s" % output_path)
    print("from the Noema project root, validate and run:")
    print("  cd %s" % project_root)
    print("  noema benchmark validate %s" % output_path)
    print("  noema benchmark run %s" % output_path)


if __name__ == "__main__":
    raise SystemExit(main())
