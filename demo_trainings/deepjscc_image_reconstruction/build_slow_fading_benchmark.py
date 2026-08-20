from __future__ import annotations

"""Build the blind slow-Rayleigh DeepJSCC demonstration campaign."""

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

from build_benchmark import (
    _assert_fresh_seeds,
    _binding_params,
    _file_spec,
    _integer_list,
    _load_json,
    _load_mapping,
    _number_id,
    _print_commands,
    _project_root,
    _relative_path,
    _validate_partition,
    _validate_source_recipes,
    _write_benchmark_recipe,
    _yaml_number,
)
from noema_lab.core.trained_artifacts import inspect_trained_artifact
from noema_lab.ops import build_registry


DEFAULT_SNR_DB = (0.0, 5.0, 10.0, 15.0, 20.0)
DEFAULT_SEEDS = (81001, 82001, 83001)
RATE_CHANNELS = (8, 16, 32)
HELD_OUT_IMAGE_IDS = ("kodim21", "kodim22", "kodim23", "kodim24")
RATE_SLICE_SNR_DB = 10.0
PREVIEW_SEED = DEFAULT_SEEDS[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the paired no-CSI slow-Rayleigh digital-separation versus "
            "nested-bandwidth DeepJSCC benchmark."
        )
    )
    parser.add_argument("--recipe", default="noema_recipe.yaml")
    parser.add_argument("--output", default="benchmark_pack.yaml")
    parser.add_argument("--project-root")
    parser.add_argument(
        "--digital-recipe",
        help="Slow-Rayleigh JPEG ideal-separation recipe.",
    )
    parser.add_argument(
        "--snr-db",
        default=",".join("%g" % value for value in DEFAULT_SNR_DB),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_SEEDS),
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    project_root = _project_root(args.project_root)
    learned_recipe = _load_mapping(
        Path(args.recipe).expanduser().resolve(), "DeepJSCC source recipe"
    )
    digital_path = (
        Path(args.digital_recipe).expanduser().resolve()
        if args.digital_recipe
        else project_root
        / "recipes"
        / "jpeg_capacity_matched_kodak_slow_rayleigh.yaml"
    )
    digital_recipe = _load_mapping(
        digital_path, "slow-Rayleigh digital source recipe"
    )
    _validate_source_recipes(learned_recipe, digital_recipe)
    _validate_slow_recipe(learned_recipe, digital_recipe)
    _validate_partition(learned_recipe)
    seeds = _integer_list(args.seeds, "seeds")
    _assert_fresh_seeds(learned_recipe, seeds)
    snr_grid = _float_list(args.snr_db)

    history_path = Path("training_history.json").resolve()
    evaluation_path = Path("evaluation_metrics.json").resolve()
    history = _load_json(history_path, "training history")
    evaluation = _load_json(evaluation_path, "validation metrics")
    if str(evaluation.get("split") or "") != "validation":
        raise ValueError("DeepJSCC example evaluation must remain validation-only")
    artifacts = _inspect_rate_artifacts(
        history,
        project_root=project_root,
        output_dir=output_path.parent,
    )

    recipe_dir = output_path.parent / "benchmark_recipes"
    learned_reference = _relative_path(
        _write_benchmark_recipe(
            learned_recipe, recipe_dir / "deepjscc_slow_rayleigh.yaml"
        ),
        output_path.parent,
    )
    digital_reference = _relative_path(
        _write_benchmark_recipe(
            digital_recipe, recipe_dir / "jpeg_slow_rayleigh.yaml"
        ),
        output_path.parent,
    )
    pack = _build_pack(
        learned_recipe_reference=learned_reference,
        digital_recipe_reference=digital_reference,
        artifacts=artifacts,
        history_evidence=_file_spec(history_path, output_path.parent),
        evaluation_evidence=_file_spec(evaluation_path, output_path.parent),
        snr_grid=snr_grid,
        seeds=seeds,
    )
    output_path.write_text(
        yaml.safe_dump(pack, sort_keys=False), encoding="utf-8"
    )
    _print_commands(output_path, project_root)
    return 0


def _inspect_rate_artifacts(
    history: Mapping[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
) -> dict[int, dict[str, Any]]:
    rows = [
        dict(item)
        for item in history.get("exported_rates") or []
        if isinstance(item, Mapping)
    ]
    by_channels: dict[int, dict[str, Any]] = {}
    registry = build_registry()
    for row in rows:
        channels = int(row.get("symbol_channels") or 0)
        if channels not in RATE_CHANNELS:
            continue
        path = Path(str(row.get("manifest") or "")).expanduser().resolve()
        inspected = inspect_trained_artifact(
            path, project_root=project_root, registry=registry
        )
        if not bool(inspected.get("ready")):
            raise ValueError(
                "κ=%.3g artifact is not runtime-ready: %s"
                % (
                    float(channels) / 64.0,
                    "; ".join(
                        str(item) for item in inspected.get("issues") or []
                    ),
                )
            )
        by_channels[channels] = {
            "sender": _binding_params(
                inspected,
                "model.deepjscc_external_encode",
                "encoder",
            ),
            "receiver": _binding_params(
                inspected,
                "model.deepjscc_external_decode",
                "decoder",
            ),
            "runtime_identity_sha256": str(
                inspected.get("runtime_identity_sha256")
                or inspected.get("package_sha256")
                or ""
            ),
            "evidence": _file_spec(path, output_dir),
        }
    missing = sorted(set(RATE_CHANNELS).difference(by_channels))
    if missing:
        raise ValueError(
            "training did not return every nested-bandwidth artifact; missing "
            "symbol-channel counts: %s" % ", ".join(map(str, missing))
        )
    return by_channels


def _build_pack(
    *,
    learned_recipe_reference: str,
    digital_recipe_reference: str,
    artifacts: Mapping[int, Mapping[str, Any]],
    history_evidence: Mapping[str, str],
    evaluation_evidence: Mapping[str, str],
    snr_grid: list[float],
    seeds: list[int],
) -> dict[str, Any]:
    recipes: list[dict[str, Any]] = []
    held_out_ids = ",".join(HELD_OUT_IMAGE_IDS)

    def append_pair(
        *,
        snr_db: float,
        channels: int,
        seed: int,
        method_suffix: str,
        preview: bool,
    ) -> None:
        kappa = float(channels) / 64.0
        cell = "snr_db=%s,kappa=%s" % (
            _number_id(snr_db),
            _number_id(kappa),
        )
        common = {
            "benchmark_held_out": True,
            "benchmark_channel_seed": int(seed),
            "aggregation_cell_id": cell,
            "channel_state_information": "none_at_transmitter_or_receiver",
            "fading_scope": "one_complex_gain_per_source_image",
            "resource_constraint": "kappa=%g" % kappa,
        }
        data_params = {
            "image_ids": held_out_ids,
            "crop_size": 256,
            "repeat_count": 1,
        }
        preview_params = (
            {"preview_count": 1, "preview_size": 160}
            if preview
            else {"preview_count": 0}
        )
        learned = artifacts[channels]
        id_suffix = method_suffix or "_snr_sweep"
        recipes.append(
            {
                "id": "digital_%s_c%d_seed%d%s"
                % (_number_id(snr_db), channels, seed, id_suffix),
                "label": (
                    "JPEG + ideal separation · %g dB · κ=%g · seed %d"
                    % (snr_db, kappa, seed)
                ),
                "role": "baseline",
                "path": digital_recipe_reference,
                "params": {
                    "method_id": "digital_separation%s" % method_suffix,
                    "metadata": {
                        **common,
                        "benchmark_method": (
                            "digital_separation%s" % method_suffix
                        ),
                    },
                    "step_params": {
                        "data": data_params,
                        "wireless_channel": {
                            "channel_model": "slow_rayleigh",
                            "snr_db": _yaml_number(snr_db),
                            "channel_uses_per_pixel": kappa,
                            "seed": int(seed),
                            "on_outage": "image_channel_mean",
                        },
                        "evaluation": preview_params,
                    },
                },
            }
        )
        recipes.append(
            {
                "id": "learned_%s_c%d_seed%d%s"
                % (_number_id(snr_db), channels, seed, id_suffix),
                "label": (
                    "Blind DeepJSCC · %g dB · κ=%g · seed %d"
                    % (snr_db, kappa, seed)
                ),
                "role": "candidate",
                "path": learned_recipe_reference,
                "params": {
                    "method_id": "learned_deepjscc%s" % method_suffix,
                    "metadata": {
                        **common,
                        "benchmark_method": (
                            "learned_deepjscc%s" % method_suffix
                        ),
                        "artifact_runtime_identity_sha256": learned[
                            "runtime_identity_sha256"
                        ],
                    },
                    "step_params": {
                        "data": data_params,
                        "sender": dict(learned["sender"]),
                        "receiver": dict(learned["receiver"]),
                        "wireless_channel": {
                            "channel": "flat_rayleigh",
                            "snr_db": _yaml_number(snr_db),
                            "wireless_backend": "numpy",
                            "receiver_processing": "none",
                            "fading_scope": "source_item",
                            "seed": int(seed),
                        },
                        "evaluation": preview_params,
                    },
                },
            }
        )

    for snr_db in snr_grid:
        for seed in seeds:
            append_pair(
                snr_db=snr_db,
                channels=32,
                seed=seed,
                method_suffix="",
                preview=(
                    seed == PREVIEW_SEED
                    and snr_db in {snr_grid[0], snr_grid[-1]}
                ),
            )
    for channels in RATE_CHANNELS:
        for seed in seeds:
            append_pair(
                snr_db=RATE_SLICE_SNR_DB,
                channels=channels,
                seed=seed,
                method_suffix="_rate_sweep",
                preview=False,
            )

    methods = [
        {
            "id": "digital_separation",
            "label": "JPEG + ideal separation",
            "role": "baseline",
        },
        {
            "id": "learned_deepjscc",
            "label": "Blind DeepJSCC",
            "role": "candidate",
        },
        {
            "id": "digital_separation_rate_sweep",
            "label": "JPEG + ideal separation (10 dB)",
            "role": "baseline",
        },
        {
            "id": "learned_deepjscc_rate_sweep",
            "label": "Blind DeepJSCC (10 dB)",
            "role": "candidate",
        },
    ]
    metrics = [
        _metric(
            "channel.snr_db",
            "wireless_channel",
            [
                "channel.jpeg_capacity_oracle",
                "wireless.channel",
            ],
            "channel",
            "dB",
            "neutral",
        ),
        _metric(
            "channel.uses_per_pixel",
            "wireless_channel",
            [
                "channel.jpeg_capacity_oracle",
                "wireless.channel",
            ],
            "rate",
            "complex_channel_use/source_pixel",
            "neutral",
        ),
        _metric(
            "quality.psnr_db",
            "evaluation",
            ["metrics.image_reconstruction"],
            "reconstruction",
            "dB",
            "higher_is_better",
        ),
        _metric(
            "quality.ms_ssim",
            "evaluation",
            ["metrics.image_reconstruction"],
            "reconstruction",
            "fraction",
            "higher_is_better",
        ),
        _metric(
            "quality.mse",
            "evaluation",
            ["metrics.image_reconstruction"],
            "reconstruction",
            "normalized_squared_error",
            "lower_is_better",
        ),
        _metric(
            "channel.outage_rate",
            "wireless_channel",
            ["channel.jpeg_capacity_oracle"],
            "reliability",
            "fraction",
            "lower_is_better",
            applicable_roles=["baseline"],
        ),
    ]
    demo = {
        "schema_version": 1,
        "slug": "deepjscc-slow-rayleigh",
        "title": "Blind DeepJSCC over slow Rayleigh fading",
        "summary": (
            "A nested-bandwidth learned image link is compared with JPEG plus "
            "an ideal capacity-achieving digital code. Neither learned endpoint "
            "receives the unknown per-image fading coefficient or pilots."
        ),
        "question": (
            "Can a continuous learned joint source-channel representation "
            "degrade gracefully when fixed-rate digital separation enters "
            "block-fading outage?"
        ),
        "tutorial": "../../../tutorials/deepjscc_slow_rayleigh.html",
        "held_constant": [
            "held-out Kodak images 21–24 and 256×256 center crops",
            "one complex Rayleigh gain held constant over each image",
            "no transmitter CSI, receiver CSI, or pilots",
            "the same per-image fading gains for learned and digital methods",
            "unit average transmitted-symbol power",
        ],
        "changed": [
            "average SNR",
            "bandwidth ratio κ",
            "held-out channel seed",
        ],
        "primary_metric": "quality.psnr_db",
        "comparison_axis": "channel.snr_db",
        "series": methods,
        "plots": [
            {
                "id": "quality-vs-average-snr",
                "title": "Reconstruction quality vs average SNR (κ=0.5)",
                "kind": "line",
                "x": "channel.snr_db",
                "y": "quality.psnr_db",
                "group": "benchmark_method",
                "method_order": [
                    "digital_separation",
                    "learned_deepjscc",
                ],
                "style": {"aggregation": "mean_ci", "y_scale": "linear"},
            },
            {
                "id": "quality-vs-bandwidth",
                "title": "Rate–distortion slice at 10 dB",
                "kind": "line",
                "x": "channel.uses_per_pixel",
                "y": "quality.psnr_db",
                "group": "benchmark_method",
                "method_order": [
                    "digital_separation_rate_sweep",
                    "learned_deepjscc_rate_sweep",
                ],
                "style": {"aggregation": "mean_ci", "y_scale": "linear"},
            },
            {
                "id": "perceptual-quality-vs-average-snr",
                "title": "MS-SSIM vs average SNR (κ=0.5)",
                "kind": "line",
                "x": "channel.snr_db",
                "y": "quality.ms_ssim",
                "group": "benchmark_method",
                "method_order": [
                    "digital_separation",
                    "learned_deepjscc",
                ],
                "style": {"aggregation": "mean_ci", "y_scale": "linear"},
            },
        ],
        "table_metrics": [
            "channel.snr_db",
            "channel.uses_per_pixel",
            "quality.psnr_db",
            "quality.ms_ssim",
            "channel.outage_rate",
        ],
        "training_evidence": [
            {
                "series": "learned_deepjscc",
                "trained_artifact_manifest": artifacts[32]["evidence"],
                "training_history": history_evidence,
                "evaluation_metrics": evaluation_evidence,
            }
        ],
    }
    return {
        "schema_version": 1,
        "id": "semantic_comm.deepjscc_slow_rayleigh_post_training_v1",
        "version": "1.0.0",
        "name": "Blind DeepJSCC over Slow Rayleigh Fading",
        "description": (
            "Held-out paired comparison of nested-bandwidth DeepJSCC and "
            "fixed-rate ideal digital separation under no-CSI block fading."
        ),
        "suite": {
            "id": "semantic_comm",
            "name": "Semantic Communication",
            "status": "experimental",
            "version": "v1-draft",
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
        "metrics": metrics,
        "baselines": [
            "digital_separation",
            "digital_separation_rate_sweep",
        ],
        "recipes": recipes,
        "metadata": {
            "benchmark_tier": "experimental",
            "protocol_revision": "blind-slow-rayleigh-deepjscc-v1",
            "snr_db_grid": snr_grid,
            "rate_slice_snr_db": RATE_SLICE_SNR_DB,
            "bandwidth_ratios": [
                float(value) / 64.0 for value in RATE_CHANNELS
            ],
            "held_out_seeds": seeds,
            "held_out_image_ids": list(HELD_OUT_IMAGE_IDS),
            "demo": demo,
        },
    }


def _metric(
    metric_id: str,
    source_step: str,
    source_operations: list[str],
    family: str,
    unit: str,
    direction: str,
    *,
    applicable_roles: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": metric_id,
        "definition_version": 1,
        "source_step": source_step,
        "family": family,
        "unit": unit,
        "direction": direction,
    }
    if len(source_operations) == 1:
        payload["source_operation"] = source_operations[0]
    else:
        payload["source_operations"] = source_operations
    if applicable_roles:
        payload["applicable_roles"] = applicable_roles
    return payload


def _validate_slow_recipe(
    learned: Mapping[str, Any], digital: Mapping[str, Any]
) -> None:
    learned_channel = next(
        dict(row)
        for row in learned.get("steps") or []
        if isinstance(row, Mapping)
        and str(row.get("id") or "") == "wireless_channel"
    )
    params = dict(learned_channel.get("params") or {})
    if (
        str(params.get("channel") or "") != "flat_rayleigh"
        or str(params.get("receiver_processing") or "") != "none"
        or str(params.get("fading_scope") or "") != "source_item"
    ):
        raise ValueError(
            "learned recipe must use blind source-item slow Rayleigh fading"
        )
    digital_channel = next(
        dict(row)
        for row in digital.get("steps") or []
        if isinstance(row, Mapping)
        and str(row.get("id") or "") == "wireless_channel"
    )
    if (
        str(
            (digital_channel.get("params") or {}).get("channel_model")
            or ""
        )
        != "slow_rayleigh"
    ):
        raise ValueError(
            "digital recipe must select channel_model=slow_rayleigh"
        )


def _float_list(value: str) -> list[float]:
    rows = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("snr-db must contain unique numeric values")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
