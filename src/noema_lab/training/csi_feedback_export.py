from __future__ import annotations

import hashlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from noema_lab.core.operations import OperationRegistry
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import Recipe, RecipeStep
from noema_lab.core.reproducibility import (
    canonical_json_sha256,
    master_seed_from_recipe,
)
from noema_lab.core.structured_input import load_strict_yaml_or_json
from noema_lab.core.training_plans import scenario_recipe_fingerprint
from noema_lab.training.capture_plan import (
    TrainingCapturePlan,
    TrainingCapturePlanError,
    resolve_training_capture_plan,
)
from noema_lab.training.starter_refresh import prepare_demo_starter_directory
from noema_lab.training.package_layout import find_demo_training_dir
from noema_lab.training.standalone_input import write_standalone_structured_input


JsonDict = Dict[str, Any]

CSI_FEEDBACK_LOSS = "csi.normalized_reconstruction_mse"
CSI_FEEDBACK_TEMPLATE = "csi_feedback.klt_initialized_quantization_residual_v3"
CSI_FEEDBACK_EXAMPLE_LOSS = "csi.hybrid_nmse_mrt_rate"
CSI_FEEDBACK_ENCODER_OP = "model.csi_feedback_encoder"
CSI_FEEDBACK_DECODER_OP = "model.csi_feedback_decoder"


class CsiFeedbackExportError(ValueError):
    pass


def _configured_capture_integer(
    value: Any,
    *,
    field: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CsiFeedbackExportError(
            "%s must be an integer greater than or equal to %d"
            % (field, minimum)
        )
    if value < minimum:
        raise CsiFeedbackExportError(
            "%s must be greater than or equal to %d" % (field, minimum)
        )
    return value


@dataclass(frozen=True)
class CsiFeedbackExportPlan:
    recipe: Recipe
    recipe_sha256: str
    channel_step: RecipeStep
    encoder_step: RecipeStep
    feedback_link_step: RecipeStep
    decoder_step: RecipeStep
    precoder_step: RecipeStep
    metrics_step: RecipeStep
    framework: str
    loss: str
    true_csi_tap: str
    true_csi_reference: str
    reconstruction_reference: str
    tx_antennas: int
    subcarrier_count: int
    feedback_dimension: int
    bits_per_latent: int
    feedback_bits_per_sample: int
    clip_value: float
    feedback_mode: str
    downlink_snr_db: float
    capture_plan: TrainingCapturePlan
    project_root: Optional[Path] = None
    template_id: str = CSI_FEEDBACK_TEMPLATE


def is_csi_feedback_slot_pair(steps: Sequence[RecipeStep]) -> bool:
    return len(steps) == 2 and {step.op for step in steps} == {
        CSI_FEEDBACK_ENCODER_OP,
        CSI_FEEDBACK_DECODER_OP,
    }


def suggested_csi_feedback_capture_total(recipe: Recipe) -> int:
    """Return a research-scale recommendation without changing explicit splits.

    ``resolve_training_capture_plan`` gives an explicitly configured split plan
    precedence over this suggestion.  The recommendation can therefore grow as
    the demonstration trainer becomes more capable without silently changing an
    existing recipe's requested train/validation/test record counts.
    """

    capture = dict(recipe.dataset_capture or {})
    split_plan = capture["split_plan"] if "split_plan" in capture else {}
    if not isinstance(split_plan, Mapping):
        raise CsiFeedbackExportError("dataset_capture.split_plan must be a mapping")
    candidates = [12288]
    if "total_samples" in split_plan:
        candidates.append(
            _configured_capture_integer(
                split_plan["total_samples"],
                field="dataset_capture.split_plan.total_samples",
                minimum=3,
            )
        )
    if "samples" in capture:
        candidates.append(
            _configured_capture_integer(
                capture["samples"],
                field="dataset_capture.samples",
                minimum=3,
            )
        )
    source = next(
        (step for step in recipe.steps if step.op == "wireless.miso_ofdm_csi"),
        None,
    )
    source_params = source.params if source is not None else {}
    one_batch = (
        _configured_capture_integer(
            source_params["sample_count"],
            field="wireless.miso_ofdm_csi.sample_count",
            minimum=1,
        )
        if "sample_count" in source_params
        else 64
    )
    candidates.append(3 * one_batch)
    return max(candidates)


def build_csi_feedback_export_plan(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    optimizable_steps: Sequence[str],
    loss: str,
    framework: str,
) -> CsiFeedbackExportPlan:
    validate_recipe_against_registry(recipe, registry)
    normalized_framework = str(framework or "torch").strip().lower().replace("_", "-")
    if normalized_framework not in {"torch", "torch-sionna"}:
        raise CsiFeedbackExportError(
            "CSI-feedback example training requires framework=torch or torch-sionna"
        )
    external_loss_identifier = str(loss or "").strip()
    if not external_loss_identifier:
        raise CsiFeedbackExportError(
            "CSI-feedback training requires a nonempty external loss identifier"
        )

    selected = [_step(recipe, str(step_id)) for step_id in optimizable_steps]
    if not is_csi_feedback_slot_pair(selected):
        raise CsiFeedbackExportError(
            "CSI-feedback training requires the paired model.csi_feedback_encoder and "
            "model.csi_feedback_decoder slots"
        )
    encoder = next(step for step in selected if step.op == CSI_FEEDBACK_ENCODER_OP)
    decoder = next(step for step in selected if step.op == CSI_FEEDBACK_DECODER_OP)
    channel = _producer(recipe, encoder, "csi")
    if channel.op != "wireless.miso_ofdm_csi":
        raise CsiFeedbackExportError(
            "CSI-feedback encoder input must come from wireless.miso_ofdm_csi.csi"
        )
    true_csi_reference = str(encoder.inputs.get("csi") or "")
    if true_csi_reference != "%s.csi" % channel.id:
        raise CsiFeedbackExportError(
            "CSI-feedback encoder input must reference %s.csi" % channel.id
        )

    feedback_link = _single_consumer(
        recipe,
        "%s.feedback_code" % encoder.id,
        expected_op="channel.csi_feedback_link",
    )
    if str(feedback_link.inputs.get("feedback_code") or "") != "%s.feedback_code" % encoder.id:
        raise CsiFeedbackExportError(
            "CSI feedback link must consume the selected encoder feedback_code"
        )
    if str(decoder.inputs.get("received_code") or "") != "%s.received_code" % feedback_link.id:
        raise CsiFeedbackExportError(
            "CSI feedback decoder must consume the feedback-link received_code"
        )
    precoder = _single_consumer(
        recipe,
        "%s.reconstruction" % decoder.id,
        expected_op="model.csi_mrt_precoder",
    )
    metrics = _single_consumer(
        recipe,
        "%s.precoder" % precoder.id,
        expected_op="metrics.csi_feedback",
    )
    expected_metric_inputs = {
        "true_csi": true_csi_reference,
        "reconstruction": "%s.reconstruction" % decoder.id,
        "precoder": "%s.precoder" % precoder.id,
    }
    mismatched = [
        name
        for name, reference in expected_metric_inputs.items()
        if str(metrics.inputs.get(name) or "") != reference
    ]
    if mismatched:
        raise CsiFeedbackExportError(
            "CSI feedback metrics must consume the common true CSI, reconstruction, and "
            "precoder; mismatched input(s): %s" % ", ".join(mismatched)
        )

    tx_antennas = _positive_int(
        channel.params.get("tx_antennas", 8), "tx_antennas"
    )
    subcarriers = _positive_int(
        channel.params.get("ofdm_fft_size", 32), "ofdm_fft_size"
    )
    feedback_dimension = _positive_int(
        encoder.params.get("feedback_dimension", 64), "feedback_dimension"
    )
    decoder_dimension = _positive_int(
        decoder.params.get("feedback_dimension", 64), "decoder feedback_dimension"
    )
    if decoder_dimension != feedback_dimension:
        raise CsiFeedbackExportError(
            "Encoder and decoder feedback_dimension values must match"
        )
    bits_per_latent = _positive_int(
        feedback_link.params.get("bits_per_latent", 8), "bits_per_latent"
    )
    clip_value = _positive_float(
        feedback_link.params.get("clip_value", 1.0), "clip_value"
    )
    downlink_snr_db = _finite_float(
        channel.params.get("downlink_snr_db", 10.0), "downlink_snr_db"
    )
    feedback_mode = str(feedback_link.params.get("mode") or "ideal_noiseless")
    if feedback_mode not in {"ideal_noiseless", "uniform_quantized"}:
        raise CsiFeedbackExportError(
            "The CSI-feedback starter supports mode=ideal_noiseless or uniform_quantized; "
            "got %s" % feedback_mode
        )

    true_csi_tap = _configured_tap_id(recipe, true_csi_reference, "true_csi")
    try:
        capture_plan = resolve_training_capture_plan(
            recipe,
            registry,
            required_taps=(
                {
                    "id": true_csi_tap,
                    "from": true_csi_reference,
                    "role": "self_supervised_csi_source_and_target",
                },
            ),
            sample_unit="CSI realizations",
            suggested_total_samples=suggested_csi_feedback_capture_total(recipe),
        )
    except TrainingCapturePlanError as exc:
        raise CsiFeedbackExportError(str(exc)) from exc
    if len(capture_plan.taps) != 1 or capture_plan.taps[0].reference != true_csi_reference:
        raise CsiFeedbackExportError(
            "CSI-feedback capture must contain exactly the true-CSI tap %s"
            % true_csi_reference
        )

    return CsiFeedbackExportPlan(
        recipe=recipe,
        recipe_sha256=scenario_recipe_fingerprint(recipe),
        channel_step=channel,
        encoder_step=encoder,
        feedback_link_step=feedback_link,
        decoder_step=decoder,
        precoder_step=precoder,
        metrics_step=metrics,
        framework="torch",
        loss=external_loss_identifier,
        true_csi_tap=true_csi_tap,
        true_csi_reference=true_csi_reference,
        reconstruction_reference="%s.reconstruction" % decoder.id,
        tx_antennas=tx_antennas,
        subcarrier_count=subcarriers,
        feedback_dimension=feedback_dimension,
        bits_per_latent=bits_per_latent,
        feedback_bits_per_sample=feedback_dimension * bits_per_latent,
        clip_value=clip_value,
        feedback_mode=feedback_mode,
        downlink_snr_db=downlink_snr_db,
        capture_plan=capture_plan,
    )


def write_csi_feedback_data_contract(
    plan: CsiFeedbackExportPlan,
    out_dir: Path,
    *,
    project_root: Path,
) -> JsonDict:
    """Write deterministic self-supervised CSI capture assets.

    Capture recipes intentionally contain only the Sionna-backed channel source.
    Export-only encoder/decoder placeholders therefore never execute while data is
    being collected.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_recipes = []
    for split_spec in plan.capture_plan.splits:
        payload = _capture_recipe(
            plan,
            split=split_spec.id,
            seed_offset=split_spec.seed_offset,
            samples=split_spec.samples,
        )
        path = out_dir / ("capture_%s_recipe.yaml" % split_spec.id)
        _write_yaml(path, payload)
        capture_recipes.append(
            {
                "split": split_spec.id,
                "path": path.name,
                "sha256": canonical_json_sha256(payload),
                "file_sha256": _file_sha256(path),
                "requested_samples": split_spec.samples,
                "seed_offset": split_spec.seed_offset,
                "taps": [{"id": plan.true_csi_tap, "from": plan.true_csi_reference}],
            }
        )

    contract: JsonDict = {
        "schema_version": 1,
        "kind": "noema.training_data_contract@1",
        "mode": "captured_self_supervised_csi",
        "source_recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
            "execution_profile": plan.recipe.execution_profile.to_dict(),
        },
        "ownership": {
            "capture_recipe_generation": "noema",
            "capture_execution": "noema",
            "capture_integrity_validation": "noema",
            "dataset_consumption": "external_researcher",
            "model_architecture": "external_researcher",
            "loss_definition": "external_researcher",
            "model_training": "external_researcher",
        },
        "capture_plan": plan.capture_plan.to_dict(),
        "feature_and_target": {
            "tap_id": plan.true_csi_tap,
            "reference": plan.true_csi_reference,
            "kind": "channel.miso_ofdm_csi.numpy",
            "stored_dtype": "float32",
            "stored_shape": [
                "sample",
                2,
                plan.tx_antennas,
                plan.subcarrier_count,
            ],
            "layout": "sample_real_imag_tx_antenna_subcarrier",
            "semantic": "real_and_imaginary_parts_of_realized_downlink_csi",
            "pairing": "autoencoder_input_equals_reconstruction_target",
        },
        "runtime_interface": {
            "encoder_input": {
                "name": "csi_ri",
                "shape": ["batch", 2, plan.tx_antennas, plan.subcarrier_count],
            },
            "decoder_output": {
                "name": "csi_hat_ri",
                "shape": ["batch", 2, plan.tx_antennas, plan.subcarrier_count],
            },
        },
        "feedback_constraint": {
            "feedback_dimension": plan.feedback_dimension,
            "bits_per_latent": plan.bits_per_latent,
            "feedback_bits_per_sample": plan.feedback_bits_per_sample,
            "clip_value": plan.clip_value,
            "transport": plan.feedback_mode,
        },
        "labels": {
            "required": False,
            "separate_target_captured": False,
            "reconstruction_or_precoder_output_captured": False,
        },
        "splits": [
            {
                "id": item.id,
                "requested_samples": item.samples,
                "seed_offset": item.seed_offset,
                "training_use": item.training_use,
            }
            for item in plan.capture_plan.splits
        ],
        "capture_recipes": capture_recipes,
    }
    contract_path = out_dir / "data_contract.yaml"
    _write_yaml(contract_path, contract)
    jobs = _capture_jobs(plan, out_dir=out_dir, project_root=project_root)
    return {
        "data_contract": contract,
        "data_contract_sha256": canonical_json_sha256(contract),
        "data_contract_file_sha256": _file_sha256(contract_path),
        "capture_recipes": capture_recipes,
        "capture_jobs": jobs,
        "files": [
            "data_contract.yaml",
            "capture_train_recipe.yaml",
            "capture_validation_recipe.yaml",
            "capture_test_recipe.yaml",
        ],
    }


def write_csi_feedback_starter(
    plan: CsiFeedbackExportPlan,
    out_dir: Path,
    *,
    force: bool = False,
) -> JsonDict:
    """Copy a replaceable demonstration trainer beside the normative contract."""

    out_dir = prepare_demo_starter_directory(
        Path(out_dir),
        force=force,
        error_type=CsiFeedbackExportError,
    )
    template_dir = _template_dir()
    static_files = (
        "model.py",
        "losses.py",
        "datamodule.py",
        "train.py",
        "evaluate.py",
        "build_benchmark.py",
        "README.md",
        "requirements.txt",
    )
    for filename in static_files:
        shutil.copy2(template_dir / filename, out_dir / filename)
    write_standalone_structured_input(out_dir)
    shutil.copy2(template_dir / "template.yaml", out_dir / "training_template.yaml")

    project_root = Path(plan.project_root or Path.cwd()).resolve()
    _write_yaml(out_dir / "train_config.yaml", _training_config(plan, project_root))
    _write_yaml(out_dir / "noema_recipe.yaml", plan.recipe.to_dict())
    manifest = _starter_project_manifest(
        plan,
        out_dir=out_dir,
        project_root=project_root,
    )
    _write_yaml(out_dir / "project_manifest.yaml", manifest)
    files = [
        *static_files,
        "structured_input.py",
        "training_template.yaml",
        "train_config.yaml",
        "noema_recipe.yaml",
        "project_manifest.yaml",
    ]
    return {
        "status": "exported",
        "recipe": plan.recipe.name,
        "recipe_sha256": plan.recipe_sha256,
        "exporter": "csi-feedback",
        "training_template": plan.template_id,
        "framework": plan.framework,
        "loss": plan.loss,
        "out_dir": str(out_dir),
        "optimizable_steps": [plan.encoder_step.id, plan.decoder_step.id],
        "features": {
            "tap_id": plan.true_csi_tap,
            "from": plan.true_csi_reference,
            "shape": [2, plan.tx_antennas, plan.subcarrier_count],
        },
        "feedback": {
            "dimension": plan.feedback_dimension,
            "bits_per_latent": plan.bits_per_latent,
            "bits_per_sample": plan.feedback_bits_per_sample,
        },
        "project_manifest": manifest,
        "capture_jobs": list(manifest["capture_jobs"]),
        "trained_artifacts": list(manifest["trained_artifacts"]),
        "files": files,
    }


def _training_config(plan: CsiFeedbackExportPlan, project_root: Path) -> JsonDict:
    capture_root = Path(project_root) / ".noema" / "dataset_captures"
    stem = _safe_stem(plan.recipe.name)
    return {
        "schema_version": 1,
        "training_template": plan.template_id,
        "framework": "torch",
        "external_training_request": {
            "loss_identifier": plan.loss,
            "ownership": "external_researcher",
        },
        "recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
            "channel_step": plan.channel_step.id,
            "encoder_step": plan.encoder_step.id,
            "feedback_link_step": plan.feedback_link_step.id,
            "decoder_step": plan.decoder_step.id,
            "precoder_step": plan.precoder_step.id,
        },
        "data": {
            "contract_path": "../data_contract.yaml",
            "contract_sha256": "",
            "contract_file_sha256": "",
            "feature_tap": plan.true_csi_tap,
            "train_capture_dirs": [str(capture_root / ("%s_train" % stem))],
            "validation_capture_dirs": [
                str(capture_root / ("%s_validation" % stem))
            ],
            "test_capture_dirs": [str(capture_root / ("%s_test" % stem))],
            "tx_antennas": plan.tx_antennas,
            "subcarrier_count": plan.subcarrier_count,
            "test_split_exposed_to_training": False,
        },
        "feedback_link": {
            "mode": plan.feedback_mode,
            "feedback_dimension": plan.feedback_dimension,
            "bits_per_latent": plan.bits_per_latent,
            "feedback_bits_per_sample": plan.feedback_bits_per_sample,
            "clip_value": plan.clip_value,
        },
        "model": {
            "class": "ReferenceCsiFeedbackAutoencoder",
            "architecture": "example_only_klt_initialized_quantization_residual_v3",
            "base_channels": 24,
            "refinement_blocks": 2,
            "latent_activation": "identity",
            "global_linear_domain": "frequency",
            "use_angular_delay_transform": True,
            "use_per_latent_adaptor": True,
            "initialization": {
                "kind": "training_split_klt",
                "scale_percentile": 99.0,
            },
        },
        "objective": {
            "loss": CSI_FEEDBACK_EXAMPLE_LOSS,
            "weights": {
                "nmse": 0.5,
                "subcarrier_direction_cosine": 0.15,
                "mrt_rate": 0.35,
            },
            "quantization_weight": 0.02,
            "snr_db_values": [0, 5, 10, 15, 20],
            "labels": "none_self_supervised_reconstruction",
            "checkpoint_selection": (
                "maximum_validation_mean_spectral_efficiency_retention_then_minimum_nmse"
            ),
            "downstream_metric": "mrt_spectral_efficiency_retention",
        },
        "training": {
            "epochs": 160,
            "batch_size": 256,
            "learning_rate": 1e-3,
            "min_learning_rate": 1e-5,
            "optimizer": "AdamW",
            "learning_rate_schedule": "cosine_annealing",
            "lr_warmup_epochs": 5,
            "quantization_warmup_epochs": 0,
            "nmse_pretraining_epochs": 50,
            "prior_freeze_epochs": 50,
            "prior_learning_rate_scale": 0.1,
            "checkpoint_rate_tolerance": 1e-5,
            "weight_decay": 1e-4,
            "early_stopping_patience": 35,
            "initialization_seeds": [23, 41],
            "gradient_clip_norm": 1.0,
            "num_workers": 0,
            "device": "cuda_if_available",
            "downlink_snr_db": plan.downlink_snr_db,
            "artifact_manifest_path": "../trained_artifact.yaml",
            "artifact_component_paths": [
                "../artifacts/csi_feedback_encoder.onnx",
                "../artifacts/csi_feedback_decoder.onnx",
            ],
            "artifact_id": "%s.csi_feedback_codec" % stem,
            "artifact_name": "Learned CSI feedback codec for %s" % plan.recipe.name,
            "artifact_label": "Learned CSI feedback · %s" % plan.recipe.name,
            "history_path": "training_history.json",
        },
        "evaluation": {
            "primary_metric": "downlink_spectral_efficiency_retention",
            "diagnostics": [
                "nmse",
                "nmse_db",
                "phase_invariant_cosine",
                "subcarrier_direction_cosine",
                "spectral_efficiency_bps_hz",
            ],
            "downlink_snr_db": plan.downlink_snr_db,
            "test_split_exposed_to_checkpoint_selection": False,
        },
        "project_root": str(project_root),
    }


def _starter_project_manifest(
    plan: CsiFeedbackExportPlan,
    *,
    out_dir: Path,
    project_root: Path,
) -> JsonDict:
    root_bundle = out_dir.parent
    return {
        "schema_version": 1,
        "kind": "noema.standalone_training_project",
        "exporter": "csi-feedback",
        "training_template": plan.template_id,
        "source_recipe": plan.recipe.name,
        "source_recipe_sha256": plan.recipe_sha256,
        "out_dir": _project_path(out_dir, project_root),
        "capture_plan": plan.capture_plan.to_dict(),
        "capture_jobs": _capture_jobs(
            plan,
            out_dir=root_bundle,
            project_root=project_root,
        ),
        "training": {
            "owner": "external",
            "working_directory": _project_path(out_dir, project_root),
            "requires_python": ">=3.11,<3.14",
            "dependency_file": _project_path(
                out_dir / "requirements.txt", project_root
            ),
            "setup_command": "python -m pip install -r requirements.txt",
            "command": "python train.py",
            "artifact_manifest_path": _project_path(
                root_bundle / "trained_artifact.yaml", project_root
            ),
            "component_paths": [
                _project_path(
                    root_bundle / "artifacts" / "csi_feedback_encoder.onnx",
                    project_root,
                ),
                _project_path(
                    root_bundle / "artifacts" / "csi_feedback_decoder.onnx",
                    project_root,
                ),
            ],
            "history_path": _project_path(
                out_dir / "training_history.json", project_root
            ),
        },
        "evaluation": {
            "owner": "external",
            "command": "python evaluate.py",
            "metrics_path": _project_path(
                out_dir / "evaluation_metrics.json", project_root
            ),
            "split": "test",
        },
        "post_training": {
            "owner": "noema_generated",
            "helper_path": _project_path(
                out_dir / "build_benchmark.py", project_root
            ),
            "command": "python build_benchmark.py",
            "benchmark_pack_path": _project_path(
                out_dir / "benchmark_pack.yaml", project_root
            ),
            "comparison": {
                "methods": [
                    "truncated_angular_delay",
                    "matched_klt",
                    "learned_codec",
                    "perfect_csit",
                ],
                "paired_held_out_seeds": True,
                "sweep": "channel.snr_db",
                "matched_feedback_budget_bits": 128,
            },
        },
        "trained_artifacts": [
            {
                "role": "paired_trained_codec",
                "operations": [plan.encoder_step.op, plan.decoder_step.op],
                "step_ids": [plan.encoder_step.id, plan.decoder_step.id],
                "binding_group": "csi_feedback_codec",
                "manifest_path": _project_path(
                    root_bundle / "trained_artifact.yaml", project_root
                ),
            }
        ],
    }


def _capture_recipe(
    plan: CsiFeedbackExportPlan,
    *,
    split: str,
    seed_offset: int,
    samples: int,
) -> JsonDict:
    source_step = plan.channel_step.to_dict()
    source_params = dict(source_step.get("params") or {})
    base_seed = int(
        source_params.get("seed")
        or master_seed_from_recipe(plan.recipe)
        or 23
    )
    source_params["sample_count"] = int(samples)
    source_params["seed"] = int(base_seed + seed_offset)
    source_step["params"] = source_params
    metadata = dict(plan.recipe.metadata or {})
    metadata.update(
        {
            "seed": int(base_seed + seed_offset),
            "training_performed": False,
            "capture_purpose": "self_supervised_csi_feedback",
            "capture_split": str(split),
            "source_recipe_name": plan.recipe.name,
            "source_recipe_sha256": plan.recipe_sha256,
            "downstream_trainable_slots_executed": False,
            "held_out_evaluation": str(split) == "test",
        }
    )
    return {
        "schema_version": 1,
        "name": "%s_capture_%s" % (plan.recipe.name, split),
        "description": (
            "Generated source-only CSI capture for the %s split; model placeholders "
            "and downstream metrics are intentionally omitted." % split
        ),
        "execution_profile": {"id": "custom", "version": 1},
        "metadata": metadata,
        "steps": [source_step],
        "dataset_capture": {
            "split": str(split),
            "samples": int(samples),
            "shard_size": min(1024, int(samples)),
            "max_runs": 1,
            "seed_mode": "fixed_seed",
            "taps": [
                {"id": plan.true_csi_tap, "from": "%s.csi" % plan.channel_step.id}
            ],
        },
    }


def _capture_jobs(
    plan: CsiFeedbackExportPlan,
    *,
    out_dir: Path,
    project_root: Path,
) -> list[JsonDict]:
    capture_root = Path(project_root) / ".noema" / "dataset_captures"
    stem = _safe_stem(plan.recipe.name)
    jobs = []
    for split_spec in plan.capture_plan.splits:
        path = Path(out_dir) / ("capture_%s_recipe.yaml" % split_spec.id)
        row: JsonDict = {
            "split": split_spec.id,
            "label": "Held-out test" if split_spec.id == "test" else split_spec.id.title(),
            "recipe_path": _project_path(path, project_root),
            "bundle_recipe_path": path.name,
            "output_dir": _project_path(
                capture_root / ("%s_%s" % (stem, split_spec.id)), project_root
            ),
            "requested_samples": split_spec.samples,
            "expected_taps": [
                {"id": plan.true_csi_tap, "from": plan.true_csi_reference}
            ],
            "sample_unit": plan.capture_plan.sample_unit,
            "seed_offset": split_spec.seed_offset,
            "seed_role": split_spec.training_use,
            "owner": "noema",
            "consumer": "external_researcher",
            "source_recipe_sha256": plan.recipe_sha256,
        }
        if path.is_file():
            payload = load_strict_yaml_or_json(path)
            if not isinstance(payload, Mapping):
                raise CsiFeedbackExportError(
                    "Capture recipe must contain a mapping: %s" % path
                )
            row["recipe_sha256"] = canonical_json_sha256(payload)
            row["recipe_file_sha256"] = _file_sha256(path)
        jobs.append(row)
    return jobs


def _configured_tap_id(recipe: Recipe, reference: str, fallback: str) -> str:
    for item in list((recipe.dataset_capture or {}).get("taps") or []):
        if isinstance(item, Mapping) and str(item.get("from") or "") == reference:
            return str(item.get("id") or fallback)
    return fallback


def _single_consumer(
    recipe: Recipe,
    reference: str,
    *,
    expected_op: str,
) -> RecipeStep:
    matches = [
        step
        for step in recipe.steps
        if step.op == expected_op and reference in step.inputs.values()
    ]
    if len(matches) != 1:
        raise CsiFeedbackExportError(
            "Expected exactly one %s consumer of %s; found %d"
            % (expected_op, reference, len(matches))
        )
    return matches[0]


def _producer(recipe: Recipe, step: RecipeStep, input_name: str) -> RecipeStep:
    reference = str(step.inputs.get(input_name) or "")
    if "." not in reference:
        raise CsiFeedbackExportError(
            "Step %s requires input %s" % (step.id, input_name)
        )
    return _step(recipe, reference.split(".", 1)[0])


def _step(recipe: Recipe, step_id: str) -> RecipeStep:
    for step in recipe.steps:
        if step.id == step_id:
            return step
    raise CsiFeedbackExportError("Recipe has no step %s" % step_id)


def _positive_int(value: Any, label: str) -> int:
    try:
        number = int(value)
        exact = float(value)
    except (TypeError, ValueError) as exc:
        raise CsiFeedbackExportError("%s must be a positive integer" % label) from exc
    if number <= 0 or not math.isfinite(exact) or exact != float(number):
        raise CsiFeedbackExportError("%s must be a positive integer" % label)
    return number


def _positive_float(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if number <= 0.0:
        raise CsiFeedbackExportError("%s must be greater than zero" % label)
    return number


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CsiFeedbackExportError("%s must be finite" % label) from exc
    if not math.isfinite(number):
        raise CsiFeedbackExportError("%s must be finite" % label)
    return number


def _template_dir() -> Path:
    relative = Path("demo_trainings") / "csi_feedback_autoencoder"
    candidate = find_demo_training_dir(relative)
    if candidate is not None:
        return candidate
    raise CsiFeedbackExportError(
        "CSI-feedback demonstration project was not found; expected %s" % relative
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(resolved)


def _safe_stem(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in str(value)
    ).strip("_")
    return safe or "csi_feedback"
