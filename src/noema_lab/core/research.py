from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from noema_lab.core.recipes import (
    Recipe,
    RecipeStep,
    RecipeValidationError,
    require_strict_recipe,
)
from noema_lab.core.research_catalog import load_research_catalog, validate_research_specs_against_catalog

JsonDict = Dict[str, Any]


class ResearchSpecError(RecipeValidationError):
    pass


@dataclass
class DatasetSpec:
    id: str
    modality: str
    version: Optional[str] = None
    split: Optional[str] = None
    source: Optional[str] = None
    params: JsonDict = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "schema_version": self.schema_version,
            "id": self.id,
            "modality": self.modality,
            "params": dict(self.params),
        }
        if self.version:
            payload["version"] = self.version
        if self.split:
            payload["split"] = self.split
        if self.source:
            payload["source"] = self.source
        return payload


@dataclass
class MetricSpec:
    id: str
    family: Optional[str] = None
    unit: Optional[str] = None
    direction: str = "neutral"
    reduction: str = "mean"
    params: JsonDict = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "schema_version": self.schema_version,
            "id": self.id,
            "direction": self.direction,
            "reduction": self.reduction,
            "params": dict(self.params),
        }
        if self.family:
            payload["family"] = self.family
        if self.unit:
            payload["unit"] = self.unit
        return payload


@dataclass
class TaskSpec:
    id: str
    kind: str
    modality: Optional[str] = None
    target: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    params: JsonDict = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "schema_version": self.schema_version,
            "id": self.id,
            "kind": self.kind,
            "metrics": list(self.metrics),
            "params": dict(self.params),
        }
        if self.modality:
            payload["modality"] = self.modality
        if self.target:
            payload["target"] = self.target
        return payload


@dataclass
class BenchmarkSpec:
    id: str
    version: str
    dataset: Optional[str] = None
    task: Optional[str] = None
    channel: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    baselines: List[str] = field(default_factory=list)
    params: JsonDict = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "schema_version": self.schema_version,
            "id": self.id,
            "version": self.version,
            "metrics": list(self.metrics),
            "baselines": list(self.baselines),
            "params": dict(self.params),
        }
        if self.dataset:
            payload["dataset"] = self.dataset
        if self.task:
            payload["task"] = self.task
        if self.channel:
            payload["channel"] = self.channel
        return payload


def validate_recipe_research_metadata(recipe: Recipe) -> None:
    _validate_seed_metadata(recipe)
    metadata = dict(recipe.metadata or {})
    if "research" in metadata and not isinstance(metadata["research"], Mapping):
        raise ResearchSpecError("metadata.research must be a mapping")
    specs = research_specs_from_recipe(recipe)
    errors = list((specs.get("catalog_validation") or {}).get("errors") or [])
    if errors:
        raise ResearchSpecError("; ".join(str(item) for item in errors))


def research_specs_from_recipe(recipe: Recipe) -> JsonDict:
    require_strict_recipe(recipe)
    metadata = dict(recipe.metadata or {})
    explicit = _explicit_research_payload(metadata)
    inferred = _infer_research_specs(recipe)
    if explicit:
        payload = _normalize_explicit_research(explicit)
        for key in ("dataset", "task", "metrics"):
            if key not in payload and key in inferred:
                payload[key] = inferred[key]
        if payload.get("source") == "recipe.metadata.research" and any(
            key not in explicit and key in inferred for key in ("dataset", "task", "metrics")
        ):
            payload["source"] = "recipe.metadata.research+inferred_from_recipe"
    else:
        payload = inferred

    raw_explicit_task = explicit.get("task")
    explicit_task_id = str(
        ((raw_explicit_task or {}) if isinstance(raw_explicit_task, Mapping) else {}).get("id") or ""
    ).strip()
    legacy_task_id = str(metadata.get("task_id") or "").strip()
    inferred_task_id = str((inferred.get("task") or {}).get("id") or "").strip()

    if explicit_task_id and legacy_task_id and explicit_task_id != legacy_task_id:
        raise ResearchSpecError(
            "Recipe task declarations disagree: metadata.research.task.id is `%s`, metadata.task_id is `%s`"
            % (explicit_task_id, legacy_task_id)
        )
    declared_task_id = explicit_task_id or legacy_task_id
    if (
        declared_task_id
        and inferred_task_id
        and inferred_task_id != "bit_transport"
        and declared_task_id != inferred_task_id
    ):
        raise ResearchSpecError(
            "Recipe declares task `%s`, but its executable metric graph implies `%s`"
            % (declared_task_id, inferred_task_id)
        )
    if legacy_task_id and not explicit_task_id:
        payload["task"] = _task_spec_for_catalog_id(legacy_task_id).to_dict()
        payload["metrics"] = _metric_specs_for_ids(payload["task"].get("metrics") or [])
        payload["source"] = "metadata.task_id+%s" % str(payload.get("source") or "recipe")

    payload["catalog_validation"] = validate_research_specs_against_catalog(payload)
    return payload


def _task_spec_for_catalog_id(task_id: str) -> TaskSpec:
    definition = load_research_catalog().task(str(task_id))
    if definition is None:
        raise ResearchSpecError(
            "metadata.task_id `%s` is not in the research catalog; use metadata.research.task for a custom task contract"
            % task_id
        )
    return TaskSpec(
        id=definition.id,
        kind=definition.kind,
        modality=definition.modality,
        target=definition.target,
        metrics=list(definition.metrics),
    )


def _explicit_research_payload(metadata: JsonDict) -> JsonDict:
    research = metadata.get("research")
    if isinstance(research, Mapping):
        return dict(research)
    payload: JsonDict = {}
    for source_key, target_key in (
        ("dataset_spec", "dataset"),
        ("task_spec", "task"),
        ("benchmark_spec", "benchmark"),
        ("metric_specs", "metrics"),
    ):
        if source_key in metadata:
            payload[target_key] = metadata[source_key]
    return payload


def _normalize_explicit_research(data: JsonDict) -> JsonDict:
    payload: JsonDict = {
        "schema_version": 1,
        "source": "recipe.metadata.research",
    }
    if "dataset" in data:
        payload["dataset"] = _dataset_from_mapping(data["dataset"], "metadata.research.dataset").to_dict()
    if "task" in data:
        payload["task"] = _task_from_mapping(data["task"], "metadata.research.task").to_dict()
    metrics = data.get("metrics")
    if metrics is not None:
        payload["metrics"] = _metrics_from_value(metrics, "metadata.research.metrics")
    elif "task" in payload:
        payload["metrics"] = _metric_specs_for_ids(payload["task"].get("metrics") or [])
    if "benchmark" in data:
        payload["benchmark"] = _benchmark_from_mapping(data["benchmark"], "metadata.research.benchmark").to_dict()
    return payload


def _infer_research_specs(recipe: Recipe) -> JsonDict:
    dataset = _infer_dataset(recipe.steps)
    task = _infer_task(recipe.steps)
    metric_ids = list(task.metrics if task else [])
    metrics = _metric_specs_for_ids(metric_ids)
    benchmark = BenchmarkSpec(
        id=str((recipe.metadata or {}).get("benchmark_id") or recipe.name),
        version=str((recipe.metadata or {}).get("benchmark_version") or "ad_hoc"),
        dataset=dataset.id if dataset else None,
        task=task.id if task else None,
        channel=_infer_channel(recipe),
        metrics=metric_ids,
        params={"recipe_name": recipe.name},
    )
    payload: JsonDict = {
        "schema_version": 1,
        "source": "inferred_from_recipe",
        "benchmark": benchmark.to_dict(),
    }
    if dataset:
        payload["dataset"] = dataset.to_dict()
    if task:
        payload["task"] = task.to_dict()
    if metrics:
        payload["metrics"] = metrics
    return payload


def _infer_dataset(steps: List[RecipeStep]) -> Optional[DatasetSpec]:
    for step in steps:
        params = dict(step.params or {})
        if step.op == "wireless.miso_ofdm_csi":
            return DatasetSpec(
                id="sionna_correlated_miso_ofdm_csi",
                modality="wireless",
                source=step.op,
                version="sionna-tdl-miso-ofdm-csi-v1",
                params={
                    "step_id": step.id,
                    "sample_count": params.get("sample_count", 128),
                    "tx_antennas": params.get("tx_antennas", 8),
                    "ofdm_fft_size": params.get("ofdm_fft_size", 32),
                    "tdl_model": params.get("tdl_model", "A"),
                    "wireless_backend": params.get("wireless_backend", "sionna"),
                },
            )
        if step.op == "source.random_bits":
            return DatasetSpec(
                id="synthetic_random_bits",
                modality="bits",
                source=step.op,
                version="synthetic-random-bits-v1",
                split="fixed_seed",
                params={
                    "step_id": step.id,
                    "bit_count": params.get("bit_count", 4096),
                    "batch_size": params.get("batch_size", 1),
                    "seed": params.get("seed", 0),
                },
            )
        if step.op == "source.modulation_frames":
            return DatasetSpec(
                id="synthetic_modulation_iq",
                modality="wireless",
                source=step.op,
                version="synthetic-modulation-iq-blind-carrier-v1",
                split="fixed_seed",
                params={
                    "step_id": step.id,
                    "frame_count": params.get("frame_count", 96),
                    "symbols_per_frame": params.get("symbols_per_frame", 128),
                    "seed": params.get("seed", 23),
                },
            )
        if step.op == "source.image_dataset":
            dataset_id = str(params.get("dataset") or "image_dataset")
            return DatasetSpec(
                id=dataset_id,
                modality="image",
                source=step.op,
                version=_dataset_version(dataset_id, params),
                params={
                    "step_id": step.id,
                    "dataset_dir": params.get("dataset_dir"),
                    "image_ids": _split_csv(params.get("image_ids")),
                    "crop_size": params.get("crop_size", 0),
                    "repeat_count": params.get("repeat_count", 1),
                },
            )
        if step.op == "source.local_npz_images":
            source_path = str(params.get("path") or "")
            return DatasetSpec(
                id=Path(source_path).stem or "local_npz_images",
                modality="image",
                source=step.op,
                params={
                    "step_id": step.id,
                    "path": source_path,
                    "array": params.get("array", "images"),
                },
            )
        if step.op == "source.text_dataset":
            dataset_id = str(params.get("dataset") or "semantic_text_smoke")
            return DatasetSpec(
                id=dataset_id,
                modality="text",
                source=step.op,
                version="semantic-text-smoke-v1" if dataset_id == "semantic_text_smoke" else None,
                params={
                    "step_id": step.id,
                    "sample_ids": _split_csv(params.get("sample_ids")),
                    "repeat_count": params.get("repeat_count", 1),
                },
            )
        if step.op == "source.task_labels_smoke":
            dataset_id = str(params.get("dataset") or "task_smoke")
            return DatasetSpec(
                id=dataset_id,
                modality="generic",
                source=step.op,
                version="task-smoke-v1" if dataset_id == "task_smoke" else None,
                params={
                    "step_id": step.id,
                    "sample_ids": _split_csv(params.get("sample_ids")),
                },
            )
        if step.op == "source.coco128_detection":
            return DatasetSpec(
                id="coco128_detection",
                modality="image",
                source=step.op,
                version="ultralytics-coco128-v1",
                params={
                    "step_id": step.id,
                    "split": params.get("split", "train2017"),
                    "limit": params.get("limit", 8),
                    "image_size": params.get("image_size", 320),
                },
            )
        if step.op == "source.coco8_segmentation":
            return DatasetSpec(
                id="coco8_segmentation",
                modality="image",
                source=step.op,
                version="ultralytics-coco8-seg-v1",
                params={
                    "step_id": step.id,
                    "split": params.get("split", "val"),
                    "limit": params.get("limit", 4),
                    "image_size": params.get("image_size", 320),
                },
            )
        if step.op == "source.semantic_artifacts_smoke":
            dataset_id = str(params.get("dataset") or "semantic_artifact_smoke")
            return DatasetSpec(
                id=dataset_id,
                modality="multimodal",
                source=step.op,
                version="semantic-artifact-smoke-v1" if dataset_id == "semantic_artifact_smoke" else None,
                params={
                    "step_id": step.id,
                    "embedding_dim": params.get("embedding_dim", 8),
                },
            )
        if step.op == "source.retrieval_smoke":
            dataset_id = str(params.get("dataset") or "retrieval_smoke")
            return DatasetSpec(
                id=dataset_id,
                modality="multimodal",
                source=step.op,
                version="retrieval-smoke-v1" if dataset_id == "retrieval_smoke" else None,
                params={
                    "step_id": step.id,
                    "sample_ids": _split_csv(params.get("sample_ids")),
                    "image_size": params.get("image_size", 224),
                },
            )
        if step.op == "source.flickr8k_retrieval":
            return DatasetSpec(
                id="flickr8k",
                modality="multimodal",
                source=step.op,
                version="flickr8k-validation-v1" if str(params.get("split") or "validation") == "validation" else None,
                params={
                    "step_id": step.id,
                    "split": params.get("split", "validation"),
                    "limit": params.get("limit", 16),
                    "image_size": params.get("image_size", 224),
                    "caption_index": params.get("caption_index", 0),
                },
            )
        if step.op == "source.vqa_smoke":
            dataset_id = str(params.get("dataset") or "coco_vqa_smoke")
            return DatasetSpec(
                id=dataset_id,
                modality="multimodal",
                source=step.op,
                version="coco-vqa-smoke-v1" if dataset_id == "coco_vqa_smoke" else None,
                params={
                    "step_id": step.id,
                    "sample_ids": _split_csv(params.get("sample_ids")),
                },
            )
        if step.op == "source.vqa_manifest":
            manifest_path = str(params.get("manifest_path") or "")
            return DatasetSpec(
                id=Path(manifest_path).stem or "local_vqa_manifest",
                modality="multimodal",
                source=step.op,
                params={
                    "step_id": step.id,
                    "manifest_path": manifest_path,
                    "image_root": params.get("image_root", ""),
                    "limit": params.get("limit", 8),
                    "image_size": params.get("image_size", 224),
                },
            )
        if step.op == "source.vqa_small_hf":
            split = str(params.get("split") or "validation")
            return DatasetSpec(
                id="vqa_small",
                modality="multimodal",
                source=step.op,
                version="soumyasj-vqa-dataset-small",
                params={
                    "step_id": step.id,
                    "repo": "soumyasj/vqa-dataset-small",
                    "split": split,
                    "limit": params.get("limit", 8),
                    "image_size": params.get("image_size", 224),
                },
            )
    return None


def _infer_task(steps: List[RecipeStep]) -> Optional[TaskSpec]:
    op_ids = {step.op for step in steps}
    if "metrics.modulation_classification" in op_ids:
        return TaskSpec(
            id="automatic_modulation_recognition",
            kind="signal_classification",
            modality="wireless",
            target="receiver.prediction",
            metrics=[
                "modulation_recognition.accuracy",
                "modulation_recognition.balanced_accuracy",
                "modulation_recognition.macro_f1",
            ],
        )
    if "metrics.csi_feedback" in op_ids:
        return TaskSpec(
            id="csi_compression_feedback",
            kind="feedback_control",
            modality="wireless",
            target="feedback_decoder.reconstruction",
            metrics=[
                "csi_feedback.achieved_spectral_efficiency_bps_hz",
                "csi_feedback.spectral_efficiency_retention",
                "csi_feedback.nmse_db",
                "csi_feedback.phase_invariant_cosine",
            ],
        )
    if "metrics.image_reconstruction" in op_ids:
        return TaskSpec(
            id="image_reconstruction",
            kind="reconstruction",
            modality="image",
            target="receiver.images",
            metrics=["quality.psnr_db", "quality.mse", "quality.mae"],
        )
    if "metrics.text_semantic_similarity" in op_ids:
        metrics = [
            "semantic.lexical_similarity",
            "text.unigram_bleu_proxy",
            "text.edit_similarity",
            "text.exact_match",
        ]
        if "metrics.semantic_state_faithfulness" in op_ids:
            metrics.extend(
                [
                    "faithfulness.concept_f1",
                    "faithfulness.fact_f1",
                    "faithfulness.unsupported_assertion_rate",
                    "faithfulness.kb_fact_precision",
                ]
            )
        return TaskSpec(
            id="text_semantic_similarity",
            kind="semantic_reconstruction",
            modality="text",
            target="receiver.texts",
            metrics=metrics,
        )
    if "metrics.classification" in op_ids:
        return TaskSpec(
            id="classification",
            kind="task_success",
            modality="generic",
            target="candidate.prediction",
            metrics=["task.accuracy", "task.exact_match", "classification.balanced_accuracy"],
        )
    if "metrics.vqa" in op_ids:
        return TaskSpec(
            id="visual_question_answering",
            kind="task_success",
            modality="multimodal",
            target="receiver.answer",
            metrics=["vqa.single_reference_exact_match"],
        )
    if "metrics.detection" in op_ids:
        detection_step = next(step for step in steps if step.op == "metrics.detection")
        threshold = float((detection_step.params or {}).get("iou_threshold", 0.5))
        threshold_suffix = (
            "0"
            if threshold == 0.0
            else format(threshold, ".12g").replace("-", "m").replace(".", "p")
        )
        return TaskSpec(
            id="object_detection",
            kind="task_success",
            modality="image",
            target="receiver.detections",
            metrics=[
                "detection.recall_at_iou_%s" % threshold_suffix,
                "detection.precision_at_iou_%s" % threshold_suffix,
                "detection.f1_at_iou_%s" % threshold_suffix,
            ],
        )
    if "metrics.segmentation" in op_ids:
        return TaskSpec(
            id="segmentation",
            kind="task_success",
            modality="image",
            target="receiver.segmentation",
            metrics=["segmentation.miou", "segmentation.pixel_accuracy"],
        )
    if "metrics.captioning" in op_ids:
        return TaskSpec(
            id="image_captioning",
            kind="task_success",
            modality="multimodal",
            target="receiver.caption",
            metrics=[
                "caption.unigram_bleu_proxy",
                "caption.lexical_similarity",
                "caption.exact_match",
            ],
        )
    if "metrics.retrieval" in op_ids:
        return TaskSpec(
            id="image_text_retrieval",
            kind="task_success",
            modality="multimodal",
            target="receiver.ranking",
            metrics=["retrieval.recall_at_1", "retrieval.recall_at_5", "retrieval.recall_at_10"],
        )
    if "foundation.diffusion_state_to_image" in op_ids or "metrics.embedding_similarity" in op_ids:
        return TaskSpec(
            id="image_generation",
            kind="generative_reconstruction",
            modality="multimodal",
            target="receiver.images",
            metrics=["generation.text_image_clip.cosine_mean", "generation.image_image_clip.cosine_mean"],
        )
    if "demodulation.neural_receiver_adapter" in op_ids and "metrics.block_error_rate" in op_ids:
        return TaskSpec(
            id="neural_receiver_demapping",
            kind="transport_integrity",
            modality="bits",
            target="demodulator.bits",
            metrics=["channel.coded.ber", "channel.coded.bler"],
        )
    if "metrics.bit_error_rate" in op_ids:
        return TaskSpec(
            id="bit_transport",
            kind="transport_integrity",
            modality="bits",
            metrics=["channel.ber"],
        )
    return None


def _infer_channel(recipe: Recipe) -> str:
    metadata = dict(recipe.metadata or {})
    if "channel_mode" in metadata:
        return str(metadata["channel_mode"])
    if metadata.get("channel_enabled") is False:
        return "disabled"
    ops = {step.op for step in recipe.steps}
    if "wireless.channel" in ops or "wireless.digital_link" in ops:
        return "wireless"
    if "channel.identity_link" in ops:
        return "identity"
    return "unspecified"


def _dataset_version(dataset_id: str, params: JsonDict) -> Optional[str]:
    if dataset_id == "kodak":
        image_ids = _split_csv(params.get("image_ids"))
        if len(image_ids) == 24:
            return "kodak-24"
    return None


def _dataset_from_mapping(value: Any, label: str) -> DatasetSpec:
    data = _require_mapping(value, label)
    return DatasetSpec(
        id=_require_str(data, "id", label),
        modality=_require_str(data, "modality", label),
        version=_optional_str(data, "version", label),
        split=_optional_str(data, "split", label),
        source=_optional_str(data, "source", label),
        params=dict(data.get("params") or {}),
    )


def _task_from_mapping(value: Any, label: str) -> TaskSpec:
    data = _require_mapping(value, label)
    task_id = _require_str(data, "id", label)
    definition = load_research_catalog().task(task_id)
    if definition is not None:
        return TaskSpec(
            id=task_id,
            kind=str(data.get("kind") or definition.kind),
            modality=(
                _optional_str(data, "modality", label)
                if "modality" in data
                else definition.modality
            ),
            target=(
                _optional_str(data, "target", label)
                if "target" in data
                else definition.target
            ),
            metrics=(
                _string_list(data["metrics"], "%s.metrics" % label)
                if "metrics" in data
                else list(definition.metrics)
            ),
            params=dict(data.get("params") or {}),
        )
    return TaskSpec(
        id=task_id,
        kind=_require_str(data, "kind", label),
        modality=_optional_str(data, "modality", label),
        target=_optional_str(data, "target", label),
        metrics=_string_list(data.get("metrics") or [], "%s.metrics" % label),
        params=dict(data.get("params") or {}),
    )


def _benchmark_from_mapping(value: Any, label: str) -> BenchmarkSpec:
    data = _require_mapping(value, label)
    return BenchmarkSpec(
        id=_require_str(data, "id", label),
        version=str(data.get("version") or "1"),
        dataset=_optional_str(data, "dataset", label),
        task=_optional_str(data, "task", label),
        channel=_optional_str(data, "channel", label),
        metrics=_string_list(data.get("metrics") or [], "%s.metrics" % label),
        baselines=_string_list(data.get("baselines") or [], "%s.baselines" % label),
        params=dict(data.get("params") or {}),
    )


def _metrics_from_value(value: Any, label: str) -> List[JsonDict]:
    if not isinstance(value, list):
        raise ResearchSpecError("%s must be a list" % label)
    metrics = []
    for index, item in enumerate(value):
        item_label = "%s[%d]" % (label, index)
        if isinstance(item, str):
            metrics.append(_metric_spec_for_id(item).to_dict())
        else:
            metrics.append(_metric_from_mapping(item, item_label).to_dict())
    return metrics


def _metric_from_mapping(value: Any, label: str) -> MetricSpec:
    data = _require_mapping(value, label)
    direction = str(data.get("direction") or "neutral")
    if direction not in ("higher_is_better", "lower_is_better", "neutral"):
        raise ResearchSpecError(
            "%s.direction must be higher_is_better, lower_is_better, or neutral" % label
        )
    return MetricSpec(
        id=_require_str(data, "id", label),
        family=_optional_str(data, "family", label),
        unit=_optional_str(data, "unit", label),
        direction=direction,
        reduction=str(data.get("reduction") or "mean"),
        params=dict(data.get("params") or {}),
    )


def _metric_specs_for_ids(metric_ids: List[str]) -> List[JsonDict]:
    return [_metric_spec_for_id(metric_id).to_dict() for metric_id in metric_ids]


def _metric_spec_for_id(metric_id: str) -> MetricSpec:
    metric = load_research_catalog().metric(str(metric_id))
    if metric:
        return MetricSpec(
            metric.id,
            family=metric.family,
            unit=metric.unit,
            direction=metric.direction,
            reduction=metric.reduction,
        )
    return MetricSpec(str(metric_id))


def _validate_seed_metadata(recipe: Recipe) -> None:
    metadata = dict(recipe.metadata or {})
    for key in ("seed", "experiment_seed", "ui_seed"):
        if key not in metadata or metadata[key] is None:
            continue
        value = metadata[key]
        if isinstance(value, bool):
            raise ResearchSpecError("metadata.%s must be an integer seed, not a boolean" % key)
        try:
            seed = int(value)
        except Exception as exc:
            raise ResearchSpecError("metadata.%s must be an integer seed" % key) from exc
        if seed < 0:
            raise ResearchSpecError("metadata.%s must be >= 0" % key)


def _require_mapping(value: Any, label: str) -> JsonDict:
    if not isinstance(value, Mapping):
        raise ResearchSpecError("%s must be a mapping" % label)
    return dict(value)


def _require_str(data: JsonDict, key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ResearchSpecError("%s.%s must be a non-empty string" % (label, key))
    return value


def _optional_str(data: JsonDict, key: str, label: str) -> Optional[str]:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ResearchSpecError("%s.%s must be a string" % (label, key))
    return value


def _string_list(value: Any, label: str) -> List[str]:
    if not isinstance(value, list):
        raise ResearchSpecError("%s must be a list" % label)
    output = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ResearchSpecError("%s[%d] must be a non-empty string" % (label, index))
        output.append(item)
    return output


def _split_csv(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [item.strip() for item in str(value).split(",") if item.strip()]
