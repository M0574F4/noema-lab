from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema
from noema_lab.core.structured_input import (
    decode_strict_json,
    decode_strict_json_object,
)
from noema_lab.ops.metrics.text import (
    _edit_similarity,
    _token_f1,
    _unigram_bleu_proxy,
)

JsonDict = Dict[str, Any]


class ClassificationMetricsOperation(Operation):
    id = "metrics.classification"
    name = "Classification task metrics"
    input_kinds = {
        "reference": ["task.labels.json"],
        "candidate": ["task.predictions.json", "task.labels.json"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        reference = _load_examples(ctx.require_input("reference").path)
        candidate = _load_examples(ctx.require_input("candidate").path)
        rows, metrics = _label_metrics(reference, candidate)
        return _write_report(ctx, "classification", rows, metrics)


class VqaMetricsOperation(Operation):
    id = "metrics.vqa"
    name = "Visual question answering task metrics"
    input_kinds = {
        "reference": ["vqa.answers.json", "task.labels.json"],
        "candidate": ["vqa.answers.json", "task.predictions.json", "task.labels.json"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        reference = _load_examples(ctx.require_input("reference").path)
        candidate = _load_examples(ctx.require_input("candidate").path)
        rows, base = _label_metrics(reference, candidate, ref_key="answer", cand_key="answer")
        metrics = {
            "vqa.single_reference_exact_match": base["task.exact_match"],
            "task.exact_match": base["task.exact_match"],
        }
        return _write_report(
            ctx,
            "vqa",
            rows,
            metrics,
            metadata={
                "metric_definition": (
                    "Literal, case-sensitive Unicode string equality against one reference answer; "
                    "this is not the consensus-based standard VQA accuracy."
                )
            },
        )


class DetectionMetricsOperation(Operation):
    id = "metrics.detection"
    name = "Object detection task metrics"
    input_kinds = {
        "reference": ["vision.detections.json"],
        "candidate": ["vision.detections.json"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema(
        {
            "iou_threshold": {"type": "number", "default": 0.5, "minimum": 0.0, "maximum": 1.0},
            "confidence_threshold": {"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 1.0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        threshold = float(ctx.params.get("iou_threshold", 0.5))
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise RuntimeError("iou_threshold must be finite and between 0 and 1")
        requested_confidence_threshold = float(
            ctx.params.get("confidence_threshold", 0.0)
        )
        if (
            not math.isfinite(requested_confidence_threshold)
            or not 0.0 <= requested_confidence_threshold <= 1.0
        ):
            raise RuntimeError("confidence_threshold must be finite and between 0 and 1")
        reference_artifact = ctx.require_input("reference")
        candidate_artifact = ctx.require_input("candidate")
        reference_payload = _load_task_payload(reference_artifact.path)
        candidate_payload = _load_task_payload(candidate_artifact.path)
        candidate_prefilter_confidence_threshold = (
            _detection_prefilter_confidence_threshold(
                candidate_payload, candidate_artifact.metadata
            )
        )
        confidence_threshold = max(
            requested_confidence_threshold,
            candidate_prefilter_confidence_threshold,
        )
        iou_suffix = _threshold_metric_suffix(threshold)
        confidence_suffix = _threshold_metric_suffix(confidence_threshold)
        metric_suffix = "%s_conf_%s" % (iou_suffix, confidence_suffix)
        reference_examples = _payload_examples(
            reference_payload, reference_artifact.path
        )
        candidate_examples = _payload_examples(
            candidate_payload, candidate_artifact.path
        )
        pairs = _paired_examples(reference_examples, candidate_examples)
        reference = _detections_by_id(reference_examples)
        candidate = _detections_by_id(candidate_examples)
        matched = 0
        predicted = 0
        total = 0
        rows = []
        for example_id, _ref, _cand in pairs:
            ref_boxes = reference[example_id]
            cand_boxes = _detections_at_confidence(
                candidate[example_id], confidence_threshold, example_id
            )
            total += len(ref_boxes)
            predicted += len(cand_boxes)
            example_matches = _maximum_detection_matches(
                ref_boxes, cand_boxes, threshold
            )
            matched += example_matches
            rows.append({"id": example_id, "reference_count": len(ref_boxes), "candidate_count": len(cand_boxes), "matches": example_matches})
        precision = float(matched) / float(predicted) if predicted else (1.0 if total == 0 else 0.0)
        recall = float(matched) / float(total) if total else 1.0
        f1 = _f1(precision, recall)
        metrics = {
            "detection.precision_at_iou_%s" % metric_suffix: precision,
            "detection.recall_at_iou_%s" % metric_suffix: recall,
            "detection.f1_at_iou_%s" % metric_suffix: f1,
        }
        deprecated_aliases: Dict[str, str] = {}
        if confidence_threshold == 0.0:
            for stem in ("precision", "recall", "f1"):
                canonical = "detection.%s_at_iou_%s" % (stem, metric_suffix)
                alias = "detection.%s_at_iou_%s" % (stem, iou_suffix)
                metrics[alias] = metrics[canonical]
                deprecated_aliases[alias] = canonical
        return _write_report(
            ctx,
            "detection",
            rows,
            metrics,
            metadata={
                "iou_threshold": threshold,
                "confidence_threshold": confidence_threshold,
                "requested_confidence_threshold": requested_confidence_threshold,
                "candidate_prefilter_confidence_threshold": (
                    candidate_prefilter_confidence_threshold
                ),
                "effective_confidence_threshold": confidence_threshold,
                "metric_suffix": metric_suffix,
                "metric_ids": sorted(metrics),
                "matching": "deterministic maximum-cardinality label-matched bipartite matching at the configured IoU threshold",
                "confidence_policy": (
                    "effective threshold is max(requested metric threshold, candidate "
                    "inference prefilter threshold); candidate score must meet the effective "
                    "threshold, and missing scores are included only when it is exactly zero"
                ),
                "deprecated_metric_aliases": deprecated_aliases,
                "not_computed": ["average_precision", "mean_average_precision"],
            },
        )


class SegmentationMetricsOperation(Operation):
    id = "metrics.segmentation"
    name = "Segmentation task metrics"
    input_kinds = {
        "reference": ["vision.segmentation_mask.numpy"],
        "candidate": ["vision.segmentation_mask.numpy"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        reference, reference_metadata = _load_masks(ctx.require_input("reference").path)
        candidate, candidate_metadata = _load_masks(ctx.require_input("candidate").path)
        if reference.shape != candidate.shape:
            raise RuntimeError(
                "Segmentation mask shape mismatch: reference %s, candidate %s"
                % (reference.shape, candidate.shape)
            )
        if reference.ndim not in {2, 3}:
            raise RuntimeError(
                "Segmentation masks must have shape [H,W] or [N,H,W], got %s"
                % (reference.shape,)
            )
        if reference.size == 0:
            raise RuntimeError("Segmentation metrics require non-empty masks")
        if not np.issubdtype(reference.dtype, np.integer):
            raise RuntimeError(
                "Reference segmentation masks must use an integer dtype, got %s"
                % reference.dtype
            )
        if not np.issubdtype(candidate.dtype, np.integer):
            raise RuntimeError(
                "Candidate segmentation masks must use an integer dtype, got %s"
                % candidate.dtype
            )
        count = int(reference.shape[0]) if reference.ndim == 3 else 1
        sample_ids, pairing_mode = _paired_mask_ids(
            reference_metadata, candidate_metadata, count
        )
        ref = reference if reference.ndim == 3 else reference[None, ...]
        cand = candidate if candidate.ndim == 3 else candidate[None, ...]
        labels = sorted(set(int(value) for value in np.unique(ref)) | set(int(value) for value in np.unique(cand)))
        ious = []
        for label in labels:
            ref_mask = ref == label
            cand_mask = cand == label
            union = int(np.logical_or(ref_mask, cand_mask).sum())
            if union:
                ious.append(float(np.logical_and(ref_mask, cand_mask).sum()) / float(union))
        pixel_accuracy = float((ref == cand).sum()) / float(ref.size) if ref.size else 1.0
        pooled_miou = float(sum(ious) / len(ious)) if ious else 1.0
        metrics = {
            "segmentation.semantic_raster_pooled_miou": pooled_miou,
            "segmentation.semantic_raster_pooled_pixel_accuracy": pixel_accuracy,
            "segmentation.semantic_raster_evaluated_class_count": int(len(ious)),
            "segmentation.semantic_raster_evaluated_pixel_count": int(ref.size),
            # Scientifically exact compatibility aliases for the historical operation.
            "segmentation.miou": pooled_miou,
            "segmentation.pixel_accuracy": pixel_accuracy,
        }
        rows = [{"label": label, "iou": iou} for label, iou in zip(labels, ious)]
        return _write_report(
            ctx,
            "segmentation",
            rows,
            metrics,
            metadata={
                "num_masks": int(count),
                "sample_ids": sample_ids,
                "sample_pairing": pairing_mode,
                "reference_dtype": str(reference.dtype),
                "candidate_dtype": str(candidate.dtype),
                "construct": "pooled semantic-class raster agreement (not instance segmentation)",
                "pooling_policy": "pool all evaluated pixels across samples before per-class IoU",
                "class_policy": {
                    "included_labels": labels,
                    "source": "union of labels observed in reference or candidate rasters",
                    "ignored_labels": [],
                    "background_handling": "no label is implicitly treated as or excluded as background",
                    "absent_in_both": "not represented and therefore not scored",
                    "macro_reduction": "unweighted mean over included labels with non-empty union",
                },
                "coverage": {
                    "mask_count": int(count),
                    "pixel_count": int(ref.size),
                    "evaluated_class_count": int(len(ious)),
                },
                "deprecated_metric_aliases": {
                    "segmentation.miou": "segmentation.semantic_raster_pooled_miou",
                    "segmentation.pixel_accuracy": "segmentation.semantic_raster_pooled_pixel_accuracy",
                },
            },
        )


class CaptioningMetricsOperation(Operation):
    id = "metrics.captioning"
    name = "Image captioning task metrics"
    input_kinds = {
        "reference": ["text.batch.json", "text.caption.json"],
        "candidate": ["text.batch.json", "text.caption.json"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        reference = _load_examples(ctx.require_input("reference").path)
        candidate = _load_examples(ctx.require_input("candidate").path)
        pairs = _paired_examples(reference, candidate)
        rows = []
        exact = unigram_proxy = lexical = edit = 0.0
        for index, (example_id, ref, cand) in enumerate(pairs):
            ref_text = _required_string_field(
                ref, ("caption", "text"), "reference caption", index
            )
            cand_text = _required_string_field(
                cand, ("caption", "text"), "candidate caption", index
            )
            row = {
                "id": example_id,
                "exact_match": 1.0 if ref_text == cand_text else 0.0,
                "unigram_bleu_proxy": _unigram_bleu_proxy(ref_text, cand_text),
                "token_f1": _token_f1(ref_text, cand_text),
                "edit_similarity": _edit_similarity(ref_text, cand_text),
            }
            rows.append(row)
            exact += row["exact_match"]
            unigram_proxy += row["unigram_bleu_proxy"]
            lexical += row["token_f1"]
            edit += row["edit_similarity"]
        n = float(len(rows) or 1)
        metrics = {
            "caption.exact_match": exact / n,
            "caption.unigram_bleu_proxy": unigram_proxy / n,
            "caption.edit_similarity": edit / n,
            "caption.lexical_similarity": lexical / n,
            "semantic.lexical_similarity": lexical / n,
        }
        return _write_report(
            ctx,
            "captioning",
            rows,
            metrics,
            metadata={
                "metric_definitions": {
                    "caption.exact_match": (
                        "Mean literal Unicode string equality, including case, punctuation, and whitespace."
                    ),
                    "caption.unigram_bleu_proxy": (
                        "Mean sentence-level clipped unigram precision with brevity penalty; "
                        "it is not corpus BLEU."
                    ),
                    "caption.lexical_similarity": (
                        "Mean lowercased bag-of-word token F1 lexical-overlap proxy."
                    ),
                }
            },
        )


class RetrievalMetricsOperation(Operation):
    id = "metrics.retrieval"
    name = "Image-text retrieval task metrics"
    input_kinds = {
        "rankings": ["retrieval.rankings.json"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema(
        {
            "k_values": {"type": "string", "default": "1,5,10"},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        ranking_payload = _load_task_payload(ctx.require_input("rankings").path)
        rankings = _payload_examples(
            ranking_payload, ctx.require_input("rankings").path
        )
        if not rankings:
            raise RuntimeError("Retrieval metrics require at least one ranking")
        query_ids = _required_ids(rankings, "retrieval ranking")
        k_values = _parse_retrieval_k_values(
            ctx.params.get("k_values") if "k_values" in ctx.params else "1,5,10"
        )
        (
            candidate_ids,
            candidate_count,
            ranking_depth,
            universe_binding,
        ) = _retrieval_candidate_universe(ranking_payload, rankings)
        invalid_cutoffs = [value for value in k_values if value > ranking_depth]
        if invalid_cutoffs:
            raise RuntimeError(
                "Retrieval K values %s exceed the bound ranking_depth %d"
                % (invalid_cutoffs, ranking_depth)
            )
        rows = []
        totals = {k: 0.0 for k in k_values}
        reciprocal_total = 0.0
        for index, item in enumerate(rankings):
            query_id = query_ids[index]
            expected = _required_scalar_field(
                item,
                ("target_id", "expected_id"),
                "retrieval target",
                index,
                allow_empty=False,
            )
            if "ranked_ids" not in item or not isinstance(item.get("ranked_ids"), list):
                raise RuntimeError(
                    "retrieval ranking example %d requires a ranked_ids list" % index
                )
            raw_ranked = item["ranked_ids"]
            if any(
                value is None or isinstance(value, (Mapping, list, tuple))
                for value in raw_ranked
            ):
                raise RuntimeError(
                    "retrieval ranking example %d ranked_ids must contain scalar IDs" % index
                )
            ranked = [str(value) for value in raw_ranked]
            if not ranked or any(not value for value in ranked):
                raise RuntimeError(
                    "retrieval ranking example %d requires non-empty ranked_ids" % index
                )
            if len(set(ranked)) != len(ranked):
                raise RuntimeError(
                    "retrieval ranking example %d ranked_ids must be unique" % index
                )
            if len(ranked) != ranking_depth:
                raise RuntimeError(
                    "Retrieval ranking example %d has depth %d but the artifact binds ranking_depth %d"
                    % (index, len(ranked), ranking_depth)
                )
            outside_pool = sorted(set(ranked) - set(candidate_ids))
            if outside_pool:
                raise RuntimeError(
                    "retrieval ranking example %d contains IDs outside the declared candidate universe: %s"
                    % (index, outside_pool)
                )
            if expected not in candidate_ids:
                raise RuntimeError(
                    "retrieval ranking example %d target_id %r is outside the declared candidate universe"
                    % (index, expected)
                )
            rank = ranked.index(expected) + 1 if expected in ranked else 0
            reciprocal_total += 1.0 / float(rank) if rank else 0.0
            row = {"id": query_id, "target_id": expected, "rank": rank}
            for k in k_values:
                hit = 1.0 if rank and rank <= k else 0.0
                row["hit_at_%d" % k] = hit
                totals[k] += hit
            rows.append(row)
        n = float(len(rows))
        cutoff = int(ranking_depth)
        mrr_metric_id = "retrieval.mrr_at_%d" % cutoff
        metrics = {mrr_metric_id: reciprocal_total / n}
        deprecated_aliases: Dict[str, str] = {}
        for k in k_values:
            canonical = "retrieval.hit_at_%d" % k
            alias = "retrieval.recall_at_%d" % k
            metrics[canonical] = totals[k] / n
            # With exactly one declared positive target per query, historical Recall@K is
            # numerically identical to Hit@K. Retain it only as an explicit qualified alias.
            metrics[alias] = metrics[canonical]
            deprecated_aliases[alias] = canonical
        _require_bounded_scores(metrics, "retrieval")
        return _write_report(
            ctx,
            "retrieval",
            rows,
            metrics,
            metadata={
                "mrr_cutoff_k": cutoff,
                "mrr_metric_id": mrr_metric_id,
                "candidate_count": candidate_count,
                "candidate_ids": candidate_ids,
                "ranking_depth": ranking_depth,
                "candidate_universe_binding": universe_binding,
                "target_cardinality_per_query": 1,
                "metric_definitions": {
                    mrr_metric_id: (
                        "Mean reciprocal rank evaluated only within the bound ranked-list depth."
                    ),
                    **{
                        "retrieval.hit_at_%d" % k: (
                            "Mean indicator that the one declared positive target is present in the first %d positions."
                            % k
                        )
                        for k in k_values
                    },
                },
                "deprecated_metric_aliases": deprecated_aliases,
            },
        )


class EmbeddingSimilarityMetricsOperation(Operation):
    id = "metrics.embedding_similarity"
    name = "Paired embedding similarity metrics"
    input_kinds = {
        "reference": ["foundation.embedding.numpy", "vision.embedding.clip.numpy", "multimodal.embedding.numpy"],
        "candidate": ["foundation.embedding.numpy", "vision.embedding.clip.numpy", "multimodal.embedding.numpy"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema(
        {
            "label": {
                "type": "string",
                "default": "embedding",
                "enum": [
                    "embedding",
                    "generation.text_image_clip",
                    "generation.image_image_clip",
                ],
            },
            "normalize": {"type": "boolean", "default": True},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        reference, reference_metadata = _load_embedding_matrix(ctx.require_input("reference").path)
        candidate, candidate_metadata = _load_embedding_matrix(ctx.require_input("candidate").path)
        if reference.ndim != 2 or candidate.ndim != 2:
            raise RuntimeError("Embedding similarity requires rank-2 embedding matrices")
        if reference.shape[1] != candidate.shape[1]:
            raise RuntimeError("Embedding dimensions differ: %d vs %d" % (reference.shape[1], candidate.shape[1]))
        if reference.shape[0] != candidate.shape[0]:
            raise RuntimeError(
                "Embedding count mismatch: reference has %d rows, candidate has %d"
                % (reference.shape[0], candidate.shape[0])
            )
        count = int(reference.shape[0])
        if count <= 0:
            raise RuntimeError("Embedding similarity requires at least one paired example")
        reference_space, reference_modality = _required_embedding_provenance(
            reference_metadata, "reference embedding", int(reference.shape[1])
        )
        candidate_space, candidate_modality = _required_embedding_provenance(
            candidate_metadata, "candidate embedding", int(candidate.shape[1])
        )
        if reference_space != candidate_space:
            raise RuntimeError(
                "Embedding provenance mismatch: reference space %s, candidate space %s"
                % (reference_space, candidate_space)
            )
        normalize = bool(ctx.params.get("normalize", True))
        if normalize:
            left = _normalize_rows(reference[:count], "reference embedding")
            right = _normalize_rows(candidate[:count], "candidate embedding")
            score_name = "cosine"
        else:
            left = reference[:count].astype(np.float64, copy=False)
            right = candidate[:count].astype(np.float64, copy=False)
            score_name = "dot_product"
        scores = np.sum(left * right, axis=1, dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("Embedding similarity produced non-finite %s scores" % score_name)
        label = _validated_embedding_metric_namespace(
            str(ctx.params.get("label") or "embedding"),
            reference_space,
            reference_modality,
            candidate_modality,
        )
        reference_ids, candidate_ids, pairing_mode = _paired_embedding_ids(
            reference_metadata, candidate_metadata, count
        )
        rows = [
            {
                "id": reference_ids[index],
                "candidate_id": candidate_ids[index],
                score_name: float(scores[index]),
            }
            for index in range(count)
        ]
        metrics = {
            "%s.%s_mean" % (label, score_name): float(np.mean(scores)),
            "%s.%s_min" % (label, score_name): float(np.min(scores)),
            "%s.%s_max" % (label, score_name): float(np.max(scores)),
            "%s.count" % label: int(count),
        }
        return _write_report(
            ctx,
            "embedding_similarity",
            rows,
            metrics,
            metadata={
                "label": label,
                "normalize": normalize,
                "similarity": score_name,
                "sample_pairing": pairing_mode,
                "embedding_space": reference_space,
                "reference_modality": reference_modality,
                "candidate_modality": candidate_modality,
                "metric_namespace_policy": (
                    "namespace is derived from paired modalities unless an allow-listed CLIP "
                    "namespace is proven by matching embedding-space provenance"
                ),
                "reference_metadata": reference_metadata,
                "candidate_metadata": candidate_metadata,
            },
        )


class ClipRetrievalRankOperation(Operation):
    id = "foundation.clip_retrieval_rank"
    name = "CLIP retrieval ranker"
    input_kinds = {
        "image_embeddings": ["foundation.embedding.numpy", "vision.embedding.clip.numpy", "multimodal.embedding.numpy"],
        "text_embeddings": ["foundation.embedding.numpy", "vision.embedding.clip.numpy", "multimodal.embedding.numpy"],
        "targets": ["retrieval.targets.json"],
    }
    output_kinds = {"rankings": "retrieval.rankings.json"}
    params_schema = object_schema(
        {
            "normalize": {"type": "boolean", "default": True},
            "top_k": {"type": "integer", "default": 10, "minimum": 1},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        image_embeddings, image_metadata = _load_embedding_matrix(ctx.require_input("image_embeddings").path)
        text_embeddings, text_metadata = _load_embedding_matrix(ctx.require_input("text_embeddings").path)
        targets = _load_examples(ctx.require_input("targets").path)
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise RuntimeError("Retrieval embeddings must be rank-2 matrices")
        if image_embeddings.shape[1] != text_embeddings.shape[1]:
            raise RuntimeError(
                "Image/text embedding dimensions differ: %d vs %d" % (image_embeddings.shape[1], text_embeddings.shape[1])
            )
        if text_embeddings.shape[0] != len(targets):
            raise RuntimeError(
                "Retrieval query count mismatch: text embeddings has %d rows, targets has %d examples"
                % (text_embeddings.shape[0], len(targets))
            )
        if image_embeddings.shape[0] <= 0 or text_embeddings.shape[0] <= 0:
            raise RuntimeError("Retrieval ranking requires non-empty image and text embeddings")
        image_space, image_modality = _required_embedding_provenance(
            image_metadata, "image embedding", int(image_embeddings.shape[1])
        )
        text_space, text_modality = _required_embedding_provenance(
            text_metadata, "text embedding", int(text_embeddings.shape[1])
        )
        if image_space != text_space:
            raise RuntimeError(
                "Retrieval embedding provenance mismatch: image space %s, text space %s"
                % (image_space, text_space)
            )
        if image_modality != "image" or text_modality != "text":
            raise RuntimeError(
                "Retrieval ranking requires image and text embedding modalities, got %s and %s"
                % (image_modality, text_modality)
            )
        raw_top_k = ctx.params.get("top_k", 10)
        if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int) or raw_top_k < 1:
            raise RuntimeError("top_k must be a positive integer")
        top_k = int(raw_top_k)
        normalize = bool(ctx.params.get("normalize", True))
        image_ids = _optional_embedding_ids(
            image_metadata, int(image_embeddings.shape[0]), "image embedding"
        )
        if image_ids is None:
            raise RuntimeError(
                "Retrieval image embeddings must declare unique sample IDs to bind the candidate universe"
            )
        target_ids = _required_ids(targets, "retrieval target")
        declared_text_ids = _optional_embedding_ids(
            text_metadata, int(text_embeddings.shape[0]), "text embedding"
        )
        if declared_text_ids is None:
            raise RuntimeError(
                "Retrieval text embeddings must declare unique sample IDs to bind query identity"
            )
        if declared_text_ids != target_ids:
            raise RuntimeError(
                "Retrieval query ID/order mismatch: text embedding IDs %s, target IDs %s"
                % (declared_text_ids, target_ids)
            )
        text_ids = declared_text_ids
        expected_ids = [
            _required_scalar_field(
                item,
                ("target_id", "expected_id"),
                "retrieval target",
                index,
                allow_empty=False,
            )
            for index, item in enumerate(targets)
        ]
        missing_candidate_ids = sorted(set(expected_ids) - set(image_ids))
        if missing_candidate_ids:
            raise RuntimeError(
                "Retrieval target IDs are absent from image candidates: %s"
                % missing_candidate_ids
            )
        if normalize:
            image_matrix = _normalize_rows(image_embeddings, "image embedding")
            text_matrix = _normalize_rows(text_embeddings, "text embedding")
            similarity = "cosine"
        else:
            image_matrix = image_embeddings.astype(np.float64, copy=False)
            text_matrix = text_embeddings.astype(np.float64, copy=False)
            similarity = "dot_product"
        scores = text_matrix @ image_matrix.T
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("Retrieval ranking produced non-finite %s scores" % similarity)
        examples = []
        for query_index in range(scores.shape[0]):
            order = np.argsort(-scores[query_index])[: min(top_k, len(image_ids))]
            examples.append(
                {
                    "id": text_ids[query_index],
                    "target_id": expected_ids[query_index],
                    "ranked_ids": [image_ids[int(index)] for index in order],
                    "scores": [float(scores[query_index, int(index)]) for index in order],
                }
            )
        payload = {
            "schema_version": 1,
            "kind": "retrieval.rankings",
            "candidate_count": len(image_ids),
            "candidate_ids": image_ids,
            "ranking_depth": min(top_k, len(image_ids)),
            "similarity": similarity,
            "embedding_space": image_space,
            "examples": examples,
        }
        metadata = {
            "query_count": len(examples),
            "candidate_count": len(image_ids),
            "top_k": top_k,
            "normalize": normalize,
            "similarity": similarity,
            "embedding_space": image_space,
            "rankings_preview": examples[:10],
            "image_embedding_metadata": image_metadata,
            "text_embedding_metadata": text_metadata,
        }
        path = ctx.output_path("rankings", ".json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={"rankings": artifact("retrieval.rankings.json", path, metadata)},
            metrics={"retrieval.query_count": len(examples), "retrieval.candidate_count": len(image_ids)},
            metadata=metadata,
        )


def _write_report(
    ctx: OperationContext,
    family: str,
    rows: List[JsonDict],
    metrics: JsonDict,
    metadata: Optional[JsonDict] = None,
) -> OperationResult:
    report = {
        "schema_version": 1,
        "metric_family": family,
        "num_examples": len(rows),
        "metrics": metrics,
        "per_example": rows,
    }
    if metadata:
        report["metadata"] = dict(metadata)
    path = ctx.output_path("report", ".json")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return OperationResult(
        outputs={"report": artifact("metrics.report", path, report)},
        metrics=metrics,
        metadata={"num_examples": len(rows), **dict(metadata or {})},
    )


def _load_examples(path) -> List[JsonDict]:
    data = _load_task_payload(path)
    return _payload_examples(data, path)


def _load_task_payload(path) -> JsonDict:
    data = decode_strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise RuntimeError("Expected JSON object artifact: %s" % path)
    return dict(data)


def _payload_examples(data: Mapping[str, Any], path) -> List[JsonDict]:
    examples = data.get("examples")
    if not isinstance(examples, list):
        raise RuntimeError("Task metric artifact requires an examples list: %s" % path)
    if any(not isinstance(item, Mapping) for item in examples):
        raise RuntimeError("Task metric examples must all be objects: %s" % path)
    return [dict(item) for item in examples]


def _detection_prefilter_confidence_threshold(
    payload: Mapping[str, Any], artifact_metadata: Mapping[str, Any]
) -> float:
    declared: List[Tuple[str, Any]] = []
    for label, source in (
        ("payload", payload),
        ("artifact metadata", artifact_metadata),
    ):
        if source.get("inference_confidence_threshold") is not None:
            declared.append(
                (
                    "%s.inference_confidence_threshold" % label,
                    source.get("inference_confidence_threshold"),
                )
            )
        elif (
            source.get("adapter_family") == "yolo_detection"
            and source.get("confidence") is not None
        ):
            declared.append(
                ("%s.confidence" % label, source.get("confidence"))
            )
    values: List[Tuple[str, float]] = []
    for label, raw in declared:
        if isinstance(raw, bool):
            raise RuntimeError("%s must be numeric" % label)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("%s must be numeric" % label) from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError("%s must be finite and between 0 and 1" % label)
        values.append((label, value))
    if not values:
        return 0.0
    expected = values[0][1]
    conflicts = [
        "%s=%s" % (label, value)
        for label, value in values[1:]
        if value != expected
    ]
    if conflicts:
        raise RuntimeError(
            "Conflicting candidate inference confidence thresholds: %s=%s, %s"
            % (values[0][0], expected, ", ".join(conflicts))
        )
    return expected


def _label_metrics(
    reference: List[JsonDict],
    candidate: List[JsonDict],
    ref_key: str = "label",
    cand_key: str = "prediction",
) -> Tuple[List[JsonDict], JsonDict]:
    pairs = _paired_examples(reference, candidate)
    rows = []
    correct = 0
    label_total = Counter()
    label_correct = Counter()
    if ref_key == "answer":
        reference_keys = ("answer",)
        candidate_keys = ("answer", "prediction", "label")
    else:
        reference_keys = ("label",)
        candidate_keys = ("prediction", "label")
    for index, (example_id, ref, cand) in enumerate(pairs):
        expected = _required_scalar_field(
            ref, reference_keys, "reference label", index, allow_empty=False
        )
        predicted = _required_scalar_field(
            cand, candidate_keys, "candidate prediction", index, allow_empty=True
        )
        match = expected == predicted
        correct += 1 if match else 0
        label_total[expected] += 1
        if match:
            label_correct[expected] += 1
        rows.append({"id": example_id, "expected": expected, "predicted": predicted, "exact_match": 1.0 if match else 0.0})
    total = len(rows)
    accuracy = float(correct) / float(total) if total else 0.0
    balanced = (
        sum(float(label_correct[label]) / float(count) for label, count in label_total.items()) / float(len(label_total))
        if label_total
        else 0.0
    )
    return rows, {
        "task.accuracy": accuracy,
        "task.exact_match": accuracy,
        "classification.accuracy": accuracy,
        "classification.balanced_accuracy": balanced,
    }


def _paired_examples(reference: List[JsonDict], candidate: List[JsonDict]) -> List[Tuple[str, JsonDict, JsonDict]]:
    if len(reference) != len(candidate):
        raise RuntimeError(
            "Task example count mismatch: reference has %d examples, candidate has %d"
            % (len(reference), len(candidate))
        )
    if not reference:
        raise RuntimeError("Task metrics require at least one paired example")
    reference_ids = _required_ids(reference, "reference")
    candidate_ids = _required_ids(candidate, "candidate")
    if reference_ids != candidate_ids:
        raise RuntimeError(
            "Task sample ID/order mismatch: reference IDs %s, candidate IDs %s"
            % (reference_ids, candidate_ids)
        )
    return [
        (reference_ids[index], reference[index], candidate[index])
        for index in range(len(reference))
    ]


def _required_ids(examples: List[JsonDict], label: str) -> List[str]:
    ids = []
    for index, item in enumerate(examples):
        if item.get("id") is None or not str(item.get("id")):
            raise RuntimeError("%s example %d requires a non-empty id" % (label, index))
        ids.append(str(item["id"]))
    if len(set(ids)) != len(ids):
        raise RuntimeError("%s example IDs must be unique" % label)
    return ids


def _required_scalar_field(
    example: Mapping[str, Any],
    keys: Tuple[str, ...],
    label: str,
    index: int,
    *,
    allow_empty: bool,
) -> str:
    key = next((candidate for candidate in keys if candidate in example), None)
    if key is None or example.get(key) is None:
        raise RuntimeError(
            "%s example %d requires one of %s"
            % (label, index, ", ".join(keys))
        )
    value = example[key]
    if isinstance(value, (Mapping, list, tuple)):
        raise RuntimeError("%s example %d must contain a scalar value" % (label, index))
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("%s example %d must contain a finite value" % (label, index))
    rendered = str(value)
    if not allow_empty and rendered == "":
        raise RuntimeError("%s example %d requires a non-empty value" % (label, index))
    return rendered


def _required_string_field(
    example: Mapping[str, Any],
    keys: Tuple[str, ...],
    label: str,
    index: int,
) -> str:
    key = next((candidate for candidate in keys if candidate in example), None)
    if key is None or example.get(key) is None:
        raise RuntimeError(
            "%s example %d requires one of %s"
            % (label, index, ", ".join(keys))
        )
    value = example[key]
    if not isinstance(value, str):
        raise RuntimeError("%s example %d must contain a string" % (label, index))
    return value


def _detections_by_id(examples: Iterable[JsonDict]) -> Dict[str, List[JsonDict]]:
    output: Dict[str, List[JsonDict]] = {}
    for item in examples:
        example_id = str(item.get("id") or "")
        detections = item.get("detections")
        if isinstance(detections, list):
            if any(not isinstance(det, Mapping) for det in detections):
                raise RuntimeError("Detection lists must contain only objects for sample %s" % example_id)
            output[example_id] = [
                _validated_detection(det, example_id, index)
                for index, det in enumerate(detections)
            ]
        elif item.get("bbox") is not None:
            output.setdefault(example_id, []).append(
                _validated_detection(item, example_id, len(output.get(example_id, [])))
            )
        else:
            raise RuntimeError("Detection sample %s requires a detections list" % example_id)
    return output


def _validated_detection(
    value: Mapping[str, Any], example_id: str, index: int
) -> JsonDict:
    if "label" not in value or value.get("label") is None or str(value.get("label")) == "":
        raise RuntimeError(
            "Detection %d for sample %s requires a non-empty label" % (index, example_id)
        )
    bbox = value.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise RuntimeError(
            "Detection %d for sample %s requires a four-value bbox" % (index, example_id)
        )
    try:
        coordinates = [float(item) for item in bbox]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Detection %d for sample %s bbox must be numeric" % (index, example_id)
        ) from exc
    if not all(math.isfinite(item) for item in coordinates):
        raise RuntimeError(
            "Detection %d for sample %s bbox must be finite" % (index, example_id)
        )
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        raise RuntimeError(
            "Detection %d for sample %s bbox must have positive area" % (index, example_id)
        )
    output = dict(value)
    output["bbox"] = coordinates
    if "score" in value:
        score = value.get("score")
        if isinstance(score, bool):
            raise RuntimeError(
                "Detection %d for sample %s score must be numeric" % (index, example_id)
            )
        try:
            confidence = float(score)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Detection %d for sample %s score must be numeric" % (index, example_id)
            ) from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise RuntimeError(
                "Detection %d for sample %s score must be finite and between 0 and 1"
                % (index, example_id)
            )
        output["score"] = confidence
    return output


def _detections_at_confidence(
    detections: List[JsonDict], threshold: float, example_id: str
) -> List[JsonDict]:
    output = []
    for index, detection in enumerate(detections):
        if "score" not in detection:
            if threshold > 0.0:
                raise RuntimeError(
                    "Detection %d for sample %s requires a score when confidence_threshold > 0"
                    % (index, example_id)
                )
            output.append(detection)
        elif float(detection["score"]) >= threshold:
            output.append(detection)
    return output


def _detection_sort_key(detection: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(detection.get("label")),
        *(float(value) for value in detection.get("bbox") or ()),
        float(detection.get("score", 1.0)),
    )


def _maximum_detection_matches(
    references: List[JsonDict], candidates: List[JsonDict], threshold: float
) -> int:
    """Return maximum-cardinality label/IoU matches independent of input order."""

    left = sorted(references, key=_detection_sort_key)
    right = sorted(candidates, key=_detection_sort_key)
    adjacency: List[List[int]] = []
    for reference in left:
        eligible = []
        for index, candidate in enumerate(right):
            if candidate.get("label") != reference.get("label"):
                continue
            iou = _box_iou(reference.get("bbox"), candidate.get("bbox"))
            if iou >= threshold:
                eligible.append((index, iou, _detection_sort_key(candidate)))
        eligible.sort(key=lambda item: (-item[1], item[2]))
        adjacency.append([item[0] for item in eligible])

    matched_left_by_right: Dict[int, int] = {}

    def augment(left_index: int, visited: set[int]) -> bool:
        for right_index in adjacency[left_index]:
            if right_index in visited:
                continue
            visited.add(right_index)
            previous = matched_left_by_right.get(right_index)
            if previous is None or augment(previous, visited):
                matched_left_by_right[right_index] = left_index
                return True
        return False

    return sum(1 for index in range(len(left)) if augment(index, set()))


def _threshold_metric_suffix(threshold: float) -> str:
    if threshold == 0.0:
        return "0"
    rendered = format(float(threshold), ".12g")
    return rendered.replace("-", "m").replace(".", "p")


def _box_iou(left: Any, right: Any) -> float:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != 4 or len(right) != 4:
        return 0.0
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    inter_x1 = max(lx1, rx1)
    inter_y1 = max(ly1, ry1)
    inter_x2 = min(lx2, rx2)
    inter_y2 = min(ly2, ry2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - inter
    return inter / union if union > 0.0 else 0.0


def _load_masks(path) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        metadata: JsonDict = {}
        if "metadata_json" in payload:
            metadata = decode_strict_json_object(
                str(payload["metadata_json"]),
                label="Segmentation mask metadata_json",
            )
        if "masks" in payload:
            return np.asarray(payload["masks"]), metadata
        if "mask" in payload:
            return np.asarray(payload["mask"]), metadata
        array_keys = [key for key in payload.files if key != "metadata_json"]
        if array_keys:
            return np.asarray(payload[array_keys[0]]), metadata
    raise RuntimeError("Segmentation mask artifact requires a mask or masks array")


def _paired_mask_ids(
    reference_metadata: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
    count: int,
) -> Tuple[List[str], str]:
    reference_ids = _optional_sample_ids(reference_metadata, count, "reference mask")
    candidate_ids = _optional_sample_ids(candidate_metadata, count, "candidate mask")
    if (reference_ids is None) != (candidate_ids is None):
        raise RuntimeError(
            "Segmentation sample identity mismatch: both inputs must declare IDs when either input does"
        )
    if reference_ids is not None:
        if reference_ids != candidate_ids:
            raise RuntimeError(
                "Segmentation sample ID/order mismatch: reference IDs %s, candidate IDs %s"
                % (reference_ids, candidate_ids)
            )
        return reference_ids, "declared_id_and_order"
    return ["mask_%03d" % (index + 1) for index in range(count)], "exact_positional_no_ids"


def _optional_sample_ids(
    metadata: Mapping[str, Any], count: int, label: str
) -> Optional[List[str]]:
    raw = metadata.get("sample_ids") if "sample_ids" in metadata else metadata.get("ids")
    if raw is None:
        source = metadata.get("source_image_metadata")
        if isinstance(source, Mapping):
            raw = source.get("sample_ids") if "sample_ids" in source else source.get("ids")
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != count:
        raise RuntimeError("%s IDs must contain exactly %d entries" % (label, count))
    if any(item is None or str(item) == "" for item in raw):
        raise RuntimeError("%s IDs must be non-empty and unique" % label)
    ids = [str(item) for item in raw]
    if len(set(ids)) != len(ids):
        raise RuntimeError("%s IDs must be non-empty and unique" % label)
    return ids


def _load_embedding_matrix(path) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        if "embeddings" not in payload:
            raise RuntimeError("Embedding artifact requires an embeddings array: %s" % path)
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        metadata: JsonDict = {}
        if "metadata_json" in payload:
            metadata = decode_strict_json_object(
                str(payload["metadata_json"]),
                label="Embedding metadata_json in %s" % path,
            )
    if embeddings.size == 0:
        raise RuntimeError("Embedding artifact must contain a non-empty matrix: %s" % path)
    if not np.all(np.isfinite(embeddings)):
        raise RuntimeError("Embedding artifact contains non-finite values: %s" % path)
    return embeddings, metadata


def _optional_embedding_ids(
    metadata: Mapping[str, Any], count: int, label: str
) -> Optional[List[str]]:
    return _optional_sample_ids(metadata, count, label)


def _paired_embedding_ids(
    reference_metadata: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
    count: int,
) -> Tuple[List[str], List[str], str]:
    reference_ids = _optional_embedding_ids(reference_metadata, count, "reference embedding")
    candidate_ids = _optional_embedding_ids(candidate_metadata, count, "candidate embedding")
    if (reference_ids is None) != (candidate_ids is None):
        raise RuntimeError(
            "Embedding sample identity mismatch: both inputs must declare IDs when either input does"
        )
    if reference_ids is not None:
        if reference_ids != candidate_ids:
            raise RuntimeError(
                "Embedding sample ID/order mismatch: reference IDs %s, candidate IDs %s"
                % (reference_ids, candidate_ids)
            )
        return reference_ids, candidate_ids, "declared_id_and_order"
    generated = ["example_%03d" % (index + 1) for index in range(count)]
    return generated, list(generated), "exact_positional_no_ids"


def _normalize_rows(values: np.ndarray, label: str = "embedding") -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] <= 0:
        raise RuntimeError("%s normalization requires a non-empty rank-2 matrix" % label)
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("%s contains non-finite values" % label)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)):
        raise RuntimeError("%s norms are non-finite" % label)
    zero_rows = np.flatnonzero(norms[:, 0] == 0.0)
    if zero_rows.size:
        raise RuntimeError(
            "%s contains zero-norm rows at indices %s; cosine similarity is undefined"
            % (label, zero_rows.tolist())
        )
    return matrix / norms


def _parse_retrieval_k_values(value: Any) -> List[int]:
    if not isinstance(value, str):
        raise RuntimeError("Retrieval k_values must be a comma-separated string")
    tokens = [item.strip() for item in value.split(",")]
    if not tokens or any(not item for item in tokens):
        raise RuntimeError("Retrieval k_values must contain only positive integers")
    if any(not item.isdigit() or item.startswith("0") for item in tokens):
        raise RuntimeError("Retrieval k_values must contain only positive integers")
    values = [int(item) for item in tokens]
    if len(set(values)) != len(values):
        raise RuntimeError("Retrieval k_values must be unique")
    return values


def _required_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError("%s must be a positive integer" % label)
    return int(value)


def _retrieval_candidate_universe(
    payload: Mapping[str, Any], rankings: List[JsonDict]
) -> Tuple[List[str], int, int, str]:
    raw_candidate_ids = payload.get("candidate_ids")
    raw_candidate_count = payload.get("candidate_count")
    raw_ranking_depth = payload.get("ranking_depth")
    if raw_candidate_ids is not None:
        if not isinstance(raw_candidate_ids, list) or not raw_candidate_ids:
            raise RuntimeError("retrieval candidate_ids must be a non-empty list")
        if any(
            value is None or isinstance(value, (Mapping, list, tuple)) or not str(value)
            for value in raw_candidate_ids
        ):
            raise RuntimeError("retrieval candidate_ids must contain non-empty scalar IDs")
        candidate_ids = [str(value) for value in raw_candidate_ids]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise RuntimeError("retrieval candidate_ids must be unique")
        candidate_count = _required_positive_int(
            raw_candidate_count, "retrieval candidate_count"
        )
        if candidate_count != len(candidate_ids):
            raise RuntimeError(
                "retrieval candidate_count %d does not match %d candidate_ids"
                % (candidate_count, len(candidate_ids))
            )
        ranking_depth = _required_positive_int(
            raw_ranking_depth, "retrieval ranking_depth"
        )
        if ranking_depth > candidate_count:
            raise RuntimeError(
                "retrieval ranking_depth %d exceeds candidate_count %d"
                % (ranking_depth, candidate_count)
            )
        return candidate_ids, candidate_count, ranking_depth, "declared_candidate_ids"

    if raw_candidate_count is not None or raw_ranking_depth is not None:
        raise RuntimeError(
            "retrieval candidate_count/ranking_depth require candidate_ids to bind the candidate universe"
        )

    # Compatibility is safe only when every row contains the same complete pool. A
    # truncated legacy list cannot establish what candidates were eligible.
    ranked_lists = []
    for index, item in enumerate(rankings):
        raw = item.get("ranked_ids")
        if not isinstance(raw, list) or not raw:
            raise RuntimeError(
                "retrieval ranking example %d requires non-empty ranked_ids" % index
            )
        if any(
            value is None or isinstance(value, (Mapping, list, tuple)) or not str(value)
            for value in raw
        ):
            raise RuntimeError(
                "retrieval ranking example %d ranked_ids must contain non-empty scalar IDs"
                % index
            )
        rendered = [str(value) for value in raw]
        if len(set(rendered)) != len(rendered):
            raise RuntimeError(
                "retrieval ranking example %d ranked_ids must be unique" % index
            )
        ranked_lists.append(rendered)
    first_set = set(ranked_lists[0])
    if any(set(row) != first_set or len(row) != len(first_set) for row in ranked_lists):
        raise RuntimeError(
            "Legacy retrieval rankings may omit candidate_ids only when every row is a complete ranking of one identical pool"
        )
    candidate_ids = sorted(first_set)
    if not candidate_ids or any(not value for value in candidate_ids):
        raise RuntimeError("retrieval candidate universe contains an empty ID")
    return (
        candidate_ids,
        len(candidate_ids),
        len(candidate_ids),
        "inferred_identical_complete_legacy_rankings",
    )


def _require_bounded_scores(metrics: Mapping[str, Any], family: str) -> None:
    for metric_id, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("%s metric %s is not numeric" % (family, metric_id))
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise RuntimeError(
                "%s metric %s must be finite and between 0 and 1, got %r"
                % (family, metric_id, value)
            )


def _required_embedding_provenance(
    metadata: Mapping[str, Any], label: str, dimensions: int
) -> Tuple[JsonDict, str]:
    raw_space = metadata.get("embedding_space")
    if not isinstance(raw_space, Mapping):
        raise RuntimeError(
            "%s requires an embedding_space provenance object" % label
        )
    required = (
        "schema_version",
        "space_family",
        "backend",
        "model_id",
        "model_revision",
        "preprocessing_contract",
        "dimensions",
    )
    missing = [key for key in required if key not in raw_space]
    if missing:
        raise RuntimeError(
            "%s embedding_space is missing provenance fields: %s"
            % (label, missing)
        )
    space = {key: raw_space[key] for key in required}
    if space["schema_version"] != 1:
        raise RuntimeError("%s embedding_space schema_version must be 1" % label)
    for key in required[1:-1]:
        if not isinstance(space[key], str) or not space[key].strip():
            raise RuntimeError(
                "%s embedding_space.%s must be a non-empty string" % (label, key)
            )
        space[key] = space[key].strip()
    if isinstance(space["dimensions"], bool) or space["dimensions"] != dimensions:
        raise RuntimeError(
            "%s embedding_space dimensions %r do not match matrix width %d"
            % (label, space["dimensions"], dimensions)
        )
    space["dimensions"] = int(space["dimensions"])
    modality = metadata.get("embedding_modality")
    if modality not in {"image", "text", "generic", "multimodal"}:
        raise RuntimeError(
            "%s requires embedding_modality image, text, generic, or multimodal" % label
        )
    return space, str(modality)


def _validated_embedding_metric_namespace(
    requested: str,
    space: Mapping[str, Any],
    reference_modality: str,
    candidate_modality: str,
) -> str:
    pair = "%s_%s" % (reference_modality, candidate_modality)
    if requested == "embedding":
        return "embedding.%s" % pair
    allowed = {
        "generation.text_image_clip": ("clip", "text", "image"),
        "generation.image_image_clip": ("clip", "image", "image"),
    }
    expected = allowed.get(requested)
    actual = (
        str(space.get("space_family")),
        reference_modality,
        candidate_modality,
    )
    if expected is None or actual != expected:
        raise RuntimeError(
            "Embedding metric namespace %r is not proven by provenance %s"
            % (requested, actual)
        )
    return requested


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
