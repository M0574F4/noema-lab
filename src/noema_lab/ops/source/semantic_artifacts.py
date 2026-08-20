from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema

JsonDict = Dict[str, Any]


class SemanticArtifactsSmokeOperation(Operation):
    id = "source.semantic_artifacts_smoke"
    name = "Typed semantic artifact smoke source"
    output_kinds = {
        "clip_embeddings": "vision.embedding.clip.numpy",
        "multimodal_embeddings": "multimodal.embedding.numpy",
        "detections_reference": "vision.detections.json",
        "detections_candidate": "vision.detections.json",
        "segmentation_reference": "vision.segmentation_mask.numpy",
        "segmentation_candidate": "vision.segmentation_mask.numpy",
        "scene_graph": "vision.scene_graph.json",
        "semantic_map": "vision.semantic_map.json",
        "captions_reference": "text.caption.json",
        "captions_candidate": "text.caption.json",
        "vqa_reference": "vqa.answers.json",
        "vqa_candidate": "vqa.answers.json",
        "rankings": "retrieval.rankings.json",
        "importance_map": "semantic.importance_map.numpy",
        "video_frames": "video.frame_sequence.numpy",
    }
    params_schema = object_schema(
        {
            "dataset": {
                "type": "string",
                "default": "semantic_artifact_smoke",
                "enum": ["semantic_artifact_smoke"],
            },
            "embedding_dim": {"type": "integer", "default": 8, "minimum": 2},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        dataset = str(ctx.params.get("dataset") or "semantic_artifact_smoke")
        if dataset != "semantic_artifact_smoke":
            raise RuntimeError("Unsupported semantic artifact smoke dataset: %s" % dataset)
        embedding_dim = max(2, int(ctx.params.get("embedding_dim") or 8))
        metadata = {
            "dataset": dataset,
            "split": "smoke",
            "artifact_contract_stage": "typed_semantic_artifacts",
            "example_count": 2,
        }
        outputs = {}

        clip = _normalized(np.arange(embedding_dim * 2, dtype=np.float32).reshape(2, embedding_dim) + 1.0)
        multimodal = _normalized(np.flip(clip, axis=1).astype(np.float32))
        outputs["clip_embeddings"] = _write_npz(ctx, "clip_embeddings", "vision.embedding.clip.numpy", {"embeddings": clip}, {**metadata, "dimensions": embedding_dim, "encoder": "clip_smoke"})
        outputs["multimodal_embeddings"] = _write_npz(ctx, "multimodal_embeddings", "multimodal.embedding.numpy", {"embeddings": multimodal}, {**metadata, "dimensions": embedding_dim, "encoder": "multimodal_smoke"})

        detections_ref = {
            "schema_version": 1,
            "kind": "vision.detections",
            "dataset": dataset,
            "examples": [
                {"id": "frame_001", "detections": [{"label": "antenna", "bbox": [10, 10, 40, 42], "score": 1.0}]},
                {"id": "frame_002", "detections": [{"label": "vehicle", "bbox": [12, 14, 52, 58], "score": 1.0}]},
            ],
        }
        detections_cand = {
            "schema_version": 1,
            "kind": "vision.detections",
            "dataset": dataset,
            "examples": [
                {"id": "frame_001", "detections": [{"label": "antenna", "bbox": [11, 11, 41, 42], "score": 0.91}]},
                {"id": "frame_002", "detections": [{"label": "vehicle", "bbox": [14, 16, 50, 57], "score": 0.88}]},
            ],
        }
        outputs["detections_reference"] = _write_json(ctx, "detections_reference", "vision.detections.json", detections_ref, {**metadata, "box_count": 2})
        outputs["detections_candidate"] = _write_json(ctx, "detections_candidate", "vision.detections.json", detections_cand, {**metadata, "box_count": 2})

        mask_ref = np.array([[[0, 0, 1], [0, 1, 1], [2, 2, 1]], [[1, 1, 0], [1, 0, 0], [2, 2, 2]]], dtype=np.int32)
        mask_cand = np.array([[[0, 0, 1], [0, 1, 1], [2, 1, 1]], [[1, 1, 0], [1, 0, 0], [2, 2, 1]]], dtype=np.int32)
        outputs["segmentation_reference"] = _write_npz(ctx, "segmentation_reference", "vision.segmentation_mask.numpy", {"masks": mask_ref}, {**metadata, "classes": [0, 1, 2]})
        outputs["segmentation_candidate"] = _write_npz(ctx, "segmentation_candidate", "vision.segmentation_mask.numpy", {"masks": mask_cand}, {**metadata, "classes": [0, 1, 2]})

        scene_graph = {
            "schema_version": 1,
            "kind": "vision.scene_graph",
            "dataset": dataset,
            "examples": [
                {
                    "id": "frame_001",
                    "nodes": [{"id": "antenna", "label": "antenna"}, {"id": "roof", "label": "roof"}],
                    "edges": [{"source": "antenna", "relation": "mounted_on", "target": "roof"}],
                }
            ],
        }
        semantic_map = {
            "schema_version": 1,
            "kind": "vision.semantic_map",
            "dataset": dataset,
            "examples": [
                {"id": "frame_001", "regions": [{"label": "antenna", "polygon": [[10, 10], [40, 10], [40, 42], [10, 42]]}]}
            ],
        }
        outputs["scene_graph"] = _write_json(ctx, "scene_graph", "vision.scene_graph.json", scene_graph, {**metadata, "node_count": 2, "edge_count": 1})
        outputs["semantic_map"] = _write_json(ctx, "semantic_map", "vision.semantic_map.json", semantic_map, {**metadata, "region_count": 1})

        captions_ref = {
            "schema_version": 1,
            "kind": "text.caption",
            "dataset": dataset,
            "examples": [
                {"id": "frame_001", "caption": "an antenna is mounted on the roof"},
                {"id": "frame_002", "caption": "a vehicle is near the roadside sensor"},
            ],
        }
        captions_cand = {
            "schema_version": 1,
            "kind": "text.caption",
            "dataset": dataset,
            "examples": [
                {"id": "frame_001", "caption": "an antenna is mounted on a roof"},
                {"id": "frame_002", "caption": "a vehicle is near the sensor"},
            ],
        }
        outputs["captions_reference"] = _write_json(ctx, "captions_reference", "text.caption.json", captions_ref, {**metadata, "caption_count": 2})
        outputs["captions_candidate"] = _write_json(ctx, "captions_candidate", "text.caption.json", captions_cand, {**metadata, "caption_count": 2})

        vqa_ref = {
            "schema_version": 1,
            "kind": "vqa.answers",
            "dataset": dataset,
            "examples": [
                {"id": "qa_001", "question": "What is mounted on the roof?", "answer": "antenna"},
                {"id": "qa_002", "question": "What is near the roadside sensor?", "answer": "vehicle"},
            ],
        }
        vqa_cand = {
            "schema_version": 1,
            "kind": "vqa.answers",
            "dataset": dataset,
            "examples": [
                {"id": "qa_001", "question": "What is mounted on the roof?", "answer": "antenna"},
                {"id": "qa_002", "question": "What is near the roadside sensor?", "answer": "vehicle"},
            ],
        }
        outputs["vqa_reference"] = _write_json(ctx, "vqa_reference", "vqa.answers.json", vqa_ref, {**metadata, "question_count": 2})
        outputs["vqa_candidate"] = _write_json(ctx, "vqa_candidate", "vqa.answers.json", vqa_cand, {**metadata, "question_count": 2})

        rankings = {
            "schema_version": 1,
            "kind": "retrieval.rankings",
            "dataset": dataset,
            "examples": [
                {"id": "query_001", "target_id": "frame_001", "ranked_ids": ["frame_001", "frame_002"]},
                {"id": "query_002", "target_id": "frame_002", "ranked_ids": ["frame_001", "frame_002"]},
            ],
        }
        outputs["rankings"] = _write_json(ctx, "rankings", "retrieval.rankings.json", rankings, {**metadata, "query_count": 2})

        importance = np.array([[[0.1, 0.2, 0.8], [0.2, 0.7, 0.9], [0.1, 0.3, 0.5]]], dtype=np.float32)
        frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        frames[0, :, :, 0] = 180
        frames[1, :, :, 1] = 160
        outputs["importance_map"] = _write_npz(ctx, "importance_map", "semantic.importance_map.numpy", {"importance": importance}, {**metadata, "dtype": "float32"})
        outputs["video_frames"] = _write_npz(ctx, "video_frames", "video.frame_sequence.numpy", {"frames": frames}, {**metadata, "frame_count": 2, "fps": 1.0})

        return OperationResult(
            outputs=outputs,
            metrics={
                "semantic_artifact.count": len(outputs),
                "semantic_artifact.example_count": metadata["example_count"],
            },
            metadata={**metadata, "artifact_kinds": {name: item.kind for name, item in outputs.items()}},
        )


def _write_json(ctx: OperationContext, name: str, kind: str, payload: JsonDict, metadata: JsonDict):
    path = ctx.output_path(name, ".json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact(kind, path, metadata)


def _write_npz(ctx: OperationContext, name: str, kind: str, arrays: Dict[str, np.ndarray], metadata: JsonDict):
    path = ctx.output_path(name, ".npz")
    np.savez_compressed(path, **arrays, metadata_json=json.dumps(metadata, sort_keys=True))
    return artifact(kind, path, metadata)


def _normalized(value: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(value, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return value / denom
