# Typed Semantic Artifacts

Noema makes semantic-task payloads explicit. A researcher can bring their own encoder or receiver
model as long as it emits one of these typed artifacts.

## Canonical Artifact Kinds

| Artifact kind | Purpose |
| --- | --- |
| `vision.embedding.clip.numpy` | CLIP-style image embedding matrix in NPZ, with `embeddings: [N,D]` and embedding-space provenance. |
| `multimodal.embedding.numpy` | Joint image/text or VLM embedding matrix in NPZ with embedding-space provenance. |
| `vision.detections.json` | Object boxes and labels, with examples containing `detections[]`. |
| `vision.segmentation_mask.numpy` | Integer segmentation masks in NPZ, usually `masks: [N,H,W]`. |
| `vision.scene_graph.json` | Object/relation graph, with `nodes[]` and `edges[]`. |
| `vision.semantic_map.json` | Region or polygon semantic map. |
| `text.caption.json` | Caption examples with `caption` fields. |
| `vqa.answers.json` | Question/answer examples with `question` and `answer` fields. |
| `retrieval.rankings.json` | Ranked candidate IDs per query plus the bound candidate IDs, candidate count, and ranking depth. |
| `semantic.importance_map.numpy` | Spatial/task importance weights in NPZ. |
| `video.frame_sequence.numpy` | Video frame sequence in NPZ, usually `frames: [T,H,W,3]` or `[N,T,H,W,3]`. |
| `image.batch.numpy` | Generated or reconstructed RGB image batch in NPZ, usually `images: [N,H,W,3]`. |

The source smoke adapter `source.semantic_artifacts_smoke` emits all of them so graph rendering,
recipe validation, artifact metadata, and task metrics can be tested without heavyweight models.

Publication-grade retrieval artifacts bind a non-empty unique `candidate_ids` universe,
`candidate_count`, and `ranking_depth`; every ranked ID and target must belong to that universe.
Legacy artifacts without these fields are accepted only when every query provides a complete ranking
of the same pool, which makes the missing declarations unambiguous. Paired embedding metrics require
matching backend/model/revision/preprocessing-space provenance rather than trusting a metric label.

## Runnable Smoke Recipe

```bash
uv run noema recipe validate recipes/semantic_artifacts_smoke.yaml
uv run noema recipe run recipes/semantic_artifacts_smoke.yaml
```

The recipe evaluates the artifacts that have task metric adapters:

```text
vision.detections.json           -> metrics.detection
vision.segmentation_mask.numpy   -> metrics.segmentation
text.caption.json                -> metrics.captioning
vqa.answers.json                 -> metrics.vqa
retrieval.rankings.json          -> metrics.retrieval
foundation.embedding.numpy       -> metrics.embedding_similarity
```

The remaining artifacts are carried as typed outputs for scene-graph VQA, semantic-map transmission,
importance-aware bit allocation, diffusion/generative receivers, and video/world-model experiments.
