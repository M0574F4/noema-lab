# Task-Oriented Benchmarks

Task-oriented benchmark contracts support semantic-communication experiments where the main result
is task success rather than pixel reconstruction.

## Supported Task Contracts

The research catalog now includes supported contracts for:

| Task | Metric operation | Primary metrics |
| --- | --- | --- |
| Classification | `metrics.classification` | `task.accuracy`, `task.exact_match`, `classification.balanced_accuracy` |
| Visual question answering | `metrics.vqa` | `vqa.single_reference_exact_match` |
| Object detection | `metrics.detection` | `detection.f1_at_iou_0p5_conf_0`, `detection.recall_at_iou_0p5_conf_0`, `detection.precision_at_iou_0p5_conf_0` |
| Segmentation | `metrics.segmentation` | `segmentation.semantic_raster_pooled_miou`, `segmentation.semantic_raster_pooled_pixel_accuracy` |
| Image-text retrieval | `metrics.retrieval` | `retrieval.hit_at_1`, `retrieval.hit_at_5`, `retrieval.hit_at_10` |
| Caption-to-image generation | `metrics.embedding_similarity` | `generation.text_image_clip.cosine_mean`, `generation.image_image_clip.cosine_mean` |

These operations are dependency-light evaluator blocks. The VQA evaluator uses one reference answer,
so it does not claim consensus-based standard VQA accuracy; its exact-match score is literal and
case-sensitive. The detection evaluator reports deterministic maximum-cardinality, label-matched
one-to-one precision/recall/F1 with both IoU and confidence thresholds encoded in each canonical
metric ID; it does not claim AP or mAP. Segmentation pools semantic-class rasters across samples and
does not claim instance-segmentation or COCO mask metrics. They define the task surface that heavier
model adapters can plug into later.
`metrics.captioning` remains available as a low-level artifact evaluator, but image captioning is
not advertised as a runnable task until a real captioning receiver is registered.

## Artifact Boundary

The task evaluators expect typed artifacts:

```text
task.labels.json                 reference class labels
task.predictions.json            predicted class labels
vqa.answers.json                 question-answer examples
vision.detections.json           bounding boxes and labels
vision.segmentation_mask.numpy   integer segmentation masks in NPZ
text.batch.json                  captions
retrieval.rankings.json          ranked candidate ids plus bound candidate_ids/count/depth
retrieval.targets.json           expected image id per text query, used by ranker blocks
image.batch.numpy                generated image batches
foundation.embedding.numpy       embedding matrices with model/revision/preprocessing-space provenance
```

VQA, detection, segmentation, CLIP retrieval, captioning, and generative receiver adapters use these
stable artifact kinds even when their internal models differ.

## CLIP Retrieval

The repository includes a runnable image-text retrieval baseline:

```bash
uv run noema recipe validate recipes/clip_retrieval_smoke.yaml
uv run noema recipe run recipes/clip_retrieval_smoke.yaml
```

The recipe uses:

```text
source.flickr8k_retrieval
  -> foundation.clip_image_embed
  -> foundation.clip_text_embed
  -> foundation.clip_retrieval_rank
  -> metrics.retrieval
```

The default source is real Flickr8k validation/test image-caption data, cached under
`.noema/datasets/flickr8k`, and the default backend is `transformers_clip` with
`openai/clip-vit-base-patch32`. Each query declares exactly one positive target, so the canonical
construct is Hit@K rather than multi-positive Recall@K. The ranker binds the complete candidate
universe, emitted ranking depth, and matching image/text embedding-space provenance; evaluators reject
duplicate or out-of-range K. Historical `retrieval.recall_at_K` outputs remain exact, explicitly
deprecated aliases only for this one-positive contract. `source.retrieval_smoke` remains only as a
fast synthetic debug source for plumbing tests; it
should not be reported as a research benchmark.

## Detection And Segmentation

The repository includes compact vision task baselines:

```bash
uv run noema recipe validate recipes/yolo_coco_detection.yaml
uv run noema recipe run recipes/yolo_coco_detection.yaml

uv run noema recipe validate recipes/yolo_coco_segmentation.yaml
uv run noema recipe run recipes/yolo_coco_segmentation.yaml
```

The detection recipe uses real COCO128 images and YOLO-format box labels:

```text
source.coco128_detection
  -> foundation.yolo_detect
  -> metrics.detection
```

The segmentation recipe uses real COCO8-seg images and YOLO polygon labels:

```text
source.coco8_segmentation
  -> foundation.yolo_segment
  -> metrics.segmentation
```

Both model adapters use pretrained Ultralytics YOLO models by default (`yolo11n.pt` and
`yolo11n-seg.pt`) and cache weights under `.noema/checkpoints/yolo`. In an installed environment use
an explicitly supplied wheel with
`python -m pip install "noema-lab[vision] @ file:///absolute/path/to/noema_lab-<version>-<platform-tag>.whl"`;
in a source checkout use `uv sync --extra vision`
(or `uv sync --all-extras`) before launching these recipes.

## Diffusion Generative Receiver

The repository includes a generative receiver baseline:

```bash
uv run noema recipe validate recipes/diffusion_flickr8k_generation.yaml
uv run noema recipe run recipes/diffusion_flickr8k_generation.yaml

uv run noema benchmark validate benchmarks/diffusion_flickr8k_generation_v1.yaml
uv run noema benchmark run benchmarks/diffusion_flickr8k_generation_v1.yaml
```

The recipe uses:

```text
source.flickr8k_retrieval
  -> foundation.text_semantic_state_encode
  -> foundation.semantic_state_payload_encode
  -> channel.bit_boundary / identity channel / bit-count checks
  -> foundation.semantic_state_payload_decode
  -> foundation.diffusion_state_to_image
  -> foundation.clip_text_embed + foundation.clip_image_embed
  -> metrics.embedding_similarity
```

The default receiver is `segmind/tiny-sd` through Diffusers on CPU using the local-quality preset:
512x512 generation, 20 denoising steps, and a simple photographic prompt template. It is a trained
compact Stable Diffusion model, so the default produces an actual generated image rather than a
random test pipeline. It still downloads a model checkpoint, but it is far lighter than ordinary
Stable Diffusion releases. The recipe stores both the raw Flickr8k caption and the exact prompt
passed to Diffusers, generates an image from the received caption/SemanticState prompt, and reports
CLIP-space text-image alignment plus source-image/generated-image alignment. Install optional
dependencies before running:

```bash
# Explicitly supplied wheel (Noema is not yet on PyPI):
python -m pip install \
  "noema-lab[foundation,retrieval-data] @ file:///absolute/path/to/noema_lab-<version>-<platform-tag>.whl"

# Source checkout:
uv sync --extra foundation --extra retrieval-data
```

This is a generative reconstruction benchmark, not a pixel-lossless reconstruction benchmark. The
source image is shown in the dashboard only as a visual and CLIP-space reference for the Flickr8k
caption pair. For a very fast plumbing-only smoke test, switch the UI quality preset to
`Plumbing only`; its tiny Diffusers pipeline output is not meaningful enough for research results.

## Smoke Benchmark

The first runnable pack is intentionally tiny:

```bash
uv run noema benchmark validate benchmarks/task_oriented_smoke_v1.yaml
uv run noema benchmark run benchmarks/task_oriented_smoke_v1.yaml
```

It uses `source.task_labels_smoke` to emit reference labels and candidate predictions, then evaluates
them with `metrics.classification`. This proves the task-success benchmark path, report generation,
catalog validation, and CLI result export.

## Shared Task Surface

CLIP retrieval, YOLO detection/segmentation, VQA, and Diffusers generation recipes all preserve the
same typed artifact and benchmark-reporting surface.

```text
image + question -> semantic encoder -> channel -> VQA receiver -> answer metrics
```
