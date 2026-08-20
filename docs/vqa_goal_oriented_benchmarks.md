# Goal-Oriented VQA Smoke Benchmark

This dependency-light visual question answering path exercises the platform shape needed for
goal-oriented semantic communication.

The reference semantic-packet smoke recipe is `recipes/vqa_goal_oriented_smoke.yaml`. It is a
plumbing example, not a canonical publication benchmark.

It runs this block sequence:

```text
source.vqa_smoke
  -> foundation.vqa_semantic_select
  -> foundation.vqa_payload_encode
  -> channel.bit_boundary
  -> channel.identity_encoder
  -> channel.bit_boundary
  -> channel.identity_link
  -> channel.bit_boundary
  -> channel.bit_count_match
  -> channel.identity_decoder
  -> foundation.vqa_payload_decode
  -> foundation.vqa_answer_from_packet
  -> metrics.vqa
```

The source emits tiny built-in images, questions, ground-truth answers, and ground-truth boxes. The selector is intentionally local-rule based: it produces a `vqa.semantic_packet.json` with the question-relevant label and region. The payload codec serializes that semantic packet as JSON UTF-8 bytes and then as canonical unpacked `uint8` bits.

This is not a final VQA model. BLIP, LLaVA, Qwen-VL, SAM, or CLIP adapters can replace
`foundation.vqa_semantic_select` and `foundation.vqa_answer_from_packet` while preserving:

- typed VQA artifacts,
- fixed channel bit boundaries,
- payload and transmitted bit accounting,
- VQA answer accuracy/exact-match metrics,
- benchmark catalog validation.

Run it from the CLI:

```bash
uv run noema recipe run recipes/vqa_goal_oriented_smoke.yaml
uv run noema benchmark validate benchmarks/vqa_goal_oriented_smoke_v1.yaml
```

In the dashboard, open **Template Recipes**, expand **Visual question answering**, choose the desired
starter, and press **Run All**. The Overview shows VQA as the recipe's read-only purpose tag.

## Pretrained VQA Baseline

The recipe `recipes/vqa_pretrained_transformers_smoke.yaml` uses the registered operation `foundation.vqa_transformers_answer`, an optional full-image VQA answerer. It is useful as a pretrained task baseline, but it is not itself a communication codec and does not exercise the channel blocks. The default model is `dandelin/vilt-b32-finetuned-vqa`; `Salesforce/blip-vqa-base` is a practical alternative.

This adapter needs the `foundation` extra:

```bash
# Explicitly supplied wheel (Noema is not yet on PyPI):
python -m pip install \
  "noema-lab[foundation] @ file:///absolute/path/to/noema_lab-<version>-<platform-tag>.whl"

# Source checkout:
uv sync --extra foundation
```

Use it when you want to compare the semantic-packet channel recipe against a conventional pretrained VQA model, or when designing a later VLM-based semantic encoder/decoder.
