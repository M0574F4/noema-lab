# Text Semantic Task

The text task is a lightweight semantic communication scaffold. It is intended for controlled
experiments, pipeline debugging, and early baselines, not as a claimed reproduction of a canonical
DeepSC checkpoint.

## Pipeline Contract

Text recipes follow the same sender/channel/receiver structure as image recipes:

```text
source.text_dataset
-> optional sender text source semantic perturbation
-> text encoder
-> fixed channel boundary
-> channel encoder / physical channel / channel decoder
-> text decoder
-> optional receiver repair
-> text metrics
```

Bit-preserving text codecs use canonical unpacked `np.uint8` bits at the payload and transmit
boundaries. Neural symbol codecs use canonical `np.complex64` channel symbols at the transmit and
receive symbol boundaries. Disabled channels still appear as explicit identity blocks, so rate and
format checks remain visible.

## Preset Recipes

The recommended starter presets are:

| Recipe | Purpose | Extra dependencies |
| --- | --- | --- |
| `recipes/text_semantic_utf8_clean.yaml` | Raw UTF-8 baseline with no source semantic perturbation and disabled physical channel. | none |
| `recipes/text_semantic_utf8_mask_repair.yaml` | Sender masks words, UTF-8 preserves literal `[MASK]`, receiver masked-LM fills them. | `textgen` |
| `recipes/text_bart_jscc_clean.yaml` | BART JSCC-lite neural semantic-symbol baseline. | `textgen` |

Run or validate the preset pack:

```bash
uv run noema benchmark validate benchmarks/text_semantic_presets_v1.yaml
uv run noema benchmark run benchmarks/text_semantic_presets_v1.yaml
```

Install optional model dependencies when using masked-LM repair or BART JSCC-lite:

```bash
# Explicitly supplied wheel (Noema is not yet on PyPI):
python -m pip install \
  "noema-lab[textgen] @ file:///absolute/path/to/noema_lab-<version>-<platform-tag>.whl"

# Source checkout:
uv sync --extra textgen
```

## Masking Rule

`[MASK]` is a control token only when the codec preserves text bytes exactly enough for the receiver
to see the same token. Therefore:

- Raw UTF-8 text can use `mask_words` sender noise and `masked_lm` receiver repair.
- BART JSCC-lite cannot use mask-token repair. Its decoder is generative, so `[MASK]` may become
  `[MASk]`, `[MAS K]`, another word, or disappear. The UI disables mask-token noise and masked-LM
  repair for this codec.

For JSCC experiments, use physical channel noise, symbol-domain effects, and text reconstruction
metrics rather than literal mask-token repair.

## Metrics

The current deterministic text metrics are:

```text
semantic.lexical_similarity   token-F1 lexical-overlap proxy (not embedding-based semantic similarity)
text.unigram_bleu_proxy       sentence unigram precision with brevity penalty (not corpus BLEU)
text.edit_similarity          normalized Levenshtein similarity
text.exact_match              literal case-, punctuation-, and whitespace-sensitive string match fraction
text.remaining_mask_count     unfilled mask-token count after receiver repair
```

These metrics are dependency-light, and the proxy name matters: it must not be reported as standard
corpus BLEU. Embedding metrics, BERTScore, task-success metrics, and LLM/VLM faithfulness judges are
not bundled with this evaluator. They require separate adapters with explicit dependency and runtime
contracts.

## Current Scope

The text task is sufficient as a platform scaffold:

- Raw UTF-8 gives a controlled, bit-preserving baseline.
- UTF-8 mask repair tests receiver-side generative repair without hiding it inside the decoder.
- BART JSCC-lite provides a neural semantic-symbol path with explicit `complex64` channel boundaries.

Published text-semantic-communication checkpoints and locally trained DeepSC variants can be
integrated as separate codec adapters; they are not represented by the BART JSCC-lite smoke baseline.
