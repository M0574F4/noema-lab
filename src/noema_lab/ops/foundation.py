from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.boundaries import validate_channel_bits
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationResult, object_schema
from noema_lab.core.semantic import (
    KnowledgeBase,
    SemanticState,
    concept_set,
    concepts_from_text,
    ensure_jsonable,
    entity_set,
    fact_key,
    fact_set,
    kb_fact_set,
    load_json,
    load_knowledge_base,
    load_semantic_state,
    overlap_scores,
    state_by_id,
    tokenize,
    write_json,
)
from noema_lab.core.structured_input import (
    decode_strict_json,
    decode_strict_json_object,
    decode_strict_yaml_or_json,
)
from noema_lab.ops.models.timing import append_measurement, codec_timing_metadata
from noema_lab.ops.source.text_dataset import TEXT_SMOKE_EXAMPLES

JsonDict = Dict[str, Any]
MASK_TOKEN_RE = re.compile(r"\[\s*mask\s*\]", re.IGNORECASE)
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_CLIP_MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
DEFAULT_DIFFUSION_MODEL_ID = "segmind/tiny-sd"
DEFAULT_DIFFUSION_MODEL_REVISION = "cad0bd7495fa6c4bcca01b19a723dc91627fe84f"
DEFAULT_MASKED_LM_MODEL_ID = "distilbert/distilbert-base-uncased"
DEFAULT_MASKED_LM_MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


class KnowledgeBaseSourceOperation(Operation):
    id = "foundation.knowledge_base"
    name = "Foundation knowledge-base source"
    output_kinds = {"kb": "foundation.kb.json"}
    params_schema = object_schema(
        {
            "kb_id": {
                "type": "string",
                "default": "semantic_text_smoke_kb",
                "enum": ["semantic_text_smoke_kb", "empty", "inline_json"],
            },
            "facts_json": {
                "type": "string",
                "default": "",
                "description": "Optional JSON list of facts. Used when kb_id is inline_json; appended for semantic_text_smoke_kb.",
            },
            "max_builtin_concepts": {"type": "integer", "default": 12, "minimum": 1},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        kb_id = str(ctx.params.get("kb_id") or "semantic_text_smoke_kb")
        max_builtin_concepts = max(1, int(ctx.params.get("max_builtin_concepts") or 12))
        facts: List[JsonDict] = []
        if kb_id == "semantic_text_smoke_kb":
            facts.extend(_builtin_text_facts(max_builtin_concepts))
        elif kb_id not in ("empty", "inline_json"):
            raise OperationError("Unsupported knowledge base id: %s" % kb_id)
        inline = str(ctx.params.get("facts_json") or "").strip()
        if inline:
            try:
                parsed = decode_strict_yaml_or_json(
                    inline,
                    input_format="json",
                )
            except Exception as exc:
                raise OperationError("facts_json must be valid JSON: %s" % exc) from exc
            if not isinstance(parsed, list):
                raise OperationError("facts_json must be a JSON list of fact objects")
            for index, item in enumerate(parsed):
                if not isinstance(item, Mapping):
                    raise OperationError(
                        "facts_json item %d must be a JSON object" % index
                    )
                facts.append(_normalize_fact(item, "inline"))
        kb = KnowledgeBase(kb_id=kb_id, facts=facts)
        payload = kb.to_dict()
        payload["fact_count"] = len(facts)
        path = ctx.output_path("kb", ".json")
        write_json(path, payload)
        return OperationResult(
            outputs={"kb": artifact("foundation.kb.json", path, {"kb_id": kb_id, "fact_count": len(facts)})},
            metrics={"kb.fact_count": len(facts)},
            metadata={"kb_id": kb_id, "fact_count": len(facts)},
        )


class TextSemanticStateEncodeOperation(Operation):
    id = "foundation.text_semantic_state_encode"
    name = "Text to SemanticState encoder"
    input_kinds = {"texts": ["text.batch.json"]}
    output_kinds = {"state": "semantic.state.json"}
    params_schema = object_schema(
        {
            "extractor": {
                "type": "string",
                "default": "local_rules",
                "enum": ["local_rules"],
            },
            "max_concepts": {"type": "integer", "default": 16, "minimum": 1},
            "include_text": {"type": "boolean", "default": False},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        extractor = str(ctx.params.get("extractor") or "local_rules")
        if extractor != "local_rules":
            raise OperationError(
                "Text SemanticState extractor `%s` is not configured; use local_rules."
                % extractor
            )
        start = time.perf_counter()
        batch = _load_text_batch(ctx.require_input("texts").path)
        examples = _examples(batch)
        max_concepts = max(1, int(ctx.params.get("max_concepts") or 16))
        include_text = bool(ctx.params.get("include_text", False))
        states = []
        total = max(len(examples), 1)
        for index, example in enumerate(examples):
            ctx.report_progress(
                "Encoding SemanticState %d/%d" % (index, total),
                phase="semantic_state_encode",
                status="running",
                completed=index,
                total=total,
                percent=float(index) / float(total) * 100.0,
                unit="texts",
            )
            states.append(_semantic_state_from_text(example, max_concepts, include_text))
        ctx.report_progress(
            "Encoding SemanticState %d/%d" % (len(examples), total),
            phase="semantic_state_encode",
            status="running",
            completed=len(examples),
            total=total,
            percent=100.0,
            unit="texts",
        )
        elapsed = time.perf_counter() - start
        state = SemanticState(modality="text", states=states, state_id="text_semantic_state")
        payload = state.to_dict()
        payload["source_dataset"] = batch.get("dataset")
        path = ctx.output_path("state", ".json")
        write_json(path, payload)
        timing_records: List[JsonDict] = []
        append_measurement(timing_records, None, "encoder.semantic_state_extract", elapsed, text_count=len(examples))
        append_measurement(timing_records, None, "encoder.total", elapsed, text_count=len(examples))
        metadata = {
            "semantic_state_count": len(states),
            "extractor": extractor,
            "semantic_form": "semantic_state",
            "modality": "text",
            "codec_timing": codec_timing_metadata(
                "encoder",
                timing_records,
                runner="local_python",
                notes={
                    "runtime_execution_language": "Python",
                    "foundation_adapter": "local rule-based SemanticState extractor",
                },
            ),
        }
        return OperationResult(
            outputs={"state": artifact("semantic.state.json", path, metadata)},
            metrics={
                "semantic_state.count": len(states),
                "semantic_state.concept_count": sum(len(item.get("concepts") or []) for item in states),
                "semantic_state.fact_count": sum(len(item.get("facts") or []) for item in states),
            },
            metadata=metadata,
        )


class SemanticStateGroundOperation(Operation):
    id = "foundation.semantic_state_ground"
    name = "Ground SemanticState against a knowledge base"
    input_kinds = {"state": ["semantic.state.json"], "kb": ["foundation.kb.json"]}
    output_kinds = {"state": "semantic.state.json"}
    params_schema = object_schema(
        {
            "min_token_overlap": {"type": "integer", "default": 1, "minimum": 1},
            "max_matches_per_state": {"type": "integer", "default": 12, "minimum": 1},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        state = load_semantic_state(ctx.require_input("state").path)
        kb = load_knowledge_base(ctx.require_input("kb").path)
        min_overlap = max(1, int(ctx.params.get("min_token_overlap") or 1))
        max_matches = max(1, int(ctx.params.get("max_matches_per_state") or 12))
        kb_facts = list(kb.facts)
        match_count = 0
        grounded_states = []
        for item in state.states:
            grounded = dict(item)
            tokens = _state_tokens(item)
            matches = []
            for fact in kb_facts:
                fact_tokens = set(tokenize("%s %s %s" % (fact.get("subject", ""), fact.get("predicate", ""), fact.get("object", ""))))
                overlap = sorted(tokens & fact_tokens)
                if len(overlap) >= min_overlap:
                    matches.append({"fact_id": fact.get("id"), "overlap": overlap, "fact": fact})
                if len(matches) >= max_matches:
                    break
            grounded["kb_matches"] = matches
            grounded["grounded"] = bool(matches)
            match_count += len(matches)
            grounded_states.append(grounded)
        payload = SemanticState(modality=state.modality, states=grounded_states, state_id=state.state_id).to_dict()
        payload["knowledge_base"] = kb.kb_id
        path = ctx.output_path("state", ".json")
        write_json(path, payload)
        metadata = {
            "semantic_state_count": len(grounded_states),
            "kb_id": kb.kb_id,
            "kb_match_count": match_count,
            "semantic_form": "semantic_state",
            "modality": state.modality,
        }
        return OperationResult(
            outputs={"state": artifact("semantic.state.json", path, metadata)},
            metrics={
                "faithfulness.kb_match_count": match_count,
                "faithfulness.kb_grounded_state_fraction": _safe_fraction(sum(1 for item in grounded_states if item.get("grounded")), len(grounded_states)),
            },
            metadata=metadata,
        )


class SemanticStatePayloadEncodeOperation(Operation):
    id = "foundation.semantic_state_payload_encode"
    name = "SemanticState payload encoder"
    input_kinds = {"state": ["semantic.state.json"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    params_schema = object_schema(
        {
            "payload_format": {
                "type": "string",
                "default": "semantic_state_json_utf8",
                "enum": ["semantic_state_json_utf8"],
            }
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        start = time.perf_counter()
        state_payload = load_json(ctx.require_input("state").path)
        transport_payload, redacted_field_count = _semantic_state_transport_payload(
            state_payload
        )
        payload_bytes = _canonical_json_bytes(transport_payload)
        bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8)).astype(np.uint8, copy=False)
        transported_states = list(transport_payload.get("states") or [])
        (
            source_item_payload_bit_counts,
            container_bit_count,
        ) = _semantic_payload_item_bit_counts(payload_bytes, transported_states)
        source_item_ids = [
            str(item.get("id") or index)
            for index, item in enumerate(transported_states)
        ]
        elapsed = time.perf_counter() - start
        metadata = {
            "codec": "semantic_state_json_utf8",
            "payload_format": "semantic_state_json_utf8",
            "payload_byte_count": len(payload_bytes),
            "payload_bit_count": int(bits.size),
            "bit_count": int(bits.size),
            "bit_role": "payload",
            "bit_storage": "unpacked_uint8",
            "channel_bit_array": "unpacked_uint8",
            "bits_per_element": 1,
            "semantic_form": "semantic_state",
            "payload_coder": "semantic_state_json_utf8",
            "payload_coder_type": "semantic_state_serialization",
            "payload_coder_label": "SemanticState JSON UTF-8 payload",
            "payload_coder_lossless": True,
            "entropy_coded": False,
            "source_item_count": len(transported_states),
            "source_item_ids": source_item_ids,
            "source_item_payload_bit_counts": source_item_payload_bit_counts,
            "source_item_use_counts_are_additive": True,
            "payload_container_bit_count": container_bit_count,
            "payload_accounting_policy": (
                "Contiguous canonical UTF-8 JSON slices cover the full payload; the "
                "first slice includes the top-level prefix, each non-final slice "
                "includes its separator, and the final slice includes the suffix."
            ),
            "semantic_transport_policy": "alignment_id_and_semantic_evidence_only_v1",
            "semantic_transport_redacted_field_count": redacted_field_count,
        }
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
        timing_records: List[JsonDict] = []
        append_measurement(timing_records, None, "encoder.payload_encode", elapsed, byte_count=len(payload_bytes))
        append_measurement(timing_records, None, "encoder.total", elapsed, byte_count=len(payload_bytes))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, metadata)},
            metrics={
                "channel.payload_bit_count": int(bits.size),
                "codec.bytes": len(payload_bytes),
                "codec.bit_count": int(bits.size),
            },
            metadata={
                **metadata,
                "byte_count": len(payload_bytes),
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner="local_python",
                    notes={
                        "runtime_execution_language": "Python + NumPy",
                        "payload_codec": "SemanticState JSON UTF-8 bytes to unpacked uint8 bits",
                    },
                ),
            },
        )


class SemanticStatePayloadDecodeOperation(Operation):
    id = "foundation.semantic_state_payload_decode"
    name = "SemanticState payload decoder"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"state": "semantic.state.json"}
    params_schema = object_schema(
        {
            "on_error": {"type": "string", "default": "replace", "enum": ["replace", "fail"]},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        start = time.perf_counter()
        bit_artifact = ctx.require_input("bits")
        bits, metadata = _load_bits(bit_artifact.path, bit_artifact.metadata)
        on_error = str(ctx.params.get("on_error") or "replace")
        bit_count = int(metadata.get("payload_bit_count") or metadata.get("bit_count") or bits.size)
        byte_count = int(metadata.get("payload_byte_count") or ((bit_count + 7) // 8))
        clipped = bits[:bit_count].astype(np.uint8, copy=False)
        if clipped.size % 8:
            clipped = np.pad(clipped, (0, 8 - clipped.size % 8), constant_values=0).astype(np.uint8, copy=False)
        payload_bytes = np.packbits(clipped)[:byte_count].tobytes()
        decode_error = ""
        decoder_redacted_field_count = 0
        try:
            raw_payload = decode_strict_yaml_or_json(
                payload_bytes.decode("utf-8", errors="strict"),
                input_format="json",
            )
            payload, decoder_redacted_field_count = (
                _semantic_state_transport_payload(raw_payload)
            )
            state = SemanticState.from_dict(payload)
        except Exception as exc:
            if on_error == "fail":
                raise OperationError("SemanticState payload decode failed: %s" % exc) from exc
            decode_error = str(exc)
            state = SemanticState(modality="unknown", states=[], state_id="decode_error")
            payload = state.to_dict()
            payload["decode_error"] = decode_error
        elapsed = time.perf_counter() - start
        path = ctx.output_path("state", ".json")
        write_json(path, payload)
        out_metadata = {
            "codec": "semantic_state_json_utf8",
            "payload_format": metadata.get("payload_format") or "semantic_state_json_utf8",
            "payload_byte_count": byte_count,
            "payload_bit_count": bit_count,
            "semantic_state_count": len(state.states),
            "semantic_form": "semantic_state",
            "payload_coder": metadata.get("payload_coder") or "semantic_state_json_utf8",
            "payload_coder_type": metadata.get("payload_coder_type") or "semantic_state_serialization",
            "payload_coder_label": metadata.get("payload_coder_label") or "SemanticState JSON UTF-8 payload",
            "payload_coder_lossless": bool(metadata.get("payload_coder_lossless", True)),
            "entropy_coded": bool(metadata.get("entropy_coded", False)),
            "decode_error": decode_error,
            "source_item_count": int(metadata.get("source_item_count") or 0),
            "source_item_ids": list(metadata.get("source_item_ids") or []),
            "source_item_payload_bit_counts": list(
                metadata.get("source_item_payload_bit_counts") or []
            ),
            "source_item_use_counts_are_additive": bool(
                metadata.get("source_item_use_counts_are_additive", False)
            ),
            "payload_container_bit_count": int(
                metadata.get("payload_container_bit_count") or 0
            ),
            "payload_accounting_policy": str(
                metadata.get("payload_accounting_policy") or ""
            ),
            "semantic_transport_policy": str(
                metadata.get("semantic_transport_policy") or ""
            ),
            "semantic_transport_redacted_field_count": int(
                metadata.get("semantic_transport_redacted_field_count") or 0
            )
            + decoder_redacted_field_count,
        }
        timing_records: List[JsonDict] = []
        append_measurement(timing_records, None, "decoder.payload_decode", elapsed, byte_count=byte_count)
        append_measurement(timing_records, None, "decoder.total", elapsed, byte_count=byte_count)
        return OperationResult(
            outputs={"state": artifact("semantic.state.json", path, out_metadata)},
            metrics={"semantic_state.decode_error": 1 if decode_error else 0},
            metadata={
                **out_metadata,
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="local_python",
                    notes={
                        "runtime_execution_language": "Python + NumPy",
                        "payload_codec": "unpacked uint8 bits to SemanticState JSON UTF-8 bytes",
                    },
                ),
            },
        )


class SemanticStateToTextOperation(Operation):
    id = "foundation.semantic_state_to_text"
    name = "SemanticState to text generator"
    input_kinds = {"state": ["semantic.state.json"], "kb": ["foundation.kb.json"]}
    output_kinds = {"texts": "text.batch.json"}
    params_schema = object_schema(
        {
            "generator": {
                "type": "string",
                "default": "local_template",
                "enum": ["local_template", "kb_reconstruct", "masked_lm"],
            },
            "prefer_source_text": {"type": "boolean", "default": True},
            "model_id": {"type": "string", "default": DEFAULT_MASKED_LM_MODEL_ID},
            "model_revision": {
                "type": "string",
                "default": "",
                "pattern": "^(|[0-9a-fA-F]{40})$",
                "description": "Full immutable Hugging Face commit. The built-in default model resolves to its pinned commit; other remote models require this explicitly.",
            },
            "device": {"type": "string", "default": "cpu"},
            "cache_dir": {"type": "string", "default": ""},
            "top_k": {"type": "integer", "default": 1, "minimum": 1},
        }
    )

    def runtime_availability(self, params: Mapping[str, Any]) -> JsonDict:
        return _text_generator_availability(
            str(params.get("generator") or "local_template"),
            allow_kb_reconstruct=True,
        )

    def run(self, ctx: OperationContext) -> OperationResult:
        generator = str(ctx.params.get("generator") or "local_template")
        if generator not in ("local_template", "kb_reconstruct", "masked_lm"):
            raise OperationError(
                "SemanticState generator `%s` is not configured; use "
                "local_template, kb_reconstruct, or masked_lm." % generator
            )
        start = time.perf_counter()
        state = load_semantic_state(ctx.require_input("state").path)
        kb = load_knowledge_base(ctx.require_input("kb").path)
        prefer_source_text = bool(ctx.params.get("prefer_source_text", True))
        masked_lm = None
        model_id = str(ctx.params.get("model_id") or DEFAULT_MASKED_LM_MODEL_ID)
        model_revision = ""
        top_k = max(1, int(ctx.params.get("top_k") or 1))
        device = str(ctx.params.get("device") or "cpu")
        if generator == "masked_lm":
            model_revision = _resolved_remote_model_revision(
                model_id,
                ctx.params.get("model_revision"),
                default_model_id=DEFAULT_MASKED_LM_MODEL_ID,
                default_revision=DEFAULT_MASKED_LM_MODEL_REVISION,
                label="Masked language model",
            )
            cache_dir = str(ctx.params.get("cache_dir") or "").strip()
            if not cache_dir:
                cache_dir = str(ctx.run_dir.parent.parent / "hf_cache")
            masked_lm = _load_fill_mask_pipeline(
                model_id, model_revision, device, cache_dir
            )
        examples = []
        total = max(len(state.states), 1)
        for index, item in enumerate(state.states):
            ctx.report_progress(
                "Generating text %d/%d" % (index, total),
                phase="semantic_state_to_text",
                status="running",
                completed=index,
                total=total,
                percent=float(index) / float(total) * 100.0,
                unit="texts",
            )
            text = ""
            kb_text = _canonical_text_from_semantic_evidence(item, kb)
            if generator == "kb_reconstruct":
                text = kb_text
            elif generator == "masked_lm":
                text, mask_fills = _masked_lm_render(
                    masked_lm,
                    str(item.get("source_text") or ""),
                    top_k,
                    kb_text,
                )
            if not text and prefer_source_text:
                text = str(item.get("source_text") or "")
            if not text:
                concepts = [str(value) for value in item.get("concepts") or []]
                if concepts:
                    text = "Semantic content: " + ", ".join(concepts) + "."
                else:
                    text = ""
            example = {"id": str(item.get("id") or len(examples)), "text": text}
            if generator == "masked_lm" and mask_fills:
                example["mask_fills"] = mask_fills
            examples.append(example)
        ctx.report_progress(
            "Generating text %d/%d" % (len(state.states), total),
            phase="semantic_state_to_text",
            status="running",
            completed=len(state.states),
            total=total,
            percent=100.0,
            unit="texts",
        )
        elapsed = time.perf_counter() - start
        payload = {
            "schema_version": 1,
            "kind": "text.batch",
            "dataset": "semantic_state_generated",
            "split": "generated",
            "examples": examples,
        }
        metadata = {
            "generator": generator,
            "model_id": model_id if generator == "masked_lm" else "",
            "model_revision": model_revision,
            "cache_dir": cache_dir if generator == "masked_lm" else "",
            "text_count": len(examples),
            "remaining_mask_count": sum(_mask_token_count(str(item.get("text") or "")) for item in examples),
            "texts_preview": [
                {
                    "id": item["id"],
                    "text": item["text"],
                    **({"mask_fills": item["mask_fills"]} if item.get("mask_fills") else {}),
                }
                for item in examples
            ],
            "semantic_form": "text",
        }
        path = ctx.output_path("texts", ".json")
        write_json(path, payload)
        timing_records: List[JsonDict] = []
        append_measurement(timing_records, None, "decoder.semantic_state_render", elapsed, text_count=len(examples))
        append_measurement(timing_records, None, "decoder.total", elapsed, text_count=len(examples))
        return OperationResult(
            outputs={"texts": artifact("text.batch.json", path, metadata)},
            metrics={"text.generated_count": len(examples)},
            metadata={
                **metadata,
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="local_python",
                    notes={
                        "runtime_execution_language": "Python",
                        "foundation_adapter": f"{generator} SemanticState renderer",
                    },
                ),
            },
        )


class TextMaskRepairOperation(Operation):
    id = "foundation.text_mask_repair"
    name = "Text receiver mask repair"
    input_kinds = {"texts": ["text.batch.json"]}
    output_kinds = {"texts": "text.batch.json"}
    params_schema = object_schema(
        {
            "generator": {
                "type": "string",
                "default": "masked_lm",
                "enum": ["local_template", "masked_lm"],
            },
            "model_id": {"type": "string", "default": DEFAULT_MASKED_LM_MODEL_ID},
            "model_revision": {
                "type": "string",
                "default": "",
                "pattern": "^(|[0-9a-fA-F]{40})$",
                "description": "Full immutable Hugging Face commit. The built-in default model resolves to its pinned commit; other remote models require this explicitly.",
            },
            "device": {"type": "string", "default": "cpu"},
            "cache_dir": {"type": "string", "default": ""},
            "top_k": {"type": "integer", "default": 1, "minimum": 1},
        }
    )

    def runtime_availability(self, params: Mapping[str, Any]) -> JsonDict:
        return _text_generator_availability(str(params.get("generator") or "masked_lm"))

    def run(self, ctx: OperationContext) -> OperationResult:
        generator = str(ctx.params.get("generator") or "masked_lm")
        if generator not in {"local_template", "masked_lm"}:
            raise OperationError(
                "Text mask-repair generator `%s` is unsupported; use "
                "local_template or masked_lm." % generator
            )
        start = time.perf_counter()
        batch = _load_text_batch(ctx.require_input("texts").path)
        examples_in = _examples(batch)
        model_id = str(ctx.params.get("model_id") or DEFAULT_MASKED_LM_MODEL_ID)
        model_revision = ""
        top_k = max(1, int(ctx.params.get("top_k") or 1))
        device = str(ctx.params.get("device") or "cpu")
        fill_mask = None
        cache_dir = ""
        if generator == "masked_lm":
            model_revision = _resolved_remote_model_revision(
                model_id,
                ctx.params.get("model_revision"),
                default_model_id=DEFAULT_MASKED_LM_MODEL_ID,
                default_revision=DEFAULT_MASKED_LM_MODEL_REVISION,
                label="Masked language model",
            )
            cache_dir = str(ctx.params.get("cache_dir") or "").strip()
            if not cache_dir:
                cache_dir = str(ctx.run_dir.parent.parent / "hf_cache")
            fill_mask = _load_fill_mask_pipeline(
                model_id, model_revision, device, cache_dir
            )
        examples: List[JsonDict] = []
        total = max(len(examples_in), 1)
        for index, item in enumerate(examples_in):
            ctx.report_progress(
                "Repairing text %d/%d" % (index, total),
                phase="text_mask_repair",
                status="running",
                completed=index,
                total=total,
                percent=float(index) / float(total) * 100.0,
                unit="texts",
            )
            text = str(item.get("text") or "")
            mask_fills: List[JsonDict] = []
            if generator == "masked_lm":
                text, mask_fills = _masked_lm_render(fill_mask, text, top_k, "")
            example = {"id": str(item.get("id") or index), "text": text}
            if mask_fills:
                example["mask_fills"] = mask_fills
            examples.append(example)
        ctx.report_progress(
            "Repairing text %d/%d" % (len(examples_in), total),
            phase="text_mask_repair",
            status="running",
            completed=len(examples_in),
            total=total,
            percent=100.0,
            unit="texts",
        )
        elapsed = time.perf_counter() - start
        payload = {
            "schema_version": 1,
            "kind": "text.batch",
            "dataset": batch.get("dataset") or "text_mask_repair",
            "split": batch.get("split") or "repaired",
            "examples": examples,
        }
        metadata = {
            "generator": generator,
            "model_id": model_id if generator == "masked_lm" else "",
            "model_revision": model_revision,
            "cache_dir": cache_dir if generator == "masked_lm" else "",
            "text_count": len(examples),
            "remaining_mask_count": sum(_mask_token_count(str(item.get("text") or "")) for item in examples),
            "texts_preview": [
                {
                    "id": item["id"],
                    "text": item["text"],
                    **({"mask_fills": item["mask_fills"]} if item.get("mask_fills") else {}),
                }
                for item in examples
            ],
            "semantic_form": "text",
            "receiver_repair": generator,
        }
        path = ctx.output_path("texts", ".json")
        write_json(path, payload)
        timing_records: List[JsonDict] = []
        append_measurement(timing_records, None, "decoder.semantic_repair", elapsed, text_count=len(examples))
        append_measurement(timing_records, None, "decoder.total", elapsed, text_count=len(examples))
        return OperationResult(
            outputs={"texts": artifact("text.batch.json", path, metadata)},
            metrics={
                "text.repaired_count": len(examples),
                "text.mask_repair_applied": 1 if generator == "masked_lm" else 0,
                "text.remaining_mask_count": metadata["remaining_mask_count"],
            },
            metadata={
                **metadata,
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="local_python",
                    notes={
                        "runtime_execution_language": "Python",
                        "foundation_adapter": f"{generator} text repair",
                    },
                ),
            },
        )


def _text_generator_availability(
    generator: str,
    *,
    allow_kb_reconstruct: bool = False,
) -> JsonDict:
    supported = {"local_template", "masked_lm"}
    if allow_kb_reconstruct:
        supported.add("kb_reconstruct")
    if generator not in supported:
        return {
            "available": False,
            "missing": [],
            "reason": (
                "Unsupported text generator `%s`; use %s."
                % (generator, " or ".join(sorted(supported)))
            ),
        }
    if generator != "masked_lm":
        return {"available": True, "extra": "", "missing": []}
    if importlib.util.find_spec("transformers") is not None:
        return {"available": True, "extra": "textgen", "missing": []}
    return {
        "available": False,
        "extra": "textgen",
        "missing": ["transformers"],
        "reason": (
            'Install with `python -m pip install "noema-lab[textgen]"` in an '
            "installed environment, or `uv sync --extra textgen` in a source "
            "checkout, to use masked-language-model text repair"
        ),
    }


def _clip_embedding_availability(backend: str) -> JsonDict:
    if str(backend or "").strip().lower() != "transformers_clip":
        return {"available": True, "extra": "", "missing": []}
    return _foundation_module_availability(
        ("torch", "transformers"),
        purpose="Transformers CLIP embeddings",
    )


def _foundation_module_availability(
    modules: Sequence[str],
    *,
    purpose: str,
) -> JsonDict:
    missing = []
    for module in modules:
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            missing.append(module)
    if not missing:
        return {"available": True, "extra": "foundation", "missing": []}
    return {
        "available": False,
        "extra": "foundation",
        "missing": missing,
        "reason": (
            'Install with `python -m pip install "noema-lab[foundation]"` in an '
            "installed environment, or `uv sync --extra foundation` in a source "
            "checkout, to use %s (missing: %s)"
            % (purpose, ", ".join(missing))
        ),
    }


def _load_fill_mask_pipeline(
    model_id: str, model_revision: str, device: str, cache_dir: str
):
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
    except Exception as exc:
        raise OperationError(
            'Install with `python -m pip install "noema-lab[textgen]"` in an '
            "installed environment, or `uv sync --extra textgen` in a source "
            "checkout, to use the masked_lm SemanticState receiver"
        ) from exc
    device_index = -1
    if device.startswith("cuda"):
        parts = device.split(":", 1)
        device_index = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
    try:
        loader_kwargs: JsonDict = {"cache_dir": cache_dir}
        if model_revision != "local_path":
            loader_kwargs["revision"] = model_revision
        tokenizer = AutoTokenizer.from_pretrained(model_id, **loader_kwargs)
        model = AutoModelForMaskedLM.from_pretrained(model_id, **loader_kwargs)
        return pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device_index)
    except Exception as exc:
        raise OperationError("Could not load masked language model `%s`: %s" % (model_id, exc)) from exc


def _masked_lm_text(fill_mask, text: str, top_k: int) -> str:
    rendered, _fills = _masked_lm_render(fill_mask, text, top_k, "")
    return rendered


def _masked_lm_render(fill_mask, text: str, top_k: int, canonical_text: str) -> Tuple[str, List[JsonDict]]:
    if not text:
        return "", []
    mask_token = getattr(getattr(fill_mask, "tokenizer", None), "mask_token", None) or "[MASK]"
    masked_text = _canonicalize_mask_tokens(text)
    normalized = masked_text.replace("[MASK]", mask_token)
    if mask_token not in normalized:
        return text, []
    expected_tokens = _expected_mask_tokens(masked_text, canonical_text)
    fills: List[JsonDict] = []
    for fill_index in range(32):
        if mask_token not in normalized:
            break
        prompt = _single_mask_prompt(
            normalized,
            mask_token,
            getattr(getattr(fill_mask, "tokenizer", None), "unk_token", None) or "something",
        )
        result = fill_mask(prompt, top_k=top_k)
        candidate = _first_fill_mask_candidate(result)
        if not isinstance(candidate, Mapping):
            break
        token = _candidate_fill_token(candidate)
        if token:
            normalized = _replace_first(normalized, mask_token, token)
        else:
            break
        expected = expected_tokens[fill_index] if fill_index < len(expected_tokens) else ""
        fills.append(
            {
                "fill_index": fill_index,
                "token": token,
                "expected": expected,
                "score": float(candidate.get("score") or 0.0),
                "output_token_index": -1,
                "correct": None,
            }
        )
    rendered = normalized.replace(mask_token, "[MASK]")
    output_tokens = tokenize(rendered)
    mask_positions = _mask_word_positions(masked_text)
    for index, fill in enumerate(fills):
        position = mask_positions[index] if index < len(mask_positions) else -1
        if position >= len(output_tokens):
            position = -1
        fill["output_token_index"] = position
        expected = str(fill.get("expected") or "")
        actual = output_tokens[position] if position >= 0 else _first_token(str(fill.get("token") or ""))
        if expected:
            fill["correct"] = actual == expected
    return rendered, fills


def _single_mask_prompt(text: str, mask_token: str, placeholder: str) -> str:
    parts = text.split(mask_token)
    if len(parts) <= 2:
        return text
    return parts[0] + mask_token + placeholder.join(parts[1:])


def _canonicalize_mask_tokens(text: str) -> str:
    return MASK_TOKEN_RE.sub("[MASK]", text)


def _mask_token_count(text: str) -> int:
    return len(MASK_TOKEN_RE.findall(text))


def _first_fill_mask_candidate(result: Any) -> Any:
    current = result
    while isinstance(current, list) and current:
        current = current[0]
    return current


def _candidate_fill_token(candidate: Mapping[str, Any]) -> str:
    token = str(candidate.get("token_str") or "").strip()
    if token.startswith("##"):
        token = token[2:]
    return token


def _replace_first(text: str, needle: str, replacement: str) -> str:
    index = text.find(needle)
    if index < 0:
        return text
    return text[:index] + replacement + text[index + len(needle) :]


def _expected_mask_tokens(masked_text: str, canonical_text: str) -> List[str]:
    if not canonical_text:
        return []
    masked_tokens = tokenize(masked_text)
    canonical_tokens = tokenize(canonical_text)
    if len(masked_tokens) != len(canonical_tokens):
        return []
    return [
        canonical_tokens[index]
        for index, token in enumerate(masked_tokens)
        if token == "mask"
    ]


def _mask_word_positions(masked_text: str) -> List[int]:
    return [index for index, token in enumerate(tokenize(masked_text)) if token == "mask"]


def _first_token(text: str) -> str:
    tokens = tokenize(text)
    return tokens[0] if tokens else ""


class SemanticStateFaithfulnessMetricsOperation(Operation):
    id = "metrics.semantic_state_faithfulness"
    name = "SemanticState and KB faithfulness metrics"
    input_kinds = {
        "reference": ["semantic.state.json"],
        "candidate": ["semantic.state.json"],
        "kb": ["foundation.kb.json"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        reference = load_semantic_state(ctx.require_input("reference").path)
        candidate = load_semantic_state(ctx.require_input("candidate").path)
        kb = load_knowledge_base(ctx.require_input("kb").path)
        ref_by_id = state_by_id(reference)
        cand_by_id = state_by_id(candidate)
        if set(ref_by_id) != set(cand_by_id):
            raise OperationError(
                "SemanticState sample ID mismatch: reference IDs %s, candidate IDs %s"
                % (sorted(ref_by_id), sorted(cand_by_id))
            )
        ids = sorted(ref_by_id)
        if not ids:
            raise OperationError("No SemanticState items were available for faithfulness metrics")
        rows = []
        concept_scores = []
        entity_scores = []
        fact_scores = []
        unsupported_assertion_scores = []
        omission_scores = []
        kb_scores = []
        kb_facts = kb_fact_set(kb)
        for item_id in ids:
            ref_item = ref_by_id.get(item_id) or {}
            cand_item = cand_by_id.get(item_id) or {}
            concepts = overlap_scores(concept_set(ref_item), concept_set(cand_item))
            entities = overlap_scores(entity_set(ref_item), entity_set(cand_item))
            facts = overlap_scores(fact_set(ref_item), fact_set(cand_item))
            reference_facts = fact_set(ref_item)
            candidate_facts = fact_set(cand_item)
            unsupported_count = len(candidate_facts - reference_facts)
            omitted_count = len(reference_facts - candidate_facts)
            unsupported_assertion_rate = (
                float(unsupported_count) / float(len(candidate_facts))
                if candidate_facts
                else 0.0
            )
            omission_rate = (
                float(omitted_count) / float(len(reference_facts))
                if reference_facts
                else 0.0
            )
            grounded = len(candidate_facts & kb_facts)
            kb_precision = float(grounded) / float(len(candidate_facts)) if candidate_facts else 1.0
            concept_scores.append(concepts)
            entity_scores.append(entities)
            fact_scores.append(facts)
            unsupported_assertion_scores.append(unsupported_assertion_rate)
            omission_scores.append(omission_rate)
            kb_scores.append(kb_precision)
            rows.append(
                {
                    "id": item_id,
                    "concept": concepts,
                    "entity": entities,
                    "fact": facts,
                    "unsupported_assertion_rate": unsupported_assertion_rate,
                    "fact_omission_rate": omission_rate,
                    "unsupported_candidate_fact_count": unsupported_count,
                    "omitted_reference_fact_count": omitted_count,
                    "kb_fact_precision": kb_precision,
                    "candidate_fact_count": len(candidate_facts),
                    "grounded_candidate_fact_count": grounded,
                }
            )
        metrics = {
            "faithfulness.concept_precision": _mean(score["precision"] for score in concept_scores),
            "faithfulness.concept_recall": _mean(score["recall"] for score in concept_scores),
            "faithfulness.concept_f1": _mean(score["f1"] for score in concept_scores),
            "faithfulness.entity_precision": _mean(score["precision"] for score in entity_scores),
            "faithfulness.entity_recall": _mean(score["recall"] for score in entity_scores),
            "faithfulness.entity_f1": _mean(score["f1"] for score in entity_scores),
            "faithfulness.fact_precision": _mean(score["precision"] for score in fact_scores),
            "faithfulness.fact_recall": _mean(score["recall"] for score in fact_scores),
            "faithfulness.fact_f1": _mean(score["f1"] for score in fact_scores),
            "faithfulness.unsupported_assertion_rate": _mean(unsupported_assertion_scores),
            "faithfulness.fact_omission_rate": _mean(omission_scores),
            "faithfulness.kb_fact_precision": _mean(kb_scores),
        }
        report = {
            "schema_version": 1,
            "metric_family": "semantic_state_faithfulness",
            "num_states": len(ids),
            "knowledge_base": kb.kb_id,
            "metrics": metrics,
            "per_example": rows,
            "metadata": {
                "construct_boundary": (
                    "Fact-set agreement against the supplied reference SemanticState; this does "
                    "not establish real-world hallucination or factual truth."
                ),
                "unsupported_assertion_definition": (
                    "candidate facts absent from the reference divided by candidate fact count"
                ),
                "omission_definition": (
                    "reference facts absent from the candidate divided by reference fact count"
                ),
                "empty_set_policy": {
                    "empty_candidate_unsupported_assertion_rate": 0.0,
                    "empty_reference_fact_omission_rate": 0.0,
                    "both_empty_fact_precision_recall_f1": 1.0,
                },
                "sample_pairing": "exact unique SemanticState IDs",
            },
        }
        path = ctx.output_path("report", ".json")
        write_json(path, report)
        return OperationResult(
            outputs={"report": artifact("metrics.report", path, report)},
            metrics=metrics,
            metadata={
                "num_states": len(ids),
                "kb_id": kb.kb_id,
                "construct": "reference-relative fact-set support and omission",
                "empty_set_policy": report["metadata"]["empty_set_policy"],
            },
        )


class ClipTextEmbeddingOperation(Operation):
    id = "foundation.clip_text_embed"
    name = "CLIP-style text embedding adapter"
    input_kinds = {"texts": ["text.batch.json"]}
    output_kinds = {"embeddings": "foundation.embedding.numpy"}
    params_schema = object_schema(
        {
            "backend": {
                "type": "string",
                "default": "local_semantic",
                "enum": ["local_semantic", "local_hash", "transformers_clip"],
            },
            "model_id": {"type": "string", "default": DEFAULT_CLIP_MODEL_ID},
            "model_revision": {
                "type": "string",
                "default": "",
                "pattern": "^(|[0-9a-fA-F]{40})$",
                "description": "Full immutable Hugging Face commit. The built-in default model resolves to its pinned commit; other remote models require this explicitly.",
            },
            "dimensions": {"type": "integer", "default": 128, "minimum": 8},
            "device": {"type": "string", "default": "cpu"},
        }
    )

    def runtime_availability(self, params: Mapping[str, Any]) -> JsonDict:
        return _clip_embedding_availability(
            str(params.get("backend") or "local_semantic")
        )

    def run(self, ctx: OperationContext) -> OperationResult:
        backend = str(ctx.params.get("backend") or "local_semantic")
        batch = _load_text_batch(ctx.require_input("texts").path)
        examples = _examples(batch)
        texts = [str(item.get("text") or "") for item in examples]
        sample_ids = [str(item.get("id") or "text_%03d" % (index + 1)) for index, item in enumerate(examples)]
        measurements: List[JsonDict] = []
        if backend == "local_semantic":
            start = time.perf_counter()
            embeddings = np.stack([_retrieval_text_embedding(text) for text in texts], axis=0).astype(np.float32)
            append_measurement(measurements, None, "encoder.inference", time.perf_counter() - start, example_count=len(texts))
            runner = "local_python"
            model_id = "local_semantic"
        elif backend == "local_hash":
            dimensions = max(8, int(ctx.params.get("dimensions") or 128))
            start = time.perf_counter()
            embeddings = np.stack([_hash_embedding(text, dimensions) for text in texts], axis=0).astype(np.float32)
            append_measurement(measurements, None, "encoder.inference", time.perf_counter() - start, example_count=len(texts))
            runner = "local_python"
            model_id = "local_hash"
        elif backend == "transformers_clip":
            model_id = str(ctx.params.get("model_id") or DEFAULT_CLIP_MODEL_ID)
            model_revision = _embedding_model_revision(
                backend, model_id, ctx.params.get("model_revision")
            )
            embeddings, runner, load_s, infer_s = _clip_text_embeddings_transformers(
                texts, model_id, model_revision, ctx
            )
            append_measurement(measurements, None, "encoder.setup", load_s, example_count=len(texts))
            append_measurement(measurements, None, "encoder.inference", infer_s, example_count=len(texts))
        else:
            raise OperationError("Unsupported CLIP text backend: %s" % backend)
        if backend != "transformers_clip":
            model_revision = _embedding_model_revision(
                backend, model_id, ctx.params.get("model_revision")
            )
        embedding_space = _embedding_space_metadata(
            backend, model_id, model_revision, int(embeddings.shape[1])
        )
        metadata = {
            "adapter_family": "clip_text",
            "backend": backend,
            "model_id": model_id,
            "model_revision": model_revision,
            "dimensions": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "count": len(examples),
            "sample_ids": sample_ids,
            "embedding_modality": "text",
            "embedding_space": embedding_space,
            "codec_timing": codec_timing_metadata(
                "encoder",
                measurements,
                runner=runner,
                notes={"runtime_execution_language": "Python", "foundation_adapter": "CLIP text embedding"},
            ),
        }
        path = ctx.output_path("embeddings", ".npz")
        np.savez_compressed(path, embeddings=embeddings, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"embeddings": artifact("foundation.embedding.numpy", path, metadata)},
            metrics={"foundation.embedding.count": len(examples), "foundation.embedding.dimensions": metadata["dimensions"]},
            metadata=metadata,
        )


class ClipImageEmbeddingOperation(Operation):
    id = "foundation.clip_image_embed"
    name = "CLIP-style image embedding adapter"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"embeddings": "foundation.embedding.numpy"}
    params_schema = object_schema(
        {
            "backend": {
                "type": "string",
                "default": "local_semantic",
                "enum": ["local_semantic", "local_color_histogram", "transformers_clip"],
            },
            "model_id": {"type": "string", "default": DEFAULT_CLIP_MODEL_ID},
            "model_revision": {
                "type": "string",
                "default": "",
                "pattern": "^(|[0-9a-fA-F]{40})$",
                "description": "Full immutable Hugging Face commit. The built-in default model resolves to its pinned commit; other remote models require this explicitly.",
            },
            "bins": {"type": "integer", "default": 16, "minimum": 2},
            "dimensions": {"type": "integer", "default": 8, "minimum": 8},
            "device": {"type": "string", "default": "cpu"},
        }
    )

    def runtime_availability(self, params: Mapping[str, Any]) -> JsonDict:
        return _clip_embedding_availability(
            str(params.get("backend") or "local_semantic")
        )

    def run(self, ctx: OperationContext) -> OperationResult:
        backend = str(ctx.params.get("backend") or "local_semantic")
        images, image_metadata = _load_images(ctx.require_input("images").path)
        sample_ids = [str(item) for item in image_metadata.get("sample_ids", [])] if isinstance(image_metadata.get("sample_ids"), list) else []
        if not sample_ids:
            sample_ids = ["image_%03d" % (index + 1) for index in range(int(images.shape[0]))]
        measurements: List[JsonDict] = []
        if backend == "local_semantic":
            start = time.perf_counter()
            embeddings = np.stack([_retrieval_image_embedding(image) for image in images], axis=0).astype(np.float32)
            append_measurement(measurements, None, "encoder.inference", time.perf_counter() - start, example_count=int(images.shape[0]))
            runner = "local_python"
            model_id = "local_semantic"
            bins_value = 0
        elif backend == "local_color_histogram":
            bins = max(2, int(ctx.params.get("bins") or 16))
            start = time.perf_counter()
            embeddings = np.stack([_color_histogram_embedding(image, bins) for image in images], axis=0).astype(np.float32)
            append_measurement(measurements, None, "encoder.inference", time.perf_counter() - start, example_count=int(images.shape[0]))
            runner = "local_python"
            model_id = "local_color_histogram"
            bins_value = bins
        elif backend == "transformers_clip":
            model_id = str(ctx.params.get("model_id") or DEFAULT_CLIP_MODEL_ID)
            model_revision = _embedding_model_revision(
                backend, model_id, ctx.params.get("model_revision")
            )
            embeddings, runner, load_s, infer_s = _clip_image_embeddings_transformers(
                images, model_id, model_revision, ctx
            )
            append_measurement(measurements, None, "encoder.setup", load_s, example_count=int(images.shape[0]))
            append_measurement(measurements, None, "encoder.inference", infer_s, example_count=int(images.shape[0]))
            bins_value = 0
        else:
            raise OperationError("Unsupported CLIP image backend: %s" % backend)
        if backend != "transformers_clip":
            model_revision = _embedding_model_revision(
                backend, model_id, ctx.params.get("model_revision")
            )
        embedding_space = _embedding_space_metadata(
            backend, model_id, model_revision, int(embeddings.shape[1])
        )
        metadata = {
            "adapter_family": "clip_image",
            "backend": backend,
            "model_id": model_id,
            "model_revision": model_revision,
            "bins": bins_value,
            "dimensions": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "count": int(embeddings.shape[0]) if embeddings.ndim == 2 else 0,
            "sample_ids": sample_ids,
            "embedding_modality": "image",
            "embedding_space": embedding_space,
            "source_image_metadata": image_metadata,
            "codec_timing": codec_timing_metadata(
                "encoder",
                measurements,
                runner=runner,
                notes={"runtime_execution_language": "Python", "foundation_adapter": "CLIP image embedding"},
            ),
        }
        path = ctx.output_path("embeddings", ".npz")
        np.savez_compressed(path, embeddings=embeddings, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"embeddings": artifact("foundation.embedding.numpy", path, metadata)},
            metrics={"foundation.embedding.count": metadata["count"], "foundation.embedding.dimensions": metadata["dimensions"]},
            metadata=metadata,
        )


class SamImageSegmentsOperation(Operation):
    id = "foundation.sam_segment"
    name = "SAM-style image segmentation adapter"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"state": "semantic.state.json"}
    params_schema = object_schema(
        {
            "backend": {
                "type": "string",
                "default": "local_grid",
                "enum": ["local_grid"],
            },
            "grid_size": {"type": "integer", "default": 2, "minimum": 1},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        backend = str(ctx.params.get("backend") or "local_grid")
        if backend != "local_grid":
            raise OperationError(
                "SAM-style segmentation backend `%s` is unsupported; use local_grid."
                % backend
            )
        images, metadata = _load_images(ctx.require_input("images").path)
        grid_size = max(1, int(ctx.params.get("grid_size") or 2))
        states = [_image_segment_state(image, index, grid_size, metadata) for index, image in enumerate(images)]
        payload = SemanticState(modality="image", states=states, state_id="image_segments").to_dict()
        path = ctx.output_path("state", ".json")
        write_json(path, payload)
        out_metadata = {"adapter_family": "sam", "backend": backend, "segment_count": sum(len(item.get("segments") or []) for item in states)}
        return OperationResult(
            outputs={"state": artifact("semantic.state.json", path, out_metadata)},
            metrics={"foundation.segment_count": out_metadata["segment_count"]},
            metadata=out_metadata,
        )


class VlmImageSemanticStateOperation(Operation):
    id = "foundation.vlm_image_to_state"
    name = "VLM-style image SemanticState adapter"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"state": "semantic.state.json"}
    params_schema = object_schema(
        {
            "backend": {
                "type": "string",
                "default": "local_image_stats",
                "enum": ["local_image_stats"],
            },
            "model_id": {"type": "string", "default": ""},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        backend = str(ctx.params.get("backend") or "local_image_stats")
        if backend != "local_image_stats":
            raise OperationError(
                "VLM-style image-state backend `%s` is unsupported; use "
                "local_image_stats." % backend
            )
        images, metadata = _load_images(ctx.require_input("images").path)
        states = [_image_stats_state(image, index, metadata) for index, image in enumerate(images)]
        payload = SemanticState(modality="image", states=states, state_id="image_semantic_state").to_dict()
        path = ctx.output_path("state", ".json")
        write_json(path, payload)
        out_metadata = {"adapter_family": "vlm", "backend": backend, "semantic_state_count": len(states)}
        return OperationResult(
            outputs={"state": artifact("semantic.state.json", path, out_metadata)},
            metrics={"semantic_state.count": len(states)},
            metadata=out_metadata,
        )


class DiffusionSemanticStateToImageOperation(Operation):
    id = "foundation.diffusion_state_to_image"
    name = "Diffusion SemanticState-to-image adapter"
    input_kinds = {"state": ["semantic.state.json"]}
    output_kinds = {"images": "image.batch.numpy"}
    params_schema = object_schema(
        {
            "backend": {
                "type": "string",
                "default": "diffusers",
                "enum": ["diffusers"],
            },
            "model_id": {"type": "string", "default": DEFAULT_DIFFUSION_MODEL_ID},
            "model_revision": {
                "type": "string",
                "default": "",
                "pattern": "^(|[0-9a-fA-F]{40})$",
                "description": "Full immutable Hugging Face commit. The built-in default model resolves to its pinned commit; other remote models require this explicitly.",
            },
            "device": {"type": "string", "default": "cpu"},
            "height": {"type": "integer", "default": 512, "minimum": 64},
            "width": {"type": "integer", "default": 512, "minimum": 64},
            "num_inference_steps": {"type": "integer", "default": 20, "minimum": 1},
            "guidance_scale": {"type": "number", "default": 7.5, "minimum": 0},
            "prompt_template": {"type": "string", "default": "A realistic photograph: {prompt}"},
            "negative_prompt": {"type": "string", "default": "low quality, blurry, abstract, distorted, text, watermark"},
            "disable_safety_checker": {"type": "boolean", "default": False},
            "use_safetensors": {"type": ["boolean", "null"], "default": None},
        }
    )

    def runtime_availability(self, params: Mapping[str, Any]) -> JsonDict:
        return _foundation_module_availability(
            ("torch", "transformers", "diffusers"),
            purpose="Diffusers image generation",
        )

    def run(self, ctx: OperationContext) -> OperationResult:
        model_id = str(ctx.params.get("model_id") or DEFAULT_DIFFUSION_MODEL_ID)
        model_revision = _resolved_remote_model_revision(
            model_id,
            ctx.params.get("model_revision"),
            default_model_id=DEFAULT_DIFFUSION_MODEL_ID,
            default_revision=DEFAULT_DIFFUSION_MODEL_REVISION,
            label="Diffusion",
        )
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except Exception as exc:
            raise OperationError(
                'Install with `python -m pip install "noema-lab[foundation]"` in an '
                "installed environment, or `uv sync --extra foundation` in a source "
                "checkout, to use diffusion adapters"
            ) from exc
        state = load_semantic_state(ctx.require_input("state").path)
        raw_prompts = [_prompt_from_state_item(item) for item in state.states]
        if not raw_prompts:
            raise OperationError("Diffusion adapter requires at least one SemanticState item")
        device = str(ctx.params.get("device") or "cpu")
        height = max(64, int(ctx.params.get("height") or 512))
        width = max(64, int(ctx.params.get("width") or 512))
        steps = max(1, int(ctx.params.get("num_inference_steps") or 20))
        guidance = float(ctx.params.get("guidance_scale", 7.5) or 0.0)
        prompt_template = str(ctx.params.get("prompt_template") or "{prompt}").strip() or "{prompt}"
        negative_prompt = str(ctx.params.get("negative_prompt") or "").strip()
        prompts = [_format_diffusion_prompt(prompt, prompt_template) for prompt in raw_prompts]
        disable_safety_checker = bool(ctx.params.get("disable_safety_checker", False))
        use_safetensors = ctx.params.get("use_safetensors")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        timing_records: List[JsonDict] = []
        try:
            setup_start = time.perf_counter()
            load_kwargs: JsonDict = {"torch_dtype": dtype}
            if model_revision != "local_path":
                load_kwargs["revision"] = model_revision
            if disable_safety_checker:
                load_kwargs.update({"safety_checker": None, "requires_safety_checker": False})
            if use_safetensors is not None:
                load_kwargs["use_safetensors"] = bool(use_safetensors)
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, **load_kwargs)
            pipe = pipe.to(device)
            append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start, example_count=len(prompts), model_id=model_id)
            generator = torch.Generator(device=device if device.startswith("cuda") else "cpu").manual_seed(ctx.seed("diffusion"))
            images = []
            total = max(len(prompts), 1)
            for index, prompt in enumerate(prompts):
                ctx.report_progress(
                    "Generating image %d/%d" % (index, total),
                    phase="diffusion",
                    status="running",
                    completed=index,
                    total=total,
                    percent=float(index) / float(total) * 100.0,
                    unit="images",
                )
                infer_start = time.perf_counter()
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                )
                append_measurement(timing_records, index, "decoder.inference", time.perf_counter() - infer_start, example_count=1, model_id=model_id)
                images.append(np.asarray(result.images[0].convert("RGB"), dtype=np.uint8))
            ctx.report_progress("Generating image %d/%d" % (len(prompts), total), phase="diffusion", status="running", completed=len(prompts), total=total, percent=100.0, unit="images")
        except Exception as exc:
            raise OperationError("Diffusion generation failed for model `%s`: %s" % (model_id, exc)) from exc
        batch = np.stack(images, axis=0)
        metadata = {
            "adapter_family": "diffusion",
            "backend": "diffusers",
            "model_id": model_id,
            "model_revision": model_revision,
            "device": device,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "prompt_template": prompt_template,
            "negative_prompt": negative_prompt,
            "disable_safety_checker": disable_safety_checker,
            "use_safetensors": use_safetensors,
            "image_count": int(batch.shape[0]),
            "shape": list(batch.shape),
            "dtype": str(batch.dtype),
            "raw_prompts_preview": raw_prompts[:5],
            "prompts_preview": prompts[:5],
            "codec_timing": codec_timing_metadata(
                "decoder",
                timing_records,
                runner="pytorch",
                notes={
                    "runtime_execution_language": "Python",
                    "foundation_adapter": "Diffusers text-to-image receiver",
                    "model_family": "diffusion",
                    "model_revision": model_revision,
                },
            ),
        }
        path = ctx.output_path("images", ".npz")
        np.savez_compressed(path, images=batch, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"images": artifact("image.batch.numpy", path, metadata)},
            metrics={"foundation.diffusion.image_count": int(batch.shape[0])},
            metadata=metadata,
        )


def _builtin_text_facts(max_concepts: int) -> List[JsonDict]:
    facts: List[JsonDict] = []
    for example in TEXT_SMOKE_EXAMPLES:
        sample_id = str(example["id"])
        concepts = concepts_from_text(str(example["text"]), max_concepts=max_concepts)
        facts.append({"id": "%s:type" % sample_id, "subject": sample_id, "predicate": "is_a", "object": "text_sample", "source": "semantic_text_smoke"})
        facts.append({"id": "%s:canonical_text" % sample_id, "subject": sample_id, "predicate": "canonical_text", "object": str(example["text"]), "source": "semantic_text_smoke"})
        for concept in concepts:
            facts.append({"id": "%s:mentions:%s" % (sample_id, concept), "subject": "semantic_item", "predicate": "mentions", "object": concept, "source": "semantic_text_smoke"})
    return facts


def _normalize_fact(item: Mapping[str, Any], source: str) -> JsonDict:
    subject = str(item.get("subject") or item.get("s") or "")
    predicate = str(item.get("predicate") or item.get("p") or "related_to")
    obj = str(item.get("object") or item.get("o") or "")
    return {
        "id": str(item.get("id") or "%s:%s:%s" % (subject, predicate, obj)),
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "source": str(item.get("source") or source),
        "confidence": float(item.get("confidence", 1.0) or 1.0),
    }


def _load_text_batch(path) -> JsonDict:
    data = load_json(path)
    if not isinstance(data, dict) or data.get("kind") != "text.batch":
        raise OperationError("Expected text.batch JSON artifact: %s" % path)
    _examples(data)
    return data


def _examples(batch: JsonDict) -> List[JsonDict]:
    examples = batch.get("examples")
    if not isinstance(examples, list):
        raise OperationError("text.batch artifact requires an examples list")
    normalized: List[JsonDict] = []
    for index, item in enumerate(examples):
        if not isinstance(item, Mapping):
            raise OperationError(
                "text.batch examples item %d must be an object" % index
            )
        normalized.append(dict(item))
    return normalized


def _semantic_state_from_text(example: Mapping[str, Any], max_concepts: int, include_text: bool) -> JsonDict:
    text = str(example.get("text") or "")
    item_id = str(example.get("id") or "text")
    concepts = concepts_from_text(text, max_concepts=max_concepts)
    entities = [{"id": concept, "text": concept, "type": "concept"} for concept in concepts[: min(8, len(concepts))]]
    facts = [
        {"id": "mentions:%s" % concept, "subject": "semantic_item", "predicate": "mentions", "object": concept, "confidence": 1.0}
        for concept in concepts
    ]
    payload: JsonDict = {
        "id": item_id,
        "concepts": concepts,
        "entities": entities,
        "facts": facts,
        "grounded": False,
    }
    if include_text:
        payload["source_text"] = text
    return payload


def _state_tokens(item: Mapping[str, Any]) -> set:
    values: List[str] = []
    values.extend(str(value) for value in item.get("concepts") or [])
    values.extend(
        str(value.get("text") or value.get("name") or "")
        for value in item.get("entities") or []
        if isinstance(value, Mapping)
    )
    values.extend(
        "%s %s" % (fact.get("predicate", ""), fact.get("object", ""))
        for fact in item.get("facts") or []
        if isinstance(fact, Mapping)
    )
    return {token for token in tokenize(" ".join(values)) if token}


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _semantic_payload_item_bit_counts(
    payload_bytes: bytes, states: Sequence[Mapping[str, Any]]
) -> Tuple[List[int], int]:
    item_bytes = [_canonical_json_bytes(item) for item in states]
    container_byte_count = len(payload_bytes) - sum(
        len(value) for value in item_bytes
    )
    if container_byte_count < 0:
        raise OperationError(
            "SemanticState payload accounting produced a negative JSON container size"
        )
    if not item_bytes:
        return [], container_byte_count * 8

    marker = b'"states":['
    marker_offset = payload_bytes.find(marker)
    if marker_offset < 0:
        raise OperationError(
            "SemanticState canonical payload is missing its states array"
        )
    cursor = marker_offset + len(marker)
    item_starts: List[int] = []
    for index, encoded_item in enumerate(item_bytes):
        if index:
            if payload_bytes[cursor : cursor + 1] != b",":
                raise OperationError(
                    "SemanticState canonical payload item separators are inconsistent"
                )
            cursor += 1
        item_starts.append(cursor)
        if payload_bytes[cursor : cursor + len(encoded_item)] != encoded_item:
            raise OperationError(
                "SemanticState canonical payload item %d is not byte-aligned" % index
            )
        cursor += len(encoded_item)

    segment_starts = [0, *item_starts[1:]]
    segment_ends = [*item_starts[1:], len(payload_bytes)]
    counts = [
        (end - start) * 8
        for start, end in zip(segment_starts, segment_ends)
    ]
    if any(value <= 0 for value in counts) or sum(counts) != len(payload_bytes) * 8:
        raise OperationError(
            "SemanticState source-item payload slices do not cover the payload"
        )
    return counts, container_byte_count * 8


def _semantic_state_transport_payload(
    payload: Mapping[str, Any],
) -> Tuple[JsonDict, int]:
    raw_states = payload.get("states") if isinstance(payload, Mapping) else None
    if not isinstance(raw_states, list) or any(
        not isinstance(item, Mapping) for item in raw_states
    ):
        raise OperationError(
            "SemanticState transport requires a states list of objects"
        )
    try:
        state = SemanticState.from_dict(payload)
    except Exception as exc:
        raise OperationError("SemanticState payload encode requires valid semantic.state JSON: %s" % exc) from exc

    redacted_count = 0
    forbidden_keys = {
        "source_id",
        "source_text",
        "reference_text",
        "target_text",
        "canonical_text",
        "kb_matches",
    }

    def sanitize(value: Any) -> Any:
        nonlocal redacted_count
        if isinstance(value, Mapping):
            if str(value.get("predicate") or "") == "canonical_text":
                redacted_count += 1
                return None
            output: JsonDict = {}
            for key, nested in value.items():
                key_text = str(key)
                if key_text in forbidden_keys:
                    redacted_count += 1
                    continue
                sanitized = sanitize(nested)
                if sanitized is not None:
                    output[key_text] = sanitized
            if "grounded" in output and "kb_matches" in value:
                output["grounded"] = False
            return output
        if isinstance(value, list):
            output_list = []
            for nested in value:
                sanitized = sanitize(nested)
                if sanitized is not None:
                    output_list.append(sanitized)
            return output_list
        return value

    states = []
    for index, item in enumerate(state.states):
        sanitized = sanitize(item)
        if not isinstance(sanitized, dict):
            raise OperationError(
                "SemanticState item %d could not be represented for transport" % index
            )
        states.append(sanitized)
    return (
        SemanticState(
            modality=state.modality,
            states=states,
            state_id=state.state_id,
            schema_version=state.schema_version,
        ).to_dict(),
        redacted_count,
    )


def _load_bits(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        bits = payload["bits"]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Foundation bit artifact metadata_json",
                )
            )
    canonical, boundary_metadata = validate_channel_bits(
        bits,
        label="foundation bits",
        backend="python_numpy",
    )
    metadata.update(boundary_metadata)
    return canonical, metadata


def _mean(values: Sequence[float]) -> float:
    rows = [float(value) for value in values]
    return float(sum(rows) / len(rows)) if rows else 0.0


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _hash_embedding(text: str, dimensions: int) -> np.ndarray:
    vector = np.zeros((dimensions,), dtype=np.float32)
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "little", signed=False) % dimensions
        vector[index] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _embedding_model_revision(backend: str, model_id: str, value: Any) -> str:
    if backend == "transformers_clip":
        return _resolved_remote_model_revision(
            model_id,
            value,
            default_model_id=DEFAULT_CLIP_MODEL_ID,
            default_revision=DEFAULT_CLIP_MODEL_REVISION,
            label="Transformers CLIP",
        )
    return {
        "local_semantic": "noema_local_semantic_v1",
        "local_hash": "noema_local_hash_v1",
        "local_color_histogram": "noema_local_color_histogram_v1",
    }.get(backend, "noema_local_unknown_v1")


def _resolved_remote_model_revision(
    model_id: str,
    value: Any,
    *,
    default_model_id: str,
    default_revision: str,
    label: str,
) -> str:
    if Path(model_id).exists():
        return "local_path"
    revision = str(value or "").strip().lower()
    if not revision and model_id == default_model_id:
        revision = default_revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise OperationError(
            "%s remote model `%s` requires model_revision as a full immutable "
            "40-character Hugging Face commit" % (label, model_id)
        )
    return revision


def _embedding_space_metadata(
    backend: str, model_id: str, model_revision: str, dimensions: int
) -> JsonDict:
    family = "clip" if backend == "transformers_clip" else {
        "local_semantic": "noema_retrieval_semantic",
        "local_hash": "noema_text_hash",
        "local_color_histogram": "noema_image_color_histogram",
    }.get(backend, "unknown")
    preprocessing = (
        "transformers_clip_processor_paired_modalities_v1"
        if backend == "transformers_clip"
        else "%s_preprocessing_v1" % backend
    )
    return {
        "schema_version": 1,
        "space_family": family,
        "backend": backend,
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "preprocessing_contract": preprocessing,
        "dimensions": int(dimensions),
    }


def _load_images(path) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        images = payload["images"]
        metadata: JsonDict = {}
        if "metadata_json" in payload:
            metadata = decode_strict_json_object(
                str(payload["metadata_json"]),
                label="Foundation image artifact metadata_json",
            )
    return images, metadata


def _color_histogram_embedding(image: np.ndarray, bins: int) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype.kind in ("f", "c"):
        array = np.clip(array, 0.0, 1.0) * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    if array.ndim == 2:
        array = array[..., None]
    channels = []
    for channel in range(min(array.shape[-1], 3)):
        hist, _edges = np.histogram(array[..., channel], bins=bins, range=(0, 255), density=False)
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        channels.append(hist)
    vector = np.concatenate(channels) if channels else np.zeros((bins,), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _retrieval_text_embedding(text: str) -> np.ndarray:
    tokens = set(tokenize(text))
    vector = np.zeros((8,), dtype=np.float32)
    color_tokens = {
        "red": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "green": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "blue": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "yellow": np.array([1.0, 1.0, 0.0], dtype=np.float32),
    }
    for token, color in color_tokens.items():
        if token in tokens:
            vector[:3] += color
    shape_index = {"square": 3, "circle": 4, "triangle": 5, "bar": 6}
    for token, index in shape_index.items():
        if token in tokens:
            vector[index] = 1.0
    vector[7] = 1.0
    return _normalize_vector(vector)


def _retrieval_image_embedding(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype.kind in ("f", "c"):
        array = np.clip(array, 0.0, 1.0) * 255.0
    array = np.clip(array, 0, 255).astype(np.float32, copy=False)
    background = np.array([24.0, 28.0, 34.0], dtype=np.float32)
    distance = np.linalg.norm(array[..., :3] - background, axis=-1)
    mask = distance > 30.0
    vector = np.zeros((8,), dtype=np.float32)
    if mask.any():
        mean_color = array[..., :3][mask].mean(axis=0) / 255.0
        vector[:3] = mean_color
        ys, xs = np.nonzero(mask)
        height = max(int(ys.max() - ys.min() + 1), 1)
        width = max(int(xs.max() - xs.min() + 1), 1)
        fill_ratio = float(mask.sum()) / float(height * width)
        aspect = float(width) / float(height)
        if aspect > 2.2:
            vector[6] = 1.0
        elif 0.68 <= fill_ratio <= 0.88:
            vector[4] = 1.0
        elif fill_ratio < 0.68:
            vector[5] = 1.0
        else:
            vector[3] = 1.0
    vector[7] = 1.0
    return _normalize_vector(vector)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _clip_text_embeddings_transformers(
    texts: Sequence[str],
    model_id: str,
    model_revision: str,
    ctx: OperationContext,
) -> Tuple[np.ndarray, str, float, float]:
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:
        raise OperationError(
            'Install with `python -m pip install "noema-lab[foundation]"` in an '
            "installed environment, or `uv sync --extra foundation` in a source "
            "checkout, to use Transformers CLIP text embeddings"
        ) from exc
    device = _torch_device(str(ctx.params.get("device") or "cpu"), torch)
    setup_start = time.perf_counter()
    loader_kwargs = {} if model_revision == "local_path" else {"revision": model_revision}
    processor = CLIPProcessor.from_pretrained(model_id, **loader_kwargs)
    model = CLIPModel.from_pretrained(model_id, **loader_kwargs).to(device)
    model.eval()
    setup_s = time.perf_counter() - setup_start
    infer_start = time.perf_counter()
    with torch.no_grad():
        inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        features = _clip_feature_tensor(model.get_text_features(**inputs))
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    infer_s = time.perf_counter() - infer_start
    return features.detach().cpu().numpy().astype(np.float32, copy=False), "pytorch", setup_s, infer_s


def _clip_image_embeddings_transformers(
    images: np.ndarray,
    model_id: str,
    model_revision: str,
    ctx: OperationContext,
) -> Tuple[np.ndarray, str, float, float]:
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:
        raise OperationError(
            'Install with `python -m pip install "noema-lab[foundation]"` in an '
            "installed environment, or `uv sync --extra foundation` in a source "
            "checkout, to use Transformers CLIP image embeddings"
        ) from exc
    device = _torch_device(str(ctx.params.get("device") or "cpu"), torch)
    setup_start = time.perf_counter()
    loader_kwargs = {} if model_revision == "local_path" else {"revision": model_revision}
    processor = CLIPProcessor.from_pretrained(model_id, **loader_kwargs)
    model = CLIPModel.from_pretrained(model_id, **loader_kwargs).to(device)
    model.eval()
    setup_s = time.perf_counter() - setup_start
    pil_images = [Image.fromarray(np.asarray(image).astype(np.uint8, copy=False)).convert("RGB") for image in images]
    infer_start = time.perf_counter()
    with torch.no_grad():
        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        features = _clip_feature_tensor(model.get_image_features(**inputs))
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    infer_s = time.perf_counter() - infer_start
    return features.detach().cpu().numpy().astype(np.float32, copy=False), "pytorch", setup_s, infer_s


def _torch_device(requested: str, torch) -> str:
    value = str(requested or "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise OperationError("CUDA was requested, but no CUDA GPU is available")
    return value


def _clip_feature_tensor(output: Any) -> Any:
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if isinstance(output, (list, tuple)) and output:
        return output[0]
    return output


def _image_segment_state(image: np.ndarray, index: int, grid_size: int, metadata: JsonDict) -> JsonDict:
    array = np.asarray(image)
    height, width = int(array.shape[0]), int(array.shape[1])
    segments = []
    for row in range(grid_size):
        for col in range(grid_size):
            y0 = int(round(row * height / grid_size))
            y1 = int(round((row + 1) * height / grid_size))
            x0 = int(round(col * width / grid_size))
            x1 = int(round((col + 1) * width / grid_size))
            crop = array[y0:y1, x0:x1]
            mean_rgb = _mean_rgb(crop)
            segments.append(
                {
                    "id": "seg_%d_%d" % (row, col),
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "mean_rgb": mean_rgb,
                    "area_fraction": _safe_fraction((y1 - y0) * (x1 - x0), height * width),
                }
            )
    concepts = ["image", "segments", "%dx%d" % (width, height)]
    return {
        "id": _image_id(index, metadata),
        "concepts": concepts,
        "entities": [{"id": item["id"], "type": "image_region", "bbox_xyxy": item["bbox_xyxy"]} for item in segments],
        "facts": [{"subject": _image_id(index, metadata), "predicate": "has_segment", "object": item["id"]} for item in segments],
        "segments": segments,
    }


def _image_stats_state(image: np.ndarray, index: int, metadata: JsonDict) -> JsonDict:
    array = np.asarray(image)
    height, width = int(array.shape[0]), int(array.shape[1])
    mean_rgb = _mean_rgb(array)
    brightness = sum(mean_rgb) / max(len(mean_rgb), 1)
    aspect = "wide" if width > height else "tall" if height > width else "square"
    lightness = "bright" if brightness >= 150 else "dark" if brightness < 85 else "medium_brightness"
    image_id = _image_id(index, metadata)
    concepts = ["image", aspect, lightness, "%dx%d" % (width, height)]
    return {
        "id": image_id,
        "concepts": concepts,
        "entities": [{"id": image_id, "type": "image", "width": width, "height": height}],
        "facts": [
            {"subject": image_id, "predicate": "has_aspect", "object": aspect},
            {"subject": image_id, "predicate": "has_lightness", "object": lightness},
        ],
        "image_stats": {"width": width, "height": height, "mean_rgb": mean_rgb},
    }


def _mean_rgb(array: np.ndarray) -> List[float]:
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.dtype.kind in ("f", "c"):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    channels = min(arr.shape[-1], 3)
    if channels <= 0:
        return []
    return [float(np.mean(arr[..., channel])) for channel in range(channels)]


def _image_id(index: int, metadata: JsonDict) -> str:
    image_ids = metadata.get("image_ids") if isinstance(metadata, Mapping) else None
    if isinstance(image_ids, list) and index < len(image_ids):
        return str(image_ids[index])
    return "image_%d" % index


def _canonical_text_from_semantic_evidence(
    item: Mapping[str, Any], kb: KnowledgeBase
) -> str:
    evidence_tokens = _state_tokens(item)
    if not evidence_tokens:
        return ""
    candidates: List[Tuple[int, str, str]] = []
    for fact in kb.facts:
        if str(fact.get("predicate") or "") != "canonical_text":
            continue
        text = str(fact.get("object") or "")
        if not text:
            continue
        overlap = evidence_tokens & set(tokenize(text))
        if overlap:
            candidates.append(
                (
                    len(overlap),
                    str(fact.get("id") or ""),
                    text,
                )
            )
    if not candidates:
        return ""
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    return candidates[0][2]


def _prompt_from_state_item(item: Mapping[str, Any]) -> str:
    source = str(item.get("source_text") or "").strip()
    if source:
        return source
    concepts = [str(value) for value in item.get("concepts") or [] if str(value).strip()]
    if concepts:
        return "A faithful image representing: " + ", ".join(concepts[:20])
    return "A faithful semantic communication reconstruction"


def _format_diffusion_prompt(prompt: str, template: str) -> str:
    base = str(prompt or "").strip()
    pattern = str(template or "{prompt}").strip() or "{prompt}"
    if "{prompt}" in pattern:
        rendered = pattern.replace("{prompt}", base)
    else:
        rendered = (pattern + " " + base).strip()
    return " ".join(rendered.split())
