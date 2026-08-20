from __future__ import annotations

import json
import math
import re
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.boundaries import validate_channel_bits
from noema_lab.core.structured_input import (
    decode_strict_json,
    decode_strict_json_object,
)
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationResult, object_schema
from noema_lab.ops.models.timing import append_measurement, codec_timing_metadata

JsonDict = Dict[str, Any]
BART_FRAME_LENGTH_BITS = 32
BART_FRAME_CRC_BITS = 32
BART_FRAME_FIXED_HEADER_SYMBOLS = BART_FRAME_LENGTH_BITS + BART_FRAME_CRC_BITS
DEFAULT_BART_MODEL_ID = "facebook/bart-base"
DEFAULT_BART_MODEL_REVISION = "aadd2ab0ae0c8268c7c9693540e9904811f36177"


class TextUtf8EncodeOperation(Operation):
    id = "model.text_utf8_encode"
    name = "Text UTF-8 payload encoder"
    input_kinds = {"texts": ["text.batch.json"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    params_schema = object_schema(
        {
            "payload_format": {
                "type": "string",
                "default": "json_utf8",
                "enum": ["json_utf8"],
            }
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        start = time.perf_counter()
        text_batch = _load_text_batch(ctx.require_input("texts").path)
        examples = _examples(text_batch)
        payload = {
            "schema_version": 1,
            "payload_format": "json_utf8",
            "ids": [str(example.get("id") or index) for index, example in enumerate(examples)],
            "texts": [str(example.get("text") or "") for example in examples],
        }
        payload_bytes = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        byte_array = np.frombuffer(payload_bytes, dtype=np.uint8)
        bits = np.unpackbits(byte_array).astype(np.uint8, copy=False)
        elapsed = time.perf_counter() - start
        timing_records: List[JsonDict] = []
        append_measurement(
            timing_records,
            None,
            "encoder.total",
            elapsed,
            text_count=len(examples),
            byte_count=len(payload_bytes),
        )
        append_measurement(
            timing_records,
            None,
            "encoder.payload_encode",
            elapsed,
            text_count=len(examples),
            byte_count=len(payload_bytes),
        )
        metadata = {
            "codec": "text_utf8_json",
            "payload_format": "json_utf8",
            "text_count": len(examples),
            "sample_ids": payload["ids"],
            "payload_byte_count": len(payload_bytes),
            "payload_bit_count": int(bits.size),
            "bit_count": int(bits.size),
            "bit_role": "payload",
            "bit_storage": "unpacked_uint8",
            "channel_bit_array": "unpacked_uint8",
            "bits_per_element": 1,
            "semantic_form": "text",
            "payload_coder": "json_utf8",
            "payload_coder_type": "text_serialization",
            "payload_coder_label": "JSON UTF-8 text payload",
            "payload_coder_lossless": True,
            "entropy_coded": False,
            "source_dataset": text_batch.get("dataset"),
        }
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, metadata)},
            metrics={
                "channel.payload_bit_count": int(bits.size),
                "codec.bytes": len(payload_bytes),
                "codec.bit_count": int(bits.size),
            },
            metadata={
                "codec": "text_utf8_json",
                "text_count": len(examples),
                "byte_count": len(payload_bytes),
                "bit_count": int(bits.size),
                "payload_coder": "json_utf8",
                "payload_coder_type": "text_serialization",
                "payload_coder_label": "JSON UTF-8 text payload",
                "payload_coder_lossless": True,
                "entropy_coded": False,
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner="local_python",
                    notes={
                        "runtime_execution_language": "Python + NumPy",
                        "payload_codec": "JSON UTF-8 bytes to unpacked uint8 bits",
                    },
                ),
            },
        )


class TextUtf8DecodeOperation(Operation):
    id = "model.text_utf8_decode"
    name = "Text UTF-8 payload decoder"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"texts": "text.batch.json"}
    params_schema = object_schema(
        {
            "on_error": {
                "type": "string",
                "default": "replace",
                "enum": ["replace", "fail"],
            }
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        start = time.perf_counter()
        bit_artifact = ctx.require_input("bits")
        bits, metadata = _load_bits(bit_artifact.path, bit_artifact.metadata)
        on_error = str(ctx.params.get("on_error") or "replace")
        bit_count = int(metadata.get("payload_bit_count") or metadata.get("bit_count") or bits.size)
        byte_count = int(metadata.get("payload_byte_count") or ((bit_count + 7) // 8))
        clipped = bits[: bit_count].astype(np.uint8, copy=False)
        if clipped.size % 8:
            clipped = np.pad(clipped, (0, 8 - clipped.size % 8), constant_values=0).astype(np.uint8, copy=False)
        payload_bytes = np.packbits(clipped)[:byte_count].tobytes()
        payload_text = payload_bytes.decode("utf-8", errors="replace")
        decode_error = ""
        try:
            payload = decode_strict_json(payload_text)
            ids = [str(item) for item in payload.get("ids") or []]
            texts = [str(item) for item in payload.get("texts") or []]
            if len(ids) < len(texts):
                ids.extend("decoded_%d" % index for index in range(len(ids), len(texts)))
        except Exception as exc:
            if on_error == "fail":
                raise RuntimeError("Text UTF-8 payload JSON decode failed: %s" % exc) from exc
            decode_error = str(exc)
            ids = [str(item) for item in metadata.get("sample_ids") or ["decoded_0"]]
            texts = [payload_text] + [""] * max(0, len(ids) - 1)
        examples = [{"id": ids[index] if index < len(ids) else "decoded_%d" % index, "text": text} for index, text in enumerate(texts)]
        elapsed = time.perf_counter() - start
        timing_records: List[JsonDict] = []
        append_measurement(
            timing_records,
            None,
            "decoder.total",
            elapsed,
            text_count=len(examples),
            byte_count=byte_count,
        )
        append_measurement(
            timing_records,
            None,
            "decoder.payload_decode",
            elapsed,
            text_count=len(examples),
            byte_count=byte_count,
        )
        payload_out = {
            "schema_version": 1,
            "kind": "text.batch",
            "dataset": metadata.get("source_dataset") or "decoded_text",
            "split": "decoded",
            "examples": examples,
        }
        out_metadata = {
            "codec": "text_utf8_json",
            "payload_format": metadata.get("payload_format") or "json_utf8",
            "text_count": len(examples),
            "sample_ids": [example["id"] for example in examples],
            "texts_preview": [{"id": example["id"], "text": example["text"]} for example in examples],
            "payload_byte_count": byte_count,
            "payload_bit_count": bit_count,
            "payload_coder": metadata.get("payload_coder") or "json_utf8",
            "payload_coder_type": metadata.get("payload_coder_type") or "text_serialization",
            "payload_coder_label": metadata.get("payload_coder_label") or "JSON UTF-8 text payload",
            "payload_coder_lossless": bool(metadata.get("payload_coder_lossless", True)),
            "entropy_coded": bool(metadata.get("entropy_coded", False)),
            "decode_error": decode_error,
        }
        path = ctx.output_path("texts", ".json")
        path.write_text(json.dumps(payload_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={"texts": artifact("text.batch.json", path, out_metadata)},
            metrics={"text.decode_error": 1 if decode_error else 0},
            metadata={
                "codec": "text_utf8_json",
                "text_count": len(examples),
                "byte_count": byte_count,
                "bit_count": bit_count,
                "decode_error": decode_error,
                "payload_coder": out_metadata["payload_coder"],
                "payload_coder_type": out_metadata["payload_coder_type"],
                "payload_coder_label": out_metadata["payload_coder_label"],
                "payload_coder_lossless": out_metadata["payload_coder_lossless"],
                "entropy_coded": out_metadata["entropy_coded"],
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="local_python",
                    notes={
                        "runtime_execution_language": "Python + NumPy",
                        "payload_codec": "unpacked uint8 bits to JSON UTF-8 bytes",
                    },
                ),
            },
        )


class TextBartJsccEncodeOperation(Operation):
    id = "model.text_bart_jscc_encode"
    name = "BART JSCC-lite text semantic encoder"
    input_kinds = {"texts": ["text.batch.json"]}
    output_kinds = {"symbols": "channel.symbols.complex_numpy"}
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": True,
        "exportable": True,
        "reason": "The semantic encoder is a PyTorch/Transformers module; benchmark execution uses pretrained eval mode.",
    }
    backends = {"benchmark_run": ["torch"], "dataset_capture": ["torch"], "differentiable_export": ["torch"]}
    materializations = [
        {"runner": "benchmark_run", "backend": "torch", "implementation": "transformers_eval_artifact"},
        {"runner": "dataset_capture", "backend": "torch", "implementation": "transformers_capture_artifact"},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "transformers_encoder_module"},
    ]
    params_schema = object_schema(
        {
            "model_id": {"type": "string", "default": DEFAULT_BART_MODEL_ID},
            "model_revision": {
                "type": "string",
                "default": DEFAULT_BART_MODEL_REVISION,
                "description": "Full immutable Hugging Face commit; required for remote models.",
            },
            "device": {"type": "string", "default": "cpu"},
            "cache_dir": {"type": "string", "default": ""},
            "max_length": {"type": "integer", "default": 64, "minimum": 4},
            "power_normalize": {"type": "boolean", "default": True},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _transformers_availability()
        payload["model_source"] = {
            "provider": "Hugging Face",
            "default_model": "facebook/bart-base",
            "url": "https://huggingface.co/facebook/bart-base",
            "note": "Pretrained denoising encoder-decoder used as a DeepSC-lite semantic bottleneck; it is not an official DeepSC text checkpoint.",
        }
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        start_total = time.perf_counter()
        text_batch = _load_text_batch(ctx.require_input("texts").path)
        examples = _examples(text_batch)
        model_id = str(ctx.params.get("model_id") or DEFAULT_BART_MODEL_ID)
        model_revision = _resolved_seq2seq_model_revision(
            model_id, ctx.params.get("model_revision")
        )
        device = str(ctx.params.get("device") or "cpu")
        max_length = max(4, int(ctx.params.get("max_length") or 64))
        power_normalize = bool(ctx.params.get("power_normalize", True))
        tokenizer, model, torch = _load_seq2seq_model(
            ctx, model_id, device, model_revision
        )
        target_device = _torch_device(torch, device)
        encoder = model.get_encoder()
        encoder.eval()
        model.eval()

        symbols_parts: List[np.ndarray] = []
        example_meta: List[JsonDict] = []
        timing_records: List[JsonDict] = []
        total = max(len(examples), 1)
        symbol_offset = 0
        for index, example in enumerate(examples):
            ctx.report_progress(
                "Encoding text %d/%d examples" % (index, total),
                phase="text_jscc_encode",
                status="running",
                completed=index,
                total=total,
                percent=float(index) / float(total) * 100.0,
                unit="texts",
            )
            text = str(example.get("text") or "")
            token_start = time.perf_counter()
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            encoded = {key: value.to(target_device) for key, value in encoded.items()}
            append_measurement(timing_records, index, "encoder.preprocess", time.perf_counter() - token_start, text_chars=len(text))

            inference_start = time.perf_counter()
            with torch.no_grad():
                hidden = encoder(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                    return_dict=True,
                ).last_hidden_state
            append_measurement(timing_records, index, "encoder.inference", time.perf_counter() - inference_start, token_count=int(encoded["input_ids"].shape[-1]))

            pack_start = time.perf_counter()
            hidden_np = hidden.detach().cpu().numpy().astype(np.float32, copy=False)
            packed, pack_meta = _float_tensor_to_complex_symbols(hidden_np, power_normalize)
            attention_mask = encoded.get("attention_mask")
            item_id = str(example.get("id") or index)
            decoder_metadata = {
                "schema_version": 1,
                "kind": "text_bart_jscc_item_header",
                "id": item_id,
                "semantic_symbol_count": int(packed.size),
                "hidden_shape": list(hidden_np.shape),
                "float_count": int(hidden_np.size),
                "padded_float_count": int(pack_meta["padded_float_count"]),
                "scale": float(pack_meta["scale"]),
                "attention_mask": (
                    attention_mask.detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                    .reshape(-1)
                    .tolist()
                    if attention_mask is not None
                    else []
                ),
                "input_token_count": int(encoded["input_ids"].shape[-1]),
            }
            framed, frame_metadata = _frame_bart_item_symbols(
                packed, decoder_metadata
            )
            symbols_parts.append(framed)
            frame_symbol_count = int(framed.size)
            example_meta.append(
                {
                    "id": item_id,
                    "symbol_offset": symbol_offset,
                    "symbol_count": frame_symbol_count,
                    "semantic_symbol_count": int(packed.size),
                    "decoder_metadata_byte_count": int(
                        frame_metadata["decoder_metadata_byte_count"]
                    ),
                    "decoder_metadata_channel_use_count": int(
                        frame_metadata["decoder_metadata_channel_use_count"]
                    ),
                    "pre_normalization_real_component_rms": float(
                        pack_meta["pre_normalization_real_component_rms"]
                    ),
                    "post_normalization_mean_symbol_energy": float(
                        pack_meta["post_normalization_mean_symbol_energy"]
                    ),
                }
            )
            symbol_offset += frame_symbol_count
            append_measurement(
                timing_records,
                index,
                "encoder.symbol_encode",
                time.perf_counter() - pack_start,
                symbol_count=frame_symbol_count,
                semantic_symbol_count=int(packed.size),
                decoder_metadata_channel_use_count=int(
                    frame_metadata["decoder_metadata_channel_use_count"]
                ),
            )

        ctx.report_progress(
            "Encoding text %d/%d examples" % (len(examples), total),
            phase="text_jscc_encode",
            status="running",
            completed=len(examples),
            total=total,
            percent=100.0,
            unit="texts",
        )
        symbols = np.concatenate(symbols_parts).astype(np.complex64, copy=False) if symbols_parts else np.zeros((0,), dtype=np.complex64)
        source_item_symbol_counts = [
            int(item["symbol_count"]) for item in example_meta
        ]
        semantic_symbol_count = sum(
            int(item["semantic_symbol_count"]) for item in example_meta
        )
        decoder_metadata_channel_use_count = sum(
            int(item["decoder_metadata_channel_use_count"])
            for item in example_meta
        )
        if (
            any(count <= 0 for count in source_item_symbol_counts)
            or sum(source_item_symbol_counts) != int(symbols.size)
        ):
            raise OperationError(
                "BART JSCC source-item symbol counts do not cover the encoded symbol stream"
            )
        metadata = {
            "codec": "text_bart_jscc_lite",
            "codec_profile": "text_bart_jscc",
            "model_id": model_id,
            "model_revision": model_revision,
            "model_source": "https://huggingface.co/%s" % model_id,
            "pretrained_model": True,
            "jscc_style": (
                "continuous semantic symbols with in-band BPSK decoder headers"
            ),
            "semantic_form": "text_encoder_hidden_state",
            "symbol_storage": "complex64",
            "symbol_count": int(symbols.size),
            "channel_use_count": int(symbols.size),
            "semantic_symbol_count": semantic_symbol_count,
            "decoder_metadata_channel_use_count": (
                decoder_metadata_channel_use_count
            ),
            "framing_channel_use_count": int(symbols.size)
            - semantic_symbol_count,
            "decoder_side_information_transport": (
                "in_band_crc32_bpsk_json_v1"
            ),
            "decoder_side_information_rate_accounted": True,
            "text_count": len(examples),
            "sample_ids": [item["id"] for item in example_meta],
            "source_item_count": len(example_meta),
            "source_item_ids": [item["id"] for item in example_meta],
            "source_item_symbol_counts": source_item_symbol_counts,
            "source_item_channel_use_counts": source_item_symbol_counts,
            "source_item_use_counts_are_additive": True,
            "examples": example_meta,
            "source_dataset": text_batch.get("dataset"),
            "max_length": max_length,
            "power_normalize": power_normalize,
            "encoder_semantic_power_normalization_scope": (
                "source_item" if power_normalize else "none"
            ),
            "encoder_semantic_power_normalization_quantity": (
                "real_component_rms"
            ),
            "encoder_semantic_power_normalization_target": (
                1.0 if power_normalize else None
            ),
            "encoder_semantic_power_normalization_domain": (
                "semantic_payload_before_in_band_framing"
                if power_normalize
                else "none"
            ),
            "power_normalization_scope": "none",
            "runtime_execution_language": "Python + PyTorch + Transformers",
        }
        path = ctx.output_path("symbols", ".npz")
        np.savez_compressed(path, symbols=symbols, metadata_json=json.dumps(metadata))
        append_measurement(timing_records, None, "encoder.total", time.perf_counter() - start_total, text_count=len(examples), symbol_count=int(symbols.size))
        return OperationResult(
            outputs={"symbols": artifact("channel.symbols.complex_numpy", path, metadata)},
            metrics={
                "channel.symbol_count": int(symbols.size),
                "channel.channel_use_count": int(symbols.size),
                "codec.semantic_symbol_count": semantic_symbol_count,
                "codec.decoder_metadata_channel_use_count": (
                    decoder_metadata_channel_use_count
                ),
            },
            metadata={
                **metadata,
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner="pytorch",
                    notes={
                        "runtime_execution_language": "Python + PyTorch + Transformers",
                        "model_source": "https://huggingface.co/%s" % model_id,
                        "codec": "BART encoder hidden states packed as complex channel symbols",
                    },
                ),
            },
        )


class TextBartJsccDecodeOperation(Operation):
    id = "model.text_bart_jscc_decode"
    name = "BART JSCC-lite text semantic decoder"
    input_kinds = {"symbols": ["channel.symbols.complex_numpy", "channel.rx_symbols.complex_numpy"]}
    output_kinds = {"texts": "text.batch.json"}
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": True,
        "exportable": True,
        "reason": "The semantic decoder is a PyTorch/Transformers module; benchmark execution uses pretrained eval mode.",
    }
    backends = {"benchmark_run": ["torch"], "dataset_capture": ["torch"], "differentiable_export": ["torch"]}
    materializations = [
        {"runner": "benchmark_run", "backend": "torch", "implementation": "transformers_eval_artifact"},
        {"runner": "dataset_capture", "backend": "torch", "implementation": "transformers_capture_artifact"},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "transformers_decoder_module"},
    ]
    params_schema = object_schema(
        {
            "model_id": {"type": "string", "default": DEFAULT_BART_MODEL_ID},
            "model_revision": {
                "type": "string",
                "default": DEFAULT_BART_MODEL_REVISION,
                "description": "Full immutable Hugging Face commit; required for remote models and matched to sender evidence.",
            },
            "device": {"type": "string", "default": "cpu"},
            "cache_dir": {"type": "string", "default": ""},
            "generation_max_length": {"type": "integer", "default": 64, "minimum": 4},
            "num_beams": {"type": "integer", "default": 1, "minimum": 1},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _transformers_availability()
        payload["model_source"] = {
            "provider": "Hugging Face",
            "default_model": "facebook/bart-base",
            "url": "https://huggingface.co/facebook/bart-base",
            "note": "Pretrained denoising encoder-decoder used as a DeepSC-lite semantic bottleneck; it is not an official DeepSC text checkpoint.",
        }
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        start_total = time.perf_counter()
        input_artifact = ctx.require_input("symbols")
        symbols, metadata = _load_symbols(input_artifact.path, input_artifact.metadata)
        received_items = _decode_bart_item_frames(symbols)
        example_meta = _validated_bart_example_metadata(
            metadata, int(symbols.size), received_items
        )
        model_id = str(
            ctx.params.get("model_id") or DEFAULT_BART_MODEL_ID
        )
        declared_model_id = str(metadata.get("model_id") or "")
        if declared_model_id and declared_model_id != model_id:
            raise OperationError(
                "Text JSCC receiver model_id `%s` does not match sender model_id `%s`"
                % (model_id, declared_model_id)
            )
        declared_model_revision = str(metadata.get("model_revision") or "")
        requested_model_revision = ctx.params.get("model_revision")
        if not requested_model_revision and declared_model_revision:
            requested_model_revision = declared_model_revision
        model_revision = _resolved_seq2seq_model_revision(
            model_id, requested_model_revision
        )
        if (
            declared_model_revision
            and declared_model_revision != model_revision
        ):
            raise OperationError(
                "Text JSCC receiver model_revision `%s` does not match sender model_revision `%s`"
                % (model_revision, declared_model_revision)
            )
        device = str(ctx.params.get("device") or "cpu")
        generation_max_length = max(4, int(ctx.params.get("generation_max_length") or 64))
        num_beams = max(1, int(ctx.params.get("num_beams") or 1))
        tokenizer, model, torch = _load_seq2seq_model(
            ctx, model_id, device, model_revision
        )
        from transformers.modeling_outputs import BaseModelOutput

        target_device = _torch_device(torch, device)
        model.eval()
        timing_records: List[JsonDict] = []
        examples = []
        total = max(len(example_meta), 1)
        for index, (declared_item, received_item) in enumerate(
            zip(example_meta, received_items)
        ):
            item = received_item["decoder_metadata"]
            semantic_symbols = received_item["semantic_symbols"]
            ctx.report_progress(
                "Decoding text %d/%d examples" % (index, total),
                phase="text_jscc_decode",
                status="running",
                completed=index,
                total=total,
                percent=float(index) / float(total) * 100.0,
                unit="texts",
            )
            unpack_start = time.perf_counter()
            hidden_np = _complex_symbols_to_float_tensor(
                semantic_symbols,
                {
                    **item,
                    "symbol_offset": 0,
                    "symbol_count": int(item["semantic_symbol_count"]),
                },
            )
            attention_values = item.get("attention_mask") or [1] * int(hidden_np.shape[1])
            attention = torch.tensor([int(value) for value in attention_values], dtype=torch.long, device=target_device).reshape(1, -1)
            hidden = torch.tensor(hidden_np, dtype=torch.float32, device=target_device)
            encoder_outputs = BaseModelOutput(last_hidden_state=hidden)
            append_measurement(
                timing_records,
                index,
                "decoder.symbol_decode",
                time.perf_counter() - unpack_start,
                symbol_count=int(declared_item["symbol_count"]),
                semantic_symbol_count=int(item["semantic_symbol_count"]),
                decoder_metadata_channel_use_count=int(
                    received_item["decoder_metadata_channel_use_count"]
                ),
            )

            inference_start = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention,
                    max_length=generation_max_length,
                    num_beams=num_beams,
                    do_sample=False,
                )
            text = tokenizer.decode(generated[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            append_measurement(timing_records, index, "decoder.inference", time.perf_counter() - inference_start, token_count=int(generated.shape[-1]))
            examples.append({"id": str(item.get("id") or index), "text": text})

        ctx.report_progress(
            "Decoding text %d/%d examples" % (len(example_meta), total),
            phase="text_jscc_decode",
            status="running",
            completed=len(example_meta),
            total=total,
            percent=100.0,
            unit="texts",
        )
        payload = {
            "schema_version": 1,
            "kind": "text.batch",
            "dataset": "bart_jscc_lite_generated",
            "split": "generated",
            "examples": examples,
        }
        out_metadata = {
            "codec": "text_bart_jscc_lite",
            "codec_profile": "text_bart_jscc",
            "model_id": model_id,
            "model_revision": model_revision,
            "text_count": len(examples),
            "sample_ids": [example["id"] for example in examples],
            "texts_preview": [{"id": example["id"], "text": example["text"]} for example in examples],
            "semantic_form": "text",
            "channel_use_count": int(symbols.size),
            "symbol_count": int(symbols.size),
            "source_item_count": len(example_meta),
            "source_item_ids": [str(item["id"]) for item in example_meta],
            "source_item_symbol_counts": [
                int(item["symbol_count"]) for item in example_meta
            ],
            "source_item_channel_use_counts": [
                int(item["symbol_count"]) for item in example_meta
            ],
            "source_item_use_counts_are_additive": True,
            "semantic_symbol_count": sum(
                int(item["decoder_metadata"]["semantic_symbol_count"])
                for item in received_items
            ),
            "decoder_metadata_channel_use_count": sum(
                int(item["decoder_metadata_channel_use_count"])
                for item in received_items
            ),
            "framing_channel_use_count": sum(
                int(item["frame_symbol_count"])
                - int(item["decoder_metadata"]["semantic_symbol_count"])
                for item in received_items
            ),
            "decoder_side_information_transport": (
                "in_band_crc32_bpsk_json_v1"
            ),
            "decoder_side_information_rate_accounted": True,
            "decoder_metadata_source": "received_symbol_frames",
            "power_normalization_scope": str(
                metadata.get("power_normalization_scope") or "none"
            ),
        }
        path = ctx.output_path("texts", ".json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        append_measurement(timing_records, None, "decoder.total", time.perf_counter() - start_total, text_count=len(examples), symbol_count=int(symbols.size))
        return OperationResult(
            outputs={"texts": artifact("text.batch.json", path, out_metadata)},
            metrics={
                "text.generated_count": len(examples),
                "channel.channel_use_count": int(out_metadata["channel_use_count"]),
                "codec.semantic_symbol_count": int(
                    out_metadata["semantic_symbol_count"]
                ),
                "codec.decoder_metadata_channel_use_count": int(
                    out_metadata["decoder_metadata_channel_use_count"]
                ),
            },
            metadata={
                **out_metadata,
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="pytorch",
                    notes={
                        "runtime_execution_language": "Python + PyTorch + Transformers",
                        "model_source": "https://huggingface.co/%s" % model_id,
                        "codec": "BART decoder generation from received encoder hidden-state symbols",
                    },
                ),
            },
        )


def _load_text_batch(path) -> JsonDict:
    data = decode_strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("kind") != "text.batch":
        raise RuntimeError("Expected text.batch JSON artifact: %s" % path)
    _examples(data)
    return data


def _examples(batch: JsonDict) -> List[JsonDict]:
    examples = batch.get("examples")
    if not isinstance(examples, list):
        raise RuntimeError("text.batch artifact requires an examples list")
    output = []
    for index, item in enumerate(examples):
        if not isinstance(item, dict):
            raise RuntimeError("text.batch examples[%d] must be an object" % index)
        output.append(dict(item))
    return output


def _load_bits(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        bits = payload["bits"]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Text codec bit metadata_json",
                )
            )
    canonical, boundary_metadata = validate_channel_bits(
        bits,
        label="text codec bits",
        backend="python_numpy",
    )
    metadata.update(boundary_metadata)
    return canonical, metadata


def _load_symbols(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        symbols = payload["symbols"]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Text codec symbol metadata_json",
                )
            )
    return symbols.astype(np.complex64, copy=False).reshape(-1), metadata


def _transformers_availability() -> JsonDict:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        return {
            "available": False,
            "extra": "textgen",
            "reason": (
                'Install with `python -m pip install "noema-lab[textgen]"` in an '
                "installed environment, or `uv sync --extra textgen` in a source "
                "checkout, to use BART JSCC-lite text codecs: %s" % exc
            ),
        }
    return {"available": True, "extra": "textgen"}


def _load_seq2seq_model(
    ctx: OperationContext,
    model_id: str,
    device: str,
    model_revision: str | None = None,
):
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as exc:
        raise OperationError(
            'Install with `python -m pip install "noema-lab[textgen]"` in an '
            "installed environment, or `uv sync --extra textgen` in a source "
            "checkout, to use BART JSCC-lite text codecs"
        ) from exc
    cache_dir = str(ctx.params.get("cache_dir") or "").strip()
    if not cache_dir:
        cache_dir = str(_default_hf_cache_dir(ctx))
    revision = _resolved_seq2seq_model_revision(model_id, model_revision)
    load_kwargs = {"cache_dir": cache_dir}
    if revision != "local_path":
        load_kwargs["revision"] = revision
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **load_kwargs)
        model.to(_torch_device(torch, device))
        return tokenizer, model, torch
    except Exception as exc:
        raise OperationError("Could not load pretrained text JSCC model `%s`: %s" % (model_id, exc)) from exc


def _resolved_seq2seq_model_revision(model_id: str, value: Any) -> str:
    if Path(str(model_id)).expanduser().exists():
        return "local_path"
    revision = str(value or "").strip().lower()
    if not revision and model_id == DEFAULT_BART_MODEL_ID:
        revision = DEFAULT_BART_MODEL_REVISION
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise OperationError(
            "Remote text model `%s` requires model_revision as a full immutable "
            "40-character commit" % model_id
        )
    return revision


def _default_hf_cache_dir(ctx: OperationContext):
    if ctx.run_dir.parent.name == "runs":
        return ctx.run_dir.parent.parent / "hf_cache"
    return ctx.run_dir / "hf_cache"


def _torch_device(torch, requested: str):
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise OperationError("CUDA was requested for the text JSCC codec, but no CUDA GPU is available")
    return torch.device(requested)


def _float_tensor_to_complex_symbols(values: np.ndarray, power_normalize: bool) -> Tuple[np.ndarray, JsonDict]:
    flat = values.astype(np.float32, copy=False).reshape(-1)
    pre_rms = (
        float(np.sqrt(np.mean(np.square(flat, dtype=np.float64))))
        if flat.size
        else 0.0
    )
    scale = 1.0
    if power_normalize and flat.size:
        if math.isfinite(pre_rms) and pre_rms > 1e-12:
            scale = pre_rms
            flat = (flat / scale).astype(np.float32, copy=False)
    if flat.size % 2:
        flat = np.pad(flat, (0, 1), constant_values=0).astype(np.float32, copy=False)
    symbols = (flat[0::2] + 1j * flat[1::2]).astype(np.complex64, copy=False)
    symbol_power = (
        float(np.mean(np.abs(symbols.astype(np.complex64, copy=False)) ** 2))
        if symbols.size
        else 0.0
    )
    return symbols, {
        "scale": scale,
        "padded_float_count": int(flat.size),
        "pre_normalization_real_component_rms": pre_rms,
        "post_normalization_mean_symbol_energy": symbol_power,
    }


def _frame_bart_item_symbols(
    semantic_symbols: np.ndarray, decoder_metadata: Mapping[str, Any]
) -> Tuple[np.ndarray, JsonDict]:
    metadata_bytes = json.dumps(
        dict(decoder_metadata),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(metadata_bytes) >= 2**BART_FRAME_LENGTH_BITS:
        raise OperationError("Text JSCC item metadata is too large to frame")
    checksum = zlib.crc32(metadata_bytes) & 0xFFFFFFFF
    fixed_header = len(metadata_bytes).to_bytes(4, "big") + checksum.to_bytes(
        4, "big"
    )
    header_bits = np.unpackbits(
        np.frombuffer(fixed_header + metadata_bytes, dtype=np.uint8)
    ).astype(np.uint8, copy=False)
    header_symbols = (1.0 - 2.0 * header_bits.astype(np.float32)).astype(
        np.complex64
    )
    semantic = np.asarray(semantic_symbols, dtype=np.complex64).reshape(-1)
    framed = np.concatenate((header_symbols, semantic)).astype(
        np.complex64, copy=False
    )
    return framed, {
        "decoder_metadata_byte_count": len(metadata_bytes),
        "decoder_metadata_channel_use_count": int(header_symbols.size),
        "semantic_symbol_count": int(semantic.size),
        "frame_symbol_count": int(framed.size),
        "decoder_metadata_crc32": "%08x" % checksum,
    }


def _decode_bart_item_frames(symbols: np.ndarray) -> List[JsonDict]:
    values = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    output: List[JsonDict] = []
    cursor = 0
    while cursor < int(values.size):
        frame_start = cursor
        if int(values.size) - cursor < BART_FRAME_FIXED_HEADER_SYMBOLS:
            raise OperationError(
                "Text JSCC received frame is truncated before its fixed header"
            )
        fixed_bits = _bart_bpsk_header_bits(
            values[cursor : cursor + BART_FRAME_FIXED_HEADER_SYMBOLS],
            "fixed header",
        )
        fixed_header = np.packbits(fixed_bits).tobytes()
        metadata_byte_count = int.from_bytes(fixed_header[:4], "big")
        expected_crc = int.from_bytes(fixed_header[4:8], "big")
        cursor += BART_FRAME_FIXED_HEADER_SYMBOLS
        metadata_symbol_count = metadata_byte_count * 8
        if (
            metadata_byte_count <= 0
            or metadata_symbol_count > int(values.size) - cursor
        ):
            raise OperationError(
                "Text JSCC received frame declares invalid or truncated decoder metadata"
            )
        metadata_bits = _bart_bpsk_header_bits(
            values[cursor : cursor + metadata_symbol_count],
            "decoder metadata",
        )
        metadata_bytes = np.packbits(metadata_bits).tobytes()
        actual_crc = zlib.crc32(metadata_bytes) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise OperationError(
                "Text JSCC received decoder metadata failed its CRC32 check"
            )
        try:
            raw_metadata = decode_strict_json_object(
                metadata_bytes.decode("utf-8", errors="strict"),
                label="Text JSCC decoder metadata",
            )
        except Exception as exc:
            raise OperationError(
                "Text JSCC received decoder metadata is not valid UTF-8 JSON: %s"
                % exc
            ) from exc
        decoder_metadata = _validated_bart_decoder_metadata(
            raw_metadata, len(output)
        )
        cursor += metadata_symbol_count
        semantic_symbol_count = int(
            decoder_metadata["semantic_symbol_count"]
        )
        if semantic_symbol_count > int(values.size) - cursor:
            raise OperationError(
                "Text JSCC received frame is truncated before its semantic symbols"
            )
        semantic_symbols = values[
            cursor : cursor + semantic_symbol_count
        ].astype(np.complex64, copy=False)
        cursor += semantic_symbol_count
        output.append(
            {
                "decoder_metadata": decoder_metadata,
                "semantic_symbols": semantic_symbols,
                "frame_symbol_offset": frame_start,
                "frame_symbol_count": cursor - frame_start,
                "decoder_metadata_byte_count": metadata_byte_count,
                "decoder_metadata_channel_use_count": (
                    BART_FRAME_FIXED_HEADER_SYMBOLS
                    + metadata_symbol_count
                ),
                "decoder_metadata_crc32": "%08x" % actual_crc,
            }
        )
    return output


def _bart_bpsk_header_bits(symbols: np.ndarray, label: str) -> np.ndarray:
    real = np.real(np.asarray(symbols, dtype=np.complex64).reshape(-1))
    if not np.all(np.isfinite(real)) or np.any(np.abs(real) <= 1e-12):
        raise OperationError(
            "Text JSCC received %s symbols are non-finite or undecidable"
            % label
        )
    return (real < 0.0).astype(np.uint8, copy=False)


def _validated_bart_decoder_metadata(
    value: Any, item_index: int
) -> JsonDict:
    if not isinstance(value, Mapping):
        raise OperationError(
            "Text JSCC received item %d decoder metadata must be an object"
            % item_index
        )
    required = {
        "schema_version",
        "kind",
        "id",
        "semantic_symbol_count",
        "hidden_shape",
        "float_count",
        "padded_float_count",
        "scale",
        "attention_mask",
        "input_token_count",
    }
    if set(value) != required:
        raise OperationError(
            "Text JSCC received item %d decoder metadata fields are incomplete "
            "or unknown" % item_index
        )
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "text_bart_jscc_item_header"
    ):
        raise OperationError(
            "Text JSCC received item %d decoder metadata schema is unsupported"
            % item_index
        )
    item_id = str(value.get("id") or "").strip()
    if not item_id:
        raise OperationError(
            "Text JSCC received item %d decoder metadata requires an id"
            % item_index
        )
    try:
        semantic_symbol_count = int(value["semantic_symbol_count"])
        float_count = int(value["float_count"])
        padded_float_count = int(value["padded_float_count"])
        input_token_count = int(value["input_token_count"])
        shape = [int(item) for item in value["hidden_shape"]]
        scale = float(value["scale"])
    except (TypeError, ValueError) as exc:
        raise OperationError(
            "Text JSCC received item %d decoder metadata has invalid numeric fields"
            % item_index
        ) from exc
    if (
        semantic_symbol_count <= 0
        or float_count <= 0
        or padded_float_count != semantic_symbol_count * 2
        or float_count > padded_float_count
        or len(shape) != 3
        or shape[0] != 1
        or any(item <= 0 for item in shape)
        or int(np.prod(shape, dtype=np.int64)) != float_count
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise OperationError(
            "Text JSCC received item %d decoder shape/count/scale metadata is inconsistent"
            % item_index
        )
    raw_attention = value.get("attention_mask")
    if not isinstance(raw_attention, list):
        raise OperationError(
            "Text JSCC received item %d attention_mask must be a list"
            % item_index
        )
    try:
        attention = [int(item) for item in raw_attention]
    except (TypeError, ValueError) as exc:
        raise OperationError(
            "Text JSCC received item %d attention_mask must contain integers"
            % item_index
        ) from exc
    if (
        len(attention) != shape[1]
        or input_token_count != shape[1]
        or any(item not in (0, 1) for item in attention)
    ):
        raise OperationError(
            "Text JSCC received item %d attention/token counts are inconsistent"
            % item_index
        )
    return {
        "schema_version": 1,
        "kind": "text_bart_jscc_item_header",
        "id": item_id,
        "semantic_symbol_count": semantic_symbol_count,
        "hidden_shape": shape,
        "float_count": float_count,
        "padded_float_count": padded_float_count,
        "scale": scale,
        "attention_mask": attention,
        "input_token_count": input_token_count,
    }


def _validated_bart_example_metadata(
    metadata: Mapping[str, Any],
    total_symbol_count: int,
    received_items: List[JsonDict] | None = None,
) -> List[JsonDict]:
    raw_examples = metadata.get("examples")
    if not isinstance(raw_examples, list) or any(
        not isinstance(item, Mapping) for item in raw_examples
    ):
        raise OperationError(
            "Text JSCC metadata requires an examples list of objects"
        )
    examples = [dict(item) for item in raw_examples]
    raw_counts = metadata.get("source_item_symbol_counts")
    if not isinstance(raw_counts, list):
        raise OperationError(
            "Text JSCC metadata requires source_item_symbol_counts"
        )
    try:
        counts = [int(value) for value in raw_counts]
    except (TypeError, ValueError) as exc:
        raise OperationError(
            "Text JSCC source_item_symbol_counts must contain integers"
        ) from exc
    if len(counts) != len(examples) or any(value <= 0 for value in counts):
        raise OperationError(
            "Text JSCC source_item_symbol_counts must contain one positive count per example"
        )
    if sum(counts) != int(total_symbol_count):
        raise OperationError(
            "Text JSCC source_item_symbol_counts do not cover the received symbol stream"
        )
    declared_count = metadata.get("source_item_count")
    try:
        normalized_declared_count = (
            int(declared_count) if declared_count is not None else None
        )
    except (TypeError, ValueError) as exc:
        raise OperationError("Text JSCC source_item_count must be an integer") from exc
    if normalized_declared_count != len(examples):
        raise OperationError(
            "Text JSCC source_item_count does not match examples metadata"
        )
    if not bool(metadata.get("source_item_use_counts_are_additive", False)):
        raise OperationError(
            "Text JSCC metadata must declare additive source-item use counts"
        )
    if (
        metadata.get("decoder_side_information_transport")
        != "in_band_crc32_bpsk_json_v1"
        or not bool(
            metadata.get("decoder_side_information_rate_accounted", False)
        )
    ):
        raise OperationError(
            "Text JSCC metadata must declare rate-accounted in-band decoder side information"
        )
    raw_channel_counts = metadata.get("source_item_channel_use_counts")
    if raw_channel_counts is not None:
        try:
            channel_counts = [int(value) for value in raw_channel_counts]
        except (TypeError, ValueError) as exc:
            raise OperationError(
                "Text JSCC source_item_channel_use_counts must contain integers"
            ) from exc
        if channel_counts != counts:
            raise OperationError(
                "Text JSCC source-item channel-use counts do not match symbol counts"
            )
    raw_ids = metadata.get("source_item_ids")
    if not isinstance(raw_ids, list) or len(raw_ids) != len(examples):
        raise OperationError(
            "Text JSCC metadata requires one source_item_id per example"
        )

    cursor = 0
    ids = []
    semantic_counts = []
    decoder_metadata_counts = []
    forbidden_decoder_fields = {
        "hidden_shape",
        "float_count",
        "padded_float_count",
        "scale",
        "attention_mask",
        "input_token_count",
    }
    for index, (item, count) in enumerate(zip(examples, counts)):
        if forbidden_decoder_fields & set(item):
            raise OperationError(
                "Text JSCC artifact examples must not carry decoder-critical "
                "metadata outside the received symbol frames"
            )
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            raise OperationError("Text JSCC example %d requires a non-empty id" % index)
        ids.append(item_id)
        try:
            offset = int(item.get("symbol_offset") or 0)
            item_count = int(item.get("symbol_count") or 0)
        except (TypeError, ValueError) as exc:
            raise OperationError(
                "Text JSCC example %d symbol offset/count must be integers" % index
            ) from exc
        if offset != cursor or item_count != count:
            raise OperationError(
                "Text JSCC example %d symbol offsets/counts are not contiguous" % index
            )
        cursor += count
        try:
            semantic_count = int(item.get("semantic_symbol_count") or 0)
            decoder_metadata_count = int(
                item.get("decoder_metadata_channel_use_count") or 0
            )
        except (TypeError, ValueError) as exc:
            raise OperationError(
                "Text JSCC example %d accounting counts must be integers" % index
            ) from exc
        if (
            semantic_count <= 0
            or decoder_metadata_count < BART_FRAME_FIXED_HEADER_SYMBOLS
            or semantic_count + decoder_metadata_count != item_count
        ):
            raise OperationError(
                "Text JSCC example %d frame accounting is inconsistent" % index
            )
        semantic_counts.append(semantic_count)
        decoder_metadata_counts.append(decoder_metadata_count)
    if [str(value) for value in raw_ids] != ids:
        raise OperationError(
            "Text JSCC source_item_ids do not match examples metadata"
        )
    try:
        declared_semantic_count = int(
            metadata.get("semantic_symbol_count") or 0
        )
        declared_decoder_metadata_count = int(
            metadata.get("decoder_metadata_channel_use_count") or 0
        )
        declared_framing_count = int(
            metadata.get("framing_channel_use_count") or 0
        )
    except (TypeError, ValueError) as exc:
        raise OperationError(
            "Text JSCC aggregate framing counts must be integers"
        ) from exc
    if declared_semantic_count != sum(semantic_counts):
        raise OperationError(
            "Text JSCC semantic_symbol_count does not match item frames"
        )
    if declared_decoder_metadata_count != sum(decoder_metadata_counts):
        raise OperationError(
            "Text JSCC decoder metadata channel-use count does not match item frames"
        )
    if declared_framing_count != sum(decoder_metadata_counts):
        raise OperationError(
            "Text JSCC framing channel-use count does not match item frames"
        )
    if received_items is not None:
        received_ids = [
            str(item["decoder_metadata"]["id"]) for item in received_items
        ]
        received_counts = [
            int(item["frame_symbol_count"]) for item in received_items
        ]
        received_semantic_counts = [
            int(item["decoder_metadata"]["semantic_symbol_count"])
            for item in received_items
        ]
        received_metadata_counts = [
            int(item["decoder_metadata_channel_use_count"])
            for item in received_items
        ]
        if (
            received_ids != ids
            or received_counts != counts
            or received_semantic_counts != semantic_counts
            or received_metadata_counts != decoder_metadata_counts
        ):
            raise OperationError(
                "Text JSCC received frames do not match declared item accounting"
            )
    return examples


def _complex_symbols_to_float_tensor(symbols: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    offset = int(metadata.get("symbol_offset") or 0)
    count = int(metadata.get("symbol_count") or 0)
    if count < 0 or offset < 0 or offset + count > int(symbols.size):
        raise OperationError("Text JSCC symbol metadata points outside the received symbol array")
    chunk = symbols[offset : offset + count].astype(np.complex64, copy=False)
    flat = np.empty(int(chunk.size) * 2, dtype=np.float32)
    flat[0::2] = np.real(chunk)
    flat[1::2] = np.imag(chunk)
    float_count = int(metadata.get("float_count") or flat.size)
    flat = flat[:float_count]
    scale = float(metadata.get("scale") or 1.0)
    flat = (flat * scale).astype(np.float32, copy=False)
    shape = tuple(int(value) for value in metadata.get("hidden_shape") or [])
    if not shape or int(np.prod(shape)) != flat.size:
        raise OperationError("Text JSCC hidden-state shape metadata is invalid")
    return flat.reshape(shape)
