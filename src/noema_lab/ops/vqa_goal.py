from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
DEFAULT_VQA_MODEL_ID = "dandelin/vilt-b32-finetuned-vqa"
DEFAULT_VQA_MODEL_REVISION = "d0a1f6ab88522427a7ae76ceb6e1e1e7b68a1d08"


class VqaSemanticSelectOperation(Operation):
    id = "foundation.vqa_semantic_select"
    name = "VQA semantic region selector"
    input_kinds = {
        "images": ["image.batch.numpy"],
        "questions": ["vqa.questions.json"],
    }
    output_kinds = {
        "packet": "vqa.semantic_packet.json",
        "detections": "vision.detections.json",
    }
    params_schema = object_schema(
        {
            "selector": {
                "type": "string",
                "default": "local_rules",
                "enum": ["local_rules"],
            }
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        selector = str(ctx.params.get("selector") or "local_rules")
        if selector != "local_rules":
            raise OperationError("VQA selector `%s` is not configured; use local_rules or add a VLM adapter." % selector)
        questions = _load_vqa_questions(ctx.require_input("questions").path)
        image_artifact = ctx.require_input("images")
        images, image_metadata = _load_image_batch(
            image_artifact.path, image_artifact.metadata
        )
        _validate_vqa_image_question_identity(
            questions, image_metadata, int(images.shape[0])
        )
        packets: List[JsonDict] = []
        detections: List[JsonDict] = []
        for item in questions:
            question = str(item.get("question") or "")
            label, bbox = _select_answer_region(question)
            packets.append(
                {
                    "id": str(item.get("id") or len(packets)),
                    "image_id": str(item["image_id"]),
                    "question": question,
                    "selected_label": label,
                    "selected_box": bbox,
                    "selector": selector,
                }
            )
            detections.append({"id": str(item.get("id") or len(detections)), "detections": [{"label": label, "bbox": bbox, "score": 0.99}]})
        packet_payload = {
            "schema_version": 1,
            "kind": "vqa.semantic_packet",
            "examples": packets,
        }
        detection_payload = {
            "schema_version": 1,
            "kind": "vision.detections",
            "examples": detections,
        }
        packet_path = _write_json(ctx, "packet", packet_payload)
        detection_path = _write_json(ctx, "detections", detection_payload)
        metadata = {
            "selector": selector,
            "example_count": len(packets),
            "packet_kind": "vqa.semantic_packet",
            "image_question_pairing": "exact ordered image_id match",
        }
        return OperationResult(
            outputs={
                "packet": artifact("vqa.semantic_packet.json", packet_path, metadata),
                "detections": artifact("vision.detections.json", detection_path, metadata),
            },
            metrics={
                "vqa.semantic_packet_count": len(packets),
                "vqa.selected_region_count": len(detections),
            },
            metadata=metadata,
        )


class VqaPayloadEncodeOperation(Operation):
    id = "foundation.vqa_payload_encode"
    name = "VQA semantic packet payload encoder"
    input_kinds = {"packet": ["vqa.semantic_packet.json"]}
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
        packet = _load_json(ctx.require_input("packet").path)
        examples = list(packet.get("examples") or [])
        payload_bytes = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8)).astype(np.uint8, copy=False)
        elapsed = time.perf_counter() - start
        timing_records: List[JsonDict] = []
        append_measurement(timing_records, None, "encoder.total", elapsed, example_count=len(examples), byte_count=len(payload_bytes))
        append_measurement(timing_records, None, "encoder.payload_encode", elapsed, example_count=len(examples), byte_count=len(payload_bytes))
        metadata = {
            "codec": "vqa_semantic_packet_json_utf8",
            "payload_format": "json_utf8",
            "payload_kind": "vqa.semantic_packet",
            "payload_byte_count": len(payload_bytes),
            "payload_bit_count": int(bits.size),
            "bit_count": int(bits.size),
            "bit_role": "payload",
            "bit_storage": "unpacked_uint8",
            "channel_bit_array": "unpacked_uint8",
            "bits_per_element": 1,
            "sample_ids": [str(item.get("id") or index) for index, item in enumerate(examples)],
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
                "codec": metadata["codec"],
                "byte_count": len(payload_bytes),
                "bit_count": int(bits.size),
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner="local_python",
                    notes={
                        "runtime_execution_language": "Python + NumPy",
                        "payload_codec": "VQA semantic packet JSON UTF-8 bytes to unpacked uint8 bits",
                    },
                ),
            },
        )


class VqaPayloadDecodeOperation(Operation):
    id = "foundation.vqa_payload_decode"
    name = "VQA semantic packet payload decoder"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"packet": "vqa.semantic_packet.json"}
    params_schema = object_schema(
        {
            "on_error": {
                "type": "string",
                "default": "fail",
                "enum": ["fail", "replace"],
            }
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        start = time.perf_counter()
        bits, metadata = _load_bits(ctx.require_input("bits").path, ctx.require_input("bits").metadata)
        bit_count = int(metadata.get("payload_bit_count") or metadata.get("bit_count") or bits.size)
        byte_count = int(metadata.get("payload_byte_count") or ((bit_count + 7) // 8))
        clipped = bits[:bit_count].astype(np.uint8, copy=False)
        if clipped.size % 8:
            clipped = np.pad(clipped, (0, 8 - clipped.size % 8), constant_values=0).astype(np.uint8, copy=False)
        payload_bytes = np.packbits(clipped)[:byte_count].tobytes()
        try:
            packet = decode_strict_json(payload_bytes.decode("utf-8", errors="strict"))
        except Exception as exc:
            if str(ctx.params.get("on_error") or "fail") == "fail":
                raise OperationError("VQA payload decode failed, likely due to corrupted payload bits: %s" % exc) from exc
            packet = {"schema_version": 1, "kind": "vqa.semantic_packet", "examples": []}
        examples = list(packet.get("examples") or [])
        elapsed = time.perf_counter() - start
        timing_records: List[JsonDict] = []
        append_measurement(timing_records, None, "decoder.total", elapsed, example_count=len(examples), byte_count=byte_count)
        append_measurement(timing_records, None, "decoder.payload_decode", elapsed, example_count=len(examples), byte_count=byte_count)
        output_metadata = {
            "codec": metadata.get("codec") or "vqa_semantic_packet_json_utf8",
            "payload_format": metadata.get("payload_format") or "json_utf8",
            "payload_kind": "vqa.semantic_packet",
            "payload_byte_count": byte_count,
            "payload_bit_count": bit_count,
            "sample_ids": [str(item.get("id") or index) for index, item in enumerate(examples)],
        }
        path = _write_json(ctx, "packet", packet)
        return OperationResult(
            outputs={"packet": artifact("vqa.semantic_packet.json", path, output_metadata)},
            metrics={},
            metadata={
                "codec": output_metadata["codec"],
                "byte_count": byte_count,
                "bit_count": bit_count,
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="local_python",
                    notes={
                        "runtime_execution_language": "Python + NumPy",
                        "payload_codec": "VQA semantic packet unpacked uint8 bits to JSON UTF-8 bytes",
                    },
                ),
            },
        )


class VqaAnswerFromPacketOperation(Operation):
    id = "foundation.vqa_answer_from_packet"
    name = "VQA answer from semantic packet"
    input_kinds = {"packet": ["vqa.semantic_packet.json"]}
    output_kinds = {"answers": "vqa.answers.json"}
    params_schema = object_schema(
        {
            "receiver": {
                "type": "string",
                "default": "selected_label",
                "enum": ["selected_label"],
            }
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        receiver = str(ctx.params.get("receiver") or "selected_label")
        if receiver != "selected_label":
            raise OperationError("VQA receiver `%s` is not configured; use selected_label or add a VQA model adapter." % receiver)
        packet = _load_json(ctx.require_input("packet").path)
        answers = []
        for item in packet.get("examples") or []:
            answers.append({"id": str(item.get("id") or len(answers)), "answer": str(item.get("selected_label") or "unknown")})
        payload = {
            "schema_version": 1,
            "kind": "vqa.answers",
            "examples": answers,
        }
        metadata = {"receiver": receiver, "example_count": len(answers), "answers_preview": answers}
        path = _write_json(ctx, "answers", payload)
        return OperationResult(
            outputs={"answers": artifact("vqa.answers.json", path, metadata)},
            metrics={"vqa.answer_count": len(answers)},
            metadata=metadata,
        )


class VqaTransformersAnswerOperation(Operation):
    id = "foundation.vqa_transformers_answer"
    name = "Pretrained Transformers VQA answerer"
    input_kinds = {
        "images": ["image.batch.numpy"],
        "questions": ["vqa.questions.json"],
    }
    output_kinds = {"answers": "vqa.answers.json"}
    params_schema = object_schema(
        {
            "model_id": {
                "type": "string",
                "default": DEFAULT_VQA_MODEL_ID,
                "description": "Hugging Face model id for a visual-question-answering pipeline. Practical defaults include dandelin/vilt-b32-finetuned-vqa and Salesforce/blip-vqa-base.",
            },
            "model_revision": {
                "type": "string",
                "default": "",
                "pattern": "^(|[0-9a-fA-F]{40})$",
                "description": "Full immutable Hugging Face commit. The built-in default model resolves to its pinned commit; other remote models require this explicitly.",
            },
            "device": {
                "type": "string",
                "default": "cpu",
                "description": "cpu or cuda device string. The operation uses transformers.pipeline device mapping.",
            },
            "top_k": {"type": "integer", "default": 1, "minimum": 1},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _transformers_vqa_availability()
        payload["model_source"] = {
            "provider": "Hugging Face",
            "default_model": DEFAULT_VQA_MODEL_ID,
            "default_revision": DEFAULT_VQA_MODEL_REVISION,
            "alternatives": ["Salesforce/blip-vqa-base"],
            "note": "Pretrained full-image VQA baseline. It is not a semantic channel codec; use it to produce/compare task answers before replacing it with a communication-aware adapter.",
        }
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        model_id = str(ctx.params.get("model_id") or DEFAULT_VQA_MODEL_ID)
        model_revision = _resolved_vqa_model_revision(
            model_id, ctx.params.get("model_revision")
        )
        device = str(ctx.params.get("device") or "cpu")
        top_k = max(1, int(ctx.params.get("top_k") or 1))
        image_artifact = ctx.require_input("images")
        images, image_metadata = _load_image_batch(
            image_artifact.path, image_artifact.metadata
        )
        questions = _load_vqa_questions(ctx.require_input("questions").path)
        _validate_vqa_image_question_identity(
            questions, image_metadata, int(images.shape[0])
        )
        ctx.report_progress("Loading pretrained VQA model", progress=0.0)
        backend = _load_vqa_backend(model_id, model_revision, device)
        answers: List[JsonDict] = []
        start = time.perf_counter()
        for index, item in enumerate(questions):
            ctx.report_progress("Answering %d/%d examples" % (index, len(questions)), progress=index / float(max(1, len(questions))))
            answer, score = _answer_vqa(backend, _pil_image(images[index]), str(item.get("question") or ""), top_k)
            answers.append(
                {
                    "id": str(item.get("id") or index),
                    "image_id": str(item["image_id"]),
                    "answer": answer,
                    "score": score,
                }
            )
        ctx.report_progress("Answering %d/%d examples" % (len(questions), len(questions)), progress=1.0)
        elapsed = time.perf_counter() - start
        payload = {
            "schema_version": 1,
            "kind": "vqa.answers",
            "model_id": model_id,
            "model_revision": model_revision,
            "examples": answers,
        }
        metadata = {
            "receiver": "transformers_vqa",
            "model_id": model_id,
            "model_revision": model_revision,
            "device": device,
            "example_count": len(answers),
            "answers_preview": answers,
            "runtime_execution_language": "Python + Transformers/PyTorch",
            "image_question_pairing": "exact ordered image_id match",
        }
        path = _write_json(ctx, "answers", payload)
        timing_records: List[JsonDict] = []
        per_example = elapsed / float(len(answers)) if answers else elapsed
        for index, answer in enumerate(answers):
            append_measurement(timing_records, index, "decoder.inference", per_example, model_id=model_id, sample_id=answer["id"])
        append_measurement(timing_records, None, "decoder.total", elapsed, model_id=model_id, example_count=len(answers))
        return OperationResult(
            outputs={"answers": artifact("vqa.answers.json", path, metadata)},
            metrics={
                "vqa.answer_count": len(answers),
                "codec_timing.decoder.inference_s": per_example,
                "codec_timing.decoder.total_s": elapsed,
            },
            metadata={
                **metadata,
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="pytorch",
                    notes={
                        "runtime_execution_language": "Python + Transformers/PyTorch",
                        "model_id": model_id,
                        "model_revision": model_revision,
                    },
                ),
            },
        )


def _select_answer_region(question: str) -> Tuple[str, List[int]]:
    text = question.lower()
    if "roadside" in text or "sensor" in text or "near" in text:
        return "vehicle", [2, 9, 9, 14]
    if "roof" in text or "mounted" in text:
        return "antenna", [9, 4, 14, 12]
    return "unknown", [0, 0, 0, 0]


def _load_image_batch(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    metadata = dict(fallback_metadata or {})
    with np.load(str(path), allow_pickle=False) as payload:
        if "images" not in payload.files:
            raise OperationError("VQA image artifact requires an images array")
        images = payload["images"].astype(np.uint8, copy=False)
        if "metadata_json" in payload.files:
            try:
                embedded = decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="VQA image metadata_json",
                )
            except Exception as exc:
                raise OperationError(
                    "VQA image metadata_json is malformed: %s" % exc
                ) from exc
            metadata.update(embedded)
    if images.ndim < 1:
        raise OperationError("VQA image artifact must have a batch dimension")
    return images, metadata


def _load_bits(path, metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    output_metadata = dict(metadata or {})
    with np.load(str(path), allow_pickle=False) as payload:
        bits = payload["bits"]
        if "metadata_json" in payload.files:
            try:
                embedded = decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="VQA bit metadata_json",
                )
            except Exception as exc:
                raise OperationError(
                    "VQA bit metadata_json is malformed: %s" % exc
                ) from exc
            output_metadata.update(embedded)
    canonical, boundary_metadata = validate_channel_bits(
        bits,
        label="VQA goal bitstream",
        backend="python_numpy",
    )
    output_metadata.update(boundary_metadata)
    return canonical, output_metadata


def _load_vqa_questions(path) -> List[JsonDict]:
    payload = _load_json(path)
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise OperationError("Expected JSON artifact with an examples list: %s" % path)
    output: List[JsonDict] = []
    question_ids = set()
    for index, item in enumerate(examples):
        if not isinstance(item, dict):
            raise OperationError("VQA question example %d must be an object" % index)
        question_id = str(item.get("id") or "").strip()
        image_id = str(item.get("image_id") or "").strip()
        question = item.get("question")
        if not question_id:
            raise OperationError("VQA question example %d requires a non-empty id" % index)
        if question_id in question_ids:
            raise OperationError("VQA question IDs must be unique; duplicate `%s`" % question_id)
        if not image_id:
            raise OperationError(
                "VQA question `%s` requires a non-empty image_id" % question_id
            )
        if not isinstance(question, str) or not question.strip():
            raise OperationError(
                "VQA question `%s` requires a non-empty question string" % question_id
            )
        question_ids.add(question_id)
        output.append(
            {
                **dict(item),
                "id": question_id,
                "image_id": image_id,
                "question": question,
            }
        )
    return output


def _validate_vqa_image_question_identity(
    questions: List[JsonDict],
    image_metadata: JsonDict,
    image_count: int,
) -> None:
    raw_image_ids = image_metadata.get("image_ids")
    if not isinstance(raw_image_ids, list):
        raise OperationError(
            "VQA image artifact requires image_ids metadata for identity-safe pairing"
        )
    image_ids = [str(value).strip() for value in raw_image_ids]
    if len(image_ids) != image_count or any(not value for value in image_ids):
        raise OperationError(
            "VQA image_ids metadata must contain one non-empty identity per image "
            "(%d IDs for %d images)" % (len(image_ids), image_count)
        )
    question_image_ids = [str(item["image_id"]) for item in questions]
    if len(question_image_ids) != image_count:
        raise OperationError(
            "VQA image/question count mismatch: %d images and %d questions"
            % (image_count, len(question_image_ids))
        )
    if question_image_ids != image_ids:
        mismatch = next(
            (
                index
                for index, (question_id, image_id) in enumerate(
                    zip(question_image_ids, image_ids)
                )
                if question_id != image_id
            ),
            0,
        )
        raise OperationError(
            "VQA image/question identity mismatch at index %d: question image_id `%s`, "
            "image artifact image_id `%s`"
            % (mismatch, question_image_ids[mismatch], image_ids[mismatch])
        )


def _pil_image(image: np.ndarray):
    try:
        from PIL import Image
    except Exception as exc:
        raise OperationError("Install Pillow to use the pretrained VQA adapter") from exc
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise OperationError("VQA image batch item must be HxWxC; got %s" % (list(image.shape),))
    if image.shape[-1] == 1:
        return Image.fromarray(image[:, :, 0], mode="L")
    return Image.fromarray(image)


def _load_vqa_backend(model_id: str, model_revision: str, device: str) -> JsonDict:
    try:
        import torch
        from transformers import AutoModelForVisualQuestionAnswering, AutoProcessor
    except Exception as exc:
        raise OperationError(
            'Install with `python -m pip install "noema-lab[foundation]"` in an '
            "installed environment, or `uv sync --extra foundation` in a source "
            "checkout, to use pretrained VQA adapters"
        ) from exc
    target = _torch_device(torch, device)
    loader_kwargs = {} if model_revision == "local_path" else {"revision": model_revision}
    try:
        processor = AutoProcessor.from_pretrained(model_id, **loader_kwargs)
        model = AutoModelForVisualQuestionAnswering.from_pretrained(model_id, **loader_kwargs)
        model.to(target)
        model.eval()
        return {"kind": "classification_vqa", "processor": processor, "model": model, "torch": torch, "device": target}
    except Exception as first_exc:
        try:
            from transformers import BlipForQuestionAnswering
            processor = AutoProcessor.from_pretrained(model_id, **loader_kwargs)
            model = BlipForQuestionAnswering.from_pretrained(model_id, **loader_kwargs)
            model.to(target)
            model.eval()
            return {"kind": "generative_blip_vqa", "processor": processor, "model": model, "torch": torch, "device": target}
        except Exception:
            raise OperationError("Could not load VQA model `%s`: %s" % (model_id, first_exc)) from first_exc


def _resolved_vqa_model_revision(model_id: str, value: Any) -> str:
    if Path(model_id).exists():
        return "local_path"
    revision = str(value or "").strip().lower()
    if not revision and model_id == DEFAULT_VQA_MODEL_ID:
        revision = DEFAULT_VQA_MODEL_REVISION
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise OperationError(
            "Remote VQA model `%s` requires model_revision as a full immutable "
            "40-character Hugging Face commit" % model_id
        )
    return revision


def _answer_vqa(backend: JsonDict, image, question: str, top_k: int) -> Tuple[str, float]:
    processor = backend["processor"]
    model = backend["model"]
    torch = backend["torch"]
    device = backend["device"]
    if backend["kind"] == "generative_blip_vqa":
        inputs = processor(image, question, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs)
        return str(processor.decode(output[0], skip_special_tokens=True)).strip(), 0.0
    inputs = processor(image, question, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
        k = min(max(1, int(top_k or 1)), int(probabilities.shape[-1]))
        values, indices = torch.topk(probabilities, k=k, dim=-1)
    index = int(indices[0, 0].detach().cpu().item())
    score = float(values[0, 0].detach().cpu().item())
    answer = str(model.config.id2label.get(index, index))
    return answer, score


def _torch_device(torch, requested: str):
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise OperationError("CUDA was requested for the pretrained VQA adapter, but no CUDA GPU is available")
    if requested and requested != "cpu" and not requested.startswith("cuda"):
        raise OperationError("Pretrained VQA adapter currently supports cpu or cuda device strings; got `%s`" % requested)
    return torch.device(requested or "cpu")


def _transformers_vqa_availability() -> JsonDict:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        return {
            "available": False,
            "extra": "foundation",
            "reason": (
                'Install with `python -m pip install "noema-lab[foundation]"` in an '
                "installed environment, or `uv sync --extra foundation` in a source "
                "checkout, to use pretrained VQA adapters: %s" % exc
            ),
        }
    return {"available": True, "extra": "foundation"}


def _load_json(path) -> JsonDict:
    try:
        payload = decode_strict_json(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise OperationError(
            "Could not decode JSON artifact %s: %s" % (path, exc)
        ) from exc
    if not isinstance(payload, dict):
        raise OperationError("Expected JSON object artifact: %s" % path)
    return payload


def _write_json(ctx: OperationContext, name: str, payload: JsonDict):
    path = ctx.output_path(name, ".json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
