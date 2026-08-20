from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.structured_input import decode_strict_json
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema

JsonDict = Dict[str, Any]
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class SourceTextPerturbationOperation(Operation):
    id = "noise.source_text_perturbation"
    name = "Source semantic perturbation for text"
    input_kinds = {"texts": ["text.batch.json"]}
    output_kinds = {"texts": "text.batch.json"}
    params_schema = object_schema(
        {
            "mode": {
                "type": "string",
                "default": "none",
                "enum": ["none", "drop_words", "mask_words", "shuffle_words"],
            },
            "probability": {"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 1.0},
            "mask_token": {"type": "string", "default": "[MASK]"},
            "seed": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        batch = _load_text_batch(ctx.require_input("texts").path)
        mode = str(ctx.params.get("mode") or "none")
        probability = float(ctx.params.get("probability") or 0.0)
        mask_token = str(ctx.params.get("mask_token") or "[MASK]")
        rng = np.random.default_rng(ctx.seed("source_text_perturbation"))
        examples = []
        changed = 0
        token_total = 0
        token_changed = 0
        source_examples = batch.get("examples") or []
        total = max(len(source_examples), 1)
        for index, example in enumerate(source_examples):
            text = str(example.get("text") or "")
            noisy, stats = _apply_noise(text, mode, probability, mask_token, rng)
            item = dict(example)
            item["text"] = noisy
            item["source_perturbation"] = {"mode": mode, "probability": probability}
            examples.append(item)
            changed += int(noisy != text)
            token_total += stats["token_count"]
            token_changed += stats["changed_count"]
            _report_text_noise_progress(ctx, "Noised %s" % str(example.get("id") or index), index + 1, total)
        payload = dict(batch)
        payload["examples"] = examples
        payload["source_perturbation"] = {
            "mode": mode,
            "probability": probability,
            "changed_text_count": changed,
            "changed_token_count": token_changed,
        }
        metadata = {
            "dataset": batch.get("dataset"),
            "text_count": len(examples),
            "texts_preview": [{"id": example.get("id"), "text": example.get("text")} for example in examples],
            "source_perturbation_mode": mode,
            "probability": probability,
            "changed_text_count": changed,
            "changed_token_count": token_changed,
            "token_count": token_total,
        }
        path = ctx.output_path("texts", ".json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={"texts": artifact("text.batch.json", path, metadata)},
            metrics={
                "source_perturbation.text.changed_fraction": float(changed) / float(len(examples)) if examples else 0.0,
                "source_perturbation.text.changed_token_fraction": float(token_changed) / float(token_total) if token_total else 0.0,
            },
            metadata=metadata,
        )


def _apply_noise(text: str, mode: str, probability: float, mask_token: str, rng) -> tuple[str, JsonDict]:
    if mode == "none" or probability <= 0.0:
        tokens = _tokens(text)
        return text, {"token_count": len(tokens), "changed_count": 0}
    tokens = _tokens(text)
    if not tokens:
        return text, {"token_count": 0, "changed_count": 0}
    changed = 0
    if mode == "drop_words":
        kept = []
        for token in tokens:
            if _is_word(token) and rng.random() < probability:
                changed += 1
                continue
            kept.append(token)
        return _join_tokens(kept), {"token_count": len(tokens), "changed_count": changed}
    if mode == "mask_words":
        output = []
        for token in tokens:
            if _is_word(token) and rng.random() < probability:
                output.append(mask_token)
                changed += 1
            else:
                output.append(token)
        return _join_tokens(output), {"token_count": len(tokens), "changed_count": changed}
    if mode == "shuffle_words":
        words = [index for index, token in enumerate(tokens) if _is_word(token) and rng.random() < probability]
        shuffled = [tokens[index] for index in words]
        rng.shuffle(shuffled)
        output = list(tokens)
        for output_index, token in zip(words, shuffled):
            if output[output_index] != token:
                changed += 1
            output[output_index] = token
        return _join_tokens(output), {"token_count": len(tokens), "changed_count": changed}
    raise RuntimeError("Unsupported source text perturbation mode: %s" % mode)


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def _is_word(token: str) -> bool:
    return bool(re.match(r"^\w+$", token, re.UNICODE))


def _join_tokens(tokens: List[str]) -> str:
    text = ""
    for token in tokens:
        if not text:
            text = token
        elif re.match(r"^[^\w\s]$", token, re.UNICODE):
            text += token
        else:
            text += " " + token
    return text


def _load_text_batch(path) -> JsonDict:
    data = decode_strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("kind") != "text.batch":
        raise RuntimeError("Expected text.batch JSON artifact: %s" % path)
    if not isinstance(data.get("examples"), list):
        raise RuntimeError("text.batch artifact requires an examples list")
    return data


def _report_text_noise_progress(ctx: OperationContext, message: str, completed: int, total: int) -> None:
    total = max(int(total), 1)
    completed = max(0, min(int(completed), total))
    ctx.report_progress(
        message,
        phase="source_perturbation",
        status="running",
        completed=completed,
        total=total,
        percent=float(completed) / float(total) * 100.0,
        unit="texts",
        op=ctx.step_id,
    )
