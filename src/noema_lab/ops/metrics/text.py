from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema
from noema_lab.core.structured_input import decode_strict_json

JsonDict = Dict[str, Any]
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TextSemanticSimilarityMetricsOperation(Operation):
    id = "metrics.text_semantic_similarity"
    name = "Text lexical similarity metrics"
    input_kinds = {
        "reference": ["text.batch.json"],
        "candidate": ["text.batch.json"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        reference = _load_text_batch(ctx.require_input("reference").path)
        candidate = _load_text_batch(ctx.require_input("candidate").path)
        ref_examples = _examples(reference)
        cand_examples = _examples(candidate)
        pairs = _strict_text_pairs(ref_examples, cand_examples)
        count = len(pairs)
        rows = []
        exact_total = 0.0
        edit_total = 0.0
        token_f1_total = 0.0
        unigram_proxy_total = 0.0
        for index, (example_id, ref_example, cand_example) in enumerate(pairs):
            ref = _required_text(ref_example, "reference", index)
            cand = _required_text(cand_example, "candidate", index)
            exact = 1.0 if ref == cand else 0.0
            edit_similarity = _edit_similarity(ref, cand)
            token_f1 = _token_f1(ref, cand)
            unigram_proxy = _unigram_bleu_proxy(ref, cand)
            exact_total += exact
            edit_total += edit_similarity
            token_f1_total += token_f1
            unigram_proxy_total += unigram_proxy
            rows.append(
                {
                    "index": index,
                    "id": example_id,
                    "exact_match": exact,
                    "edit_similarity": edit_similarity,
                    "token_f1": token_f1,
                    "unigram_bleu_proxy": unigram_proxy,
                }
            )
        metrics = {
            "text.exact_match": exact_total / float(count),
            "text.edit_similarity": edit_total / float(count),
            "semantic.lexical_similarity": token_f1_total / float(count),
            "text.unigram_bleu_proxy": unigram_proxy_total / float(count),
        }
        report = {
            "schema_version": 2,
            "metric_family": "text_lexical_similarity",
            "num_texts": count,
            "sample_pairing": "declared_id_and_order",
            "metric_definitions": {
                "text.exact_match": (
                    "Mean literal Unicode string equality, including case, punctuation, and whitespace."
                ),
                "text.unigram_bleu_proxy": (
                    "Mean sentence-level clipped unigram precision with BLEU brevity penalty; "
                    "not corpus BLEU."
                ),
                "semantic.lexical_similarity": (
                    "Mean lowercased bag-of-word token F1 lexical-overlap proxy; "
                    "not an embedding-based semantic metric."
                ),
            },
            "metrics": metrics,
            "per_example": rows,
        }
        path = ctx.output_path("report", ".json")
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={"report": artifact("metrics.report", path, report)},
            metrics=metrics,
            metadata={"num_texts": count},
        )


def _load_text_batch(path) -> JsonDict:
    data = decode_strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("kind") != "text.batch":
        raise RuntimeError("Expected text.batch JSON artifact: %s" % path)
    return data


def _examples(batch: JsonDict) -> List[JsonDict]:
    examples = batch.get("examples")
    if not isinstance(examples, list):
        raise RuntimeError("text.batch artifact requires an examples list")
    if any(not isinstance(item, dict) for item in examples):
        raise RuntimeError("text.batch examples must all be objects")
    return [dict(item) for item in examples]


def _strict_text_pairs(
    reference: List[JsonDict], candidate: List[JsonDict]
) -> List[Tuple[str, JsonDict, JsonDict]]:
    if len(reference) != len(candidate):
        raise RuntimeError(
            "Text example count mismatch: reference has %d examples, candidate has %d"
            % (len(reference), len(candidate))
        )
    if not reference:
        raise RuntimeError("Text metrics require at least one paired example")
    reference_ids = _strict_ids(reference, "reference")
    candidate_ids = _strict_ids(candidate, "candidate")
    if reference_ids != candidate_ids:
        raise RuntimeError(
            "Text sample ID/order mismatch: reference IDs %s, candidate IDs %s"
            % (reference_ids, candidate_ids)
        )
    return [
        (reference_ids[index], reference[index], candidate[index])
        for index in range(len(reference))
    ]


def _strict_ids(examples: List[JsonDict], label: str) -> List[str]:
    ids = []
    for index, item in enumerate(examples):
        if item.get("id") is None or not str(item.get("id")):
            raise RuntimeError("%s text example %d requires a non-empty id" % (label, index))
        ids.append(str(item["id"]))
    if len(set(ids)) != len(ids):
        raise RuntimeError("%s text example IDs must be unique" % label)
    return ids


def _required_text(example: JsonDict, label: str, index: int) -> str:
    if "text" not in example or example.get("text") is None:
        raise RuntimeError("%s text example %d requires a text value" % (label, index))
    value = example["text"]
    if not isinstance(value, str):
        raise RuntimeError("%s text example %d text value must be a string" % (label, index))
    return value


def _word_tokens(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if re.match(r"^\w+$", token, re.UNICODE)]


def _token_f1(reference: str, candidate: str) -> float:
    ref = Counter(_word_tokens(reference))
    cand = Counter(_word_tokens(candidate))
    if not ref and not cand:
        return 1.0
    if not ref or not cand:
        return 0.0
    overlap = sum((ref & cand).values())
    precision = float(overlap) / float(sum(cand.values())) if cand else 0.0
    recall = float(overlap) / float(sum(ref.values())) if ref else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _unigram_bleu_proxy(reference: str, candidate: str) -> float:
    ref = Counter(_word_tokens(reference))
    cand_tokens = _word_tokens(candidate)
    if not cand_tokens:
        return 1.0 if not ref else 0.0
    cand = Counter(cand_tokens)
    clipped = sum((cand & ref).values())
    precision = float(clipped) / float(len(cand_tokens))
    if precision <= 0.0:
        return 0.0
    ref_len = sum(ref.values())
    cand_len = len(cand_tokens)
    brevity = 1.0 if cand_len > ref_len else math.exp(1.0 - float(ref_len) / float(max(cand_len, 1)))
    return brevity * precision


def _edit_similarity(reference: str, candidate: str) -> float:
    if reference == candidate:
        return 1.0
    max_len = max(len(reference), len(candidate))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(reference, candidate)
    return max(0.0, 1.0 - float(distance) / float(max_len))


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            insert = current[right_index - 1] + 1
            delete = previous[right_index] + 1
            replace = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insert, delete, replace))
        previous = current
    return int(previous[-1])
