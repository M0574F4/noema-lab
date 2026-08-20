from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from noema_lab.core.structured_input import decode_strict_json

JsonDict = Dict[str, Any]

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "near",
    "of",
    "on",
    "or",
    "over",
    "should",
    "the",
    "to",
    "while",
    "with",
}


@dataclass
class KnowledgeBase:
    kb_id: str
    facts: List[JsonDict] = field(default_factory=list)
    schema_version: int = 1

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "kind": "knowledge_base",
            "id": self.kb_id,
            "facts": [dict(fact) for fact in self.facts],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgeBase":
        if data.get("kind") != "knowledge_base":
            raise ValueError("Expected knowledge_base JSON")
        facts = data.get("facts")
        if not isinstance(facts, list):
            raise ValueError("knowledge_base requires a facts list")
        normalized_facts: List[JsonDict] = []
        for index, item in enumerate(facts):
            if not isinstance(item, Mapping):
                raise ValueError(
                    "knowledge_base facts item %d must be an object" % index
                )
            normalized_facts.append(dict(item))
        return cls(
            kb_id=str(data.get("id") or "knowledge_base"),
            facts=normalized_facts,
        )


@dataclass
class SemanticState:
    modality: str
    states: List[JsonDict] = field(default_factory=list)
    state_id: str = "semantic_state"
    schema_version: int = 1

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "kind": "semantic.state",
            "id": self.state_id,
            "modality": self.modality,
            "states": [dict(item) for item in self.states],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticState":
        if data.get("kind") != "semantic.state":
            raise ValueError("Expected semantic.state JSON")
        states = data.get("states")
        if not isinstance(states, list):
            raise ValueError("semantic.state requires a states list")
        normalized_states: List[JsonDict] = []
        for index, item in enumerate(states):
            if not isinstance(item, Mapping):
                raise ValueError(
                    "semantic.state states item %d must be an object" % index
                )
            normalized_states.append(dict(item))
        return cls(
            modality=str(data.get("modality") or "unknown"),
            states=normalized_states,
            state_id=str(data.get("id") or "semantic_state"),
        )


def load_json(path: Path) -> JsonDict:
    payload = decode_strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Semantic JSON artifact must contain an object: %s" % path)
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_knowledge_base(path: Path) -> KnowledgeBase:
    return KnowledgeBase.from_dict(load_json(path))


def load_semantic_state(path: Path) -> SemanticState:
    return SemanticState.from_dict(load_json(path))


def normalize_token(token: str) -> str:
    return token.strip().lower()


def tokenize(text: str) -> List[str]:
    return [normalize_token(item) for item in TOKEN_RE.findall(text or "") if item.strip()]


def concepts_from_text(text: str, max_concepts: int = 16) -> List[str]:
    counts = Counter(token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1)
    return [token for token, _count in counts.most_common(max(1, int(max_concepts)))]


def fact_key(fact: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        normalize_token(str(fact.get("subject") or "")),
        normalize_token(str(fact.get("predicate") or "")),
        normalize_token(str(fact.get("object") or "")),
    )


def concept_set(state_item: Mapping[str, Any]) -> Set[str]:
    return {normalize_token(str(item)) for item in state_item.get("concepts") or [] if str(item).strip()}


def entity_set(state_item: Mapping[str, Any]) -> Set[str]:
    entities = state_item.get("entities") or []
    output: Set[str] = set()
    for item in entities:
        if isinstance(item, Mapping):
            value = item.get("text") or item.get("id") or item.get("name")
        else:
            value = item
        if str(value).strip():
            output.add(normalize_token(str(value)))
    return output


def fact_set(state_item: Mapping[str, Any]) -> Set[Tuple[str, str, str]]:
    facts = state_item.get("facts")
    if facts is None:
        return set()
    if not isinstance(facts, list):
        raise ValueError("semantic state facts must be a list")
    normalized: Set[Tuple[str, str, str]] = set()
    for index, fact in enumerate(facts):
        if not isinstance(fact, Mapping):
            raise ValueError(
                "semantic state facts item %d must be an object" % index
            )
        normalized.add(fact_key(fact))
    return normalized


def kb_fact_set(kb: KnowledgeBase) -> Set[Tuple[str, str, str]]:
    return {fact_key(fact) for fact in kb.facts}


def overlap_scores(reference: Set[Any], candidate: Set[Any]) -> JsonDict:
    if not reference and not candidate:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "overlap": 0, "reference_count": 0, "candidate_count": 0}
    overlap = len(reference & candidate)
    precision = float(overlap) / float(len(candidate)) if candidate else 0.0
    recall = float(overlap) / float(len(reference)) if reference else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap": overlap,
        "reference_count": len(reference),
        "candidate_count": len(candidate),
    }


def state_by_id(state: SemanticState) -> Dict[str, JsonDict]:
    output: Dict[str, JsonDict] = {}
    for index, item in enumerate(state.states):
        raw_id = item.get("id")
        if raw_id is None or not str(raw_id):
            raise ValueError(
                "semantic.state item %d requires a non-empty id for paired metrics" % index
            )
        item_id = str(raw_id)
        if item_id in output:
            raise ValueError("semantic.state item IDs must be unique; duplicate %r" % item_id)
        output[item_id] = item
    return output


def ensure_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): ensure_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
