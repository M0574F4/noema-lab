from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from noema_lab.core.structured_input import load_strict_yaml_or_json

JsonDict = Dict[str, Any]


@dataclass
class SuiteBenchmarkPack:
    id: str
    path: str
    status: str
    task: str

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "path": self.path,
            "status": self.status,
            "task": self.task,
        }


@dataclass
class SuiteDefinition:
    id: str
    name: str
    status: str
    version: str
    summary: str = ""
    docs: JsonDict = field(default_factory=dict)
    supported_tasks: List[str] = field(default_factory=list)
    planned_tasks: List[str] = field(default_factory=list)
    benchmark_packs: List[SuiteBenchmarkPack] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "version": self.version,
            "summary": self.summary,
            "docs": dict(self.docs),
            "supported_tasks": list(self.supported_tasks),
            "planned_tasks": list(self.planned_tasks),
            "benchmark_packs": [item.to_dict() for item in self.benchmark_packs],
        }


@dataclass
class SuitesCatalog:
    suites: Dict[str, SuiteDefinition]
    schema_version: int = 1

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "suites": [suite.to_dict() for suite in self.suites.values()],
        }

    def suite(self, suite_id: str) -> Optional[SuiteDefinition]:
        return self.suites.get(str(suite_id))

    def active_suites(self) -> List[SuiteDefinition]:
        return [suite for suite in self.suites.values() if suite.status == "active"]


@lru_cache(maxsize=1)
def load_suites_catalog() -> SuitesCatalog:
    path = Path(__file__).resolve().parent.parent / "suites_catalog.yaml"
    data = load_strict_yaml_or_json(path)
    if not isinstance(data, Mapping):
        raise ValueError("suites catalog must contain a mapping")
    return SuitesCatalog(
        schema_version=int(data.get("schema_version") or 1),
        suites=_suites_from_list(data.get("suites") or []),
    )


def _suites_from_list(value: Any) -> Dict[str, SuiteDefinition]:
    output: Dict[str, SuiteDefinition] = {}
    for index, item in enumerate(_list(value, "suites")):
        data = _mapping(item, "suites[%d]" % index)
        suite = SuiteDefinition(
            id=_string(data, "id", "suites[%d]" % index),
            name=str(data.get("name") or data.get("id")),
            status=_status(data.get("status") or "planned", "suites[%d].status" % index),
            version=str(data.get("version") or "v1"),
            summary=str(data.get("summary") or ""),
            docs=dict(_mapping(data.get("docs") or {}, "suites[%d].docs" % index)),
            supported_tasks=[str(task) for task in data.get("supported_tasks") or []],
            planned_tasks=[str(task) for task in data.get("planned_tasks") or []],
            benchmark_packs=_benchmark_packs_from_list(
                data.get("benchmark_packs") or [], "suites[%d].benchmark_packs" % index
            ),
        )
        if suite.id in output:
            raise ValueError("duplicate suite id: %s" % suite.id)
        output[suite.id] = suite
    return output


def _benchmark_packs_from_list(value: Any, label: str) -> List[SuiteBenchmarkPack]:
    packs: List[SuiteBenchmarkPack] = []
    seen_ids = set()
    for index, item in enumerate(_list(value, label)):
        data = _mapping(item, "%s[%d]" % (label, index))
        pack = SuiteBenchmarkPack(
            id=_string(data, "id", "%s[%d]" % (label, index)),
            path=_string(data, "path", "%s[%d]" % (label, index)),
            status=str(data.get("status") or "experimental"),
            task=_string(data, "task", "%s[%d]" % (label, index)),
        )
        if pack.id in seen_ids:
            raise ValueError("duplicate benchmark pack id in %s: %s" % (label, pack.id))
        seen_ids.add(pack.id)
        packs.append(pack)
    return packs


def _status(value: Any, label: str) -> str:
    status = str(value)
    allowed = {"active", "experimental", "planned"}
    if status not in allowed:
        raise ValueError("%s must be one of %s" % (label, sorted(allowed)))
    return status


def _list(value: Any, label: str) -> List[Any]:
    if not isinstance(value, list):
        raise ValueError("%s must be a list" % label)
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be a mapping" % label)
    return value


def _string(data: Mapping[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError("%s requires non-empty string `%s`" % (label, key))
    return value
