from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from noema_lab.core.structured_input import load_strict_yaml_or_json

JsonDict = Dict[str, Any]


@dataclass
class ResearchAreaDefinition:
    id: str
    name: str
    order: int
    description: str = ""

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "name": self.name,
            "order": self.order,
            "description": self.description,
        }


@dataclass
class DatasetDefinition:
    id: str
    name: str
    modality: str
    status: str = "supported"
    source_ops: List[str] = field(default_factory=list)
    versions: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "name": self.name,
            "modality": self.modality,
            "status": self.status,
            "source_ops": list(self.source_ops),
            "versions": list(self.versions),
            "description": self.description,
        }


@dataclass
class TaskDefinition:
    id: str
    name: str
    area_id: str
    kind: str
    modality: str
    status: str = "supported"
    target: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    optional_metrics: List[str] = field(default_factory=list)
    required_artifacts: JsonDict = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "id": self.id,
            "name": self.name,
            "area_id": self.area_id,
            "kind": self.kind,
            "modality": self.modality,
            "status": self.status,
            "metrics": list(self.metrics),
            "optional_metrics": list(self.optional_metrics),
            "required_artifacts": dict(self.required_artifacts),
            "description": self.description,
        }
        if self.target:
            payload["target"] = self.target
        return payload


@dataclass
class MetricDefinition:
    id: str
    name: str
    family: str
    unit: str
    direction: str = "neutral"
    reduction: str = "mean"
    status: str = "supported"
    description: str = ""

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "unit": self.unit,
            "direction": self.direction,
            "reduction": self.reduction,
            "status": self.status,
            "description": self.description,
        }


@dataclass
class ResearchCatalog:
    research_areas: Dict[str, ResearchAreaDefinition]
    datasets: Dict[str, DatasetDefinition]
    tasks: Dict[str, TaskDefinition]
    metrics: Dict[str, MetricDefinition]
    schema_version: int = 1

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "research_areas": [
                item.to_dict()
                for item in sorted(
                    self.research_areas.values(),
                    key=lambda area: (area.order, area.id),
                )
            ],
            "datasets": [item.to_dict() for item in self.datasets.values()],
            "tasks": [item.to_dict() for item in self.tasks.values()],
            "metrics": [item.to_dict() for item in self.metrics.values()],
        }

    def dataset(self, dataset_id: str) -> Optional[DatasetDefinition]:
        return self.datasets.get(str(dataset_id))

    def research_area(self, area_id: str) -> Optional[ResearchAreaDefinition]:
        return self.research_areas.get(str(area_id))

    def task(self, task_id: str) -> Optional[TaskDefinition]:
        return self.tasks.get(str(task_id))

    def metric(self, metric_id: str) -> Optional[MetricDefinition]:
        return self.metrics.get(str(metric_id))


@lru_cache(maxsize=1)
def load_research_catalog() -> ResearchCatalog:
    path = Path(__file__).resolve().parent.parent / "research_catalog.yaml"
    data = load_strict_yaml_or_json(path)
    if not isinstance(data, Mapping):
        raise ValueError("research catalog must contain a mapping")
    research_areas = _research_areas_from_list(data.get("research_areas") or [])
    tasks = _tasks_from_list(data.get("tasks") or [])
    unknown_area_refs = sorted(
        {
            task.area_id
            for task in tasks.values()
            if task.area_id not in research_areas
        }
    )
    if unknown_area_refs:
        raise ValueError(
            "research catalog tasks reference unknown research areas: %s"
            % ", ".join(unknown_area_refs)
        )
    return ResearchCatalog(
        schema_version=int(data.get("schema_version") or 1),
        research_areas=research_areas,
        datasets=_datasets_from_list(data.get("datasets") or []),
        tasks=tasks,
        metrics=_metrics_from_list(data.get("metrics") or []),
    )


def validate_research_specs_against_catalog(
    specs: JsonDict,
    catalog: Optional[ResearchCatalog] = None,
    strict: bool = False,
) -> JsonDict:
    catalog = catalog or load_research_catalog()
    errors: List[str] = []
    warnings: List[str] = []
    dataset = _mapping_or_empty(specs.get("dataset"))
    task = _mapping_or_empty(specs.get("task"))
    metric_ids = _metric_ids(specs.get("metrics"))
    if not metric_ids and task.get("metrics"):
        metric_ids = [str(item) for item in task.get("metrics") or []]

    dataset_def = None
    dataset_id = str(dataset.get("id") or "")
    if dataset_id:
        dataset_def = catalog.dataset(dataset_id)
        if not dataset_def:
            _add_issue(warnings, errors, strict, "Dataset `%s` is not in the research catalog." % dataset_id)
        elif dataset.get("modality") and dataset_def.modality != dataset.get("modality"):
            errors.append(
                "Dataset `%s` has modality `%s`, expected `%s`."
                % (dataset_id, dataset.get("modality"), dataset_def.modality)
            )

    task_def = None
    task_id = str(task.get("id") or "")
    if task_id:
        task_def = catalog.task(task_id)
        if not task_def:
            _add_issue(warnings, errors, strict, "Task `%s` is not in the research catalog." % task_id)
        else:
            if task.get("kind") and task_def.kind != task.get("kind"):
                errors.append(
                    "Task `%s` has kind `%s`, expected `%s`."
                    % (task_id, task.get("kind"), task_def.kind)
                )
            if task.get("modality") and task_def.modality != task.get("modality"):
                errors.append(
                    "Task `%s` has modality `%s`, expected `%s`."
                    % (task_id, task.get("modality"), task_def.modality)
                )
            if task_def.status != "supported":
                warnings.append("Task `%s` is cataloged as `%s`." % (task_id, task_def.status))

    if dataset_def and task_def and dataset_def.modality != task_def.modality and task_def.modality != "multimodal":
        errors.append(
            "Dataset `%s` modality `%s` does not match task `%s` modality `%s`."
            % (dataset_def.id, dataset_def.modality, task_def.id, task_def.modality)
        )

    allowed_metrics = set()
    if task_def:
        allowed_metrics.update(task_def.metrics)
        allowed_metrics.update(task_def.optional_metrics)
    for metric_id in metric_ids:
        metric_def = catalog.metric(metric_id)
        if not metric_def:
            _add_issue(warnings, errors, strict, "Metric `%s` is not in the research catalog." % metric_id)
            continue
        if metric_def.status != "supported":
            _add_issue(
                warnings,
                errors,
                strict,
                "Metric `%s` is cataloged as `%s`." % (metric_id, metric_def.status),
            )
        if task_def and metric_id not in allowed_metrics:
            _add_issue(
                warnings,
                errors,
                strict,
                "Metric `%s` is not registered for task `%s`." % (metric_id, task_def.id),
            )

    return {
        "schema_version": 1,
        "status": "invalid" if errors else "valid",
        "errors": errors,
        "warnings": warnings,
        "dataset": dataset_def.to_dict() if dataset_def else None,
        "task": task_def.to_dict() if task_def else None,
        "metrics": [
            catalog.metric(metric_id).to_dict()
            for metric_id in metric_ids
            if catalog.metric(metric_id) is not None
        ],
    }


def _datasets_from_list(value: Any) -> Dict[str, DatasetDefinition]:
    output: Dict[str, DatasetDefinition] = {}
    for index, item in enumerate(_list(value, "datasets")):
        data = _mapping(item, "datasets[%d]" % index)
        definition = DatasetDefinition(
            id=_string(data, "id", "datasets[%d]" % index),
            name=str(data.get("name") or data.get("id")),
            modality=_string(data, "modality", "datasets[%d]" % index),
            status=str(data.get("status") or "supported"),
            source_ops=[str(op) for op in data.get("source_ops") or []],
            versions=[str(version) for version in data.get("versions") or []],
            description=str(data.get("description") or ""),
        )
        if definition.id in output:
            raise ValueError("duplicate dataset id: %s" % definition.id)
        output[definition.id] = definition
    return output


def _research_areas_from_list(value: Any) -> Dict[str, ResearchAreaDefinition]:
    output: Dict[str, ResearchAreaDefinition] = {}
    for index, item in enumerate(_list(value, "research_areas")):
        label = "research_areas[%d]" % index
        data = _mapping(item, label)
        definition = ResearchAreaDefinition(
            id=_string(data, "id", label),
            name=str(data.get("name") or data.get("id")),
            order=_integer(data, "order", label),
            description=str(data.get("description") or ""),
        )
        if definition.id in output:
            raise ValueError("duplicate research area id: %s" % definition.id)
        output[definition.id] = definition
    return output


def _tasks_from_list(value: Any) -> Dict[str, TaskDefinition]:
    output: Dict[str, TaskDefinition] = {}
    for index, item in enumerate(_list(value, "tasks")):
        data = _mapping(item, "tasks[%d]" % index)
        definition = TaskDefinition(
            id=_string(data, "id", "tasks[%d]" % index),
            name=str(data.get("name") or data.get("id")),
            area_id=_string(data, "area_id", "tasks[%d]" % index),
            kind=_string(data, "kind", "tasks[%d]" % index),
            modality=_string(data, "modality", "tasks[%d]" % index),
            status=str(data.get("status") or "supported"),
            target=str(data["target"]) if data.get("target") is not None else None,
            metrics=[str(metric) for metric in data.get("metrics") or []],
            optional_metrics=[str(metric) for metric in data.get("optional_metrics") or []],
            required_artifacts=dict(data.get("required_artifacts") or {}),
            description=str(data.get("description") or ""),
        )
        if definition.id in output:
            raise ValueError("duplicate task id: %s" % definition.id)
        output[definition.id] = definition
    return output


def _metrics_from_list(value: Any) -> Dict[str, MetricDefinition]:
    output: Dict[str, MetricDefinition] = {}
    for index, item in enumerate(_list(value, "metrics")):
        data = _mapping(item, "metrics[%d]" % index)
        definition = MetricDefinition(
            id=_string(data, "id", "metrics[%d]" % index),
            name=str(data.get("name") or data.get("id")),
            family=str(data.get("family") or "custom"),
            unit=str(data.get("unit") or ""),
            direction=str(data.get("direction") or "neutral"),
            reduction=str(data.get("reduction") or "mean"),
            status=str(data.get("status") or "supported"),
            description=str(data.get("description") or ""),
        )
        if definition.id in output:
            raise ValueError("duplicate metric id: %s" % definition.id)
        output[definition.id] = definition
    return output


def _metric_ids(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        if isinstance(item, str):
            output.append(item)
        elif isinstance(item, Mapping) and item.get("id"):
            output.append(str(item["id"]))
    return output


def _mapping_or_empty(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any, label: str) -> List[Any]:
    if not isinstance(value, list):
        raise ValueError("%s must be a list" % label)
    return value


def _mapping(value: Any, label: str) -> JsonDict:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be a mapping" % label)
    return dict(value)


def _string(data: JsonDict, key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError("%s.%s must be a non-empty string" % (label, key))
    return value


def _integer(data: JsonDict, key: str, label: str) -> int:
    value = data.get(key)
    if type(value) is not int:
        raise ValueError("%s.%s must be an integer" % (label, key))
    return value


def _add_issue(warnings: List[str], errors: List[str], strict: bool, message: str) -> None:
    if strict:
        errors.append(message)
    else:
        warnings.append(message)
