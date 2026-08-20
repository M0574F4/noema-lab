from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional

from noema_lab.core.operations import (
    BACKEND_ALIASES,
    BACKEND_NAMES,
    MATERIALIZATION_RUNNERS,
    Operation,
    OperationError,
    OperationRegistry,
    normalize_backends,
    normalize_materializations,
    _normalize_runner_name,
)

JsonDict = Dict[str, Any]


def normalize_materialization_backend(value: Any, field_name: str = "backend") -> str:
    backend = str(value or "").strip().lower().replace("-", "_")
    backend = BACKEND_ALIASES.get(backend, backend)
    if backend not in BACKEND_NAMES:
        raise OperationError(
            "%s must be one of %s" % (field_name, ", ".join(sorted(BACKEND_NAMES)))
        )
    return backend


def normalize_materialization_runner(value: Any, field_name: str = "runner") -> str:
    runner = _normalize_runner_name(value)
    if runner not in MATERIALIZATION_RUNNERS:
        raise OperationError(
            "%s must be one of %s" % (field_name, ", ".join(sorted(MATERIALIZATION_RUNNERS)))
        )
    return runner


@dataclass(frozen=True)
class MaterializationSpec:
    operation_id: str
    runner: str
    backend: str
    implementation: str = "default"
    status: str = "implemented"
    notes: str = ""
    parameter_bindings: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @classmethod
    def from_mapping(cls, operation_id: str, payload: Mapping[str, Any]) -> "MaterializationSpec":
        runner = normalize_materialization_runner(payload.get("runner"), "materialization.runner")
        backend = normalize_materialization_backend(payload.get("backend"), "materialization.backend")
        notes = payload.get("notes")
        return cls(
            operation_id=str(operation_id),
            runner=runner,
            backend=backend,
            implementation=str(payload.get("implementation") or "default"),
            status=str(payload.get("status") or "implemented"),
            notes=str(notes) if notes is not None and str(notes).strip() else "",
            parameter_bindings=_freeze_mapping(
                payload.get("parameter_bindings")
                if isinstance(payload.get("parameter_bindings"), Mapping)
                else {}
            ),
        )

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "operation_id": self.operation_id,
            "runner": self.runner,
            "backend": self.backend,
            "implementation": self.implementation,
            "status": self.status,
        }
        if self.notes:
            payload["notes"] = self.notes
        if self.parameter_bindings:
            payload["parameter_bindings"] = _thaw_json(self.parameter_bindings)
        return payload


@dataclass(frozen=True)
class ResolvedMaterialization:
    operation: Operation
    spec: MaterializationSpec

    @property
    def operation_id(self) -> str:
        return self.spec.operation_id

    @property
    def runner(self) -> str:
        return self.spec.runner

    @property
    def backend(self) -> str:
        return self.spec.backend

    @property
    def implementation(self) -> str:
        return self.spec.implementation

    @property
    def status(self) -> str:
        return self.spec.status

    def to_dict(self) -> JsonDict:
        payload = self.spec.to_dict()
        payload["operation_name"] = self.operation.name
        return payload


class MaterializationRegistry:
    """Resolve stable operation IDs into runner/backend materializations.

    Recipes continue to refer to operation IDs such as ``wireless.channel``. A runner asks this
    registry for the compatible materialization it should use for its execution purpose and backend.
    """

    def __init__(self) -> None:
        self._operations: Dict[str, Operation] = {}
        self._materializations: Dict[str, List[MaterializationSpec]] = {}

    @classmethod
    def from_operation_registry(cls, registry: OperationRegistry) -> "MaterializationRegistry":
        materializations = cls()
        for operation in registry.list():
            materializations.register_operation(operation)
        return materializations

    def register_operation(self, operation: Operation) -> None:
        if operation.id in self._operations:
            raise OperationError("Operation already registered in materialization registry: %s" % operation.id)
        backends = normalize_backends(getattr(operation, "backends", None), operation.id)
        raw_materializations = normalize_materializations(
            getattr(operation, "materializations", None),
            backends,
            operation.id,
        )
        self._operations[operation.id] = operation
        self._materializations[operation.id] = [
            MaterializationSpec.from_mapping(operation.id, item) for item in raw_materializations
        ]

    def list(
        self,
        operation_id: Optional[str] = None,
        runner: Optional[str] = None,
        backend: Optional[str] = None,
        include_unimplemented: bool = True,
    ) -> List[ResolvedMaterialization]:
        runner_value = normalize_materialization_runner(runner, "runner") if runner is not None else None
        backend_value = normalize_materialization_backend(backend, "backend") if backend is not None else None
        operation_ids: Iterable[str]
        if operation_id is None:
            operation_ids = sorted(self._operations)
        else:
            if operation_id not in self._operations:
                raise OperationError("Unknown operation: %s" % operation_id)
            operation_ids = [operation_id]
        matches: List[ResolvedMaterialization] = []
        for op_id in operation_ids:
            operation = self._operations[op_id]
            for spec in self._materializations.get(op_id, []):
                if runner_value is not None and spec.runner != runner_value:
                    continue
                if backend_value is not None and spec.backend != backend_value:
                    continue
                if not include_unimplemented and spec.status != "implemented":
                    continue
                matches.append(ResolvedMaterialization(operation=operation, spec=spec))
        return matches

    def resolve(
        self,
        operation_id: str,
        runner: str,
        backend: Optional[str] = None,
        implementation: Optional[str] = None,
        require_implemented: bool = True,
    ) -> ResolvedMaterialization:
        candidates = self.list(
            operation_id=operation_id,
            runner=runner,
            backend=backend,
            include_unimplemented=not require_implemented,
        )
        if implementation is not None:
            candidates = [item for item in candidates if item.implementation == implementation]
        if require_implemented:
            candidates = [item for item in candidates if item.status == "implemented"]
        if not candidates:
            pieces = ["operation %s" % operation_id, "runner %s" % runner]
            if backend is not None:
                pieces.append("backend %s" % normalize_materialization_backend(backend, "backend"))
            if implementation is not None:
                pieces.append("implementation %s" % implementation)
            if require_implemented:
                pieces.append("implemented materialization")
            raise OperationError("No compatible materialization for %s" % ", ".join(pieces))
        return candidates[0]

    def to_dict(self) -> JsonDict:
        return {
            "operations": {
                operation_id: [spec.to_dict() for spec in self._materializations.get(operation_id, [])]
                for operation_id in sorted(self._operations)
            }
        }


def build_materialization_registry(registry: OperationRegistry) -> MaterializationRegistry:
    return MaterializationRegistry.from_operation_registry(registry)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(key): _freeze_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return copy.deepcopy(value)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return copy.deepcopy(value)
