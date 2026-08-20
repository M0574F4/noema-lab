from __future__ import annotations

import copy
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from noema_lab.core.artifacts import Artifact
from noema_lab.core.reproducibility import canonical_json_sha256, derive_seed

JsonDict = Dict[str, Any]

DIFFERENTIABILITY_FRAMEWORKS = {"torch", "sionna", "tensorflow", "numpy", "blackbox", "none"}
DIFFERENTIABILITY_FRAMEWORK_ALIASES = {
    "pytorch": "torch",
    "tf": "tensorflow",
    "python_numpy": "numpy",
    "python": "numpy",
    "black_box": "blackbox",
}
DIFFERENTIABILITY_GRADIENTS = {"full", "stop", "surrogate", "none"}
DIFFERENTIABILITY_KEYS = {"framework", "gradient", "trainable_params", "exportable", "reason"}
DEFAULT_DIFFERENTIABILITY: JsonDict = {
    "framework": "numpy",
    "gradient": "none",
    "trainable_params": False,
    "exportable": False,
}
MATERIALIZATION_RUNNERS = {"benchmark_run", "dataset_capture", "differentiable_export"}
MATERIALIZATION_RUNNER_ALIASES = {
    "benchmark_run": "benchmark_run",
    "dataset_capture": "dataset_capture",
    "dataset-capture": "dataset_capture",
    "differentiable_export": "differentiable_export",
    "differentiable-export": "differentiable_export",
}
BACKEND_NAMES = {
    "blackbox",
    "cpp",
    "external",
    "none",
    "numpy",
    "onnxruntime",
    "openvino",
    "python",
    "sionna",
    "tensorflow",
    "torch",
}
BACKEND_ALIASES = {
    "c++": "cpp",
    "native": "cpp",
    "onnx": "onnxruntime",
    "onnx_runtime": "onnxruntime",
    "ort": "onnxruntime",
    "pytorch": "torch",
    "tf": "tensorflow",
}
DEFAULT_BACKENDS: JsonDict = {
    # An undeclared Operation.run implementation is a Python dispatch path.
    # Do not turn missing metadata into an affirmative NumPy implementation
    # claim: operations that guarantee NumPy semantics declare that backend.
    "benchmark_run": ["python"],
    "dataset_capture": ["python"],
    "differentiable_export": [],
}
EQUIVALENCE_TYPES = {"exact", "numerical", "statistical", "behavioral"}
EQUIVALENCE_KEYS = {"type", "tolerance", "reason"}
DEFAULT_EQUIVALENCE: JsonDict = {
    "type": "behavioral",
    "reason": "No cross-backend equivalence claim declared.",
}
DEFAULT_FORMATS: JsonDict = {
    "artifact": "operation-defined",
    "tensor": "none",
}
MATERIALIZATION_KEYS = {
    "runner",
    "backend",
    "implementation",
    "status",
    "notes",
    "parameter_bindings",
}
METADATA_REQUIREMENT_KEYS = {"all_of", "any_of"}
_CONTRACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_METADATA_PATH_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*$"
)


class OperationError(RuntimeError):
    pass


class ExecutionCancelled(RuntimeError):
    """Raised at cooperative cancellation checkpoints."""

    pass


def normalize_differentiability(value: Optional[Mapping[str, Any]], operation_id: str = "operation") -> JsonDict:
    if value is None or value == {}:
        return dict(DEFAULT_DIFFERENTIABILITY)
    if not isinstance(value, Mapping):
        raise OperationError("Operation %s differentiability metadata must be a mapping" % operation_id)
    unknown = set(value.keys()) - DIFFERENTIABILITY_KEYS
    if unknown:
        raise OperationError(
            "Operation %s differentiability metadata has unknown field(s): %s"
            % (operation_id, ", ".join(sorted(str(item) for item in unknown)))
        )
    payload = dict(DEFAULT_DIFFERENTIABILITY)
    framework = str(value.get("framework", payload["framework"])).strip().lower()
    framework = DIFFERENTIABILITY_FRAMEWORK_ALIASES.get(framework, framework)
    if framework not in DIFFERENTIABILITY_FRAMEWORKS:
        raise OperationError(
            "Operation %s differentiability.framework must be one of %s"
            % (operation_id, ", ".join(sorted(DIFFERENTIABILITY_FRAMEWORKS)))
        )
    gradient = str(value.get("gradient", payload["gradient"])).strip().lower()
    if gradient not in DIFFERENTIABILITY_GRADIENTS:
        raise OperationError(
            "Operation %s differentiability.gradient must be one of %s"
            % (operation_id, ", ".join(sorted(DIFFERENTIABILITY_GRADIENTS)))
        )
    payload["framework"] = framework
    payload["gradient"] = gradient
    payload["trainable_params"] = _bool_metadata_value(
        value.get("trainable_params", payload["trainable_params"]),
        operation_id,
        "trainable_params",
    )
    payload["exportable"] = _bool_metadata_value(
        value.get("exportable", payload["exportable"]),
        operation_id,
        "exportable",
    )
    reason = value.get("reason")
    if reason is not None and str(reason).strip():
        payload["reason"] = str(reason)
    return payload


def _bool_metadata_value(value: Any, operation_id: str, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise OperationError("Operation %s differentiability.%s must be a boolean" % (operation_id, field_name))


def _validated_trained_artifact_abi(
    value: Optional[Mapping[str, Any]],
    *,
    operation_id: str,
    input_names: set[str],
    params_schema: Mapping[str, Any],
) -> JsonDict:
    """Validate the minimum closed-loop contract required for replacement."""

    if value is None or value == {}:
        return {}
    if not isinstance(value, Mapping):
        raise OperationError("Operation %s trained_artifact_abi must be a mapping" % operation_id)
    payload = copy.deepcopy(dict(value))
    for field_name in ("component_id", "component_role", "entrypoint_id"):
        if not str(payload.get(field_name) or "").strip():
            raise OperationError(
                "Operation %s trained_artifact_abi.%s must be non-empty"
                % (operation_id, field_name)
            )
    for field_name in ("inputs", "outputs", "binding_params"):
        field_value = payload.get(field_name)
        if not isinstance(field_value, Mapping) or not field_value:
            raise OperationError(
                "Operation %s trained_artifact_abi.%s must be a non-empty mapping"
                % (operation_id, field_name)
            )
    required_inputs = payload.get("required_operation_inputs") or []
    if not isinstance(required_inputs, (list, tuple)):
        raise OperationError(
            "Operation %s trained_artifact_abi.required_operation_inputs must be a list"
            % operation_id
        )
    normalized_required = [str(item).strip() for item in required_inputs]
    if any(not item for item in normalized_required) or len(set(normalized_required)) != len(normalized_required):
        raise OperationError(
            "Operation %s trained_artifact_abi.required_operation_inputs must be unique non-empty names"
            % operation_id
        )
    unknown_inputs = sorted(set(normalized_required) - set(input_names))
    if unknown_inputs:
        raise OperationError(
            "Operation %s trained_artifact_abi requires unknown operation input(s): %s"
            % (operation_id, ", ".join(unknown_inputs))
        )
    payload["required_operation_inputs"] = normalized_required

    schema = dict(params_schema or {})
    declared_params = set(dict(schema.get("properties") or {}))
    additional = schema.get("additionalProperties", False)
    binding_params = dict(payload.get("binding_params") or {})
    unknown_bindings = sorted(set(binding_params) - declared_params) if additional is False else []
    if unknown_bindings:
        raise OperationError(
            "Operation %s trained_artifact_abi binds undeclared operation parameter(s): %s"
            % (operation_id, ", ".join(unknown_bindings))
        )
    for field_name in ("artifact_manifest_path", "artifact_entrypoint"):
        if not str(binding_params.get(field_name) or "").strip():
            raise OperationError(
                "Operation %s trained_artifact_abi.binding_params.%s must be non-empty"
                % (operation_id, field_name)
            )
    if str(binding_params["artifact_entrypoint"]).strip() != str(
        payload["entrypoint_id"]
    ).strip():
        raise OperationError(
            "Operation %s trained_artifact_abi binding artifact_entrypoint must match entrypoint_id"
            % operation_id
        )
    return payload


def normalize_backends(value: Optional[Mapping[str, Any]], operation_id: str = "operation") -> JsonDict:
    if value is None or value == {}:
        return {runner: list(backends) for runner, backends in DEFAULT_BACKENDS.items()}
    if not isinstance(value, Mapping):
        raise OperationError("Operation %s backends metadata must be a mapping" % operation_id)
    normalized_keys = {_normalize_runner_name(key) for key in value.keys()}
    unknown = normalized_keys - MATERIALIZATION_RUNNERS
    if unknown:
        raise OperationError(
            "Operation %s backends metadata has unknown runner(s): %s"
            % (operation_id, ", ".join(sorted(str(item) for item in unknown)))
        )
    payload = {runner: list(backends) for runner, backends in DEFAULT_BACKENDS.items()}
    for runner, raw_backends in value.items():
        runner_id = _normalize_runner_name(runner)
        if isinstance(raw_backends, str):
            items = [raw_backends]
        elif isinstance(raw_backends, (list, tuple)):
            items = list(raw_backends)
        else:
            raise OperationError(
                "Operation %s backends.%s must be an ordered string/list, not %s"
                % (operation_id, runner_id, type(raw_backends).__name__)
            )
        normalized = []
        for item in items:
            backend = _normalize_backend_name(item, operation_id, "backends.%s" % runner_id)
            if backend not in normalized:
                normalized.append(backend)
        payload[runner_id] = normalized
    return payload


def normalize_equivalence(value: Optional[Mapping[str, Any]], operation_id: str = "operation") -> JsonDict:
    if value is None or value == {}:
        return dict(DEFAULT_EQUIVALENCE)
    if not isinstance(value, Mapping):
        raise OperationError("Operation %s equivalence metadata must be a mapping" % operation_id)
    unknown = set(value.keys()) - EQUIVALENCE_KEYS
    if unknown:
        raise OperationError(
            "Operation %s equivalence metadata has unknown field(s): %s"
            % (operation_id, ", ".join(sorted(str(item) for item in unknown)))
        )
    equivalence_type = str(value.get("type") or DEFAULT_EQUIVALENCE["type"]).strip().lower()
    if equivalence_type not in EQUIVALENCE_TYPES:
        raise OperationError(
            "Operation %s equivalence.type must be one of %s"
            % (operation_id, ", ".join(sorted(EQUIVALENCE_TYPES)))
        )
    payload = {"type": equivalence_type}
    if "tolerance" in value and value.get("tolerance") is not None:
        payload["tolerance"] = value.get("tolerance")
    reason = value.get("reason")
    if reason is not None and str(reason).strip():
        payload["reason"] = str(reason)
    elif equivalence_type == DEFAULT_EQUIVALENCE["type"]:
        payload["reason"] = DEFAULT_EQUIVALENCE["reason"]
    return payload


def normalize_formats(value: Optional[Mapping[str, Any]], operation_id: str = "operation") -> JsonDict:
    if value is None or value == {}:
        return dict(DEFAULT_FORMATS)
    if not isinstance(value, Mapping):
        raise OperationError("Operation %s formats metadata must be a mapping" % operation_id)
    payload = dict(DEFAULT_FORMATS)
    for key, raw_value in value.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            raise OperationError("Operation %s formats metadata has an empty key" % operation_id)
        if isinstance(raw_value, (list, tuple)):
            payload[normalized_key] = [str(item) for item in raw_value]
        elif isinstance(raw_value, set):
            raise OperationError(
                "Operation %s formats.%s must be ordered; sets are not allowed"
                % (operation_id, normalized_key)
            )
        elif raw_value is None:
            payload[normalized_key] = "none"
        else:
            payload[normalized_key] = str(raw_value)
    return payload


def normalize_input_metadata_requirements(
    value: Optional[Mapping[str, Any]],
    operation_id: str = "operation",
    *,
    input_names: Optional[set[str]] = None,
) -> JsonDict:
    """Normalize typed metadata requirements for operation inputs.

    ``all_of`` paths must all be guaranteed by the producer. ``any_of`` paths
    express runtime-compatible alternatives, such as one of
    ``original_shapes``, ``original_shape``, or ``shape``.
    """

    if value is None or value == {}:
        return {}
    if not isinstance(value, Mapping):
        raise OperationError(
            "Operation %s input_metadata_requirements must be an object"
            % operation_id
        )
    payload: JsonDict = {}
    for raw_name, raw_requirement in value.items():
        name = _validate_contract_name(
            operation_id, "input_metadata_requirements", raw_name
        )
        if input_names is not None and name not in input_names:
            raise OperationError(
                "Operation %s input_metadata_requirements references unknown input %s"
                % (operation_id, name)
            )
        if not isinstance(raw_requirement, Mapping):
            raise OperationError(
                "Operation %s input_metadata_requirements.%s must be an object"
                % (operation_id, name)
            )
        unknown = set(raw_requirement) - METADATA_REQUIREMENT_KEYS
        if unknown:
            raise OperationError(
                "Operation %s input_metadata_requirements.%s has unknown field(s): %s"
                % (
                    operation_id,
                    name,
                    ", ".join(sorted(str(item) for item in unknown)),
                )
            )
        requirement: JsonDict = {}
        for key in ("all_of", "any_of"):
            if key not in raw_requirement:
                continue
            requirement[key] = _normalize_metadata_paths(
                raw_requirement[key],
                operation_id,
                "input_metadata_requirements.%s.%s" % (name, key),
            )
        if not any(requirement.values()):
            raise OperationError(
                "Operation %s input_metadata_requirements.%s must declare all_of or any_of paths"
                % (operation_id, name)
            )
        payload[name] = requirement
    return payload


def normalize_output_metadata_guarantees(
    value: Optional[Mapping[str, Any]],
    operation_id: str = "operation",
    *,
    output_names: Optional[set[str]] = None,
) -> JsonDict:
    if value is None or value == {}:
        return {}
    if not isinstance(value, Mapping):
        raise OperationError(
            "Operation %s output_metadata_guarantees must be an object"
            % operation_id
        )
    payload: JsonDict = {}
    for raw_name, raw_paths in value.items():
        name = _validate_contract_name(
            operation_id, "output_metadata_guarantees", raw_name
        )
        if output_names is not None and name not in output_names:
            raise OperationError(
                "Operation %s output_metadata_guarantees references unknown output %s"
                % (operation_id, name)
            )
        payload[name] = _normalize_metadata_paths(
            raw_paths,
            operation_id,
            "output_metadata_guarantees.%s" % name,
        )
    return payload


def _normalize_metadata_paths(
    value: Any,
    operation_id: str,
    field_name: str,
) -> List[str]:
    if not isinstance(value, (list, tuple)) or isinstance(value, str):
        raise OperationError(
            "Operation %s %s must be an ordered list of metadata paths"
            % (operation_id, field_name)
        )
    paths = list(value)
    if not paths:
        raise OperationError(
            "Operation %s %s must not be empty" % (operation_id, field_name)
        )
    if any(
        not isinstance(path, str) or not _METADATA_PATH_RE.fullmatch(path)
        for path in paths
    ):
        raise OperationError(
            "Operation %s %s contains an invalid metadata path"
            % (operation_id, field_name)
        )
    if len(paths) != len(set(paths)):
        raise OperationError(
            "Operation %s %s contains duplicate metadata paths"
            % (operation_id, field_name)
        )
    return paths


def normalize_materializations(
    value: Optional[Any],
    backends: Mapping[str, Any],
    operation_id: str = "operation",
) -> List[JsonDict]:
    if value is None:
        return _default_materializations(backends)
    if not isinstance(value, list):
        raise OperationError("Operation %s materializations metadata must be a list" % operation_id)
    payload = []
    seen_identities: Dict[tuple[str, str, str], int] = {}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise OperationError(
                "Operation %s materializations[%d] must be a mapping" % (operation_id, index)
            )
        unknown = set(item.keys()) - MATERIALIZATION_KEYS
        if unknown:
            raise OperationError(
                "Operation %s materializations[%d] has unknown field(s): %s"
                % (operation_id, index, ", ".join(sorted(str(field) for field in unknown)))
            )
        runner = _normalize_runner_name(item.get("runner"))
        if runner not in MATERIALIZATION_RUNNERS:
            raise OperationError(
                "Operation %s materializations[%d].runner must be one of %s"
                % (operation_id, index, ", ".join(sorted(MATERIALIZATION_RUNNERS)))
            )
        backend = _normalize_backend_name(item.get("backend"), operation_id, "materializations[%d].backend" % index)
        implementation = str(item.get("implementation") or "default")
        identity = (runner, backend, implementation)
        if identity in seen_identities:
            raise OperationError(
                "Operation %s materializations[%d] duplicates materialization identity "
                "%s/%s/%s from materializations[%d]"
                % (
                    operation_id,
                    index,
                    runner,
                    backend,
                    implementation,
                    seen_identities[identity],
                )
            )
        seen_identities[identity] = index
        materialization = {
            "runner": runner,
            "backend": backend,
            "implementation": implementation,
            "status": str(item.get("status") or "implemented"),
        }
        parameter_bindings = item.get("parameter_bindings")
        if parameter_bindings is not None:
            if not isinstance(parameter_bindings, Mapping):
                raise OperationError(
                    "Operation %s materializations[%d].parameter_bindings must be a mapping"
                    % (operation_id, index)
                )
            normalized_bindings: JsonDict = {}
            for raw_name, value in parameter_bindings.items():
                name = str(raw_name).strip()
                if not name:
                    raise OperationError(
                        "Operation %s materializations[%d].parameter_bindings has an empty parameter name"
                        % (operation_id, index)
                    )
                if name in normalized_bindings:
                    raise OperationError(
                        "Operation %s materializations[%d].parameter_bindings repeats parameter %s"
                        % (operation_id, index, name)
                    )
                normalized_bindings[name] = copy.deepcopy(value)
            if normalized_bindings:
                materialization["parameter_bindings"] = normalized_bindings
        notes = item.get("notes")
        if notes is not None and str(notes).strip():
            materialization["notes"] = str(notes)
        payload.append(materialization)
    return payload


def _default_materializations(backends: Mapping[str, Any]) -> List[JsonDict]:
    payload = []
    for runner in sorted(MATERIALIZATION_RUNNERS):
        for backend in backends.get(runner, []) or []:
            payload.append(
                {
                    "runner": runner,
                    "backend": str(backend),
                    "implementation": "default",
                    "status": "implemented",
                }
            )
    return payload


def _normalize_runner_name(value: Any) -> str:
    runner = str(value or "").strip().lower().replace(" ", "_")
    return MATERIALIZATION_RUNNER_ALIASES.get(runner, runner)


def _normalize_backend_name(value: Any, operation_id: str, field_name: str) -> str:
    backend = str(value or "").strip().lower().replace("-", "_")
    backend = BACKEND_ALIASES.get(backend, backend)
    if backend not in BACKEND_NAMES:
        raise OperationError(
            "Operation %s %s must be one of %s"
            % (operation_id, field_name, ", ".join(sorted(BACKEND_NAMES)))
        )
    return backend


@dataclass
class OperationResult:
    outputs: Dict[str, Artifact] = field(default_factory=dict)
    metrics: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "outputs": {name: output.to_dict() for name, output in self.outputs.items()},
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }


@dataclass
class OperationContext:
    recipe_name: str
    step_id: str
    params: JsonDict
    inputs: Mapping[str, Artifact]
    run_dir: Path
    step_dir: Path
    progress_sink: Optional[Callable[[JsonDict], None]] = None
    master_seed: Optional[int] = None
    seed_namespace: Optional[str] = None
    cancellation_token: Optional[Any] = None

    def require_input(self, name: str) -> Artifact:
        if name not in self.inputs:
            raise OperationError("Step %s requires input '%s'" % (self.step_id, name))
        return self.inputs[name]

    def output_path(self, name: str, suffix: str) -> Path:
        if not isinstance(name, str) or not _CONTRACT_ID_RE.fullmatch(name):
            raise OperationError("Step %s output name is unsafe: %r" % (self.step_id, name))
        if not isinstance(suffix, str) or not suffix or any(
            separator and separator in suffix for separator in ("/", "\\")
        ):
            raise OperationError("Step %s output suffix is unsafe: %r" % (self.step_id, suffix))
        safe_name = name
        safe_suffix = suffix if suffix.startswith(".") else "." + suffix
        path = self.step_dir / ("%s%s" % (safe_name, safe_suffix))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def report_progress(self, message: str = "progress", **fields: Any) -> None:
        # Progress boundaries are natural cooperative-cancellation checkpoints.
        # This gives existing batched operations cancellation support without
        # requiring every loop to duplicate token polling boilerplate.
        self.raise_if_cancelled()
        if self.progress_sink is None:
            return
        event = {
            "kind": "step_progress",
            "message": message,
            "run_id": self.run_dir.name,
            "step_id": self.step_id,
        }
        event.update(fields)
        self.progress_sink(event)

    def is_cancelled(self) -> bool:
        """Return whether cooperative cancellation has been requested.

        Operation implementations should poll this at natural batch or loop
        boundaries.  The token is deliberately duck-typed so the executor can
        adapt existing ``threading.Event`` callers without coupling operation
        contracts to a particular scheduler implementation.
        """

        token = self.cancellation_token
        if token is None:
            return False
        checker = getattr(token, "is_cancelled", None)
        if callable(checker):
            return bool(checker())
        checker = getattr(token, "is_set", None)
        return bool(checker()) if callable(checker) else False

    def raise_if_cancelled(self, message: Optional[str] = None) -> None:
        """Raise the executor's cancellation exception when requested."""

        token = self.cancellation_token
        raiser = getattr(token, "raise_if_cancelled", None)
        if callable(raiser):
            if message is None:
                raiser()
            else:
                raiser(message)
            return
        if self.is_cancelled():
            raise ExecutionCancelled(message or "Operation canceled")

    def seed(self, stream: str = "default", default: int = 0) -> int:
        if "seed" in self.params and self.params["seed"] is not None:
            return int(self.params["seed"])
        if self.master_seed is None:
            return int(default)
        namespace = self.seed_namespace or self.recipe_name
        return derive_seed(int(self.master_seed), namespace, self.step_id, stream)


class Operation(ABC):
    id: str
    name: str
    status: str = "implemented"
    input_kinds: Mapping[str, List[str]] = {}
    optional_input_kinds: Mapping[str, List[str]] = {}
    output_kinds: Mapping[str, str] = {}
    input_metadata_requirements: Mapping[str, Mapping[str, List[str]]] = {}
    output_metadata_guarantees: Mapping[str, List[str]] = {}
    entropy_info: Mapping[str, Any] = {}
    differentiability: Mapping[str, Any] = {}
    backends: Mapping[str, Any] = {}
    equivalence: Mapping[str, Any] = {}
    formats: Mapping[str, Any] = {}
    trained_artifact_abi: Mapping[str, Any] = {}
    # Explicitly describes the current built-in implementation. External
    # replacement is a separate capability derived from a validated
    # trained_artifact_abi. A true declaration must be backed by a callable
    # operation-owned provider; metadata alone cannot advertise an action the
    # runtime cannot perform.
    fine_tuning_supported: bool = False
    fine_tuning_provider: Optional[Callable[..., Any]] = None
    # Safe default: registry singletons are not assumed re-entrant. The
    # parallel scheduler runs undeclared operations exclusively. Implementers
    # may opt in only when their operation and all called libraries avoid
    # mutable process-global state (including global RNGs).
    thread_safe: bool = False
    materializations: Optional[List[Mapping[str, Any]]] = None
    params_schema: Mapping[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def describe(self) -> JsonDict:
        backends = normalize_backends(getattr(self, "backends", None), self.id)
        materializations = normalize_materializations(
            getattr(self, "materializations", None),
            backends,
            self.id,
        )
        payload = {
            "schema_version": 1,
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "input_kinds": dict(self.input_kinds),
            "optional_input_kinds": dict(self.optional_input_kinds),
            "output_kinds": dict(self.output_kinds),
            "input_metadata_requirements": normalize_input_metadata_requirements(
                getattr(self, "input_metadata_requirements", None),
                self.id,
                input_names=set(self.input_kinds)
                | set(self.optional_input_kinds),
            ),
            "output_metadata_guarantees": normalize_output_metadata_guarantees(
                getattr(self, "output_metadata_guarantees", None),
                self.id,
                output_names=set(self.output_kinds),
            ),
            "params_schema": dict(self.params_schema),
            "differentiability": normalize_differentiability(getattr(self, "differentiability", None), self.id),
            "backends": backends,
            "equivalence": normalize_equivalence(getattr(self, "equivalence", None), self.id),
            "formats": normalize_formats(getattr(self, "formats", None), self.id),
            "materializations": materializations,
        }
        thread_safe = getattr(self, "thread_safe", False)
        if not isinstance(thread_safe, bool):
            raise OperationError(
                "Operation %s thread_safe must be a boolean" % self.id
            )
        payload["execution_safety"] = {
            "thread_safe": thread_safe,
            "dispatch_instance": "shallow_copy",
            "undeclared_parallel_policy": "exclusive",
        }
        if self.entropy_info:
            payload["entropy_info"] = dict(self.entropy_info)
        artifact_abi = _validated_trained_artifact_abi(
            self.trained_artifact_abi,
            operation_id=self.id,
            input_names=set(self.input_kinds) | set(self.optional_input_kinds),
            params_schema=self.params_schema,
        )
        if artifact_abi:
            payload["trained_artifact_abi"] = artifact_abi
        fine_tuning_supported = getattr(self, "fine_tuning_supported", False)
        if not isinstance(fine_tuning_supported, bool):
            raise OperationError(
                "Operation %s fine_tuning_supported must be a boolean" % self.id
            )
        fine_tuning_provider = getattr(self, "fine_tuning_provider", None)
        if fine_tuning_supported and not callable(fine_tuning_provider):
            raise OperationError(
                "Operation %s fine_tuning_supported requires a callable fine_tuning_provider"
                % self.id
            )
        if not fine_tuning_supported and fine_tuning_provider is not None:
            raise OperationError(
                "Operation %s fine_tuning_provider requires fine_tuning_supported=true"
                % self.id
            )
        payload["training_capabilities"] = {
            "built_in_fine_tuning": fine_tuning_supported,
            "portable_replacement": bool(artifact_abi),
        }
        return payload

    def runtime_availability(self, params: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        """Return parameter-aware availability for an ordinary execution.

        Most operations have one implementation and can rely on the static
        availability exposed by :meth:`describe`.  Operations whose runtime
        dependency changes with a parameter (for example a local renderer
        versus a masked-language-model renderer) override this hook.
        """

        availability = self.describe().get("availability")
        return availability if isinstance(availability, Mapping) else None

    def validate_preflight(
        self,
        params: Mapping[str, Any],
        inputs: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Validate parameter/input combinations before evidence is created.

        JSON-schema validation handles individual fields. Operations override
        this hook for cross-field or conditionally-required combinations that
        otherwise could fail only after dispatch.
        """

        return None

    def execution_instance(self) -> "Operation":
        """Return a per-step dispatch view of this registered operation.

        A shallow copy isolates ordinary scalar instance state while retaining
        deliberate references such as loaded models and callbacks. Operations
        that need deeper isolation can override this hook. Undeclared
        operations are nevertheless scheduled exclusively.
        """

        try:
            instance = copy.copy(self)
        except Exception as exc:
            raise OperationError(
                "Operation %s cannot create an isolated execution instance: %s"
                % (self.id, exc)
            ) from exc
        if not isinstance(instance, Operation):
            raise OperationError(
                "Operation %s execution_instance returned an invalid object" % self.id
            )
        return instance

    @abstractmethod
    def run(self, ctx: OperationContext) -> OperationResult:
        raise NotImplementedError


class OperationRegistry:
    def __init__(self) -> None:
        self._operations: Dict[str, Operation] = {}

    def register(self, operation: Operation) -> None:
        if not isinstance(operation, Operation):
            raise OperationError("Registered object must be an Operation")
        _validate_operation_contract(operation)
        if operation.id in self._operations:
            raise OperationError("Operation already registered: %s" % operation.id)
        self._operations[operation.id] = operation

    def get(self, operation_id: str) -> Operation:
        try:
            return self._operations[operation_id]
        except KeyError as exc:
            raise OperationError("Unknown operation: %s" % operation_id) from exc

    def list(self) -> List[Operation]:
        return [self._operations[key] for key in sorted(self._operations)]

    def describe(self) -> List[JsonDict]:
        return [operation.describe() for operation in self.list()]


def object_schema(
    properties: Optional[Mapping[str, Any]] = None,
    required: Optional[List[str]] = None,
    additional: bool = False,
) -> JsonDict:
    return {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required or []),
        "additionalProperties": additional,
    }


def _validate_operation_contract(operation: Operation) -> None:
    operation_id = getattr(operation, "id", None)
    if not isinstance(operation_id, str) or not _CONTRACT_ID_RE.fullmatch(operation_id):
        raise OperationError("Operation id must be a non-empty safe identifier")
    if not isinstance(getattr(operation, "name", None), str) or not operation.name.strip():
        raise OperationError("Operation %s name must be non-empty" % operation_id)
    if not isinstance(getattr(operation, "status", None), str) or not operation.status.strip():
        raise OperationError("Operation %s status must be non-empty" % operation_id)

    required = _validate_input_kind_map(
        operation_id,
        "input_kinds",
        getattr(operation, "input_kinds", None),
    )
    optional = _validate_input_kind_map(
        operation_id,
        "optional_input_kinds",
        getattr(operation, "optional_input_kinds", None),
    )
    overlap = sorted(set(required) & set(optional))
    if overlap:
        raise OperationError(
            "Operation %s declares inputs as required and optional: %s"
            % (operation_id, ", ".join(overlap))
        )
    _validate_output_kind_map(
        operation_id,
        getattr(operation, "output_kinds", None),
    )

    from noema_lab.core.params import validate_params_schema

    validate_params_schema(operation_id, getattr(operation, "params_schema", None))
    # describe() normalizes and validates all remaining metadata, including
    # duplicate materialization identities and concurrency declarations.
    contract = operation.describe()
    repeated_contract = operation.describe()
    if canonical_json_sha256(contract) != canonical_json_sha256(repeated_contract):
        raise OperationError(
            "Operation %s describe() is nondeterministic at registration"
            % operation_id
        )
    _validate_materialization_ambiguity(operation_id, contract["materializations"])


def _validate_contract_name(operation_id: str, field_name: str, name: Any) -> str:
    if not isinstance(name, str) or not _CONTRACT_ID_RE.fullmatch(name):
        raise OperationError(
            "Operation %s %s contains an unsafe name: %r"
            % (operation_id, field_name, name)
        )
    return name


def _validate_input_kind_map(
    operation_id: str,
    field_name: str,
    value: Any,
) -> Mapping[str, List[str]]:
    if not isinstance(value, Mapping):
        raise OperationError("Operation %s %s must be an object" % (operation_id, field_name))
    for raw_name, raw_kinds in value.items():
        name = _validate_contract_name(operation_id, field_name, raw_name)
        if not isinstance(raw_kinds, (list, tuple)) or isinstance(raw_kinds, str):
            raise OperationError(
                "Operation %s %s.%s must be an ordered list of artifact kinds"
                % (operation_id, field_name, name)
            )
        kinds = list(raw_kinds)
        if any(not isinstance(kind, str) or not kind.strip() for kind in kinds):
            raise OperationError(
                "Operation %s %s.%s contains an empty/non-string artifact kind"
                % (operation_id, field_name, name)
            )
        if len(kinds) != len(set(kinds)):
            raise OperationError(
                "Operation %s %s.%s contains duplicate artifact kinds"
                % (operation_id, field_name, name)
            )
    return value


def _validate_output_kind_map(operation_id: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        raise OperationError("Operation %s output_kinds must be an object" % operation_id)
    for raw_name, kind in value.items():
        name = _validate_contract_name(operation_id, "output_kinds", raw_name)
        if not isinstance(kind, str) or not kind.strip():
            raise OperationError(
                "Operation %s output_kinds.%s must be a non-empty artifact kind"
                % (operation_id, name)
            )


def _validate_materialization_ambiguity(
    operation_id: str,
    materializations: List[Mapping[str, Any]],
) -> None:
    groups: Dict[tuple[str, str], List[Mapping[str, Any]]] = {}
    for item in materializations:
        if item.get("status") != "implemented":
            continue
        groups.setdefault((str(item["runner"]), str(item["backend"])), []).append(item)
    for (runner, backend), group in groups.items():
        for index, left in enumerate(group):
            left_bindings = dict(left.get("parameter_bindings") or {})
            for right in group[index + 1 :]:
                right_bindings = dict(right.get("parameter_bindings") or {})
                conflicting = any(
                    key in right_bindings and right_bindings[key] != value
                    for key, value in left_bindings.items()
                )
                if conflicting or len(left_bindings) != len(right_bindings):
                    continue
                raise OperationError(
                    "Operation %s has ambiguous implemented materializations for %s/%s: "
                    "%s and %s can match the same parameters at equal specificity"
                    % (
                        operation_id,
                        runner,
                        backend,
                        left.get("implementation"),
                        right.get("implementation"),
                    )
                )
