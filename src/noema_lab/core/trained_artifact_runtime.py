from __future__ import annotations

"""Safe runtime support for executable trained-artifact components.

The trained-artifact manifest describes *what* a returned model implements.  This
module is deliberately limited to data-only runtimes: it validates executable
graphs and invokes them without importing Python supplied by the artifact.
"""

import importlib.util
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


JsonDict = Dict[str, Any]

SUPPORTED_RUNTIME_BACKENDS = {"onnxruntime"}
SUPPORTED_TRAINED_ARTIFACT_ABI_VERSIONS = frozenset({1, 2})
SUPPORTED_TENSOR_DTYPES = {
    "bool",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
}

_ONNX_TYPE_TO_DTYPE = {
    "tensor(bool)": "bool",
    "tensor(float16)": "float16",
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(int8)": "int8",
    "tensor(int16)": "int16",
    "tensor(int32)": "int32",
    "tensor(int64)": "int64",
    "tensor(uint8)": "uint8",
    "tensor(uint16)": "uint16",
    "tensor(uint32)": "uint32",
    "tensor(uint64)": "uint64",
}
_ALLOWED_ONNX_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}
_MAX_ONNX_NODES = 100_000


class TrainedArtifactRuntimeError(RuntimeError):
    pass


class TrainedArtifactRuntimeUnavailable(TrainedArtifactRuntimeError):
    pass


@dataclass(frozen=True)
class _AdmittedEntrypoint:
    specification: Mapping[str, Any]
    component: Any


class AdmittedTrainedArtifactRuntime:
    """Digest-bound, explicitly admitted executable artifact.

    Construction performs the full package inspection and materializes every
    declared ONNX session. Calls through :meth:`run_entrypoint` reuse those
    admitted in-memory sessions while continuing to validate the selected
    entrypoint and its tensor inputs and outputs.

    The handle is intentionally a snapshot. Package changes on disk are not
    observed by an existing handle; callers must admit a new handle, presenting
    the expected package identity again, to execute changed artifact bytes.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        expected_package_sha256: str,
        project_root: Optional[Path] = None,
    ) -> None:
        # Import lazily to avoid a module cycle: trained_artifacts calls this
        # module's entrypoint validator while inspecting schema-v2 artifacts.
        from noema_lab.core.trained_artifacts import inspect_trained_artifact
        from noema_lab.ops.portable_onnx import load_portable_onnx_component

        expected_digest = _validated_package_sha256(expected_package_sha256)
        path = Path(manifest_path).resolve()
        root = Path(project_root or path.parent).resolve()
        package_root = path.parent.resolve()
        inspected = inspect_trained_artifact(path, project_root=root)
        _require_admissible_inspection(inspected, expected_digest)

        runtime = dict(inspected.get("runtime") or {})
        _require_supported_runtime_abi(runtime)
        backend = _normalize_backend(runtime.get("backend"))
        if not runtime.get("available"):
            reason = "; ".join(
                str(item)
                for item in runtime.get("unavailable_reasons") or []
            )
            raise TrainedArtifactRuntimeUnavailable(
                reason or "trained artifact runtime is unavailable"
            )
        if backend != "onnxruntime":
            raise TrainedArtifactRuntimeUnavailable(
                "unsupported trained-artifact runtime backend: %s" % backend
            )

        components = {
            str(item.get("id") or ""): dict(item)
            for item in inspected.get("components") or []
        }
        loaded_components: Dict[str, Any] = {}
        admitted_entrypoints: Dict[str, _AdmittedEntrypoint] = {}
        for raw_entrypoint in runtime.get("entrypoints") or []:
            entrypoint = deepcopy(dict(raw_entrypoint))
            entrypoint_id = str(entrypoint.get("id") or "")
            if not entrypoint_id:
                raise TrainedArtifactRuntimeError(
                    "trained artifact runtime has an empty entrypoint id"
                )
            if entrypoint_id in admitted_entrypoints:
                raise TrainedArtifactRuntimeError(
                    "trained artifact runtime has duplicate entrypoint id %s"
                    % entrypoint_id
                )
            component_id = str(entrypoint.get("component") or "")
            component_row = components.get(component_id)
            if component_row is None:
                raise TrainedArtifactRuntimeError(
                    "runtime entrypoint %s references missing component %s"
                    % (entrypoint_id, component_id)
                )
            component_path = _resolve_confined_inspected_component_path(
                component_row.get("path"),
                project_root=root,
                package_root=package_root,
            )
            component = loaded_components.get(component_id)
            if component is None:
                input_names = [
                    str(item.get("name") or "")
                    for item in entrypoint.get("inputs") or []
                ]
                output_names = [
                    str(item.get("name") or "")
                    for item in entrypoint.get("outputs") or []
                ]
                try:
                    component = load_portable_onnx_component(
                        str(component_path),
                        str(component_row.get("sha256") or ""),
                        expected_inputs=input_names,
                        expected_outputs=output_names,
                    )
                except Exception as exc:
                    raise TrainedArtifactRuntimeError(
                        "could not admit ONNX component %s: %s"
                        % (component_id or "<empty>", exc)
                    ) from exc
                loaded_components[component_id] = component
            _compare_signature(
                entrypoint.get("inputs") or [],
                component.session.get_inputs(),
                "input",
            )
            _compare_signature(
                entrypoint.get("outputs") or [],
                component.session.get_outputs(),
                "output",
            )
            admitted_entrypoints[entrypoint_id] = _AdmittedEntrypoint(
                specification=MappingProxyType(entrypoint),
                component=component,
            )

        if not admitted_entrypoints:
            raise TrainedArtifactRuntimeError(
                "trained artifact runtime declares no entrypoints"
            )
        self._manifest_path = path
        self._package_sha256 = expected_digest
        self._backend = backend
        self._entrypoints = MappingProxyType(admitted_entrypoints)

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def package_sha256(self) -> str:
        return self._package_sha256

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def entrypoint_ids(self) -> Tuple[str, ...]:
        return tuple(self._entrypoints)

    def entrypoint_session_evidence(self, entrypoint_id: str) -> JsonDict:
        """Return verified portable-session evidence for an admitted entrypoint."""

        requested = str(entrypoint_id or "").strip()
        admitted = self._entrypoints.get(requested)
        if admitted is None:
            raise TrainedArtifactRuntimeError(
                "unknown trained-artifact runtime entrypoint %s"
                % (requested or "<empty>")
            )
        from noema_lab.ops.portable_onnx import portable_onnx_session_evidence

        return portable_onnx_session_evidence(admitted.component)

    def run_entrypoint(
        self,
        entrypoint_id: str,
        inputs: Mapping[str, np.ndarray],
        *,
        inference_batch_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Run a declared entrypoint through its admitted in-memory session."""

        normalized_id = str(entrypoint_id or "")
        admitted = self._entrypoints.get(normalized_id)
        if admitted is None:
            raise TrainedArtifactRuntimeError(
                "trained artifact has no runtime entrypoint %r" % normalized_id
            )
        return _run_admitted_onnx(
            admitted.component,
            admitted.specification,
            inputs,
            inference_batch_size=inference_batch_size,
        )


def admit_trained_artifact_runtime(
    manifest_path: Path,
    *,
    expected_package_sha256: str,
    project_root: Optional[Path] = None,
) -> AdmittedTrainedArtifactRuntime:
    """Inspect and admit a digest-bound schema-v2 artifact for repeated calls."""

    return AdmittedTrainedArtifactRuntime(
        manifest_path,
        expected_package_sha256=expected_package_sha256,
        project_root=project_root,
    )


def runtime_backend_availability(backend: str) -> JsonDict:
    """Return availability without importing an artifact or executing its code."""

    normalized = str(backend or "").strip().lower().replace("_", "-")
    if normalized == "onnx-runtime":
        normalized = "onnxruntime"
    if normalized not in SUPPORTED_RUNTIME_BACKENDS:
        return {
            "backend": normalized,
            "available": False,
            "reason": "unsupported trained-artifact runtime backend: %s"
            % (normalized or "<empty>"),
        }
    missing = [
        package
        for package in ("onnx", "onnxruntime")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        return {
            "backend": normalized,
            "available": False,
            "reason": (
                "runtime backend onnxruntime requires optional package(s): %s; "
                'install with `python -m pip install "noema-lab[onnx]"` in an '
                "installed environment, or `uv sync --extra onnx` in a source checkout"
                % ", ".join(missing)
            ),
        }
    return {"backend": normalized, "available": True, "reason": ""}


def validate_runtime_entrypoints(
    manifest_dir: Path,
    components: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> JsonDict:
    """Validate runtime availability, graph safety, and declared tensor signatures.

    Schema validation is performed by ``core.trained_artifacts``.  This function
    assumes normalized component and entrypoint mappings and concentrates on the
    executable graph.  Missing optional dependencies make an otherwise valid
    artifact unavailable rather than corrupt.
    """

    backend = _normalize_backend(runtime.get("backend"))
    availability = runtime_backend_availability(backend)
    result: JsonDict = {
        "backend": backend,
        "available": bool(availability.get("available")),
        "issues": [],
        "unavailable_reasons": [],
        "entrypoints": [],
    }
    raw_abi_version = runtime.get("abi_version")
    if (
        isinstance(raw_abi_version, bool)
        or not isinstance(raw_abi_version, int)
        or raw_abi_version not in SUPPORTED_TRAINED_ARTIFACT_ABI_VERSIONS
    ):
        result["issues"].append(
            "unsupported trained-artifact ABI version: %s (supported: %s)"
            % (
                raw_abi_version,
                ", ".join(
                    str(value)
                    for value in sorted(
                        SUPPORTED_TRAINED_ARTIFACT_ABI_VERSIONS
                    )
                ),
            )
        )
        result["available"] = False
        result["entrypoints"] = [
            dict(item) for item in runtime.get("entrypoints") or []
        ]
        return result
    if not availability.get("available"):
        result["unavailable_reasons"].append(str(availability.get("reason") or "runtime unavailable"))
        result["entrypoints"] = [dict(item) for item in runtime.get("entrypoints") or []]
        return result

    if backend != "onnxruntime":  # Defensive: availability currently rejects this.
        result["issues"].append("unsupported trained-artifact runtime backend: %s" % backend)
        result["available"] = False
        return result

    manifest_root = Path(manifest_dir).resolve()
    for raw_entrypoint in runtime.get("entrypoints") or []:
        entrypoint = dict(raw_entrypoint)
        entrypoint_id = str(entrypoint.get("id") or "")
        component_id = str(entrypoint.get("component") or "")
        component = components.get(component_id)
        if component is None:
            result["issues"].append(
                "runtime entrypoint %s references unknown component %s"
                % (entrypoint_id or "<empty>", component_id or "<empty>")
            )
            continue
        component_path = _confined_path(manifest_root, component.get("path"))
        try:
            session_signature = _validate_onnx_component(component_path, entrypoint)
            entrypoint["validated_signature"] = session_signature
        except TrainedArtifactRuntimeError as exc:
            result["issues"].append(
                "runtime entrypoint %s: %s" % (entrypoint_id or "<empty>", exc)
            )
        result["entrypoints"].append(entrypoint)
    if result["issues"]:
        result["available"] = False
    return result


def run_trained_artifact_entrypoint(
    manifest_path: Path,
    entrypoint_id: str,
    inputs: Mapping[str, np.ndarray],
    *,
    expected_package_sha256: str,
    project_root: Optional[Path] = None,
    inference_batch_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Execute through a freshly inspected, digest-bound compatibility path.

    This function deliberately admits a new handle for every invocation. Hot
    loops should explicitly retain :func:`admit_trained_artifact_runtime`'s
    returned handle instead.
    """

    admitted = admit_trained_artifact_runtime(
        manifest_path,
        expected_package_sha256=expected_package_sha256,
        project_root=project_root,
    )
    return admitted.run_entrypoint(
        entrypoint_id,
        inputs,
        inference_batch_size=inference_batch_size,
    )


def _validate_onnx_component(path: Path, entrypoint: Mapping[str, Any]) -> JsonDict:
    availability = runtime_backend_availability("onnxruntime")
    if not availability.get("available"):
        raise TrainedArtifactRuntimeUnavailable(str(availability.get("reason") or "ONNX unavailable"))
    try:
        import onnx
    except Exception as exc:  # pragma: no cover - protected by availability check
        raise TrainedArtifactRuntimeUnavailable(
            "could not import optional ONNX runtime dependencies: %s" % exc
        ) from exc

    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:
        raise TrainedArtifactRuntimeError("could not parse ONNX component: %s" % exc) from exc
    external_location = int(getattr(onnx.TensorProto, "EXTERNAL", 1))
    if any(int(initializer.data_location) == external_location for initializer in model.graph.initializer):
        raise TrainedArtifactRuntimeError(
            "ONNX components with external tensor data are not supported; package one self-contained model file"
        )
    if len(model.graph.node) > _MAX_ONNX_NODES:
        raise TrainedArtifactRuntimeError(
            "ONNX graph has %d nodes; maximum allowed is %d"
            % (len(model.graph.node), _MAX_ONNX_NODES)
        )
    custom_domains = sorted(
        {
            str(node.domain or "")
            for node in model.graph.node
            if str(node.domain or "") not in _ALLOWED_ONNX_DOMAINS
        }
    )
    if custom_domains:
        raise TrainedArtifactRuntimeError(
            "ONNX custom operator domain(s) are not allowed: %s" % ", ".join(custom_domains)
        )
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        raise TrainedArtifactRuntimeError("ONNX checker rejected the component: %s" % exc) from exc

    from noema_lab.ops.portable_onnx import load_portable_onnx_component

    input_names = [str(item.get("name") or "") for item in entrypoint.get("inputs") or []]
    output_names = [str(item.get("name") or "") for item in entrypoint.get("outputs") or []]
    try:
        component = load_portable_onnx_component(
            str(path),
            _file_sha256(path),
            expected_inputs=input_names,
            expected_outputs=output_names,
        )
    except Exception as exc:
        raise TrainedArtifactRuntimeError("ONNX Runtime rejected the component: %s" % exc) from exc
    session = component.session
    _compare_signature(entrypoint.get("inputs") or [], session.get_inputs(), "input")
    _compare_signature(entrypoint.get("outputs") or [], session.get_outputs(), "output")
    return {
        "inputs": [_runtime_value_info(item) for item in session.get_inputs()],
        "outputs": [_runtime_value_info(item) for item in session.get_outputs()],
    }


def _run_admitted_onnx(
    component: Any,
    entrypoint: Mapping[str, Any],
    inputs: Mapping[str, np.ndarray],
    *,
    inference_batch_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    declared_inputs = list(entrypoint.get("inputs") or [])
    expected_names = [str(item.get("name") or "") for item in declared_inputs]
    missing = sorted(set(expected_names) - set(inputs.keys()))
    extra = sorted(set(inputs.keys()) - set(expected_names))
    if missing or extra:
        details = []
        if missing:
            details.append("missing input(s): %s" % ", ".join(missing))
        if extra:
            details.append("unexpected input(s): %s" % ", ".join(extra))
        raise TrainedArtifactRuntimeError("; ".join(details))
    feed: Dict[str, np.ndarray] = {}
    symbolic_dimensions: Dict[str, int] = {}
    for spec in declared_inputs:
        name = str(spec["name"])
        array = np.asarray(inputs[name])
        _validate_array_against_tensor_spec(
            array,
            spec,
            "input %s" % name,
            symbolic_dimensions=symbolic_dimensions,
        )
        feed[name] = np.ascontiguousarray(array)

    try:
        from noema_lab.ops.portable_onnx import run_portable_onnx

        output_specs = list(entrypoint.get("outputs") or [])
        output_names = [str(item["name"]) for item in output_specs]
        model_batch_size = int(inference_batch_size or 0)
        if model_batch_size > 0:
            batch_symbol, total_batch = _batchable_entrypoint_shape(
                declared_inputs,
                output_specs,
                feed,
            )
        else:
            batch_symbol, total_batch = "", 0
        if model_batch_size > 0 and total_batch > model_batch_size:
            batched_outputs: Dict[str, np.ndarray] = {}
            for start in range(0, total_batch, model_batch_size):
                stop = min(total_batch, start + model_batch_size)
                chunk_feed = {
                    name: np.ascontiguousarray(value[start:stop])
                    for name, value in feed.items()
                }
                chunk_map = run_portable_onnx(component, chunk_feed, outputs=output_names)
                for name in output_names:
                    chunk = np.asarray(chunk_map[name])
                    if chunk.ndim < 1 or int(chunk.shape[0]) != stop - start:
                        raise TrainedArtifactRuntimeError(
                            "ONNX batched output %s does not preserve leading %s dimension"
                            % (name, batch_symbol)
                        )
                    if name not in batched_outputs:
                        batched_outputs[name] = np.empty(
                            (total_batch, *chunk.shape[1:]),
                            dtype=chunk.dtype,
                        )
                    elif batched_outputs[name].shape[1:] != chunk.shape[1:]:
                        raise TrainedArtifactRuntimeError(
                            "ONNX batched output %s changed non-batch shape between chunks"
                            % name
                        )
                    batched_outputs[name][start:stop] = chunk
            values = [batched_outputs[name] for name in output_names]
        else:
            output_map = run_portable_onnx(component, feed, outputs=output_names)
            values = [output_map[name] for name in output_names]
    except TrainedArtifactRuntimeError:
        raise
    except Exception as exc:
        raise TrainedArtifactRuntimeError("ONNX Runtime inference failed: %s" % exc) from exc
    result: Dict[str, np.ndarray] = {}
    for spec, value in zip(output_specs, values):
        name = str(spec["name"])
        array = np.asarray(value)
        _validate_array_against_tensor_spec(
            array,
            spec,
            "output %s" % name,
            symbolic_dimensions=symbolic_dimensions,
        )
        result[name] = array
    return result


def _batchable_entrypoint_shape(
    input_specs: Sequence[Mapping[str, Any]],
    output_specs: Sequence[Mapping[str, Any]],
    feed: Mapping[str, np.ndarray],
) -> Tuple[str, int]:
    """Require an ABI whose leading symbolic dimension is independently batchable."""

    all_specs = [*input_specs, *output_specs]
    leading_dimensions = [
        list(spec.get("shape") or [])[0]
        for spec in all_specs
        if list(spec.get("shape") or [])
    ]
    if len(leading_dimensions) != len(all_specs):
        raise TrainedArtifactRuntimeError(
            "inference batching requires every input and output to have a leading batch dimension"
        )
    batch_symbol = leading_dimensions[0] if leading_dimensions else None
    if not isinstance(batch_symbol, str) or not batch_symbol or any(
        dimension != batch_symbol for dimension in leading_dimensions
    ):
        raise TrainedArtifactRuntimeError(
            "inference batching requires one shared symbolic leading dimension in the tensor ABI"
        )
    batch_sizes = {int(np.asarray(value).shape[0]) for value in feed.values()}
    if len(batch_sizes) != 1:
        raise TrainedArtifactRuntimeError(
            "inference batching requires every input to have the same leading dimension"
        )
    return batch_symbol, next(iter(batch_sizes))


def _compare_signature(declared: Sequence[Mapping[str, Any]], actual: Sequence[Any], role: str) -> None:
    declared_by_name = {str(item.get("name") or ""): item for item in declared}
    actual_by_name = {str(item.name): item for item in actual}
    if set(declared_by_name) != set(actual_by_name):
        raise TrainedArtifactRuntimeError(
            "ONNX %s names do not match the tensor ABI; declared %s, graph has %s"
            % (role, sorted(declared_by_name), sorted(actual_by_name))
        )
    for name, spec in declared_by_name.items():
        value_info = actual_by_name[name]
        runtime_dtype = _ONNX_TYPE_TO_DTYPE.get(str(value_info.type))
        declared_dtype = str(spec.get("dtype") or "")
        if runtime_dtype != declared_dtype:
            raise TrainedArtifactRuntimeError(
                "ONNX %s %s has dtype %s; tensor ABI declares %s"
                % (role, name, runtime_dtype or value_info.type, declared_dtype)
            )
        declared_shape = list(spec.get("shape") or [])
        actual_shape = list(value_info.shape or [])
        if len(declared_shape) != len(actual_shape):
            raise TrainedArtifactRuntimeError(
                "ONNX %s %s rank %d does not match tensor ABI rank %d"
                % (role, name, len(actual_shape), len(declared_shape))
            )
        for axis, (declared_dim, actual_dim) in enumerate(zip(declared_shape, actual_shape)):
            if isinstance(declared_dim, int) and isinstance(actual_dim, int) and declared_dim != actual_dim:
                raise TrainedArtifactRuntimeError(
                    "ONNX %s %s axis %d has size %d; tensor ABI declares %d"
                    % (role, name, axis, actual_dim, declared_dim)
                )


def _validate_array_against_tensor_spec(
    array: np.ndarray,
    spec: Mapping[str, Any],
    label: str,
    *,
    symbolic_dimensions: Optional[Dict[str, int]] = None,
) -> None:
    dtype = str(spec.get("dtype") or "")
    if str(array.dtype) != dtype:
        raise TrainedArtifactRuntimeError(
            "%s has dtype %s; expected %s" % (label, array.dtype, dtype)
        )
    shape = list(spec.get("shape") or [])
    if array.ndim != len(shape):
        raise TrainedArtifactRuntimeError(
            "%s has rank %d; expected %d" % (label, array.ndim, len(shape))
        )
    for axis, expected in enumerate(shape):
        if isinstance(expected, int) and int(array.shape[axis]) != expected:
            raise TrainedArtifactRuntimeError(
                "%s axis %d has size %d; expected %d"
                % (label, axis, int(array.shape[axis]), expected)
            )
        if isinstance(expected, str) and symbolic_dimensions is not None:
            actual = int(array.shape[axis])
            previous = symbolic_dimensions.get(expected)
            if previous is None:
                symbolic_dimensions[expected] = actual
            elif previous != actual:
                raise TrainedArtifactRuntimeError(
                    "%s axis %d binds symbolic dimension %s to %d; expected %d"
                    % (label, axis, expected, actual, previous)
                )
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise TrainedArtifactRuntimeError("%s contains NaN or infinite values" % label)


def _runtime_value_info(value: Any) -> JsonDict:
    return {
        "name": str(value.name),
        "dtype": _ONNX_TYPE_TO_DTYPE.get(str(value.type), str(value.type)),
        "shape": [int(item) if isinstance(item, int) else str(item or "?") for item in value.shape or []],
    }


def _normalize_backend(value: Any) -> str:
    backend = str(value or "").strip().lower().replace("_", "-")
    return "onnxruntime" if backend == "onnx-runtime" else backend


def _validated_package_sha256(value: Any) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest
    ):
        raise TrainedArtifactRuntimeError(
            "trained-artifact execution requires artifact_package_sha256"
        )
    return digest


def _require_admissible_inspection(
    inspected: Mapping[str, Any],
    expected_digest: str,
) -> None:
    if int(inspected.get("schema_version") or 0) != 2:
        raise TrainedArtifactRuntimeError(
            "generic executable entrypoints require "
            "noema.trained_block_artifact schema_version=2"
        )
    if inspected.get("issues"):
        raise TrainedArtifactRuntimeError(
            "trained artifact is invalid: %s"
            % "; ".join(str(item) for item in inspected.get("issues") or [])
        )
    actual_digest = str(
        inspected.get("package_sha256") or ""
    ).strip().lower()
    if actual_digest != expected_digest:
        raise TrainedArtifactRuntimeError(
            "trained-artifact package identity mismatch: expected %s, got %s"
            % (expected_digest, actual_digest or "<missing>")
        )


def _require_supported_runtime_abi(runtime: Mapping[str, Any]) -> None:
    raw_abi_version = runtime.get("abi_version")
    if (
        isinstance(raw_abi_version, bool)
        or not isinstance(raw_abi_version, int)
        or raw_abi_version not in SUPPORTED_TRAINED_ARTIFACT_ABI_VERSIONS
    ):
        raise TrainedArtifactRuntimeError(
            "unsupported trained-artifact ABI version: %s (supported: %s)"
            % (
                raw_abi_version,
                ", ".join(
                    str(value)
                    for value in sorted(
                        SUPPORTED_TRAINED_ARTIFACT_ABI_VERSIONS
                    )
                ),
            )
        )


def _confined_path(root: Path, value: Any) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise TrainedArtifactRuntimeError("component path must not be empty")
    relative = Path(raw)
    if relative.is_absolute():
        raise TrainedArtifactRuntimeError("schema-v2 component paths must be relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TrainedArtifactRuntimeError("component path escapes the artifact package: %s" % raw) from exc
    return resolved


def _resolve_inspected_path(value: Any, project_root: Path, manifest_dir: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_candidate = (Path(project_root).resolve() / path).resolve()
    if project_candidate.is_file():
        return project_candidate
    return (Path(manifest_dir).resolve() / path).resolve()


def _resolve_confined_inspected_component_path(
    value: Any,
    *,
    project_root: Path,
    package_root: Path,
) -> Path:
    resolved_package_root = Path(package_root).resolve()
    path = _resolve_inspected_path(
        value,
        Path(project_root).resolve(),
        resolved_package_root,
    )
    try:
        path.relative_to(resolved_package_root)
    except ValueError as exc:
        raise TrainedArtifactRuntimeError(
            "component path escapes the artifact package: %s"
            % str(value or "")
        ) from exc
    return path


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
