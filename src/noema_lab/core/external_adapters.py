from __future__ import annotations

import importlib
import importlib.util
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationRegistry,
    OperationResult,
    normalize_backends,
    normalize_differentiability,
    normalize_equivalence,
    normalize_formats,
    normalize_materializations,
    object_schema,
)
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.structured_input import (
    StructuredInputError,
    load_strict_yaml_or_json,
)

JsonDict = Dict[str, Any]

ADAPTER_ENV_VAR = "NOEMA_ADAPTER_PATHS"
MANIFEST_FILENAMES = (
    "noema_adapter.yaml",
    "noema_adapter.yml",
    "adapter.yaml",
    "adapter.yml",
    "noema_adapter.json",
    "adapter.json",
)


@dataclass
class ExternalAdapterOperationSpec:
    id: str
    name: str
    wraps: str
    adapter_params: JsonDict = field(default_factory=dict)
    fixed_params: JsonDict = field(default_factory=dict)
    params_schema: JsonDict = field(default_factory=lambda: object_schema())
    description: str = ""
    differentiability: Optional[JsonDict] = None
    backends: Optional[JsonDict] = None
    equivalence: Optional[JsonDict] = None
    formats: Optional[JsonDict] = None
    materializations: Optional[List[JsonDict]] = None


@dataclass
class ExternalAdapterManifest:
    path: Path
    schema_version: int
    name: str
    version: str = ""
    description: str = ""
    training: JsonDict = field(default_factory=dict)
    operations: List[ExternalAdapterOperationSpec] = field(default_factory=list)


class ManifestWrappedOperation(Operation):
    def __init__(
        self,
        manifest: ExternalAdapterManifest,
        spec: ExternalAdapterOperationSpec,
        wrapped: Operation,
    ) -> None:
        self.id = spec.id
        self.name = spec.name
        self.status = "implemented"
        self.input_kinds = dict(wrapped.input_kinds)
        self.optional_input_kinds = dict(getattr(wrapped, "optional_input_kinds", {}) or {})
        self.output_kinds = dict(wrapped.output_kinds)
        self.entropy_info = dict(getattr(wrapped, "entropy_info", {}) or {})
        self.differentiability = dict(spec.differentiability or getattr(wrapped, "differentiability", {}) or {})
        self.backends = dict(spec.backends or getattr(wrapped, "backends", {}) or {})
        self.equivalence = dict(spec.equivalence or getattr(wrapped, "equivalence", {}) or {})
        self.formats = dict(spec.formats or getattr(wrapped, "formats", {}) or {})
        self.materializations = list(spec.materializations) if spec.materializations is not None else getattr(wrapped, "materializations", None)
        self.params_schema = dict(spec.params_schema or object_schema())
        # External Python and the wrapped runtime are not presumed re-entrant.
        # They execute exclusively unless a future manifest schema introduces
        # a separately validated safety declaration.
        self.thread_safe = False
        self._manifest = manifest
        self._spec = spec
        self._wrapped = wrapped
        self._source_identity = _adapter_source_identity(manifest, spec)
        # Import and resolve the callable during registration, before an
        # execution plan can create durable run storage.
        _load_adapter_callable(manifest, spec)

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["external_adapter"] = {
            "schema_version": self._manifest.schema_version,
            "manifest": str(self._manifest.path),
            "name": self._manifest.name,
            "version": self._manifest.version,
            "description": self._manifest.description,
            "wraps": self._spec.wraps,
            "operation_description": self._spec.description,
            **self._source_identity,
        }
        if self._manifest.training:
            payload["external_adapter"]["training"] = dict(self._manifest.training)
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        current_identity = _adapter_source_identity(self._manifest, self._spec)
        if current_identity != self._source_identity:
            raise OperationError(
                "external adapter bytes changed after registration; rebuild the execution plan"
            )
        merged_params = _merged_operation_params(self._manifest, self._spec, ctx.params)
        wrapped_ctx = OperationContext(
            recipe_name=ctx.recipe_name,
            step_id=ctx.step_id,
            params=merged_params,
            inputs=ctx.inputs,
            run_dir=ctx.run_dir,
            step_dir=ctx.step_dir,
            progress_sink=ctx.progress_sink,
            master_seed=ctx.master_seed,
            seed_namespace=ctx.seed_namespace,
            cancellation_token=ctx.cancellation_token,
        )
        result = self._wrapped.run(wrapped_ctx)
        result.metadata.update(
            {
                "external_adapter_sdk": {
                    "manifest": str(self._manifest.path),
                    "adapter": self._manifest.name,
                    "adapter_version": self._manifest.version,
                    "operation_id": self.id,
                    "wraps": self._spec.wraps,
                    **self._source_identity,
                }
            }
        )
        if self._manifest.training:
            result.metadata["external_adapter_sdk"]["training"] = dict(self._manifest.training)
        for output in result.outputs.values():
            output.metadata.setdefault("external_adapter_sdk", result.metadata["external_adapter_sdk"])
        return result


def adapter_paths_from_env() -> List[Path]:
    raw = os.environ.get(ADAPTER_ENV_VAR, "")
    if not raw.strip():
        return []
    return [Path(item).expanduser() for item in raw.split(os.pathsep) if item.strip()]


def discover_adapter_manifests(paths: Iterable[Path | str]) -> List[Path]:
    manifests: List[Path] = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        candidates: List[Path]
        if path.is_dir():
            candidates = [path / name for name in MANIFEST_FILENAMES]
        else:
            candidates = [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            if resolved.suffix.lower() not in {".yaml", ".yml", ".json"}:
                continue
            manifests.append(resolved)
            seen.add(resolved)
            break
    return manifests


def register_external_adapter_operations(registry: OperationRegistry, paths: Optional[Iterable[Path | str]] = None) -> List[ExternalAdapterManifest]:
    requested_paths = adapter_paths_from_env() if paths is None else list(paths)
    manifests = [load_adapter_manifest(path) for path in discover_adapter_manifests(requested_paths)]
    pending: List[ManifestWrappedOperation] = []
    pending_ids = set()
    for manifest in manifests:
        for spec in manifest.operations:
            if spec.id in pending_ids:
                raise OperationError(
                    "duplicate external adapter operation id: %s" % spec.id
                )
            try:
                registry.get(spec.id)
            except OperationError:
                pass
            else:
                raise OperationError(
                    "adapter operation id conflicts with an existing operation: %s"
                    % spec.id
                )
            wrapped = registry.get(spec.wraps)
            pending.append(ManifestWrappedOperation(manifest, spec, wrapped))
            pending_ids.add(spec.id)
    # Registration is deliberately two-phase: a missing import in the final
    # adapter cannot leave earlier operations partially installed.
    for operation in pending:
        registry.register(operation)
    return manifests


def load_adapter_manifest(path: Path | str) -> ExternalAdapterManifest:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("adapter manifest does not exist: %s" % manifest_path)
    try:
        payload = load_strict_yaml_or_json(manifest_path)
    except StructuredInputError as exc:
        raise OperationError(
            "invalid adapter YAML/JSON manifest %s: %s" % (manifest_path, exc)
        ) from exc
    if not isinstance(payload, dict):
        raise OperationError("adapter manifest must be a mapping: %s" % manifest_path)
    schema_version = int(payload.get("schema_version") or 1)
    if schema_version != 1:
        raise OperationError("unsupported adapter manifest schema_version: %s" % schema_version)
    name = str(payload.get("name") or manifest_path.parent.name or manifest_path.stem)
    operations_payload = payload.get("operations") or []
    if not isinstance(operations_payload, list) or not operations_payload:
        raise OperationError("adapter manifest must define at least one operation")
    operations = [_operation_spec_from_payload(item, manifest_path, index) for index, item in enumerate(operations_payload)]
    return ExternalAdapterManifest(
        path=manifest_path,
        schema_version=schema_version,
        name=name,
        version=str(payload.get("version") or ""),
        description=str(payload.get("description") or ""),
        training=_training_from_payload(payload.get("training"), manifest_path),
        operations=operations,
    )


def validate_adapter_manifest(path: Path | str, registry: OperationRegistry, import_callables: bool = True) -> JsonDict:
    manifest = load_adapter_manifest(path)
    operation_ids = set()
    rows = []
    for spec in manifest.operations:
        if spec.id in operation_ids:
            raise OperationError("duplicate operation id in adapter manifest: %s" % spec.id)
        operation_ids.add(spec.id)
        try:
            registry.get(spec.id)
        except OperationError:
            pass
        else:
            raise OperationError("adapter operation id conflicts with an existing operation: %s" % spec.id)
        wrapped = registry.get(spec.wraps)
        if import_callables:
            _load_adapter_callable(manifest, spec)
        rows.append(
            {
                "id": spec.id,
                "name": spec.name,
                "wraps": spec.wraps,
                "input_kinds": dict(wrapped.input_kinds),
                "optional_input_kinds": dict(getattr(wrapped, "optional_input_kinds", {}) or {}),
                "output_kinds": dict(wrapped.output_kinds),
                "callable": _callable_label(spec),
            }
        )
    return {
        "status": "valid",
        "manifest": str(manifest.path),
        "name": manifest.name,
        "version": manifest.version,
        "training": dict(manifest.training),
        "operation_count": len(rows),
        "operations": rows,
    }


def scaffold_adapter(directory: Path | str, name: str = "example_external_codec", kind: str = "bits", force: bool = False) -> JsonDict:
    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "noema_adapter.yaml"
    module_path = target / "adapter.py"
    if not force:
        existing = [path for path in (manifest_path, module_path) if path.exists()]
        if existing:
            raise FileExistsError("adapter scaffold target already contains: %s" % ", ".join(str(path) for path in existing))
    safe_name = _safe_identifier(name)
    manifest_path.write_text(_scaffold_manifest(safe_name, kind), encoding="utf-8")
    module_path.write_text(_scaffold_module(kind), encoding="utf-8")
    return {
        "status": "created",
        "directory": str(target),
        "manifest": str(manifest_path),
        "module": str(module_path),
        "name": safe_name,
        "kind": kind,
    }


def _operation_spec_from_payload(payload: Any, manifest_path: Path, index: int) -> ExternalAdapterOperationSpec:
    if not isinstance(payload, dict):
        raise OperationError("adapter operation %d must be a mapping" % (index + 1))
    operation_id = str(payload.get("id") or "")
    if not operation_id:
        raise OperationError("adapter operation %d is missing id" % (index + 1))
    wraps = str(payload.get("wraps") or "")
    if not wraps:
        raise OperationError("adapter operation %s is missing wraps" % operation_id)
    adapter = dict(payload.get("adapter") or {})
    fixed_params = dict(payload.get("fixed_params") or {})
    if not adapter.get("callable"):
        raise OperationError("adapter operation %s is missing adapter.callable" % operation_id)
    if adapter.get("path"):
        adapter["path"] = _resolve_manifest_relative_path(manifest_path, str(adapter["path"]))
    params_schema = payload.get("params_schema") or object_schema()
    if not isinstance(params_schema, dict):
        raise OperationError("adapter operation %s params_schema must be a mapping" % operation_id)
    differentiability = payload.get("differentiability")
    if differentiability is not None and not isinstance(differentiability, dict):
        raise OperationError("adapter operation %s differentiability must be a mapping" % operation_id)
    if differentiability is not None:
        differentiability = normalize_differentiability(differentiability, operation_id)
    backends = payload.get("backends")
    if backends is not None and not isinstance(backends, dict):
        raise OperationError("adapter operation %s backends must be a mapping" % operation_id)
    if backends is not None:
        backends = normalize_backends(backends, operation_id)
    equivalence = payload.get("equivalence")
    if equivalence is not None and not isinstance(equivalence, dict):
        raise OperationError("adapter operation %s equivalence must be a mapping" % operation_id)
    if equivalence is not None:
        equivalence = normalize_equivalence(equivalence, operation_id)
    formats = payload.get("formats")
    if formats is not None and not isinstance(formats, dict):
        raise OperationError("adapter operation %s formats must be a mapping" % operation_id)
    if formats is not None:
        formats = normalize_formats(formats, operation_id)
    materializations = payload.get("materializations")
    if materializations is not None:
        materialization_backends = backends or normalize_backends(None, operation_id)
        materializations = normalize_materializations(materializations, materialization_backends, operation_id)
    return ExternalAdapterOperationSpec(
        id=operation_id,
        name=str(payload.get("name") or operation_id),
        wraps=wraps,
        adapter_params=adapter,
        fixed_params=fixed_params,
        params_schema=params_schema,
        description=str(payload.get("description") or ""),
        differentiability=dict(differentiability) if differentiability is not None else None,
        backends=dict(backends) if backends is not None else None,
        equivalence=dict(equivalence) if equivalence is not None else None,
        formats=dict(formats) if formats is not None else None,
        materializations=list(materializations) if materializations is not None else None,
    )


def _training_from_payload(payload: Any, manifest_path: Path) -> JsonDict:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise OperationError("adapter manifest training must be a mapping")
    allowed = {
        "source_recipe",
        "source_recipe_sha256",
        "differentiable_export_id",
        "capture_id",
        "framework",
        "checkpoint_path",
        "checkpoint_sha256",
        "input_schema",
        "output_schema",
        "model_card",
    }
    unknown = set(payload.keys()) - allowed
    if unknown:
        raise OperationError(
            "adapter manifest training has unknown field(s): %s"
            % ", ".join(sorted(str(item) for item in unknown))
        )
    training: JsonDict = {}
    for key, value in payload.items():
        if value is None:
            continue
        if key in {"input_schema", "output_schema"}:
            if not isinstance(value, Mapping):
                raise OperationError("adapter manifest training.%s must be a mapping" % key)
            training[key] = dict(value)
        else:
            training[key] = str(value)
    for key in ("source_recipe", "checkpoint_path", "model_card"):
        if key in training and str(training[key]).strip():
            training[key] = _resolve_manifest_relative_path(manifest_path, str(training[key]))
    checkpoint = str(training.get("checkpoint_path") or "")
    if checkpoint:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise OperationError("adapter training.checkpoint_path does not exist: %s" % checkpoint_path)
        actual_sha = _file_sha256(checkpoint_path)
        declared_sha = str(training.get("checkpoint_sha256") or "").strip().lower()
        if declared_sha and declared_sha != actual_sha:
            raise OperationError(
                "adapter training.checkpoint_sha256 mismatch for %s: expected %s, got %s"
                % (checkpoint_path, declared_sha, actual_sha)
            )
        training["checkpoint_path"] = str(checkpoint_path)
        training["checkpoint_sha256"] = actual_sha
        training["checkpoint_size_bytes"] = int(checkpoint_path.stat().st_size)
    return training


def _resolve_manifest_relative_path(manifest_path: Path, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return str(path.resolve())


def _merged_operation_params(manifest: ExternalAdapterManifest, spec: ExternalAdapterOperationSpec, user_params: Mapping[str, Any]) -> JsonDict:
    params: JsonDict = {}
    params.update(spec.adapter_params)
    params.update(spec.fixed_params)
    params.update(dict(user_params or {}))
    if manifest.training:
        params.setdefault("training", dict(manifest.training))
        if manifest.training.get("checkpoint_path"):
            params.setdefault("checkpoint_path", manifest.training["checkpoint_path"])
            params.setdefault("checkpoint_sha256", manifest.training.get("checkpoint_sha256", ""))
    if params.get("path"):
        params["path"] = _resolve_manifest_relative_path(manifest.path, str(params["path"]))
    return params


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_source_identity(
    manifest: ExternalAdapterManifest,
    spec: ExternalAdapterOperationSpec,
) -> JsonDict:
    """Return content identities for the manifest and executable adapter source."""

    manifest_path = Path(manifest.path).resolve()
    params = _merged_operation_params(manifest, spec, {})
    raw_path = str(params.get("path") or "").strip()
    module_name = str(params.get("module") or "").strip()
    if raw_path:
        source_path = Path(raw_path).expanduser().resolve()
        if source_path.is_dir():
            if not module_name:
                raise OperationError(
                    "directory-backed external adapter %s must declare adapter.module"
                    % spec.id
                )
            source_path = source_path.joinpath(*module_name.split("."))
            if source_path.is_dir():
                source_path = source_path / "__init__.py"
            elif source_path.suffix != ".py":
                source_path = source_path.with_suffix(".py")
    elif module_name:
        module_spec = importlib.util.find_spec(module_name)
        origin = getattr(module_spec, "origin", None) if module_spec else None
        if not origin or origin in {"built-in", "frozen"}:
            raise OperationError(
                "external adapter %s module source cannot be content-identified"
                % spec.id
            )
        source_path = Path(origin).resolve()
    else:
        raise OperationError(
            "external adapter %s requires a file-backed path or module" % spec.id
        )
    if not source_path.is_file():
        raise OperationError(
            "external adapter %s source does not exist: %s" % (spec.id, source_path)
        )
    source_root = source_path.parent
    if raw_path and Path(raw_path).expanduser().resolve().is_dir():
        source_root = Path(raw_path).expanduser().resolve()
    elif not raw_path:
        if source_path.name == "__init__.py":
            source_root = source_path.parent
        elif "." in module_name:
            source_root = source_path.parent
        else:
            # A top-level installed module should not accidentally bind every
            # Python file in site-packages.
            source_root = source_path
    inventory = _python_source_inventory(source_root)
    return {
        "manifest_sha256": _file_sha256(manifest_path),
        "callable_source": str(source_path),
        "callable_sha256": _file_sha256(source_path),
        "callable_source_root": str(source_root),
        "python_source_inventory": inventory,
        "python_source_tree_sha256": canonical_json_sha256(inventory),
    }


def _python_source_inventory(root: Path) -> List[JsonDict]:
    """Best-effort transitive identity for a local Python adapter package."""

    resolved_root = root.resolve()
    if resolved_root.is_file():
        if resolved_root.is_symlink() or resolved_root.suffix != ".py":
            raise OperationError("external adapter source is unsafe: %s" % resolved_root)
        return [
            {
                "path": resolved_root.name,
                "sha256": _file_sha256(resolved_root),
                "size_bytes": int(resolved_root.stat().st_size),
            }
        ]
    if not resolved_root.is_dir():
        raise OperationError("external adapter source root is missing: %s" % resolved_root)
    paths = sorted(resolved_root.rglob("*.py"), key=lambda item: item.as_posix())
    if len(paths) > 4096:
        raise OperationError(
            "external adapter source tree is unexpectedly large (%d Python files)"
            % len(paths)
        )
    inventory: List[JsonDict] = []
    total_bytes = 0
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise OperationError("external adapter source is unsafe: %s" % path)
        size = int(path.stat().st_size)
        total_bytes += size
        if total_bytes > 256 * 1024 * 1024:
            raise OperationError("external adapter Python source tree exceeds 256 MiB")
        inventory.append(
            {
                "path": path.relative_to(resolved_root).as_posix(),
                "sha256": _file_sha256(path),
                "size_bytes": size,
            }
        )
    if not inventory:
        raise OperationError(
            "external adapter source tree contains no Python files: %s" % resolved_root
        )
    return inventory


def _load_adapter_callable(manifest: ExternalAdapterManifest, spec: ExternalAdapterOperationSpec):
    params = _merged_operation_params(manifest, spec, {})
    callable_name = str(params.get("callable") or "")
    module_name = str(params.get("module") or "")
    path = str(params.get("path") or "")
    if ":" in callable_name and not module_name:
        module_name, callable_name = callable_name.split(":", 1)
    if not callable_name:
        raise OperationError("adapter operation %s is missing callable" % spec.id)
    if path:
        path_obj = Path(path).expanduser().resolve()
        if path_obj.is_file():
            module = _load_module_from_file(path_obj)
        else:
            if not module_name:
                raise OperationError("adapter operation %s path is a directory, so adapter.module is required" % spec.id)
            module = _import_module_from_directory(path_obj, module_name)
    else:
        if not module_name:
            raise OperationError("adapter operation %s requires adapter.path or adapter.module" % spec.id)
        module = importlib.import_module(module_name)
    target = module
    try:
        for part in callable_name.split("."):
            if not part:
                raise AttributeError("empty callable path component")
            target = getattr(target, part)
    except AttributeError as exc:
        raise OperationError(
            "adapter operation %s cannot resolve callable %s: %s"
            % (spec.id, callable_name, exc)
        ) from exc
    if not callable(target):
        raise OperationError("adapter operation %s target is not callable: %s" % (spec.id, callable_name))
    return target


def _load_module_from_file(path: Path):
    module_name = "noema_sdk_adapter_%s" % hashlib.sha256(
        str(path).encode("utf-8")
    ).hexdigest()[:20]
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise OperationError("could not load adapter module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_module_from_directory(path: Path, module_name: str):
    import sys

    sys.path.insert(0, str(path))
    try:
        return importlib.import_module(module_name)
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass


def _callable_label(spec: ExternalAdapterOperationSpec) -> str:
    params = {}
    params.update(spec.adapter_params)
    params.update(spec.fixed_params)
    module = params.get("module")
    path = params.get("path")
    target = params.get("callable")
    if module:
        return "%s:%s" % (module, target)
    if path:
        return "%s:%s" % (path, target)
    return str(target or "")


def _safe_identifier(value: str) -> str:
    text = "".join(char.lower() if char.isalnum() else "_" for char in str(value or "adapter"))
    text = "_".join(part for part in text.split("_") if part)
    return text or "adapter"


def _scaffold_manifest(name: str, kind: str) -> str:
    if kind not in {"bits", "indices", "latents", "deepjscc_symbols", "classification_dataset", "classification_metric"}:
        raise OperationError("unsupported adapter scaffold kind: %s" % kind)
    if kind == "bits":
        operations = [
            {
                "id": "model.%s_encode_bits" % name,
                "name": "%s encode to payload bits" % name,
                "wraps": "model.external_encode_bits",
                "adapter": {"path": "adapter.py", "callable": "encode_bits", "call_style": "array_params"},
                "fixed_params": {"bit_storage": "unpacked_bits", "bit_order": "big"},
                "params_schema": object_schema({"image_shape": {"type": "array", "default": [1, 8, 8, 3]}}, additional=True),
            },
            {
                "id": "model.%s_decode_bits" % name,
                "name": "%s decode from payload bits" % name,
                "wraps": "model.external_decode_bits",
                "adapter": {"path": "adapter.py", "callable": "decode_bits", "call_style": "array_params"},
                "fixed_params": {"bit_storage": "unpacked_bits", "bit_order": "big"},
                "params_schema": object_schema({"image_shape": {"type": "array", "default": [1, 8, 8, 3]}}, additional=True),
            },
        ]
    elif kind == "indices":
        operations = [
            {
                "id": "model.%s_encode_indices" % name,
                "name": "%s encode to indices" % name,
                "wraps": "model.external_encode_indices",
                "adapter": {"path": "adapter.py", "callable": "encode_indices", "call_style": "array_params"},
                "params_schema": object_schema(additional=True),
            },
            {
                "id": "model.%s_decode_indices" % name,
                "name": "%s decode from indices" % name,
                "wraps": "model.external_decode_indices",
                "adapter": {"path": "adapter.py", "callable": "decode_indices", "call_style": "array_params"},
                "params_schema": object_schema({"image_shape": {"type": "array", "default": [1, 8, 8, 3]}}, additional=True),
            },
        ]
    elif kind == "latents":
        operations = [
            {
                "id": "model.%s_encode_latents" % name,
                "name": "%s encode to latents" % name,
                "wraps": "model.external_encode_latents",
                "adapter": {"path": "adapter.py", "callable": "encode_latents", "call_style": "array_params"},
                "params_schema": object_schema(additional=True),
            },
            {
                "id": "model.%s_decode_latents" % name,
                "name": "%s decode from latents" % name,
                "wraps": "model.external_decode_latents",
                "adapter": {"path": "adapter.py", "callable": "decode_latents", "call_style": "array_params"},
                "params_schema": object_schema({"image_shape": {"type": "array", "default": [1, 8, 8, 3]}}, additional=True),
            },
        ]
    elif kind == "deepjscc_symbols":
        operations = [
            {
                "id": "model.%s_deepjscc_encode" % name,
                "name": "%s DeepJSCC encode to channel symbols" % name,
                "wraps": "model.deepjscc_external_encode",
                "adapter": {"path": "adapter.py", "callable": "encode_symbols", "call_style": "array_params"},
                "params_schema": object_schema(additional=True),
            },
            {
                "id": "model.%s_deepjscc_decode" % name,
                "name": "%s DeepJSCC decode from channel symbols" % name,
                "wraps": "model.deepjscc_external_decode",
                "adapter": {"path": "adapter.py", "callable": "decode_symbols", "call_style": "array_params"},
                "params_schema": object_schema({"image_shape": {"type": "array", "default": [1, 8, 8, 3]}}, additional=True),
            },
        ]
    elif kind == "classification_dataset":
        operations = [
            {
                "id": "source.%s_classification_dataset" % name,
                "name": "%s classification dataset" % name,
                "wraps": "source.external_classification_dataset",
                "adapter": {"path": "adapter.py", "callable": "load_classification_examples", "call_style": "dict"},
                "params_schema": object_schema({"dataset": {"type": "string", "default": name}}, additional=True),
            }
        ]
    else:
        operations = [
            {
                "id": "metrics.%s_classification" % name,
                "name": "%s classification metric" % name,
                "wraps": "metrics.external_classification",
                "adapter": {"path": "adapter.py", "callable": "score_classification", "call_style": "dict"},
                "params_schema": object_schema(additional=True),
            }
        ]
    payload = {
        "schema_version": 1,
        "name": name,
        "version": "0.1.0",
        "description": "Noema external adapter manifest.",
        "operations": operations,
    }
    return yaml.safe_dump(payload, sort_keys=False)


def _scaffold_module(kind: str) -> str:
    common = '''from __future__ import annotations

import numpy as np


def _image_shape(params, default_count=1):
    shape = params.get("image_shape") or [default_count, 8, 8, 3]
    return tuple(int(value) for value in shape)

'''
    if kind == "bits":
        return common + '''
def encode_bits(images, params):
    bits = np.zeros(8, dtype=np.uint8)
    return {
        "array": bits,
        "metadata": {
            "bit_count": int(bits.size),
            "example_count": int(images.shape[0]) if getattr(images, "ndim", 0) >= 4 else 1,
            "adapter_note": "Replace this template with your payload bit encoder.",
        },
    }


def decode_bits(bits, params):
    shape = _image_shape(params)
    return np.zeros(shape, dtype=np.uint8)
'''
    if kind == "indices":
        return common + '''
def encode_indices(images, params):
    count = int(images.shape[0]) if getattr(images, "ndim", 0) >= 4 else 1
    indices = np.zeros((count, 1), dtype=np.int64)
    return {"array": indices, "metadata": {"codebook_size": 2}}


def decode_indices(indices, params):
    shape = _image_shape(params, int(indices.shape[0]) if getattr(indices, "ndim", 0) else 1)
    return np.zeros(shape, dtype=np.uint8)
'''
    if kind == "latents":
        return common + '''
def encode_latents(images, params):
    count = int(images.shape[0]) if getattr(images, "ndim", 0) >= 4 else 1
    latents = np.zeros((count, 1, 1, 1), dtype=np.float32)
    return {"array": latents, "metadata": {"latent_layout": "NCHW"}}


def decode_latents(latents, params):
    shape = _image_shape(params, int(latents.shape[0]) if getattr(latents, "ndim", 0) else 1)
    return np.zeros(shape, dtype=np.uint8)
'''
    if kind == "deepjscc_symbols":
        return common + '''
def encode_symbols(images, params):
    symbols = np.zeros(8, dtype=np.complex64)
    return {"array": symbols, "metadata": {"channel_use_count": int(symbols.size)}}


def decode_symbols(symbols, params):
    shape = _image_shape(params)
    return np.zeros(shape, dtype=np.uint8)
'''
    if kind == "classification_dataset":
        return '''from __future__ import annotations


def load_classification_examples(request):
    return {
        "metadata": {"dataset": request["params"].get("dataset", "external_classification")},
        "examples": [
            {"id": "a", "label": "clear", "prediction": "clear"},
            {"id": "b", "label": "faded", "prediction": "clear"},
            {"id": "c", "label": "blocked", "prediction": "blocked"},
        ],
    }
'''
    return '''from __future__ import annotations


def score_classification(request):
    reference = {item["id"]: item.get("label", "") for item in request["reference"]}
    candidate = {item["id"]: item.get("prediction", item.get("label", "")) for item in request["candidate"]}
    rows = []
    correct = 0
    for example_id, expected in reference.items():
        predicted = candidate.get(example_id, "")
        match = str(expected).strip().lower() == str(predicted).strip().lower()
        correct += 1 if match else 0
        rows.append({"id": example_id, "expected": expected, "predicted": predicted, "exact_match": 1.0 if match else 0.0})
    total = len(rows) or 1
    accuracy = float(correct) / float(total)
    return {
        "metric_family": "external_classification",
        "metrics": {
            "external.classification.accuracy": accuracy,
            "task.accuracy": accuracy,
        },
        "per_example": rows,
    }
'''
