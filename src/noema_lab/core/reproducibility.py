from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from noema_lab.core.recipes import Recipe
from noema_lab.core.research import research_specs_from_recipe

JsonDict = Dict[str, Any]

SEED_MODULUS = 2**31 - 1

_DIRECT_DEPENDENCIES = (
    "jsonschema",
    "matplotlib",
    "numpy",
    "Pillow",
    "PyYAML",
)
_OPTIONAL_DEPENDENCY_GROUPS = {
    "compressai": ("compressai",),
    "diffusers": ("diffusers", "torch"),
    "eflic": ("torch", "torchvision"),
    "onnx": ("onnx", "onnxruntime", "onnxscript", "torch", "compressai"),
    "openvino": ("openvino", "onnx", "onnxscript", "torch", "compressai"),
    # ``sionna`` is the canonical runtime identity recorded in operation
    # contracts. ``sionna-no-rt`` is the actual lightweight distribution used
    # by Noema's PHY/SYS-only wireless extra.
    "wireless": ("torch", "sionna", "sionna-no-rt"),
    "textgen": ("torch", "transformers", "safetensors"),
    "vqa-data": ("pandas", "pyarrow"),
    "retrieval-data": ("pandas", "pyarrow"),
    "vision": ("torch", "torchvision", "ultralytics"),
    "foundation": (
        "torch",
        "torchvision",
        "transformers",
        "sentence-transformers",
        "diffusers",
        "accelerate",
        "safetensors",
    ),
    "upstream-lic": (
        "torch",
        "torchvision",
        "compressai",
        "einops",
        "timm",
        "pytorch-msssim",
        "pybind11",
        "cmake",
    ),
    "neural": (
        "compressai",
        "diffusers",
        "torch",
        "torchvision",
        "einops",
        "timm",
        "pytorch-msssim",
        "pybind11",
        "cmake",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "openvino",
    ),
}
_RUNTIME_ENVIRONMENT_ALLOWLIST = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_LAUNCH_BLOCKING",
    "NVIDIA_TF32_OVERRIDE",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTHONHASHSEED",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_json_sha256(payload: Any) -> str:
    """Hash an actual JSON value using a deterministic, fail-closed encoding.

    Provenance identities must not silently turn ``Path``/``datetime`` objects
    into strings, conflate integer object keys with string keys, or admit
    non-finite numbers which are outside JSON.  Callers must normalize such
    values explicitly before asking for a content identity.
    """

    _validate_canonical_json_value(payload, path="$")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_canonical_json_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON contains a non-finite number at %s" % path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_canonical_json_value(item, path="%s[%d]" % (path, index))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "canonical JSON object keys must be strings at %s, got %s"
                    % (path, type(key).__name__)
                )
            _validate_canonical_json_value(item, path="%s.%s" % (path, key))
        return
    raise TypeError(
        "canonical JSON contains unsupported %s at %s"
        % (type(value).__name__, path)
    )


def recipe_fingerprint(recipe: Recipe) -> str:
    return canonical_json_sha256(recipe.to_dict())


def master_seed_from_recipe(recipe: Recipe) -> Optional[int]:
    metadata = dict(recipe.metadata or {})
    for key in ("seed", "experiment_seed", "ui_seed"):
        if key in metadata and metadata[key] is not None:
            return _coerce_seed(metadata[key], key)
    return None


def seed_namespace_from_recipe(recipe: Recipe) -> str:
    """Return the stable namespace used to derive operation RNG streams.

    A recipe's display name is the default namespace. Internal runners may use
    distinct execution names for provenance while retaining a stable seed
    namespace in metadata. This is important for Dataset Capture, where the
    configured seed mode -- rather than an internal run-name suffix -- must be
    the sole authority over whether consecutive runs share or change seeds.
    """

    metadata = dict(recipe.metadata or {})
    value = metadata.get("seed_namespace")
    if value is None:
        return str(recipe.name)
    namespace = str(value).strip()
    return namespace or str(recipe.name)


def derive_seed(master_seed: int, recipe_name: str, step_id: str, stream: str = "default") -> int:
    payload = "%d|%s|%s|%s" % (int(master_seed), recipe_name, step_id, stream)
    value = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    return int(value % SEED_MODULUS)


def seed_policy(recipe: Recipe) -> JsonDict:
    master_seed = master_seed_from_recipe(recipe)
    return {
        "schema_version": 1,
        "master_seed": master_seed,
        "master_seed_source": _master_seed_source(recipe),
        "seed_namespace": seed_namespace_from_recipe(recipe),
        "derivation": "sha256(master_seed|seed_namespace|step_id|stream) mod %d"
        % SEED_MODULUS,
        "operation_param_seed_overrides_master": True,
    }


def environment_snapshot() -> JsonDict:
    project_root = _project_root()
    dependency_names = sorted(
        {
            *_DIRECT_DEPENDENCIES,
            *(
                name
                for dependencies in _OPTIONAL_DEPENDENCY_GROUPS.values()
                for name in dependencies
            ),
            # Retain TensorFlow version capture solely so frozen Sionna 1.x
            # provenance remains inspectable. The supported runtime is Sionna
            # 2.x on PyTorch.
            "tensorflow",
        },
        key=str.lower,
    )
    return {
        "schema_version": 1,
        "captured_at_utc": utc_now_iso(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "hardware": {
            "cpu": _cpu_snapshot(),
            "accelerators": {
                **_accelerator_snapshot(),
                "linux_drm": _linux_drm_accelerator_snapshot(),
            },
        },
        "process": {
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
        },
        "noema": {
            "version": _distribution_version("noema-lab"),
        },
        "dependencies": _dependency_versions(dependency_names),
        "dependency_groups": {
            "base": list(_DIRECT_DEPENDENCIES),
            "optional": {
                group: list(dependencies)
                for group, dependencies in sorted(
                    _OPTIONAL_DEPENDENCY_GROUPS.items()
                )
            },
        },
        "runtime": {
            "thread_and_determinism_environment": (
                _runtime_environment_snapshot()
            ),
            "linear_algebra": _linear_algebra_runtime_snapshot(),
        },
        "native": _native_snapshot(),
        "project_files": _project_file_hashes(project_root),
        "git": git_snapshot(project_root),
    }


def _cpu_snapshot() -> JsonDict:
    payload: JsonDict = {
        "logical_core_count": os.cpu_count(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return payload
    try:
        first_processor: JsonDict = {}
        for line in cpuinfo.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if not line.strip() and first_processor:
                break
            key, separator, value = line.partition(":")
            if separator:
                first_processor[key.strip().lower()] = value.strip()
        model = first_processor.get("model name") or first_processor.get(
            "hardware"
        )
        if model:
            payload["model"] = str(model)[:256]
        raw_flags = str(
            first_processor.get("flags") or first_processor.get("features") or ""
        ).split()
        relevant_flags = {
            "aes",
            "avx",
            "avx2",
            "avx512f",
            "fma",
            "neon",
            "sse4_1",
            "sse4_2",
        }
        payload["numeric_instruction_flags"] = sorted(
            set(raw_flags).intersection(relevant_flags)
        )
    except OSError as exc:
        payload["inspection_error"] = type(exc).__name__
    return payload


def _accelerator_snapshot() -> JsonDict:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2.0,
        )
    except FileNotFoundError:
        return {"nvidia_smi": {"available": False, "reason": "not_installed"}}
    except subprocess.TimeoutExpired:
        return {"nvidia_smi": {"available": False, "reason": "timeout"}}
    except OSError as exc:
        return {
            "nvidia_smi": {
                "available": False,
                "reason": type(exc).__name__,
            }
        }
    if completed.returncode != 0:
        return {
            "nvidia_smi": {
                "available": False,
                "returncode": int(completed.returncode),
                "error": _bounded_diagnostic(completed.stderr),
            }
        }
    devices = []
    for row in csv.reader(completed.stdout.splitlines()):
        if len(row) != 4:
            continue
        try:
            index = int(row[0].strip())
            memory_total_mb = int(row[3].strip())
        except ValueError:
            continue
        devices.append(
            {
                "index": index,
                "name": row[1].strip()[:256],
                "driver_version": row[2].strip()[:64],
                "memory_total_mb": memory_total_mb,
            }
        )
    return {
        "nvidia_smi": {
            "available": True,
            "devices": devices,
        }
    }


def _linux_drm_accelerator_snapshot() -> JsonDict:
    drm_root = Path("/sys/class/drm")
    if not drm_root.is_dir():
        return {"available": False, "reason": "not_supported"}
    devices = []
    for card in sorted(drm_root.glob("card*"), key=lambda path: path.name):
        suffix = card.name.removeprefix("card")
        if not suffix.isdigit():
            continue
        device_root = card / "device"
        if not device_root.is_dir():
            continue
        record: JsonDict = {"card_index": int(suffix)}
        for field_name in (
            "vendor",
            "device",
            "subsystem_vendor",
            "subsystem_device",
        ):
            value_path = device_root / field_name
            try:
                value = value_path.read_text(encoding="utf-8").strip().lower()
            except OSError:
                continue
            if value:
                record[field_name + "_id"] = value[:32]
        driver = device_root / "driver"
        if driver.is_symlink():
            try:
                record["driver"] = driver.resolve(strict=True).name[:128]
            except OSError:
                record["driver"] = "unavailable"
        devices.append(record)
    return {"available": True, "devices": devices}


def _runtime_environment_snapshot() -> JsonDict:
    """Capture only reproducibility-relevant, explicitly allowlisted values."""

    payload: JsonDict = {}
    for name in _RUNTIME_ENVIRONMENT_ALLOWLIST:
        if name not in os.environ:
            continue
        value = os.environ[name]
        # Device UUIDs are stable machine identifiers. Preserve whether such a
        # selection exists and its content identity without serializing it.
        if name == "CUDA_VISIBLE_DEVICES" and (
            "GPU-" in value.upper() or "MIG-" in value.upper()
        ):
            payload[name] = {
                "set": True,
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                "length": len(value),
            }
            continue
        if len(value) <= 128 and all(
            32 <= ord(character) < 127 for character in value
        ):
            payload[name] = value
        else:
            payload[name] = {
                "set": True,
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                "length": len(value),
            }
    return payload


def _linear_algebra_runtime_snapshot() -> JsonDict:
    payload: JsonDict = {"numpy_version": _distribution_version("numpy")}
    try:
        # Importing the required base dependency activates its BLAS runtime so
        # threadpoolctl can report the implementation that will execute work.
        import numpy  # noqa: F401
    except Exception as exc:
        payload["numpy_import_error"] = type(exc).__name__
        return payload
    try:
        from threadpoolctl import threadpool_info
    except Exception:
        payload["threadpoolctl_available"] = False
        return payload
    payload["threadpoolctl_available"] = True
    implementations = []
    for raw in threadpool_info():
        if not isinstance(raw, dict):
            continue
        implementations.append(
            {
                key: raw.get(key)
                for key in (
                    "user_api",
                    "internal_api",
                    "version",
                    "num_threads",
                    "threading_layer",
                    "architecture",
                )
                if raw.get(key) is not None
            }
        )
    payload["implementations"] = implementations
    return payload


def _bounded_diagnostic(value: str) -> str:
    normalized = " ".join(str(value or "").split())
    return normalized[:512]


def git_snapshot(root: Path) -> JsonDict:
    payload: JsonDict = {
        "root": str(root),
        "available": False,
    }
    if not (root / ".git").exists():
        payload["error"] = "not a git checkout"
        return payload
    revision = _git(root, ["rev-parse", "HEAD"])
    if revision["ok"]:
        payload["available"] = True
        payload["commit"] = revision["stdout"]
    else:
        payload["error"] = revision["stderr"] or revision["stdout"]
        return payload
    branch = _git(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch["ok"]:
        payload["branch"] = branch["stdout"]
    else:
        payload.setdefault("command_errors", {})["branch"] = (
            branch["stderr"] or branch["stdout"] or "git branch inspection failed"
        )
    status = _git(root, ["status", "--porcelain"])
    if status["ok"]:
        lines = [line for line in status["stdout"].splitlines() if line.strip()]
        payload["dirty"] = bool(lines)
        payload["status_porcelain"] = lines
        payload["status_available"] = True
    else:
        payload["dirty"] = None
        payload["status_available"] = False
        payload.setdefault("command_errors", {})["status"] = (
            status["stderr"] or status["stdout"] or "git status inspection failed"
        )
    diff = _git(root, ["diff", "--stat"])
    if diff["ok"] and diff["stdout"]:
        payload["diff_stat"] = diff["stdout"]
    elif not diff["ok"]:
        payload.setdefault("command_errors", {})["diff_stat"] = (
            diff["stderr"] or diff["stdout"] or "git diff inspection failed"
        )
    if payload.get("dirty"):
        payload.update(_git_worktree_content_identity(root))
    return payload


def _git_worktree_content_identity(root: Path) -> JsonDict:
    """Identify dirty tracked and untracked bytes without embedding their content."""

    diff = _git(root, ["diff", "--binary", "--no-ext-diff", "HEAD"])
    untracked = _git(root, ["ls-files", "--others", "--exclude-standard"])
    if not diff["ok"] or not untracked["ok"]:
        errors: JsonDict = {}
        if not diff["ok"]:
            errors["tracked_diff"] = (
                diff["stderr"] or diff["stdout"] or "git diff identity failed"
            )
        if not untracked["ok"]:
            errors["untracked_files"] = (
                untracked["stderr"]
                or untracked["stdout"]
                or "git untracked-file inspection failed"
            )
        return {
            "dirty_content_identity": {
                "available": False,
                "errors": errors,
            }
        }
    untracked_rows = []
    for relative in sorted(
        line for line in untracked["stdout"].splitlines() if line.strip()
    ):
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            return {
                "dirty_content_identity": {
                    "available": False,
                    "errors": {
                        "untracked_path": "%s escapes repository root: %s"
                        % (relative, exc)
                    },
                }
            }
        if path.is_symlink():
            target = os.readlink(path)
            target_bytes = target.encode("utf-8", errors="surrogatepass")
            untracked_rows.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target_sha256": hashlib.sha256(target_bytes).hexdigest(),
                    "target_size_bytes": len(target_bytes),
                }
            )
        elif path.is_file():
            untracked_rows.append(
                {
                    "path": relative,
                    "kind": "file",
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    identity = {
        "available": True,
        "tracked_diff_sha256": hashlib.sha256(
            diff["stdout"].encode("utf-8")
        ).hexdigest(),
        "untracked": untracked_rows,
    }
    identity["sha256"] = canonical_json_sha256(identity)
    return {"dirty_content_identity": identity}


def initial_run_manifest(recipe: Recipe, operation_contracts: JsonDict, run_id: str) -> JsonDict:
    return {
        "schema_version": 1,
        "kind": "noema.run_manifest",
        "run_id": run_id,
        "recipe_name": recipe.name,
        "status": "running",
        "created_at_utc": utc_now_iso(),
        "recipe": {
            "schema_version": recipe.schema_version,
            "sha256": recipe_fingerprint(recipe),
            "step_count": len(recipe.steps),
            "metadata_keys": sorted((recipe.metadata or {}).keys()),
            "research": research_specs_from_recipe(recipe),
        },
        "seed_policy": seed_policy(recipe),
        "environment": environment_snapshot(),
        "operation_contracts": operation_contracts,
        "steps": [],
        "artifacts": [],
    }


def append_step_to_manifest(manifest: JsonDict, step_summary: JsonDict, run_dir: Path) -> None:
    step_metadata = dict(step_summary.get("metadata") or {})
    step_metrics = dict(step_summary.get("metrics") or {})
    step_record = {
        "id": step_summary.get("id"),
        "op": step_summary.get("op"),
        "status": step_summary.get("status"),
        "metrics_keys": sorted(step_metrics),
        # Preserve the values independently of summary.json.  A verifier can
        # then detect a rewritten metric even if an attacker refreshes the
        # manifest's summary-file descriptor.
        "metrics": step_metrics,
        "metrics_sha256": canonical_json_sha256(step_metrics),
        "outputs": {},
    }
    if step_metadata.get("external_adapter_sdk"):
        step_record["external_adapter_sdk"] = dict(step_metadata["external_adapter_sdk"])
    outputs = dict(step_summary.get("outputs") or {})
    for name, output in outputs.items():
        artifact_record = _artifact_manifest_record(
            step_summary.get("id"),
            name,
            output,
            run_dir,
            producer_metrics_sha256=step_record["metrics_sha256"],
        )
        step_record["outputs"][name] = artifact_record
        manifest.setdefault("artifacts", []).append(artifact_record)
    manifest.setdefault("steps", []).append(step_record)


def finalize_manifest(manifest: JsonDict, status: str, error: Optional[str] = None) -> None:
    manifest["status"] = status
    manifest["completed_at_utc"] = utc_now_iso()
    if error:
        manifest["error"] = str(error)


def _artifact_manifest_record(
    step_id: Any,
    output_name: str,
    output: JsonDict,
    run_dir: Path,
    *,
    producer_metrics_sha256: str,
) -> JsonDict:
    path = Path(str(output.get("path", ""))).resolve()
    run_root = Path(run_dir).resolve()
    try:
        relative_path = str(path.relative_to(run_root))
    except ValueError as exc:
        raise ValueError(
            "Artifact %s.%s is outside run directory %s: %s"
            % (step_id, output_name, run_root, path)
        ) from exc
    metadata = dict(output.get("metadata") or {})
    return {
        "step_id": step_id,
        "output_name": output_name,
        "kind": output.get("kind"),
        "path": str(path),
        "relative_path": relative_path,
        "sha256": output.get("sha256"),
        "dtype": metadata.get("dtype"),
        "shape": metadata.get("shape"),
        "array": metadata.get("array"),
        "metadata": metadata,
        "metadata_sha256": canonical_json_sha256(metadata),
        "producer_metrics_sha256": producer_metrics_sha256,
    }


def _master_seed_source(recipe: Recipe) -> Optional[str]:
    metadata = dict(recipe.metadata or {})
    for key in ("seed", "experiment_seed", "ui_seed"):
        if key in metadata and metadata[key] is not None:
            return "recipe.metadata.%s" % key
    return None


def _coerce_seed(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError("%s must be an integer seed, not a boolean" % label)
    try:
        seed = int(value)
    except Exception as exc:
        raise ValueError("%s must be an integer seed" % label) from exc
    if seed < 0:
        raise ValueError("%s must be >= 0" % label)
    return seed


def _dependency_versions(names: Iterable[str]) -> JsonDict:
    return {name: installed_dependency_version(name) for name in names}


def installed_dependency_version(name: str) -> Optional[str]:
    """Return an installed distribution version, including known wheel aliases."""

    aliases = {
        # The PHY/SYS-only wheel exposes the same ``sionna`` import namespace
        # without pulling in the unrelated RT stack.
        "sionna": ("sionna-no-rt", "sionna"),
        "tensorflow": ("tensorflow", "tensorflow-cpu", "tensorflow-intel"),
    }
    candidates = aliases.get(name, (name,))
    for candidate in candidates:
        version = _distribution_version(candidate)
        if version is not None:
            return version
    return None


def _distribution_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _native_snapshot() -> JsonDict:
    try:
        from noema_lab.core import dataplane

        return {"dataplane": dataplane.native_status()}
    except Exception as exc:
        return {"dataplane": {"available": False, "error": str(exc)}}


def _project_file_hashes(root: Path) -> JsonDict:
    files: JsonDict = {}
    for name in ("pyproject.toml", "uv.lock"):
        path = root / name
        if path.is_file():
            files[name] = {
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
    return files


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def _git(root: Path, args: Iterable[str]) -> JsonDict:
    try:
        completed = subprocess.run(
            ["git", *list(args)],
            cwd=str(root),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2.0,
        )
    except Exception as exc:
        return {"ok": False, "stdout": "", "stderr": str(exc)}
    return {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
