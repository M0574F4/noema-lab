from __future__ import annotations

"""Small, explicit runtime policies for isolated scientific workers.

The policy in this module is deliberately narrower than a machine image.  It
fixes the process and numerical-backend controls that can change the computed
result while allowing harmless host details such as cache paths, total memory,
and absolute virtual-environment locations to vary.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping


JsonDict = Dict[str, Any]

DETERMINISTIC_CPU_RUNTIME_POLICY_V1: JsonDict = {
    "schema_version": 1,
    "kind": "noema.deterministic_cpu_runtime_policy",
    "policy_id": "torch-sionna-cpu-single-thread-v1",
    "scope": "deterministic_numerical_cpu_backend_path_not_cross_host_bitwise",
    "process_environment": {
        "set": {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
        },
        "unset": [
            "PYTHONPATH",
            "PYTHONHOME",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "CONDA_PREFIX",
            "CUDA_DEVICE_ORDER",
            "NVIDIA_VISIBLE_DEVICES",
            "CUBLAS_WORKSPACE_CONFIG",
            "ATEN_CPU_CAPABILITY",
            "OPENBLAS_CORETYPE",
            "MKL_CBWR",
            "TF_CPP_MIN_LOG_LEVEL",
            "TF_FORCE_GPU_ALLOW_GROWTH",
            "XLA_FLAGS",
            "JAX_PLATFORM_NAME",
            "XDG_CACHE_HOME",
            "HF_HOME",
            "TORCH_HOME",
            "NUMBA_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
        ],
        "workspace_relative_set": {
            "MPLCONFIGDIR": "runtime-cache/matplotlib",
        },
    },
    "torch": {
        "device": "cpu",
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "deterministic_algorithms": True,
        "default_dtype": "float32",
    },
    "sionna": {
        "backend": "torch",
        "device": "cpu",
        "precision": "single",
    },
}


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def deterministic_cpu_runtime_policy_v1() -> JsonDict:
    """Return a caller-owned copy of the only accepted v1 worker policy."""

    return copy.deepcopy(DETERMINISTIC_CPU_RUNTIME_POLICY_V1)


def deterministic_cpu_runtime_policy_sha256_v1() -> str:
    return canonical_json_sha256(DETERMINISTIC_CPU_RUNTIME_POLICY_V1)


def scientific_runtime_identity() -> JsonDict:
    """Return exact imported scientific-runtime versions without host paths."""

    try:
        import noema_lab
        import numpy
        import onnx
        import onnxruntime
        import sionna
        import torch
        import yaml
    except Exception as exc:
        raise RuntimeError("cannot import the scientific runtime identity") from exc
    build_info_function = getattr(onnxruntime, "get_build_info", None)
    onnxruntime_build_info = (
        str(build_info_function()).strip()
        if callable(build_info_function)
        else None
    )
    torch_version = getattr(torch, "version", None)
    return {
        "schema_version": 2,
        "python": {
            "version": ".".join(str(value) for value in sys.version_info[:3]),
            "implementation": str(sys.implementation.name),
            "cache_tag": sys.implementation.cache_tag,
            "byteorder": sys.byteorder,
        },
        "modules": {
            "numpy": {"version": str(numpy.__version__)},
            "onnx": {"version": str(onnx.__version__)},
            "onnxruntime": {
                "version": str(onnxruntime.__version__),
                "build_info": onnxruntime_build_info,
            },
            "yaml": {"version": str(yaml.__version__)},
            "torch": {
                "version": str(torch.__version__),
                "git_version": (
                    None
                    if getattr(torch_version, "git_version", None) is None
                    else str(torch_version.git_version)
                ),
                "cuda": (
                    None
                    if getattr(torch_version, "cuda", None) is None
                    else str(torch_version.cuda)
                ),
            },
            "sionna": {"version": str(sionna.__version__)},
            "noema_lab": {
                "version": (
                    None
                    if getattr(noema_lab, "__version__", None) is None
                    else str(noema_lab.__version__)
                )
            },
        },
    }


def worker_runtime_identity_evidence(
    expected: Mapping[str, Any],
    *,
    pre_registry: Mapping[str, Any],
    post_execution: Mapping[str, Any],
) -> JsonDict:
    """Authenticate exact pre/post equality to a launcher-declared identity."""

    expected_copy = copy.deepcopy(dict(expected))
    pre_copy = copy.deepcopy(dict(pre_registry))
    post_copy = copy.deepcopy(dict(post_execution))
    if expected_copy != pre_copy or expected_copy != post_copy:
        raise RuntimeError("worker scientific runtime identity changed")
    digest = canonical_json_sha256(expected_copy)
    evidence: JsonDict = {
        "schema_version": 1,
        "kind": "noema.worker_scientific_runtime_identity_evidence",
        "status": "passed",
        "expected_identity_sha256": digest,
        "pre_registry_identity_sha256": canonical_json_sha256(pre_copy),
        "post_execution_identity_sha256": canonical_json_sha256(post_copy),
        "identity": expected_copy,
    }
    evidence["evidence_sha256"] = canonical_json_sha256(evidence)
    return evidence


def validate_worker_runtime_identity_evidence(
    value: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> JsonDict:
    evidence = copy.deepcopy(dict(value))
    commitment = evidence.pop("evidence_sha256", None)
    if commitment != canonical_json_sha256(evidence):
        raise ValueError("worker runtime identity evidence commitment changed")
    expected_digest = canonical_json_sha256(dict(expected))
    if (
        set(evidence)
        != {
            "schema_version",
            "kind",
            "status",
            "expected_identity_sha256",
            "pre_registry_identity_sha256",
            "post_execution_identity_sha256",
            "identity",
        }
        or evidence.get("schema_version") != 1
        or evidence.get("kind")
        != "noema.worker_scientific_runtime_identity_evidence"
        or evidence.get("status") != "passed"
        or evidence.get("identity") != dict(expected)
        or evidence.get("expected_identity_sha256") != expected_digest
        or evidence.get("pre_registry_identity_sha256") != expected_digest
        or evidence.get("post_execution_identity_sha256") != expected_digest
    ):
        raise ValueError("worker runtime identity evidence changed")
    return {**evidence, "evidence_sha256": commitment}


def runtime_policy_environment(
    policy: Mapping[str, Any],
    *,
    workspace: Path,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Validate the exact policy and return its launch-time environment.

    The worker recognizes one closed policy rather than accepting arbitrary
    process-global backend mutations from a serialized request.
    """

    if dict(policy) != DETERMINISTIC_CPU_RUNTIME_POLICY_V1:
        raise ValueError("isolated worker runtime policy is not the exact v1 policy")
    process = policy.get("process_environment")
    if not isinstance(process, Mapping):  # guarded by the exact check above
        raise ValueError("isolated worker runtime policy lacks process controls")
    environment_set = process.get("set")
    environment_unset = process.get("unset")
    workspace_relative_set = process.get("workspace_relative_set")
    if not isinstance(environment_set, Mapping) or not isinstance(
        environment_unset, list
    ) or not isinstance(workspace_relative_set, Mapping):
        raise ValueError("isolated worker runtime policy process controls are invalid")
    lexical_workspace = Path(os.path.abspath(os.fspath(workspace)))
    if lexical_workspace.is_symlink():
        raise ValueError("isolated worker runtime-policy workspace is unsafe")
    workspace = lexical_workspace.resolve()
    if not workspace.is_dir():
        raise ValueError("isolated worker runtime-policy workspace is unsafe")
    resolved_set = {str(name): str(value) for name, value in environment_set.items()}
    for name, relative_raw in workspace_relative_set.items():
        relative = Path(str(relative_raw))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("runtime-policy cache path escapes the worker workspace")
        target = (workspace / relative).resolve()
        try:
            target.relative_to(workspace)
        except ValueError as exc:  # pragma: no cover - closed policy guard
            raise ValueError(
                "runtime-policy cache path escapes the worker workspace"
            ) from exc
        resolved_set[str(name)] = str(target)
    return (
        resolved_set,
        tuple(str(name) for name in environment_unset),
    )


def prepare_runtime_policy_workspace(
    policy: Mapping[str, Any],
    *,
    workspace: Path,
) -> None:
    """Create only the closed policy's private workspace-relative cache dirs."""

    runtime_policy_environment(policy, workspace=workspace)
    process = DETERMINISTIC_CPU_RUNTIME_POLICY_V1["process_environment"]
    workspace = Path(workspace).resolve()
    for relative_raw in process["workspace_relative_set"].values():
        relative = Path(str(relative_raw))
        current = workspace
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("runtime-policy cache path contains a symlink")
        current.mkdir(parents=True, exist_ok=True, mode=0o700)
        if current.is_symlink() or not current.is_dir():
            raise ValueError("runtime-policy cache path is unsafe")


def apply_deterministic_cpu_runtime_policy(
    policy: Mapping[str, Any],
    *,
    workspace: Path,
) -> JsonDict:
    """Apply and verify the exact CPU policy before registry/PHY construction."""

    prepare_runtime_policy_workspace(policy, workspace=workspace)
    environment_set, environment_unset = runtime_policy_environment(
        policy,
        workspace=workspace,
    )
    for name, expected in environment_set.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(
                "runtime policy environment mismatch for %s" % name
            )
    for name in environment_unset:
        if os.environ.get(name) not in (None, ""):
            raise RuntimeError(
                "runtime policy requires %s to be absent" % name
            )

    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("runtime policy requires PyTorch") from exc

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    if torch.cuda.is_available() or int(torch.cuda.device_count()) != 0:
        raise RuntimeError("runtime policy did not hide CUDA devices")

    try:
        import sionna  # type: ignore
        from sionna.phy import config as sionna_config  # type: ignore
    except Exception as exc:
        raise RuntimeError("runtime policy requires Sionna 2.x") from exc

    sionna_config.device = "cpu"
    sionna_config.precision = "single"

    # Torch/Sionna imports may populate optional cache variables themselves.
    # The bootstrap boundary deliberately neutralizes those library defaults
    # once, before any registry or PHY object is constructed.  The final
    # worker observation below is read-only and fails on later drift.
    for name in environment_unset:
        os.environ.pop(name, None)

    return observe_deterministic_cpu_runtime_policy(
        policy,
        workspace=workspace,
        observation_phase="initial_after_backend_bootstrap",
        prior_evidence_sha256=None,
        torch_module=torch,
        sionna_module=sionna,
        sionna_config=sionna_config,
    )


def observe_deterministic_cpu_runtime_policy(
    policy: Mapping[str, Any],
    *,
    workspace: Path,
    observation_phase: str,
    prior_evidence_sha256: str | None,
    torch_module: Any | None = None,
    sionna_module: Any | None = None,
    sionna_config: Any | None = None,
) -> JsonDict:
    """Read and authenticate live policy state without mutating it."""

    if observation_phase not in {
        "initial_after_backend_bootstrap",
        "worker_final",
    }:
        raise ValueError("runtime policy observation phase is invalid")
    if observation_phase == "initial_after_backend_bootstrap":
        if prior_evidence_sha256 is not None:
            raise ValueError("initial runtime-policy evidence cannot have a predecessor")
    elif (
        not isinstance(prior_evidence_sha256, str)
        or len(prior_evidence_sha256) != 64
        or any(character not in "0123456789abcdef" for character in prior_evidence_sha256)
    ):
        raise ValueError("final runtime-policy evidence lacks its initial commitment")

    environment_set, environment_unset = runtime_policy_environment(
        policy,
        workspace=workspace,
    )
    for name, expected in environment_set.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(
                "runtime policy environment drift for %s" % name
            )
    for name in environment_unset:
        if os.environ.get(name) not in (None, ""):
            raise RuntimeError("runtime policy environment drift for %s" % name)
    process_policy = DETERMINISTIC_CPU_RUNTIME_POLICY_V1[
        "process_environment"
    ]
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore
        except Exception as exc:  # pragma: no cover - import error path
            raise RuntimeError("runtime policy requires PyTorch") from exc
    if sionna_module is None or sionna_config is None:
        try:
            import sionna as sionna_module  # type: ignore
            from sionna.phy import config as sionna_config  # type: ignore
        except Exception as exc:  # pragma: no cover - import error path
            raise RuntimeError("runtime policy requires Sionna 2.x") from exc

    observed = {
        "process_environment": {
            "set": {
                name: os.environ.get(name)
                for name in sorted(process_policy["set"])
            },
            "unset": {
                name: os.environ.get(name)
                for name in sorted(environment_unset)
            },
            "workspace_relative_set": {
                name: {
                    "relative_path": str(relative),
                    "confined_to_workspace": True,
                    "live_value_matches_resolved_path": True,
                }
                for name, relative in sorted(
                    process_policy["workspace_relative_set"].items()
                )
            },
        },
        "torch": {
            "version": str(getattr(torch_module, "__version__", "")),
            "device": str(torch_module.get_default_device()),
            "cuda_available": bool(torch_module.cuda.is_available()),
            "cuda_device_count": int(torch_module.cuda.device_count()),
            "intra_op_threads": int(torch_module.get_num_threads()),
            "inter_op_threads": int(torch_module.get_num_interop_threads()),
            "deterministic_algorithms": bool(
                torch_module.are_deterministic_algorithms_enabled()
            ),
            "default_dtype": str(torch_module.get_default_dtype()).removeprefix(
                "torch."
            ),
        },
        "sionna": {
            "version": str(getattr(sionna_module, "__version__", "")),
            "backend": "torch",
            "device": str(sionna_config.device),
            "precision": str(sionna_config.precision),
        },
    }
    expected_torch = DETERMINISTIC_CPU_RUNTIME_POLICY_V1["torch"]
    expected_sionna = DETERMINISTIC_CPU_RUNTIME_POLICY_V1["sionna"]
    if (
        observed["torch"]["device"] != expected_torch["device"]
        or observed["torch"]["cuda_available"] is not False
        or observed["torch"]["cuda_device_count"] != 0
        or observed["torch"]["intra_op_threads"]
        != expected_torch["intra_op_threads"]
        or observed["torch"]["inter_op_threads"]
        != expected_torch["inter_op_threads"]
        or observed["torch"]["deterministic_algorithms"]
        is not expected_torch["deterministic_algorithms"]
        or observed["torch"]["default_dtype"]
        != expected_torch["default_dtype"]
        or observed["sionna"]["backend"] != expected_sionna["backend"]
        or observed["sionna"]["device"] != expected_sionna["device"]
        or observed["sionna"]["precision"] != expected_sionna["precision"]
    ):
        raise RuntimeError("runtime policy backend verification failed")

    evidence: JsonDict = {
        "schema_version": 1,
        "kind": "noema.deterministic_cpu_runtime_policy_evidence",
        "status": "passed",
        "policy_id": DETERMINISTIC_CPU_RUNTIME_POLICY_V1["policy_id"],
        "policy_sha256": deterministic_cpu_runtime_policy_sha256_v1(),
        "observation_phase": observation_phase,
        "prior_evidence_sha256": prior_evidence_sha256,
        "observed": observed,
    }
    evidence["evidence_sha256"] = canonical_json_sha256(evidence)
    return evidence


def validate_deterministic_cpu_runtime_policy_evidence(
    value: Mapping[str, Any],
    *,
    required_phase: str | None = None,
) -> JsonDict:
    """Validate authenticated, metric-free evidence from one worker process."""

    evidence = copy.deepcopy(dict(value))
    observed_digest = evidence.pop("evidence_sha256", None)
    if observed_digest != canonical_json_sha256(evidence):
        raise ValueError("runtime policy evidence commitment changed")
    if set(evidence) != {
        "schema_version",
        "kind",
        "status",
        "policy_id",
        "policy_sha256",
        "observation_phase",
        "prior_evidence_sha256",
        "observed",
    }:
        raise ValueError("runtime policy evidence schema changed")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("kind")
        != "noema.deterministic_cpu_runtime_policy_evidence"
        or evidence.get("status") != "passed"
        or evidence.get("policy_id")
        != DETERMINISTIC_CPU_RUNTIME_POLICY_V1["policy_id"]
        or evidence.get("policy_sha256")
        != deterministic_cpu_runtime_policy_sha256_v1()
    ):
        raise ValueError("runtime policy evidence identity changed")
    observed = evidence.get("observed")
    phase = evidence.get("observation_phase")
    prior = evidence.get("prior_evidence_sha256")
    if phase not in {
        "initial_after_backend_bootstrap",
        "worker_final",
    } or (required_phase is not None and phase != required_phase):
        raise ValueError("runtime policy evidence phase changed")
    if phase == "initial_after_backend_bootstrap" and prior is not None:
        raise ValueError("initial runtime policy evidence predecessor changed")
    if phase == "worker_final" and (
        not isinstance(prior, str)
        or len(prior) != 64
        or any(character not in "0123456789abcdef" for character in prior)
    ):
        raise ValueError("final runtime policy evidence predecessor changed")
    if not isinstance(observed, Mapping) or set(observed) != {
        "process_environment",
        "torch",
        "sionna",
    }:
        raise ValueError("runtime policy observed state schema changed")
    process = observed.get("process_environment")
    torch = observed.get("torch")
    sionna = observed.get("sionna")
    expected_process = DETERMINISTIC_CPU_RUNTIME_POLICY_V1[
        "process_environment"
    ]
    expected_torch = DETERMINISTIC_CPU_RUNTIME_POLICY_V1["torch"]
    expected_sionna = DETERMINISTIC_CPU_RUNTIME_POLICY_V1["sionna"]
    if (
        not isinstance(process, Mapping)
        or set(process) != {"set", "unset", "workspace_relative_set"}
        or dict(process.get("set") or {}) != dict(expected_process["set"])
        or dict(process.get("workspace_relative_set") or {})
        != {
            name: {
                "relative_path": str(relative),
                "confined_to_workspace": True,
                "live_value_matches_resolved_path": True,
            }
            for name, relative in sorted(
                expected_process["workspace_relative_set"].items()
            )
        }
        or set(dict(process.get("unset") or {}))
        != set(expected_process["unset"])
        or any(
            raw not in (None, "")
            for raw in dict(process.get("unset") or {}).values()
        )
        or not isinstance(torch, Mapping)
        or set(torch)
        != {
            "version",
            "device",
            "cuda_available",
            "cuda_device_count",
            "intra_op_threads",
            "inter_op_threads",
            "deterministic_algorithms",
            "default_dtype",
        }
        or not str(torch.get("version") or "")
        or torch.get("device") != expected_torch["device"]
        or torch.get("cuda_available") is not False
        or torch.get("cuda_device_count") != 0
        or torch.get("intra_op_threads") != expected_torch["intra_op_threads"]
        or torch.get("inter_op_threads") != expected_torch["inter_op_threads"]
        or torch.get("deterministic_algorithms")
        is not expected_torch["deterministic_algorithms"]
        or torch.get("default_dtype") != expected_torch["default_dtype"]
        or not isinstance(sionna, Mapping)
        or set(sionna) != {"version", "backend", "device", "precision"}
        or not str(sionna.get("version") or "")
        or sionna.get("backend") != expected_sionna["backend"]
        or sionna.get("device") != expected_sionna["device"]
        or sionna.get("precision") != expected_sionna["precision"]
    ):
        raise ValueError("runtime policy evidence state changed")
    return {**evidence, "evidence_sha256": observed_digest}


def runtime_policy_lifecycle_evidence(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
) -> JsonDict:
    """Bind initial and final live observations into one terminal record."""

    initial_valid = validate_deterministic_cpu_runtime_policy_evidence(
        initial,
        required_phase="initial_after_backend_bootstrap",
    )
    final_valid = validate_deterministic_cpu_runtime_policy_evidence(
        final,
        required_phase="worker_final",
    )
    if final_valid["prior_evidence_sha256"] != initial_valid["evidence_sha256"]:
        raise ValueError("runtime policy lifecycle evidence chain changed")
    evidence: JsonDict = {
        "schema_version": 1,
        "kind": "noema.deterministic_cpu_runtime_policy_lifecycle_evidence",
        "status": "passed",
        "policy_id": DETERMINISTIC_CPU_RUNTIME_POLICY_V1["policy_id"],
        "policy_sha256": deterministic_cpu_runtime_policy_sha256_v1(),
        "initial": initial_valid,
        "final": final_valid,
    }
    evidence["evidence_sha256"] = canonical_json_sha256(evidence)
    return evidence


def validate_runtime_policy_lifecycle_evidence(
    value: Mapping[str, Any],
) -> JsonDict:
    evidence = copy.deepcopy(dict(value))
    commitment = evidence.pop("evidence_sha256", None)
    if commitment != canonical_json_sha256(evidence):
        raise ValueError("runtime policy lifecycle commitment changed")
    if (
        set(evidence)
        != {
            "schema_version",
            "kind",
            "status",
            "policy_id",
            "policy_sha256",
            "initial",
            "final",
        }
        or evidence.get("schema_version") != 1
        or evidence.get("kind")
        != "noema.deterministic_cpu_runtime_policy_lifecycle_evidence"
        or evidence.get("status") != "passed"
        or evidence.get("policy_id")
        != DETERMINISTIC_CPU_RUNTIME_POLICY_V1["policy_id"]
        or evidence.get("policy_sha256")
        != deterministic_cpu_runtime_policy_sha256_v1()
    ):
        raise ValueError("runtime policy lifecycle identity changed")
    rebuilt = runtime_policy_lifecycle_evidence(
        evidence.get("initial") or {},
        evidence.get("final") or {},
    )
    if rebuilt["evidence_sha256"] != commitment:
        raise ValueError("runtime policy lifecycle evidence changed")
    return rebuilt
