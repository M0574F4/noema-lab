from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from noema_lab.core.reproducibility import installed_dependency_version

JsonDict = Dict[str, Any]

WIRELESS_INSTALL_MESSAGE = (
    'Install with `python -m pip install "noema-lab[wireless]"` in an installed '
    'environment, or `uv sync --extra wireless` in a source checkout, to use '
    "Sionna/PyTorch differentiable export blocks."
)
TORCH_INSTALL_MESSAGE = (
    'Install optional dependencies from a PyTorch-enabled Noema extra, for example `python -m pip install '
    '"noema-lab[wireless]"` in an installed environment or `uv sync --extra wireless` '
    "in a source checkout, to use differentiable export blocks."
)


class TrainingDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class DifferentiabilitySpec:
    framework: str
    gradient: str
    trainable_params: bool
    exportable: bool
    reason: str = ""

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "framework": self.framework,
            "gradient": self.gradient,
            "trainable_params": bool(self.trainable_params),
            "exportable": bool(self.exportable),
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


def differentiability_spec(
    *,
    framework: str = "torch",
    gradient: str = "full",
    trainable_params: bool = False,
    exportable: bool = True,
    reason: str = "",
) -> JsonDict:
    return DifferentiabilitySpec(
        framework=framework,
        gradient=gradient,
        trainable_params=trainable_params,
        exportable=exportable,
        reason=reason,
    ).to_dict()


def torch_available() -> bool:
    try:
        module = importlib.import_module("torch")
    except Exception:  # pragma: no cover - depends on optional dependency state
        return False
    return bool(hasattr(module, "nn") and hasattr(module.nn, "Module"))


def sionna_available() -> bool:
    version = installed_dependency_version("sionna") or ""
    major = version.split(".", 1)[0]
    return (
        importlib.util.find_spec("sionna") is not None
        and torch_available()
        and major.isdigit()
        and int(major) >= 2
    )


def require_torch_available():
    try:
        module = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - exercised by dependency-missing tests through availability checks
        raise TrainingDependencyError(TORCH_INSTALL_MESSAGE) from exc
    if not (hasattr(module, "nn") and hasattr(module.nn, "Module")):
        raise TrainingDependencyError(TORCH_INSTALL_MESSAGE)
    return module


def require_sionna_available() -> None:
    if not sionna_available():
        raise TrainingDependencyError(WIRELESS_INSTALL_MESSAGE)


def operation_differentiability(operation_description: Mapping[str, Any]) -> JsonDict:
    value = operation_description.get("differentiability")
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
    }


def gradient_is_differentiable(metadata: Mapping[str, Any]) -> bool:
    return str(metadata.get("gradient") or "none") in {"full", "surrogate"}
