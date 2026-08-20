from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from noema_lab.core.artifacts import file_sha256
from noema_lab.ops.models.onnx_evidence import onnxruntime_native_evidence

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class CppOnnxValueInfo:
    name: str
    type: str = "tensor"
    shape: tuple = ()


class CppOnnxSession:
    def __init__(
        self,
        model_path: str | Path,
        provider: str = "CPUExecutionProvider",
        library_path: str | Path | None = None,
        intra_op_num_threads: int = 0,
    ) -> None:
        native = _require_native_module()
        self.model_path = str(Path(model_path))
        self.provider = _normalize_provider(provider)
        self.library_path = str(library_path or _default_library_path())
        self._session = native.Session(
            self.library_path,
            self.model_path,
            self.provider,
            int(intra_op_num_threads or 0),
        )
        self._inputs = [CppOnnxValueInfo(name=str(name)) for name in self._session.input_names]
        self._outputs = [CppOnnxValueInfo(name=str(name)) for name in self._session.output_names]

    def run(self, output_names: Iterable[str] | None, inputs: Dict[str, Any]):
        del output_names
        feed = {}
        for value_info in self._inputs:
            if value_info.name not in inputs:
                raise RuntimeError("ONNX Runtime C++ session missing input %s" % value_info.name)
            feed[value_info.name] = np.ascontiguousarray(inputs[value_info.name])
        return list(self._session.run(feed))

    def get_providers(self):
        return [self.provider]

    def get_inputs(self):
        return list(self._inputs)

    def get_outputs(self):
        return list(self._outputs)


def require_onnxruntime_cpp() -> Any:
    return _require_native_module()


def cpp_runtime_evidence(session: CppOnnxSession, model_path: str | Path, provider: str) -> JsonDict:
    model = Path(str(model_path))
    return {
        "runtime": "onnxruntime_cpp",
        "runtime_engine": "onnxruntime_cxx",
        "runtime_execution_language": "C++",
        "runtime_api": "onnxruntime_c_api",
        "runtime_api_binding": "pybind11_native_extension",
        "native_inference_engine": True,
        "session_class": "%s.%s" % (session.__class__.__module__, session.__class__.__name__),
        "native_library_path": session.library_path,
        "native_library_sha256": file_sha256(Path(session.library_path)) if Path(session.library_path).is_file() else "",
        "requested_provider": _normalize_provider(provider),
        "active_providers": session.get_providers(),
        "model_path": str(model),
        "model_sha256": file_sha256(model) if model.is_file() else "",
        "session_inputs": [_value_info(item) for item in session.get_inputs()],
        "session_outputs": [_value_info(item) for item in session.get_outputs()],
        "runtime_note": "Inference is executed through a Noema pybind11 adapter calling the ONNX Runtime C API directly.",
    }


def cpp_runtime_available() -> JsonDict:
    missing = []
    if importlib.util.find_spec("noema_lab._onnxruntime_cpp") is None:
        missing.append("noema_lab._onnxruntime_cpp")
    try:
        _default_library_path()
    except Exception:
        missing.append("libonnxruntime.so")
    if missing:
        return {
            "available": False,
            "extra": "onnx",
            "missing": missing,
            "reason": (
                'Install with `python -m pip install "noema-lab[onnx]"` in an '
                "installed environment, or `uv sync --extra onnx` in a source "
                "checkout, so the Noema ONNX Runtime C++ bridge and libonnxruntime "
                "are available."
            ),
        }
    return {"available": True, "extra": "onnx", "missing": []}


def _require_native_module() -> Any:
    try:
        import noema_lab._onnxruntime_cpp as native
    except Exception as exc:
        raise RuntimeError(
            'Noema ONNX Runtime C++ bridge is not built. Install with `python -m '
            'pip install "noema-lab[onnx]"` in an installed environment, or '
            "`uv sync --extra onnx` in a source checkout, and restart the dashboard."
        ) from exc
    return native


def _default_library_path() -> Path:
    evidence = onnxruntime_native_evidence()
    library = Path(str(evidence.get("native_library_path") or ""))
    if not library.is_file():
        raise RuntimeError("Could not find libonnxruntime from the installed onnxruntime package")
    return library


def _normalize_provider(provider: str) -> str:
    value = str(provider or "CPUExecutionProvider")
    if value == "CPU":
        return "CPUExecutionProvider"
    if value != "CPUExecutionProvider":
        raise RuntimeError("ONNX Runtime C++ bridge currently supports CPUExecutionProvider only; requested %s" % value)
    return value


def _value_info(value: Any) -> JsonDict:
    return {
        "name": str(getattr(value, "name", "")),
        "type": str(getattr(value, "type", "")),
        "shape": list(getattr(value, "shape", ()) or ()),
    }
