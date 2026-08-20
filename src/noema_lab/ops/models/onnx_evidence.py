from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from noema_lab.core.artifacts import file_sha256

JsonDict = Dict[str, Any]


def onnxruntime_native_evidence(ort: Any = None) -> JsonDict:
    """Describe the native ONNX Runtime engine behind the Python API."""
    package_root = _onnxruntime_package_root(ort)
    libraries = _find_native_libraries(package_root)
    headers = _find_headers(package_root)
    primary_library = libraries[0] if libraries else None
    return {
        "runtime_engine": "onnxruntime_cxx",
        "runtime_execution_language": "C++",
        "runtime_api": "onnxruntime_python",
        "runtime_api_binding": "Python",
        "native_inference_engine": True,
        "native_library_path": str(primary_library) if primary_library else "",
        "native_library_sha256": file_sha256(primary_library) if primary_library and primary_library.is_file() else "",
        "native_library_paths": [str(path) for path in libraries],
        "native_headers_available": bool(headers),
        "native_header_paths": [str(path) for path in headers],
        "native_cxx_api_buildable": bool(headers and libraries),
        "runtime_note": (
            "Inference is executed by the ONNX Runtime native C++ engine. "
            "This adapter enters that engine through the installed Python binding; "
            "a separate C++ API adapter requires ONNX Runtime C/C++ headers."
        ),
    }


def _onnxruntime_package_root(ort: Any = None) -> Path | None:
    module_file = str(getattr(ort, "__file__", "") or "")
    if not module_file:
        try:
            import onnxruntime as imported_ort

            module_file = str(getattr(imported_ort, "__file__", "") or "")
        except Exception:
            module_file = ""
    if not module_file:
        return None
    try:
        return Path(module_file).resolve().parent
    except OSError:
        return None


def _find_native_libraries(package_root: Path | None) -> list[Path]:
    if package_root is None:
        return []
    patterns = (
        "capi/libonnxruntime.so*",
        "capi/onnxruntime.dll",
        "capi/onnxruntime*.dll",
        "capi/libonnxruntime*.dylib",
    )
    libraries: list[Path] = []
    for pattern in patterns:
        libraries.extend(path for path in package_root.glob(pattern) if path.is_file())
    return sorted(set(libraries), key=lambda path: str(path))


def _find_headers(package_root: Path | None) -> list[Path]:
    if package_root is None:
        return []
    headers: list[Path] = []
    for header_name in ("onnxruntime_c_api.h", "onnxruntime_cxx_api.h"):
        headers.extend(path for path in package_root.rglob(header_name) if path.is_file())
    return sorted(set(headers), key=lambda path: str(path))
