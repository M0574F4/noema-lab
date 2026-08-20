from __future__ import annotations

import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


OPTIMIZATION_ARGS = [] if sys.platform == "win32" else ["-O3"]
RUNTIME_LINK_LIBRARIES = ["dl"] if sys.platform.startswith("linux") else []

ext_modules = [
    Pybind11Extension(
        "noema_lab._native_dataplane",
        ["src/noema_lab/native/dataplane.cpp"],
        cxx_std=17,
        extra_compile_args=OPTIMIZATION_ARGS,
    ),
    Pybind11Extension(
        "noema_lab._onnxruntime_cpp",
        ["src/noema_lab/native/onnxruntime_cpp.cpp"],
        cxx_std=17,
        include_dirs=["src/noema_lab/native"],
        extra_compile_args=OPTIMIZATION_ARGS,
        libraries=RUNTIME_LINK_LIBRARIES,
    ),
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
