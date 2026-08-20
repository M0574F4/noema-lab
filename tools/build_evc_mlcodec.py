from __future__ import annotations

import argparse
import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).resolve().parents[1]


def default_ryg_rans_dir(repo: Path) -> Path:
    candidates = [
        repo / "src" / "build" / "3rdparty" / "ryg_rans" / "ryg_rans-src",
        ROOT
        / ".noema"
        / "upstreams"
        / "LIC-HPCM"
        / "src"
        / "entropy_models"
        / "entropy_coders"
        / "unbounded_rans"
        / "third_party"
        / "ryg_rans",
    ]
    for candidate in candidates:
        if (candidate / "rans64.h").is_file():
            return candidate
    raise SystemExit(
        "Could not find ryg_rans headers. Expected rans64.h in EVC build cache or LIC-HPCM upstream clone."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EVC MLCodec pybind11 extensions in-place.")
    parser.add_argument(
        "--repo",
        default=str(ROOT / ".noema" / "upstreams" / "DCVC" / "DCVC-family" / "EVC"),
        help="Path to the DCVC-family/EVC upstream repository.",
    )
    parser.add_argument("--ryg-rans-dir", default="", help="Path containing ryg_rans headers.")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    src = repo / "src"
    models = src / "models"
    cpp = src / "cpp"
    if not models.is_dir() or not cpp.is_dir():
        raise SystemExit(f"EVC repo path does not look valid: {repo}")

    ryg_rans = Path(args.ryg_rans_dir).expanduser().resolve() if args.ryg_rans_dir else default_ryg_rans_dir(repo)
    compile_args = ["-std=c++17", "-O3"]
    ext_modules = [
        Pybind11Extension(
            "MLCodec_CXX",
            [str(cpp / "ops" / "ops.cpp")],
            include_dirs=[str(cpp / "ops")],
            cxx_std=17,
            extra_compile_args=compile_args,
        ),
        Pybind11Extension(
            "MLCodec_rans",
            [str(cpp / "py_rans" / "py_rans.cpp"), str(cpp / "rans" / "rans.cpp")],
            include_dirs=[str(cpp / "py_rans"), str(cpp / "rans"), str(ryg_rans)],
            cxx_std=17,
            extra_compile_args=compile_args,
        ),
    ]

    os.chdir(models)
    setup(
        name="evc-mlcodec-local",
        version="0.0.0",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        script_args=["build_ext", "--inplace"],
    )


if __name__ == "__main__":
    main()
