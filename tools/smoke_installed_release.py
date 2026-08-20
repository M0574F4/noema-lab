#!/usr/bin/env python3
"""Exercise installed-package CLI, execution, UI, and optional native ONNX paths."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "noema_lab", *args]
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _smoke_cli(root: Path) -> None:
    version = _run("--version", cwd=root).stdout.strip()
    if not version.startswith("noema "):
        raise RuntimeError("unexpected version output: %r" % version)
    recipe_path = root / "quickstart.yaml"
    instantiated = _run(
        "template",
        "instantiate",
        "semantic_comm.text_semantic_similarity.default",
        cwd=root,
    )
    recipe_path.write_text(instantiated.stdout, encoding="utf-8")
    _run("recipe", "lint", str(recipe_path), cwd=root)
    completed = _run(
        "--workspace",
        str(root / "workspace"),
        "recipe",
        "run",
        str(recipe_path),
        "--strict-lint",
        cwd=root,
    )
    if "completed" not in completed.stdout.lower():
        raise RuntimeError("installed quickstart did not report completion")


def _smoke_ui(root: Path) -> None:
    from noema_lab.ui.server import start_ui_server_in_thread

    server, thread = start_ui_server_in_thread(
        "127.0.0.1",
        0,
        root / "ui-workspace",
        root,
    )
    try:
        port = int(server.server_address[1])
        with urllib.request.urlopen(
            "http://127.0.0.1:%d/api/health" % port,
            timeout=10,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("status") != "ok":
            raise RuntimeError("installed UI health response is not healthy: %r" % payload)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def _smoke_native_onnx(root: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    from noema_lab.ops.models.onnx_cpp_runtime import CppOnnxSession

    model_path = root / "identity.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "noema-release-smoke",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, model_path)
    session = CppOnnxSession(model_path)
    source = np.asarray([[1.25, -3.5]], dtype=np.float32)
    outputs = session.run(None, {"input": source})
    if len(outputs) != 1 or not np.array_equal(outputs[0], source):
        raise RuntimeError("native ONNX identity session returned the wrong value")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-native-onnx", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="noema-installed-smoke-") as raw:
        root = Path(raw)
        _smoke_cli(root)
        _smoke_ui(root)
        if args.require_native_onnx:
            _smoke_native_onnx(root)
    print("installed release smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
