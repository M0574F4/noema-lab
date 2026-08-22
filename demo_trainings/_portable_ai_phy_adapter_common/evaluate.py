from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import onnxruntime as ort
import torch

import task
from datamodule import load_capture_dataset

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def main() -> int:
    config = _mapping(Path("train_config.yaml"))
    data = dict(config.get("data") or {})
    training = dict(config.get("training") or {})
    dataset = load_capture_dataset(
        data.get("test_capture_dirs") or [],
        feature_tap=str(data.get("feature_tap") or ""),
        target_tap=str(data.get("target_tap") or ""),
        expected_split="test",
    )
    manifest_path = Path(str(training.get("artifact_manifest_path") or "../trained_artifact.yaml"))
    manifest = _mapping(manifest_path)
    components = list(manifest.get("components") or [])
    component = next(
        row for row in components if str(row.get("id") or "") == task.COMPONENT_ID
    )
    component_path = manifest_path.parent / str(component.get("path") or "")
    expected_sha = str(component.get("sha256") or "")
    if _sha256(component_path) != expected_sha:
        raise ValueError("trained artifact component SHA-256 does not match")
    session = ort.InferenceSession(str(component_path), providers=["CPUExecutionProvider"])
    raw_output = session.run(None, task.onnx_feed(dataset.features))[0]
    output = torch.from_numpy(np.asarray(raw_output, dtype=np.float32))
    metrics = task.metrics_from_predictions(
        output,
        torch.from_numpy(dataset.features),
        torch.from_numpy(dataset.targets),
    )
    payload = {
        "schema_version": 1,
        "kind": "noema.reference_training_evaluation",
        "task": task.TASK_ID,
        "split": "test",
        "sample_count": int(dataset.features.shape[0]),
        "test_split_used_for_training_or_selection": False,
        "metrics": metrics,
        "trained_artifact": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "components": [{"id": task.COMPONENT_ID, "sha256": expected_sha}],
        },
    }
    Path("evaluation_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _mapping(path: Path) -> dict[str, Any]:
    payload = load_strict_yaml_or_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a mapping: %s" % path)
    return dict(payload)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
