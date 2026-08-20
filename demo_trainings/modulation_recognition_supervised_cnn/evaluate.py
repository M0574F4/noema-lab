from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from datamodule import (
    RECORD_FINGERPRINT_ALGORITHM,
    load_capture_dataset,
    split_fingerprint_report,
)
try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def main() -> int:
    config = load_strict_yaml_or_json(Path("train_config.yaml"))
    data = dict(config.get("data") or {})
    training = dict(config.get("training") or {})
    manifest_path = Path(str(training.get("artifact_manifest_path") or "trained_artifact.yaml"))
    manifest = load_strict_yaml_or_json(manifest_path)
    component = dict((manifest.get("components") or [])[0])
    component_path = manifest_path.parent / str(component.get("path") or "")
    actual_sha = hashlib.sha256(component_path.read_bytes()).hexdigest()
    if actual_sha != str(component.get("sha256") or ""):
        raise ValueError("AMC component SHA-256 does not match the manifest")
    dataset = load_capture_dataset(
        data.get("test_capture_dirs") or [],
        feature_tap=str(data.get("feature_tap") or "iq_frames"),
        target_tap=str(data.get("target_tap") or "modulation_labels"),
        expected_split="test",
    )
    training_records = _training_split_fingerprints(manifest)
    split_record_fingerprints = split_fingerprint_report(
        {
            "train": training_records["train"],
            "validation": training_records["validation"],
            "test": dataset,
        },
        include_record_sha256=True,
    )
    session = ort.InferenceSession(str(component_path), providers=["CPUExecutionProvider"])
    logits = np.asarray(session.run(["class_logits"], {"iq_ri": dataset.iq_frames})[0], dtype=np.float32)
    metrics = _metrics(logits, dataset.class_ids)
    metrics.update(
        {
            "split": "test",
            "evaluated_frames": int(dataset.class_ids.size),
            "component_sha256": actual_sha,
            "test_capture_schema_sha256": list(dataset.capture_schema_sha256),
            "split_record_fingerprints": split_record_fingerprints,
        }
    )
    Path("evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
    return 0


def _training_split_fingerprints(manifest: dict) -> dict[str, tuple[str, ...]]:
    training = manifest.get("training")
    if not isinstance(training, dict):
        raise ValueError(
            "AMC artifact is missing training split record-fingerprint evidence"
        )
    evidence = training.get("split_record_fingerprints")
    if not isinstance(evidence, dict):
        raise ValueError(
            "AMC artifact is missing training split record-fingerprint evidence"
        )
    if str(evidence.get("algorithm") or "") != RECORD_FINGERPRINT_ALGORITHM:
        raise ValueError("AMC artifact uses an unknown record-fingerprint algorithm")
    raw_splits = evidence.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("AMC artifact split record-fingerprint evidence is malformed")
    records: dict[str, tuple[str, ...]] = {}
    for name in ("train", "validation"):
        row = raw_splits.get(name)
        raw_records = row.get("record_sha256") if isinstance(row, dict) else None
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(
                "AMC artifact lacks %s record byte fingerprints" % name
            )
        records[name] = tuple(str(value or "") for value in raw_records)
    recomputed = split_fingerprint_report(
        records,
        include_record_sha256=False,
    )
    for name in ("train", "validation"):
        declared = raw_splits[name]
        actual = recomputed["splits"][name]
        for field in (
            "record_count",
            "unique_record_count",
            "fingerprint_set_sha256",
        ):
            if declared.get(field) != actual[field]:
                raise ValueError(
                    "AMC artifact %s record-fingerprint %s is inconsistent"
                    % (name, field)
                )
    return records


def _metrics(logits: np.ndarray, truth: np.ndarray) -> dict:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_prob = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    prediction = np.argmax(logits, axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    diagonal = np.diag(confusion).astype(np.float64)
    recall = np.divide(diagonal, support, out=np.zeros(3), where=support > 0)
    precision = np.divide(diagonal, predicted, out=np.zeros(3), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall > 0)
    return {
        "accuracy": float(np.mean(prediction == truth)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "cross_entropy": float(-np.mean(log_prob[np.arange(truth.size), truth])),
        "confusion_counts": confusion.tolist(),
        "class_names": ["bpsk", "qpsk", "qam16"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
