from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from noema_lab.core.recipes import load_recipe
from noema_lab.core.reproducibility import recipe_fingerprint
from noema_lab.training.capture_integrity import (
    CaptureSplitIntegrityError,
    assert_disjoint_capture_splits,
    fingerprint_capture_split,
)
from noema_lab.training.contracts import _validator_py
from noema_lab.ui.server import _exported_project_payload


class CaptureSplitIntegrityTests(unittest.TestCase):
    def test_capture_schema_rejects_duplicate_identity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "train"
            _write_capture(
                capture,
                "train",
                np.ones((1, 2), dtype=np.float32),
                np.ones((1,), dtype=np.int64),
            )
            schema_path = capture / "schema.json"
            content = schema_path.read_text(encoding="utf-8").replace(
                '"split": "train",',
                '"split": "test", "split": "train",',
            )
            schema_path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(
                CaptureSplitIntegrityError,
                "Duplicate JSON object key",
            ):
                fingerprint_capture_split(capture, expected_split="train")

    def test_declared_tap_dtype_and_shape_are_checked_against_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "train"
            _write_capture(
                capture,
                "train",
                np.ones((2, 2, 3), dtype=np.float32),
                np.ones((2,), dtype=np.int64),
            )
            schema_path = capture / "schema.json"
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            schema["tap_schemas"]["signal"] = {
                "dtype": "uint8",
                "record_shape": [8],
            }
            schema_path.write_text(
                json.dumps(schema, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                CaptureSplitIntegrityError,
                "has dtype float32, declared uint8",
            ):
                fingerprint_capture_split(capture, expected_split="train")

    def test_rejects_complete_record_overlap_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            train_labels = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
            _write_capture(root / "train", "train", train_features, train_labels)
            _write_capture(
                root / "validation",
                "validation",
                np.asarray([[3.0, 4.0], [9.0, 10.0]], dtype=np.float32),
                np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
            )

            with self.assertRaisesRegex(
                CaptureSplitIntegrityError,
                r"dataset leakage detected: train and validation share 1 byte-identical complete record",
            ):
                assert_disjoint_capture_splits(
                    {
                        "train": root / "train",
                        "validation": root / "validation",
                    },
                    expected_taps={
                        "train": ["signal", "label"],
                        "validation": ["signal", "label"],
                    },
                )

    def test_repeated_labels_with_distinct_signals_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repeated_labels = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
            _write_capture(
                root / "train",
                "train",
                np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                repeated_labels,
            )
            _write_capture(
                root / "validation",
                "validation",
                np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
                repeated_labels,
            )

            report = assert_disjoint_capture_splits(
                {
                    "train": root / "train",
                    "validation": root / "validation",
                },
                expected_taps={
                    "train": ["signal", "label"],
                    "validation": ["signal", "label"],
                },
            )

        self.assertEqual(report["status"], "disjoint")
        self.assertEqual(report["splits"]["train"]["records"], 2)

    def test_generated_bundle_validator_uses_tensor_records_not_schema_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "validate_contract.py"
            script.write_text(_validator_py(), encoding="utf-8")
            values = np.asarray([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
            labels = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
            jobs = []
            for split in ("train", "validation", "test"):
                # Vary schema-only metadata to prove that different schema files
                # cannot hide identical tensor records.
                _write_capture(
                    root / "data" / split,
                    split,
                    values,
                    labels,
                    metadata_note="schema metadata for %s" % split,
                )
                jobs.append(
                    {
                        "split": split,
                        "output_dir": "elsewhere/%s" % split,
                        "expected_taps": [
                            {"id": "signal", "from": "source.signal"},
                            {"id": "label", "from": "source.label"},
                        ],
                    }
                )
            validator = runpy.run_path(str(script))

            with self.assertRaisesRegex(ValueError, "captured dataset leakage detected"):
                validator["_validate_capture_split_integrity"](
                    root,
                    {"capture_jobs": jobs},
                )

    def test_workbench_handoff_marks_leaking_captures_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            values = np.asarray([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
            labels = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
            jobs = []
            for split in ("train", "validation", "test"):
                recipe_path = bundle / ("capture_%s_recipe.yaml" % split)
                recipe_path.write_text(
                    yaml.safe_dump(
                        {
                            "schema_version": 1,
                            "name": "capture_%s" % split,
                            "execution_profile": {"id": "custom", "version": 1},
                            "steps": [
                                {
                                    "id": "source",
                                    "op": "source.random_bits",
                                    "params": {"bit_count": 4},
                                }
                            ],
                            "dataset_capture": {
                                "split": split,
                                "samples": 2,
                                "taps": [
                                    {"id": "signal", "from": "source.bits"},
                                    {"id": "label", "from": "source.bits"},
                                ],
                            },
                        },
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                recipe = load_recipe(recipe_path)
                output_dir = bundle / "data" / split
                _write_capture(
                    output_dir,
                    split,
                    values,
                    labels,
                    recipe_sha256=recipe_fingerprint(recipe),
                    requested_samples=2,
                )
                jobs.append(
                    {
                        "split": split,
                        "recipe_path": str(recipe_path.relative_to(root)),
                        "output_dir": str(output_dir.relative_to(root)),
                        "requested_samples": 2,
                        "expected_taps": [
                            {"id": "signal", "from": "source.bits"},
                            {"id": "label", "from": "source.bits"},
                        ],
                    }
                )
            (bundle / "project_manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "kind": "noema.training_interface_bundle@1",
                        "capture_jobs": jobs,
                        "training": {
                            "owner": "external_researcher",
                            "working_directory": str(bundle.relative_to(root)),
                        },
                        "evaluation": {"owner": "noema_ordinary_recipe_or_benchmark"},
                        "trained_artifacts": [],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            payload = _exported_project_payload(bundle, root)

        self.assertFalse(payload["ready_for_external_training"])
        self.assertEqual(payload["capture_integrity"]["status"], "invalid")
        self.assertTrue(all(row["status"] == "invalid" for row in payload["captures"]))
        self.assertIn(
            "dataset leakage detected",
            payload["captures"][0]["issues"][0].lower(),
        )


def _write_capture(
    directory: Path,
    split: str,
    signals: np.ndarray,
    labels: np.ndarray,
    *,
    metadata_note: str = "",
    recipe_sha256: str = "",
    requested_samples: int | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shards = directory / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shards / "shard_0000.npz",
        signal=np.asarray(signals),
        label=np.asarray(labels),
    )
    schema = {
        "schema_version": 1,
        "kind": "noema.capture_dataset",
        "split": split,
        "captured_samples": int(signals.shape[0]),
        "requested_samples": int(
            signals.shape[0] if requested_samples is None else requested_samples
        ),
        "recipe_sha256": recipe_sha256,
        "metadata_note": metadata_note,
        "shards": [
            {
                "path": "shards/shard_0000.npz",
                "captured_samples": int(signals.shape[0]),
            }
        ],
        "tap_schemas": {
            "signal": {
                "dtype": str(signals.dtype),
                "record_shape": list(signals.shape[1:]),
            },
            "label": {
                "dtype": str(labels.dtype),
                "record_shape": list(labels.shape[1:]),
            },
        },
    }
    (directory / "schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
