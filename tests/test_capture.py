import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.cli.main import main
from noema_lab.core.capture import (
    DatasetCaptureError,
    MAX_CAPTURE_RUNS,
    MAX_CAPTURE_SAMPLES,
    MAX_CAPTURE_SHARD_SIZE,
    _load_npz_tap,
    _sweep_values,
    run_dataset_capture_recipe,
    validate_dataset_capture_contract,
)
from noema_lab.core.matrix import matrix_variant_id
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationRegistry,
    OperationResult,
    object_schema,
)
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.ops import build_registry


def _capture_recipe():
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "semantic_artifact_capture_smoke",
            "metadata": {"seed": 123},
            "dataset_capture": {
                "split": "train",
                "samples": 2,
                "taps": [
                    {"id": "received_embedding", "from": "data.clip_embeddings"},
                    {"id": "target_mask", "from": "data.segmentation_reference"},
                ],
            },
            "steps": [
                {
                    "id": "data",
                    "op": "source.semantic_artifacts_smoke",
                    "params": {"embedding_dim": 4},
                }
            ],
        }
    )


def _multi_shard_capture_recipe():
    payload = _capture_recipe().to_dict()
    payload["name"] = "semantic_artifact_capture_multishard"
    payload["dataset_capture"].update(
        {
            "samples": 10,
            "shard_size": 4,
            "max_runs": 10,
            "seed_mode": "increment_run_seed",
        }
    )
    return recipe_from_dict(payload)


def _capture_payload_with_variant_control():
    payload = _capture_recipe().to_dict()
    payload["steps"].append(
        {
            "id": "variant_control",
            "op": "source.random_bits",
            "params": {"bit_count": 8, "batch_size": 1, "seed": 0},
        }
    )
    return payload


class _CaptureBackendProbe(Operation):
    id = "test.capture_backend_probe"
    name = "Dataset capture runner preflight probe"
    output_kinds = {"records": "test.capture.records.numpy"}
    backends = {
        "benchmark_run": ["numpy", "sionna"],
        "dataset_capture": ["numpy"],
        "differentiable_export": [],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "benchmark_numpy",
            "status": "implemented",
        },
        {
            "runner": "benchmark_run",
            "backend": "sionna",
            "implementation": "benchmark_sionna",
            "status": "implemented",
        },
        {
            "runner": "dataset_capture",
            "backend": "numpy",
            "implementation": "capture_numpy",
            "status": "implemented",
        },
    ]
    params_schema = object_schema(
        {
            "wireless_backend": {
                "type": "string",
                "default": "auto",
                "enum": ["auto", "numpy", "sionna"],
            }
        }
    )

    def __init__(self) -> None:
        self.run_calls = 0

    def run(self, ctx: OperationContext) -> OperationResult:
        self.run_calls += 1
        return OperationResult()


def _capture_backend_probe_recipe():
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "capture_backend_preflight",
            "dataset_capture": {
                "samples": 1,
                "taps": [{"id": "records", "from": "source.records"}],
            },
            "steps": [
                {
                    "id": "source",
                    "op": _CaptureBackendProbe.id,
                    "params": {"wireless_backend": "auto"},
                }
            ],
        }
    )


def _capture_backend_probe_registry(operation):
    registry = OperationRegistry()
    registry.register(operation)
    return registry


class CaptureTests(unittest.TestCase):
    def test_capture_contract_enforces_finite_samples_shards_and_runs(self):
        cases = (
            ({"samples": MAX_CAPTURE_SAMPLES + 1}, "samples"),
            ({"samples": True}, "samples"),
            ({"samples": 1.5}, "samples"),
            ({"shard_size": MAX_CAPTURE_SHARD_SIZE + 1}, "shard_size"),
            ({"shard_size": 0}, "shard_size"),
            ({"max_runs": MAX_CAPTURE_RUNS + 1}, "max_runs"),
            ({"max_runs": False}, "max_runs"),
        )
        for overrides, expected in cases:
            payload = _capture_recipe().to_dict()
            payload["dataset_capture"].update(overrides)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                DatasetCaptureError, expected
            ):
                validate_dataset_capture_contract(
                    recipe_from_dict(payload), build_registry()
                )

    def test_numeric_sweep_accepts_ranges_and_explicit_lists(self):
        self.assertEqual(_sweep_values("1:2:8"), [1, 3, 5, 7])
        self.assertEqual(_sweep_values("1,2,8"), [1, 2, 8])

    def test_capture_contract_rejects_unknown_tap_output(self):
        payload = _capture_recipe().to_dict()
        payload["dataset_capture"]["taps"] = [
            {"id": "missing", "from": "data.missing"}
        ]
        with self.assertRaisesRegex(
            DatasetCaptureError,
            r"tap missing references unknown output data\.missing",
        ):
            validate_dataset_capture_contract(
                recipe_from_dict(payload),
                build_registry(),
            )

    def test_capture_contract_rejects_reserved_metadata_tap_id(self):
        payload = _capture_recipe().to_dict()
        payload["dataset_capture"]["taps"] = [
            {"id": "metadata_json", "from": "data.clip_embeddings"}
        ]
        with self.assertRaisesRegex(
            DatasetCaptureError,
            r"Tap id is reserved: metadata_json",
        ):
            validate_dataset_capture_contract(
                recipe_from_dict(payload),
                build_registry(),
            )

    def test_dataset_capture_writes_npz_bundle_and_manifests(self):
        recipe = _capture_recipe()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "capture_dataset"
            payload = run_dataset_capture_recipe(
                recipe,
                build_registry(),
                LocalStore(root / ".noema"),
                out_dir,
            )
            self.assertEqual(payload["status"], "captured")
            self.assertEqual(payload["tap_count"], 2)
            for relative in [
                "shards/shard_0000.npz",
                "schema.json",
                "tap_manifest.json",
                "recipe.json",
                "channel_distribution.json",
                "split.json",
            ]:
                self.assertTrue((out_dir / relative).is_file(), relative)
            with np.load(str(out_dir / "shards" / "shard_0000.npz"), allow_pickle=False) as shard:
                self.assertEqual(tuple(shard["received_embedding"].shape), (2, 4))
                self.assertEqual(tuple(shard["target_mask"].shape), (2, 3, 3))
                self.assertIn("metadata_json", shard.files)
            schema = json.loads((out_dir / "schema.json").read_text(encoding="utf-8"))
            self.assertEqual(schema["recipe"], recipe.name)
            self.assertEqual(schema["split"], "train")
            self.assertEqual(schema["requested_samples"], 2)
            self.assertEqual(schema["captured_samples"], 2)
            self.assertEqual([tap["id"] for tap in schema["taps"]], ["received_embedding", "target_mask"])
            self.assertEqual(schema["seed_policy"]["master_seed"], 123)

    def test_force_capture_replaces_existing_dataset_after_success(self):
        recipe = _capture_recipe()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "capture_dataset"
            out_dir.mkdir()
            marker = out_dir / "previous.txt"
            marker.write_text("previous dataset", encoding="utf-8")

            payload = run_dataset_capture_recipe(
                recipe,
                build_registry(),
                LocalStore(root / ".noema"),
                out_dir,
                force=True,
            )

            self.assertEqual(payload["out_dir"], str(out_dir))
            self.assertFalse(marker.exists())
            self.assertTrue((out_dir / "schema.json").is_file())
            self.assertTrue((out_dir / "shards" / "shard_0000.npz").is_file())
            self.assertFalse(
                any(path.name.startswith(".capture_dataset.") for path in root.iterdir())
            )

    def test_failed_force_capture_preserves_existing_dataset(self):
        payload = _capture_recipe().to_dict()
        payload["name"] = "semantic_artifact_recapture_mismatch"
        payload["dataset_capture"]["taps"] = [
            {"id": "received_embedding", "from": "data.clip_embeddings"},
            {"id": "importance", "from": "data.importance_map"},
        ]
        recipe = recipe_from_dict(payload)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "capture_dataset"
            out_dir.mkdir()
            marker = out_dir / "previous.txt"
            marker.write_text("previous dataset", encoding="utf-8")

            with self.assertRaisesRegex(Exception, "incompatible record counts"):
                run_dataset_capture_recipe(
                    recipe,
                    build_registry(),
                    LocalStore(root / ".noema"),
                    out_dir,
                    force=True,
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "previous dataset")
            self.assertEqual([path.name for path in out_dir.iterdir()], ["previous.txt"])
            self.assertFalse(
                any(path.name.startswith(".capture_dataset.") for path in root.iterdir())
            )

    def test_recipe_matrix_variants_cycle_with_canonical_run_provenance(self):
        raw = _capture_payload_with_variant_control()
        raw["dataset_capture"].update(
            {"samples": 6, "shard_size": 6, "max_runs": 3}
        )
        raw["metadata"]["matrix"] = {
            "dimensions": {"control_seed": [3, 5]},
            "step_params": {
                "variant_control": {"seed": {"matrix": "control_seed"}}
            },
        }
        recipe = recipe_from_dict(raw)
        first_selection = {"control_seed": 3}
        second_selection = {"control_seed": 5}
        expected_ids = [
            matrix_variant_id(first_selection),
            matrix_variant_id(second_selection),
            matrix_variant_id(first_selection),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "capture_dataset"
            result = run_dataset_capture_recipe(
                recipe,
                build_registry(),
                LocalStore(root / ".noema"),
                out_dir,
            )
            schema = json.loads(
                (out_dir / "schema.json").read_text(encoding="utf-8")
            )

            self.assertEqual(result["matrix_variant_ids"], expected_ids)
            self.assertEqual(
                [record["matrix_selection"] for record in schema["runs"]],
                [first_selection, second_selection, first_selection],
            )
            self.assertEqual(
                [record["matrix_index"] for record in schema["runs"]],
                [0, 1, 0],
            )
            self.assertEqual(
                [record["matrix_variant_id"] for record in schema["runs"]],
                expected_ids,
            )
            self.assertEqual(
                schema["matrix_distribution"]["source"], "metadata.matrix"
            )
            self.assertEqual(
                len(schema["matrix_distribution"]["variants"]), 2
            )
            self.assertFalse(schema["sweep_distribution"]["enabled"])
            self.assertTrue((out_dir / "matrix_distribution.json").is_file())

            for record, expected_dim, expected_id in zip(
                schema["runs"], [3, 5, 3], expected_ids
            ):
                run_dir = Path(record["run_dir"])
                authored = json.loads(
                    (run_dir / "recipe.authored.json").read_text(encoding="utf-8")
                )
                summary = json.loads(
                    (run_dir / "summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(authored["steps"][1]["params"]["seed"], expected_dim)
                self.assertNotIn("matrix", authored["metadata"])
                self.assertEqual(authored["metadata"]["matrix_variant_id"], expected_id)
                self.assertEqual(summary["execution_plan"]["runner"], "dataset_capture")

    def test_matrix_and_legacy_capture_sweep_conflict_before_output_creation(self):
        raw = _capture_payload_with_variant_control()
        raw["metadata"]["matrix"] = {
            "dimensions": {"control_seed": [3, 5]},
            "step_params": {
                "variant_control": {"seed": {"matrix": "control_seed"}}
            },
        }
        raw["dataset_capture"]["sweep"] = {"variant_control.seed": [7, 9]}
        recipe = recipe_from_dict(raw)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "must_not_exist"
            with self.assertRaisesRegex(
                DatasetCaptureError,
                "cannot combine a recipe metadata.matrix with dataset_capture.sweep",
            ):
                run_dataset_capture_recipe(
                    recipe,
                    build_registry(),
                    LocalStore(root / ".noema"),
                    out_dir,
                )
            self.assertFalse(out_dir.exists())

    def test_invalid_matrix_variant_fails_before_output_creation(self):
        raw = _capture_payload_with_variant_control()
        raw["metadata"]["matrix"] = {
            "dimensions": {"control_seed": [-1]},
            "step_params": {
                "variant_control": {"seed": {"matrix": "control_seed"}}
            },
        }
        recipe = recipe_from_dict(raw)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "must_not_be_replaced"
            out_dir.mkdir()
            marker = out_dir / "keep.txt"
            marker.write_text("preserved", encoding="utf-8")
            with self.assertRaisesRegex(
                DatasetCaptureError,
                "recipe variants are invalid",
            ):
                run_dataset_capture_recipe(
                    recipe,
                    build_registry(),
                    LocalStore(root / ".noema"),
                    out_dir,
                    force=True,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserved")

    def test_matrix_runner_preflight_checks_every_variant_before_output_creation(self):
        operation = _CaptureBackendProbe()
        registry = _capture_backend_probe_registry(operation)
        raw = _capture_backend_probe_recipe().to_dict()
        raw["metadata"] = {
            "matrix": {
                "dimensions": {"backend": ["numpy", "sionna"]},
                "step_params": {
                    "source": {"wireless_backend": {"matrix": "backend"}}
                },
            }
        }
        recipe = recipe_from_dict(raw)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "must_not_exist"
            with self.assertRaisesRegex(
                DatasetCaptureError,
                r"preflight failed for matrix variant 1 .*runner=dataset_capture, backend=sionna",
            ):
                run_dataset_capture_recipe(
                    recipe,
                    registry,
                    LocalStore(root / ".noema"),
                    out_dir,
                )
            self.assertFalse(out_dir.exists())
            self.assertFalse((root / ".noema").exists())
            self.assertEqual(operation.run_calls, 0)

    def test_legacy_sweep_runner_preflight_preserves_force_output(self):
        operation = _CaptureBackendProbe()
        registry = _capture_backend_probe_registry(operation)
        raw = _capture_backend_probe_recipe().to_dict()
        raw["dataset_capture"]["sweep"] = {
            "source.wireless_backend": ["numpy", "sionna"]
        }
        recipe = recipe_from_dict(raw)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "must_not_be_replaced"
            out_dir.mkdir()
            marker = out_dir / "keep.txt"
            marker.write_text("preserved", encoding="utf-8")
            with self.assertRaisesRegex(
                DatasetCaptureError,
                r"preflight failed for dataset_capture\.sweep variant 1 .*runner=dataset_capture, backend=sionna",
            ):
                run_dataset_capture_recipe(
                    recipe,
                    registry,
                    LocalStore(root / ".noema"),
                    out_dir,
                    force=True,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserved")
            self.assertFalse((root / ".noema").exists())
            self.assertEqual(operation.run_calls, 0)

    def test_legacy_capture_sweep_adds_canonical_variant_identity(self):
        raw = _capture_payload_with_variant_control()
        raw["dataset_capture"].update(
            {
                "samples": 4,
                "shard_size": 4,
                "max_runs": 2,
                "sweep": {"variant_control.seed": [3, 5]},
            }
        )
        recipe = recipe_from_dict(raw)
        selections = [
            {"variant_control.seed": 3},
            {"variant_control.seed": 5},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "capture_dataset"
            run_dataset_capture_recipe(
                recipe,
                build_registry(),
                LocalStore(root / ".noema"),
                out_dir,
            )
            schema = json.loads(
                (out_dir / "schema.json").read_text(encoding="utf-8")
            )

            self.assertEqual(
                [record["matrix_selection"] for record in schema["runs"]],
                selections,
            )
            self.assertEqual(
                [record["matrix_variant_id"] for record in schema["runs"]],
                [matrix_variant_id(item) for item in selections],
            )
            self.assertEqual(
                schema["matrix_distribution"]["source"],
                "dataset_capture.sweep",
            )
            for record in schema["runs"]:
                authored = json.loads(
                    (Path(record["run_dir"]) / "recipe.authored.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertNotIn("sweep", authored["dataset_capture"])
                self.assertEqual(
                    authored["metadata"]["matrix_variant_id"],
                    record["matrix_variant_id"],
                )

    def test_dataset_capture_reports_monotonic_progress_and_events(self):
        recipe = _multi_shard_capture_recipe()
        events = []
        progress = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = run_dataset_capture_recipe(
                recipe,
                build_registry(),
                LocalStore(root / ".noema"),
                root / "capture_dataset",
                event_sink=events.append,
                progress_sink=progress.append,
            )
        self.assertEqual(payload["status"], "captured")
        self.assertTrue(progress)
        percents = [float(item["percent"]) for item in progress]
        self.assertEqual(percents, sorted(percents))
        self.assertEqual(progress[0]["phase"], "validating")
        self.assertEqual(progress[0]["percent"], 0.0)
        self.assertEqual(progress[-1]["percent"], 100.0)
        self.assertEqual(progress[-1]["phase"], "completed")
        self.assertEqual(progress[-1]["completed_samples"], 10)
        capture_updates = [
            item for item in progress if item["phase"] == "capturing"
        ]
        self.assertEqual(
            [item["completed_samples"] for item in capture_updates],
            [2, 4, 6, 8, 10],
        )
        self.assertLess(capture_updates[0]["percent"], 30.0)
        self.assertEqual(capture_updates[-1]["percent"], 96.0)
        self.assertTrue(all(item["split"] == "train" for item in progress))
        self.assertEqual(progress[-2]["phase"], "finalizing")
        self.assertEqual(progress[-2]["percent"], 99.0)
        kinds = [item["kind"] for item in events]
        self.assertIn("capture_started", kinds)
        self.assertIn("step_started", kinds)
        self.assertIn("capture_shard_written", kinds)
        self.assertIn("capture_completed", kinds)

    def test_dataset_capture_cli_json(self):
        recipe = _capture_recipe()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "capture_recipe.yaml"
            recipe_path.write_text(yaml.safe_dump(recipe.to_dict(), sort_keys=False), encoding="utf-8")
            out_dir = root / "capture_cli"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([
                    "--workspace",
                    str(root / ".noema"),
                    "dataset-capture",
                    "run",
                    str(recipe_path),
                    "--out",
                    str(out_dir),
                    "--json",
                ])
            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "captured")
            self.assertTrue((out_dir / "schema.json").is_file())
            self.assertTrue((out_dir / "shards" / "shard_0000.npz").is_file())

    def test_capture_samples_and_shard_size_write_multiple_shards(self):
        recipe = _multi_shard_capture_recipe()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "capture_dataset"
            payload = run_dataset_capture_recipe(
                recipe,
                build_registry(),
                LocalStore(root / ".noema"),
                out_dir,
            )
            self.assertEqual(payload["captured_samples"], 10)
            self.assertEqual(payload["number_of_shards"], 3)
            self.assertEqual(
                payload["shards"],
                ["shards/shard_0000.npz", "shards/shard_0001.npz", "shards/shard_0002.npz"],
            )
            expected_counts = [4, 4, 2]
            for index, count in enumerate(expected_counts):
                with np.load(str(out_dir / "shards" / ("shard_%04d.npz" % index)), allow_pickle=False) as shard:
                    self.assertEqual(tuple(shard["received_embedding"].shape), (count, 4))
                    self.assertEqual(tuple(shard["target_mask"].shape), (count, 3, 3))
                    metadata = json.loads(str(shard["metadata_json"].item()))
                    self.assertEqual(metadata["captured_samples"], count)
                    self.assertEqual(metadata["shard_index"], index)
            schema = json.loads((out_dir / "schema.json").read_text(encoding="utf-8"))
            self.assertEqual(schema["requested_samples"], 10)
            self.assertEqual(schema["captured_samples"], 10)
            self.assertEqual(schema["number_of_shards"], 3)
            self.assertEqual(schema["shard_size"], 4)
            self.assertEqual(len(schema["runs"]), 5)
            run_seeds = schema["seed_policy"]["run_seeds"]
            self.assertEqual(run_seeds, [123, 124, 125, 126, 127])
            tap = schema["taps"][0]
            self.assertEqual(tap["record_count"], 10)
            self.assertEqual(len(tap["source_artifacts"]), 5)
            self.assertEqual(
                [row["captured_record_count"] for row in tap["source_artifacts"]],
                [2, 2, 2, 2, 2],
            )
            self.assertNotIn("artifact_sha256", tap)
            self.assertEqual(
                [row["source_run_ids"] for row in schema["shards"]],
                [
                    [schema["runs"][0]["run_id"], schema["runs"][1]["run_id"]],
                    [schema["runs"][2]["run_id"], schema["runs"][3]["run_id"]],
                    [schema["runs"][4]["run_id"]],
                ],
            )

    def test_capture_rejects_malformed_embedded_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tap.npz"
            np.savez_compressed(
                path,
                values=np.arange(4, dtype=np.float32),
                metadata_json="{not-json",
            )
            with self.assertRaisesRegex(
                DatasetCaptureError,
                "contains malformed metadata_json",
            ):
                _load_npz_tap(
                    {"path": str(path), "metadata": {"array": "values"}},
                    "source.values",
                )

    def test_capture_mismatched_tap_record_counts_fail_clearly(self):
        payload = _capture_recipe().to_dict()
        payload["name"] = "semantic_artifact_capture_mismatch"
        payload["dataset_capture"]["taps"] = [
            {"id": "received_embedding", "from": "data.clip_embeddings"},
            {"id": "importance", "from": "data.importance_map"},
        ]
        recipe = recipe_from_dict(payload)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(Exception, "incompatible record counts"):
                run_dataset_capture_recipe(
                    recipe,
                    build_registry(),
                    LocalStore(root / ".noema"),
                    root / "capture_dataset",
                )


if __name__ == "__main__":
    unittest.main()
