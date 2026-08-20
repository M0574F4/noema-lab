from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from noema_lab.core.benchmarks import load_benchmark_pack, run_benchmark_pack, validate_benchmark_pack
from noema_lab.core.execution_profiles import inspect_execution_profile
from noema_lab.core.executor import LocalExecutor
from noema_lab.core.matrix import materialize_recipe_matrix_selection
from noema_lab.core.recipes import load_recipe, recipe_from_dict
from noema_lab.core.reproducibility import derive_seed
from noema_lab.core.storage import LocalStore
from noema_lab.core.training_plans import TrainingPlan, apply_training_plan
from noema_lab.ops import build_registry
from noema_lab.ops.modulation_recognition import _blind_cumulant_scores
from noema_lab.training.exporter import (
    export_differentiable_scenario,
    exporter_ids,
    inspect_training_capture,
)


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "recipes" / "modulation_recognition_awgn.yaml"
BENCHMARK_PATH = ROOT / "benchmarks" / "neural_receiver_ai_phy" / "modulation_recognition_awgn_v1.yaml"


def _step(summary, step_id: str):
    return next(item for item in summary["steps"] if item["id"] == step_id)


class ModulationRecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()

    def test_controlled_recipe_is_balanced_label_separated_and_accurate(self) -> None:
        authored = load_recipe(RECIPE_PATH)
        inspection = inspect_execution_profile(authored)
        self.assertEqual(inspection.status, "conformant", inspection.issues)
        concrete = recipe_from_dict(
            materialize_recipe_matrix_selection(authored, {"channel.snr_db": 14.0})
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = LocalExecutor(
                self.registry,
                LocalStore(Path(tmp) / ".noema"),
            ).run(concrete)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            data_outputs = _step(summary, "data")["outputs"]
            with np.load(data_outputs["frames"]["path"], allow_pickle=False) as payload:
                self.assertEqual(set(payload.files), {"symbols", "metadata_json"})
                frames = np.asarray(payload["symbols"])
                frame_metadata = json.loads(str(payload["metadata_json"]))
            with np.load(data_outputs["labels"]["path"], allow_pickle=False) as payload:
                labels = np.asarray(payload["class_ids"])
            self.assertEqual(frames.shape, (96, 128))
            self.assertEqual(np.bincount(labels, minlength=3).tolist(), [32, 32, 32])
            self.assertTrue(frame_metadata["truth_separated"])
            self.assertNotIn("class_ids", frame_metadata)
            observation_output = _step(summary, "observation")["outputs"]["observation"]
            with np.load(observation_output["path"], allow_pickle=False) as payload:
                self.assertEqual(set(payload.files), {"iq_ri", "metadata_json"})
                self.assertEqual(np.asarray(payload["iq_ri"]).shape, (96, 128, 2))
            metrics = _step(summary, "evaluation")["metrics"]
            self.assertGreaterEqual(metrics["modulation_recognition.accuracy"], 0.98)
            self.assertGreaterEqual(metrics["modulation_recognition.balanced_accuracy"], 0.98)
            self.assertGreaterEqual(metrics["modulation_recognition.macro_f1"], 0.98)
            matrix_output = _step(summary, "evaluation")["outputs"]["confusion_matrix"]
            with np.load(matrix_output["path"], allow_pickle=False) as payload:
                self.assertEqual(np.asarray(payload["counts"]).shape, (3, 3))
                self.assertEqual(int(np.sum(payload["counts"])), 96)

    def test_classifier_exposes_exact_portable_artifact_abi(self) -> None:
        operation = self.registry.get("model.modulation_classifier_adapter").describe()
        abi = operation["trained_artifact_abi"]
        self.assertEqual(abi["entrypoint_id"], "modulation_classifier")
        self.assertEqual(abi["required_operation_inputs"], ["observation"])
        self.assertEqual(abi["inputs"]["iq_ri"]["shape"], ["batch", "sample", 2])
        self.assertEqual(abi["outputs"]["class_logits"]["shape"], ["batch", 3])
        modes = operation["params_schema"]["properties"]["mode"]["enum"]
        self.assertEqual(
            modes,
            [
                "classical_cumulant",
                "classical_likelihood",
                "classical_evm",
                "learned_artifact",
            ],
        )

    def test_reference_model_starts_from_the_blind_cumulant_prior(self) -> None:
        rng = np.random.default_rng(8123)
        constellations = (
            np.asarray([-1.0, 1.0], dtype=np.complex64),
            np.asarray(
                [-1.0 - 1.0j, -1.0 + 1.0j, 1.0 - 1.0j, 1.0 + 1.0j],
                dtype=np.complex64,
            )
            / np.sqrt(2.0),
            (
                np.asarray(
                    [
                        complex(in_phase, quadrature)
                        for in_phase in (-3.0, -1.0, 1.0, 3.0)
                        for quadrature in (-3.0, -1.0, 1.0, 3.0)
                    ],
                    dtype=np.complex64,
                )
                / np.sqrt(10.0)
            ),
        )
        labels = np.repeat(np.arange(3, dtype=np.int64), 12)
        received = np.empty((labels.size, 128), dtype=np.complex64)
        symbol_index = np.arange(128, dtype=np.float32)
        for frame_index, class_id in enumerate(labels):
            points = constellations[int(class_id)]
            symbols = rng.choice(points, size=128)
            phase = rng.uniform(-np.pi, np.pi)
            frequency = rng.uniform(-0.006, 0.006)
            carrier = np.exp(
                1j * (phase + 2.0 * np.pi * frequency * symbol_index)
            )
            noise = 0.02 * (
                rng.standard_normal(128) + 1j * rng.standard_normal(128)
            )
            received[frame_index] = symbols * carrier + noise
        baseline_scores, _ = _blind_cumulant_scores(received)
        model_module = runpy.run_path(
            str(
                ROOT
                / "demo_trainings"
                / "modulation_recognition_supervised_cnn"
                / "model.py"
            ),
            run_name="amc_reference_model",
        )
        model = model_module["ReferenceModulationCNN1D"]().eval()
        iq_ri = np.stack([received.real, received.imag], axis=2).astype(
            np.float32
        )
        with torch.no_grad():
            logits, residual, analytic_scores = model.forward_components(
                torch.from_numpy(iq_ri)
            )
        self.assertEqual(float(torch.max(torch.abs(residual))), 0.0)
        np.testing.assert_array_equal(
            np.argmax(logits.numpy(), axis=1),
            np.argmax(analytic_scores.numpy(), axis=1),
        )
        np.testing.assert_array_equal(
            np.argmax(analytic_scores.numpy(), axis=1),
            np.argmax(baseline_scores, axis=1),
        )
        self.assertGreaterEqual(
            float(np.mean(np.argmax(analytic_scores.numpy(), axis=1) == labels)),
            0.98,
        )

    def test_benchmark_runs_classical_grid_and_skips_unbound_learned_slot(self) -> None:
        pack = load_benchmark_pack(BENCHMARK_PATH)
        validation = validate_benchmark_pack(pack, self.registry, ROOT)
        self.assertEqual(validation["catalog_validation"]["status"], "valid")
        with tempfile.TemporaryDirectory() as tmp:
            benchmark_dir = run_benchmark_pack(
                pack,
                self.registry,
                LocalStore(Path(tmp) / ".noema"),
                ROOT,
            )
            result = json.loads((benchmark_dir / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual([item["status"] for item in result["recipes"]], ["completed", "completed", "completed", "skipped"])
        self.assertIn("bind", result["recipes"][-1]["skip_reason"].lower())
        for row in result["recipes"][:3]:
            self.assertIn("modulation_recognition.balanced_accuracy", row["metrics"])

    def test_training_export_writes_valid_capture_contract_and_optional_cnn(self) -> None:
        recipe = apply_training_plan(
            load_recipe(RECIPE_PATH),
            TrainingPlan(
                dataset_capture={
                    "taps": [
                        {"id": "iq_frames", "from": "observation.observation"},
                        {"id": "modulation_labels", "from": "data.labels"},
                    ],
                    "split_plan": {
                        "total_samples": 2304,
                        "percentages": {"train": 50, "validation": 25, "test": 25},
                    },
                }
            ),
        )
        capture = inspect_training_capture(
            recipe,
            self.registry,
            optimizable_steps=["receiver"],
        )
        self.assertTrue(capture["ready"], capture.get("issue"))
        self.assertEqual(capture["sample_unit"], "recipe records")
        self.assertEqual(capture["split_plan"]["counts"], {"train": 1152, "validation": 576, "test": 576})
        self.assertIn("modulation-recognition", exporter_ids())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "amc_training"
            payload = export_differentiable_scenario(
                recipe,
                self.registry,
                optimizable_steps=["receiver"],
                loss="classification.cross_entropy",
                framework="torch",
                out_dir=out_dir,
                project_root=root,
                exporter="modulation-recognition",
                include_starter=True,
            )
            validator = runpy.run_path(str(out_dir / "validate_contract.py"), run_name="amc_bundle_validator")
            validator["validate_bundle"](out_dir)
            self.assertEqual(payload["starter_exporter"], "modulation-recognition")
            self.assertEqual(payload["data_contract"]["mode"], "captured_generic_tensors")
            self.assertEqual(
                {item["reference"] for item in payload["data_contract"]["signals"]},
                {"observation.observation", "data.labels"},
            )
            starter = out_dir / "reference_training"
            for relative in (
                "model.py",
                "losses.py",
                "datamodule.py",
                "train.py",
                "evaluate.py",
                "build_benchmark.py",
                "train_config.yaml",
                "training_contract.yaml",
                "training_template.yaml",
            ):
                self.assertTrue((starter / relative).is_file(), relative)
            config = yaml.safe_load((starter / "train_config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["training_template"], "modulation_recognition.supervised_cnn1d")
            self.assertEqual(config["data"]["feature_tap"], "iq_frames")
            self.assertEqual(config["data"]["target_tap"], "modulation_labels")
            self.assertEqual(
                config["model"]["architecture"],
                "cumulant_prior_residual_temporal_cnn",
            )
            self.assertEqual(config["model"]["channels"], [32, 64])
            self.assertTrue(config["training"]["rotation_augmentation"])
            self.assertEqual(
                config["objective"]["checkpoint_selection"],
                "minimum_validation_cross_entropy_then_maximum_balanced_accuracy",
            )
            self.assertEqual(
                config["training"]["artifact_manifest_path"],
                "../trained_artifact.yaml",
            )
            self.assertEqual(
                config["training"]["artifact_component_path"],
                "../artifacts/modulation_classifier.onnx",
            )
            train_capture = None
            for split in ("train", "validation", "test"):
                capture_recipe = yaml.safe_load(
                    (out_dir / ("capture_%s_recipe.yaml" % split)).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    capture_recipe["dataset_capture"]["max_runs"],
                    capture_recipe["dataset_capture"]["samples"],
                )
                if split == "train":
                    train_capture = capture_recipe
            self.assertFalse((out_dir / "trained_artifact.yaml").exists())
            post_training = payload["project_manifest"]["external_training"]["optional_demo_scaffold"]["post_training"]
            self.assertEqual(post_training["command"], "python build_benchmark.py")
            component_path = _write_stub_artifact(
                out_dir / "trained_artifact.yaml"
            )
            _write_training_evidence(starter)
            completed = subprocess.run(
                [
                    sys.executable,
                    "build_benchmark.py",
                    "--snr-db=-2,6",
                    "--seeds",
                    "81001,82001",
                ],
                cwd=starter,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            pack = load_benchmark_pack(starter / "benchmark_pack.yaml")
            validation = validate_benchmark_pack(pack, self.registry, starter)
            self.assertEqual(validation["recipe_count"], 12)
            self.assertTrue(
                all(
                    entry.params["method_id"]
                    == entry.params["metadata"]["benchmark_method"]
                    for entry in pack.recipes
                )
            )
            self.assertNotIn("tier", pack.metadata)
            self.assertFalse(
                any(
                    "modulation_recognition.frame_count" in warning
                    for warning in validation["catalog_validation"]["warnings"]
                )
            )
            demo = pack.metadata["demo"]
            self.assertEqual(demo["primary_metric"], "modulation_recognition.balanced_accuracy")
            self.assertEqual(demo["training_evidence"][0]["series"], "learned_classifier")
            self.assertEqual(
                demo["training_evidence"][0]["trained_artifact_manifest"]["path"],
                "../trained_artifact.yaml",
            )
            for field in (
                "trained_artifact_manifest",
                "training_history",
                "evaluation_metrics",
            ):
                self.assertRegex(
                    demo["training_evidence"][0][field]["sha256"],
                    r"^[0-9a-f]{64}$",
                )
            self.assertEqual(
                demo["series"][0]["label"],
                "Blind differential cumulant",
            )
            self.assertEqual(pack.metadata["evaluation_frames_per_run"], 1536)
            self.assertIn(
                "modulation_recognition.frame_count",
                [metric["id"] for metric in pack.metrics],
            )
            metric_provenance = {
                metric["id"]: (
                    metric.get("source_step"),
                    metric.get("source_operation"),
                )
                for metric in pack.metrics
            }
            self.assertEqual(
                metric_provenance["channel.snr_db"],
                ("observation", "wireless.modulation_awgn_observation"),
            )
            for metric_id in (
                "modulation_recognition.accuracy",
                "modulation_recognition.balanced_accuracy",
                "modulation_recognition.macro_f1",
            ):
                self.assertEqual(
                    metric_provenance[metric_id],
                    ("evaluation", "metrics.modulation_classification"),
                )
            self.assertEqual(
                metric_provenance["modulation_recognition.frame_count"],
                ("data", "source.modulation_frames"),
            )
            self.assertTrue(all(plot["kind"] == "line" for plot in demo["plots"]))
            self.assertTrue(
                all(
                    plot["style"] == {"aggregation": "mean_ci", "y_scale": "linear"}
                    for plot in demo["plots"]
                )
            )
            paired = {}
            paired_metadata = {}
            for entry in pack.recipes:
                params = entry.params
                metadata = params["metadata"]
                selection = params["matrix_selection"]
                key = (selection["channel.snr_db"], metadata["benchmark_paired_seed"])
                paired.setdefault(key, {})[metadata["benchmark_method"]] = params["step_params"]
                paired_metadata.setdefault(key, {})[
                    metadata["benchmark_method"]
                ] = metadata
            self.assertEqual(len(paired), 4)
            for key, methods in paired.items():
                self.assertEqual(
                    set(methods),
                    {
                        "blind_cumulant",
                        "learned_classifier",
                        "oracle_likelihood",
                    },
                )
                common_seeds = {
                    (row["data"]["seed"], row["observation"]["seed"])
                    for row in methods.values()
                }
                self.assertEqual(len(common_seeds), 1)
                method_metadata = paired_metadata[key]
                self.assertEqual(
                    {
                        row["statistical_unit"]
                        for row in method_metadata.values()
                    },
                    {"paired held-out symbol/carrier/noise seed"},
                )
                self.assertEqual(
                    len(
                        {
                            row["aggregation_cell_id"]
                            for row in method_metadata.values()
                        }
                    ),
                    1,
                )
                self.assertEqual(
                    len({row["pairing_id"] for row in method_metadata.values()}),
                    1,
                )
                self.assertEqual(
                    {row["data"]["frame_count"] for row in methods.values()},
                    {1536},
                )
                self.assertEqual(
                    methods["learned_classifier"]["receiver"]["artifact_manifest_path"],
                    "amc_training/trained_artifact.yaml",
                )
                self.assertRegex(
                    methods["learned_classifier"]["receiver"][
                        "artifact_package_sha256"
                    ],
                    r"^[0-9a-f]{64}$",
                )
                self.assertTrue(
                    (
                        root
                        / methods["learned_classifier"]["receiver"][
                            "artifact_manifest_path"
                        ]
                    ).is_file()
                )

            leaked_evaluation_path = starter / "evaluation_metrics.json"
            leaked_evaluation = json.loads(
                leaked_evaluation_path.read_text(encoding="utf-8")
            )
            leaked_splits = leaked_evaluation["split_record_fingerprints"][
                "splits"
            ]
            leaked_splits["test"] = json.loads(
                json.dumps(leaked_splits["train"])
            )
            leaked_evaluation_path.write_text(
                json.dumps(leaked_evaluation),
                encoding="utf-8",
            )
            leaked_result = subprocess.run(
                [
                    sys.executable,
                    "build_benchmark.py",
                    "--output",
                    "record_overlap_pack.yaml",
                ],
                cwd=starter,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(leaked_result.returncode, 0)
            self.assertIn("byte-identical record", leaked_result.stderr)
            self.assertFalse((starter / "record_overlap_pack.yaml").exists())
            _write_training_evidence(starter)

            self.assertIsNotNone(train_capture)
            overlapping_data_seed = derive_seed(
                int(train_capture["metadata"]["seed"]),
                str(train_capture["name"]),
                "data",
                "modulation_frames",
            )
            seed_overlap = subprocess.run(
                [
                    sys.executable,
                    "build_benchmark.py",
                    "--seeds",
                    str(overlapping_data_seed),
                    "--output",
                    "seed_overlap_pack.yaml",
                ],
                cwd=starter,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(seed_overlap.returncode, 0)
            self.assertIn(
                "not held out from dataset capture", seed_overlap.stderr
            )
            self.assertFalse((starter / "seed_overlap_pack.yaml").exists())

            (starter / "evaluation_metrics.json").write_text("{}", encoding="utf-8")
            invalid_evidence = subprocess.run(
                [
                    sys.executable,
                    "build_benchmark.py",
                    "--output",
                    "invalid_evidence_pack.yaml",
                ],
                cwd=starter,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(invalid_evidence.returncode, 0)
            self.assertIn(
                "evaluation metrics must contain a non-empty JSON",
                invalid_evidence.stderr,
            )
            self.assertFalse((starter / "invalid_evidence_pack.yaml").exists())

            _write_training_evidence(starter)
            stale_evaluation_path = starter / "evaluation_metrics.json"
            stale_evaluation = json.loads(
                stale_evaluation_path.read_text(encoding="utf-8")
            )
            stale_evaluation["component_sha256"] = "f" * 64
            stale_evaluation_path.write_text(
                json.dumps(stale_evaluation), encoding="utf-8"
            )
            stale_evaluation_result = subprocess.run(
                [
                    sys.executable,
                    "build_benchmark.py",
                    "--output",
                    "stale_evaluation_pack.yaml",
                ],
                cwd=starter,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(stale_evaluation_result.returncode, 0)
            self.assertIn(
                "component SHA-256 does not match the selected artifact component",
                stale_evaluation_result.stderr,
            )
            self.assertFalse((starter / "stale_evaluation_pack.yaml").exists())

            _write_training_evidence(starter)
            component_path.write_bytes(component_path.read_bytes() + b"tampered")
            tampered_artifact = subprocess.run(
                [
                    sys.executable,
                    "build_benchmark.py",
                    "--output",
                    "tampered_artifact_pack.yaml",
                ],
                cwd=starter,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(tampered_artifact.returncode, 0)
            self.assertIn("SHA-256 does not match", tampered_artifact.stderr)
            self.assertFalse((starter / "tampered_artifact_pack.yaml").exists())

    def test_capture_record_fingerprints_reject_cross_split_byte_overlap(
        self,
    ) -> None:
        module = runpy.run_path(
            str(
                ROOT
                / "demo_trainings"
                / "modulation_recognition_supervised_cnn"
                / "datamodule.py"
            )
        )
        load_capture_dataset = module["load_capture_dataset"]
        split_fingerprint_report = module["split_fingerprint_report"]
        frames = np.asarray(
            [
                [[0.25, -0.5], [1.0, 0.0]],
                [[-0.75, 0.5], [0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 2], dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = _write_amc_capture(root / "train", "train", frames, labels)
            validation_path = _write_amc_capture(
                root / "validation",
                "validation",
                frames.copy(),
                labels.copy(),
            )
            test_path = _write_amc_capture(
                root / "test",
                "test",
                frames + np.float32(0.125),
                labels,
            )
            common = {
                "feature_tap": "iq_frames",
                "target_tap": "modulation_labels",
            }
            train = load_capture_dataset(
                [str(train_path)], expected_split="train", **common
            )
            validation = load_capture_dataset(
                [str(validation_path)],
                expected_split="validation",
                **common,
            )
            test = load_capture_dataset(
                [str(test_path)], expected_split="test", **common
            )
            self.assertEqual(train.record_sha256, validation.record_sha256)
            with self.assertRaisesRegex(
                ValueError,
                "overlap between train and validation",
            ):
                split_fingerprint_report(
                    {"train": train, "validation": validation, "test": test}
                )


def _write_stub_artifact(path: Path) -> Path:
    artifact_files = path.parent / "artifact_files"
    artifact_files.mkdir(parents=True, exist_ok=True)
    contract_path = artifact_files / "training_contract.yaml"
    contract = {
        "schema_version": 1,
        "kind": "noema.trainable_slot_contract@1",
        "id": "test.amc_training_contract",
        "version": 1,
        "artifact_return": {
            "bindings": [
                {"operation": "model.modulation_classifier_adapter"}
            ],
        },
    }
    contract_path.write_text(
        yaml.safe_dump(contract, sort_keys=False), encoding="utf-8"
    )
    component_path = artifact_files / "modulation_classifier.onnx"
    component_path.write_bytes(b"test-amc-onnx-component")
    payload = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": "test.amc_classifier",
        "name": "Test AMC classifier",
        "contract": {
            "id": contract["id"],
            "version": contract["version"],
            "path": "artifact_files/training_contract.yaml",
            "sha256": _canonical_sha256(contract),
            "file_sha256": _file_sha256(contract_path),
        },
        "components": [
            {
                "id": "classifier",
                "role": "modulation_classifier",
                "path": "artifact_files/modulation_classifier.onnx",
                "sha256": _file_sha256(component_path),
                "format": "onnx",
            }
        ],
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 1,
            "entrypoints": [
                {
                    "id": "modulation_classifier",
                    "component": "classifier",
                    "inputs": [
                        {
                            "name": "iq_ri",
                            "dtype": "float32",
                            "shape": ["batch", "sample", 2],
                        }
                    ],
                    "outputs": [
                        {
                            "name": "class_logits",
                            "dtype": "float32",
                            "shape": ["batch", 3],
                        }
                    ],
                }
            ],
        },
        "compatible_operations": [
            {
                "operation": "model.modulation_classifier_adapter",
                "runtime_entrypoint": "modulation_classifier",
                "params": {},
            }
        ],
        "training": {
            "split_record_fingerprints": _record_fingerprint_report(
                {
                    "train": ["1" * 64],
                    "validation": ["2" * 64],
                }
            )
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return component_path


def _write_training_evidence(starter: Path) -> None:
    manifest_path = starter.parent / "trained_artifact.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    component = dict(manifest["components"][0])
    (starter / "training_history.json").write_text(
        json.dumps([{"epoch": 1, "train_cross_entropy": 0.5}]),
        encoding="utf-8",
    )
    (starter / "evaluation_metrics.json").write_text(
        json.dumps(
            {
                "balanced_accuracy": 0.75,
                "component_sha256": component["sha256"],
                "test_capture_schema_sha256": ["a" * 64],
                "split_record_fingerprints": _record_fingerprint_report(
                    {
                        "train": ["1" * 64],
                        "validation": ["2" * 64],
                        "test": ["3" * 64],
                    }
                ),
            }
        ),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_fingerprint_report(
    split_records: dict[str, list[str]],
) -> dict:
    rows = {}
    for name, records in split_records.items():
        unique = sorted(set(records))
        digest = hashlib.sha256(b"noema.amc.capture-record-set@1\0")
        for fingerprint in unique:
            digest.update(fingerprint.encode("ascii"))
            digest.update(b"\0")
        rows[name] = {
            "record_count": len(records),
            "unique_record_count": len(unique),
            "fingerprint_set_sha256": digest.hexdigest(),
            "record_sha256": list(records),
        }
    names = list(split_records)
    pairwise = []
    disjoint = True
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = len(
                set(split_records[left]).intersection(split_records[right])
            )
            disjoint = disjoint and overlap == 0
            pairwise.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "overlap_count": overlap,
                }
            )
    return {
        "algorithm": "sha256:noema.amc.capture-record@1",
        "disjoint": disjoint,
        "splits": rows,
        "pairwise_overlap": pairwise,
    }


def _write_amc_capture(
    path: Path,
    split: str,
    frames: np.ndarray,
    labels: np.ndarray,
) -> Path:
    shards = path / "shards"
    shards.mkdir(parents=True)
    np.savez(
        shards / "part-00000.npz",
        iq_frames=np.asarray(frames, dtype=np.float32),
        modulation_labels=np.asarray(labels, dtype=np.int64),
    )
    (path / "schema.json").write_text(
        json.dumps(
            {
                "kind": "noema.capture_dataset",
                "split": split,
                "tap_schemas": {
                    "iq_frames": {},
                    "modulation_labels": {},
                },
                "shards": [{"path": "shards/part-00000.npz"}],
            }
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
