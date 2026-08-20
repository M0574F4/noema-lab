from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from noema_lab.core.trained_artifact_runtime import (
    TrainedArtifactRuntimeError,
    _validate_array_against_tensor_spec,
    admit_trained_artifact_runtime,
    run_trained_artifact_entrypoint,
)
from noema_lab.core.trained_artifacts import (
    TrainedArtifactError,
    _validate_csi_feedback_pair_invariants,
    discover_trained_artifacts,
    import_external_trained_artifact_package,
    inspect_trained_artifact,
    trained_artifact_recipe_compatibility_issues,
    validate_trained_artifact_publication_readiness,
    validate_trained_artifact_package,
)
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.ops import build_registry
from tools.check_paper import (
    Reporter,
    sha256_path,
    validate_publication_artifact_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainedArtifactV2Tests(unittest.TestCase):
    def test_phase_tracking_receiver_is_discovered_from_workbench_training_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = (
                root
                / ".noema"
                / "training_exports"
                / "qpsk_phase_tracking"
            )
            _write_phase_tracking_receiver_package(package)

            rows = discover_trained_artifacts(
                root,
                registry=build_registry(),
                operation="demodulation.phase_tracking_receiver_adapter",
            )

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertTrue(row["ready"], row["issues"])
            self.assertEqual(row["source_class"], "project_trained")
            self.assertEqual(
                row["manifest_path"],
                ".noema/training_exports/qpsk_phase_tracking/trained_artifact.yaml",
            )
            binding = row["compatible_operations"][0]
            self.assertEqual(
                binding["operation"],
                "demodulation.phase_tracking_receiver_adapter",
            )
            self.assertEqual(
                binding["required_inputs"],
                ["rx_symbols", "pilot_context"],
            )
            self.assertEqual(binding["params"]["mode"], "learned_artifact")
            self.assertEqual(
                binding["params"]["artifact_manifest_path"],
                row["manifest_path"],
            )

    def test_legacy_five_channel_phase_receiver_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / ".noema" / "training_exports" / "legacy_phase"
            _write_phase_tracking_receiver_package(package, legacy=True)

            rows = discover_trained_artifacts(
                root,
                registry=build_registry(),
                operation="demodulation.phase_tracking_receiver_adapter",
            )

            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["ready"])
            issues = "; ".join(rows[0]["issues"])
            self.assertIn("missing receiver_features_v3", issues)
            self.assertIn("unexpected receiver_features", issues)

    def test_imported_package_cannot_self_assert_published_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            package = base / "published-reference"
            manifest_path = _write_single_package(package)
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            payload["label"] = "Authoritative published label"
            payload["source"] = {"origin": "published_reference_checkpoint"}
            manifest_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            _attach_publication_selection_history(manifest_path)
            notice = package / "LICENSE.upstream"
            notice.write_text("Example permissive license\n", encoding="utf-8")
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            payload["support_files"].append(
                {
                    "role": "license_notice",
                    "path": notice.name,
                    "sha256": _sha256(notice),
                }
            )
            manifest_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            asserted = inspect_trained_artifact(
                manifest_path,
                project_root=base,
                registry=build_registry(),
            )
            self.assertTrue(asserted["publication_ready"], asserted["publication_issues"])
            self.assertEqual(asserted["source_class"], "published_reference")

            imported = import_external_trained_artifact_package(
                base / "project",
                package,
                label="browser-upload-filename",
                registry=build_registry(),
            )

            self.assertTrue(imported["ready"], imported["issues"])
            self.assertEqual(imported["label"], "Authoritative published label")
            self.assertEqual(imported["source_class"], "imported")
            self.assertEqual(
                imported["source"]["origin"],
                "external_artifact_import",
            )
            self.assertEqual(
                imported["source"]["upstream_claimed_source"]["origin"],
                "published_reference_checkpoint",
            )
            self.assertFalse(imported["publication_ready"])
            self.assertIn(
                "imported artifacts cannot self-assert publication readiness",
                "; ".join(imported["publication_issues"]),
            )
            license_row = next(
                item
                for item in imported["support_files"]
                if item["role"] == "license_notice"
            )
            self.assertEqual(
                license_row["actual_sha256"],
                license_row["sha256"],
            )
            managed_manifest = base / "project" / imported["manifest_path"]
            self.assertTrue((managed_manifest.parent / notice.name).is_file())

            notice.write_text("tampered\n", encoding="utf-8")
            invalid = inspect_trained_artifact(
                manifest_path, project_root=base, registry=build_registry()
            )
            self.assertFalse(invalid["valid"])
            self.assertIn("SHA-256 does not match", "; ".join(invalid["issues"]))

    def test_csi_pair_invariants_and_recipe_protocol_gate(self):
        entrypoints = {
            "encoder": {
                "id": "encoder",
                "validated_signature": {
                    "inputs": [
                        {"name": "csi_ri", "shape": ["batch", 2, 8, 32]}
                    ],
                    "outputs": [
                        {"name": "feedback_code", "shape": ["batch", 32]}
                    ],
                },
            },
            "decoder": {
                "id": "decoder",
                "validated_signature": {
                    "inputs": [
                        {"name": "feedback_code", "shape": ["batch", 64]}
                    ],
                    "outputs": [
                        {"name": "csi_hat_ri", "shape": ["batch", 2, 8, 32]}
                    ],
                },
            },
        }
        bindings = _csi_bindings(feedback_dimension=32)
        issues = []
        _validate_csi_feedback_pair_invariants(
            bindings, entrypoints, "reference_baseline", issues
        )
        self.assertIn(
            "CSI encoder feedback output and decoder feedback input axis 1 differs",
            "; ".join(issues),
        )

        entrypoints["decoder"]["validated_signature"]["inputs"][0]["shape"][-1] = 32
        artifact = {
            "compatible_operations": [
                {
                    **bindings[0],
                    "tensor_abi": entrypoints["encoder"],
                },
                {
                    **bindings[1],
                    "tensor_abi": entrypoints["decoder"],
                },
            ],
            "training": {
                "feedback_constraint": {
                    "mode": "uniform_quantized",
                    "feedback_bits_per_sample": 128,
                }
            },
        }
        recipe = _csi_recipe()
        self.assertEqual(
            trained_artifact_recipe_compatibility_issues(artifact, recipe), []
        )
        recipe["steps"][0]["params"]["tx_antennas"] = 4
        recipe["steps"][2]["params"]["bits_per_latent"] = 2
        compatibility = "; ".join(
            trained_artifact_recipe_compatibility_issues(artifact, recipe)
        )
        self.assertIn("requires 8 transmit antennas", compatibility)
        self.assertIn("requires 128 feedback bits", compatibility)

        recipe = _csi_recipe()
        recipe["steps"][0]["params"].pop("tx_antennas")
        recipe["steps"][1]["params"]["feedback_dimension"] = "invalid"
        compatibility = "; ".join(
            trained_artifact_recipe_compatibility_issues(artifact, recipe)
        )
        self.assertIn("recipe has no valid integer value", compatibility)

        artifact["training"]["feedback_constraint"]["feedback_bits_per_sample"] = "invalid"
        compatibility = "; ".join(
            trained_artifact_recipe_compatibility_issues(artifact, _csi_recipe())
        )
        self.assertIn(
            "feedback_bits_per_sample constraint is not a valid integer",
            compatibility,
        )

    def test_runtime_binds_repeated_symbolic_axes_across_tensors(self):
        bindings = {}
        _validate_array_against_tensor_spec(
            np.ones((2, 4), dtype=np.float32),
            {"dtype": "float32", "shape": ["batch", 4]},
            "first input",
            symbolic_dimensions=bindings,
        )
        with self.assertRaisesRegex(
            TrainedArtifactRuntimeError, "symbolic dimension batch"
        ):
            _validate_array_against_tensor_spec(
                np.ones((3, 1), dtype=np.float32),
                {"dtype": "float32", "shape": ["batch", 1]},
                "second input",
                symbolic_dimensions=bindings,
            )

    def test_valid_package_is_architecture_and_loss_neutral_and_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "returned-model"
            manifest_path = _write_single_package(package)

            row = inspect_trained_artifact(
                manifest_path,
                project_root=Path(tmp),
                registry=build_registry(),
            )

            self.assertTrue(row["valid"], row["issues"])
            self.assertTrue(row["ready"], row["issues"])
            self.assertEqual(row["contract"]["id"], "noema.slot.power-policy")
            self.assertEqual(row["components"][0]["role"], "power_policy")
            self.assertEqual(row["training"]["loss"], "a completely external objective")
            self.assertEqual(row["training"]["architecture"], "user-defined")
            binding = row["compatible_operations"][0]
            self.assertEqual(binding["runtime_entrypoint"], "power_policy")
            self.assertEqual(
                binding["tensor_abi"]["inputs"][0]["semantic"],
                "per-subcarrier channel power gain",
            )

            values = np.asarray(
                [[0.2, 1.5], [3.0, 0.5], [0.7, 0.9], [4.0, 0.1], [1.1, 1.2]],
                dtype=np.float32,
            )
            outputs = run_trained_artifact_entrypoint(
                manifest_path,
                "power_policy",
                {
                    "channel_gain": values,
                    "noise_variance": np.full((5, 1), 0.2, dtype=np.float32),
                    "average_power_budget": np.ones((5, 1), dtype=np.float32),
                },
                expected_package_sha256=row["package_sha256"],
                project_root=Path(tmp),
            )
            np.testing.assert_array_equal(outputs["allocation_scores"], values)
            chunked_outputs = run_trained_artifact_entrypoint(
                manifest_path,
                "power_policy",
                {
                    "channel_gain": values,
                    "noise_variance": np.full((5, 1), 0.2, dtype=np.float32),
                    "average_power_budget": np.ones((5, 1), dtype=np.float32),
                },
                expected_package_sha256=row["package_sha256"],
                project_root=Path(tmp),
                inference_batch_size=2,
            )
            np.testing.assert_array_equal(
                chunked_outputs["allocation_scores"],
                outputs["allocation_scores"],
            )

    def test_admitted_runtime_reuses_inspection_loader_and_session(self):
        from noema_lab.ops.portable_onnx import (
            _load_component_cached,
            load_portable_onnx_component,
        )

        try:
            import onnxruntime
        except Exception as exc:  # pragma: no cover - optional dependency gate
            self.skipTest("onnxruntime is unavailable: %s" % exc)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_single_package(root / "returned-model")
            inspected = inspect_trained_artifact(
                manifest_path,
                project_root=root,
            )
            _load_component_cached.cache_clear()
            values = np.asarray(
                [[0.2, 1.5], [3.0, 0.5], [0.7, 0.9]],
                dtype=np.float32,
            )
            inputs = {
                "channel_gain": values,
                "noise_variance": np.full(
                    (3, 1), 0.2, dtype=np.float32
                ),
                "average_power_budget": np.ones(
                    (3, 1), dtype=np.float32
                ),
            }

            with (
                patch(
                    "noema_lab.core.trained_artifacts."
                    "inspect_trained_artifact",
                    wraps=inspect_trained_artifact,
                ) as inspect_mock,
                patch(
                    "noema_lab.ops.portable_onnx."
                    "load_portable_onnx_component",
                    wraps=load_portable_onnx_component,
                ) as load_mock,
                patch(
                    "onnxruntime.InferenceSession",
                    wraps=onnxruntime.InferenceSession,
                ) as session_mock,
            ):
                admitted = admit_trained_artifact_runtime(
                    manifest_path,
                    expected_package_sha256=inspected["package_sha256"],
                    project_root=root,
                )
                admission_counts = (
                    inspect_mock.call_count,
                    load_mock.call_count,
                    session_mock.call_count,
                )
                self.assertEqual(admission_counts[0], 1)
                self.assertGreaterEqual(admission_counts[1], 1)
                self.assertEqual(admission_counts[2], 1)

                first = admitted.run_entrypoint(
                    "power_policy",
                    inputs,
                )
                second = admitted.run_entrypoint(
                    "power_policy",
                    inputs,
                    inference_batch_size=2,
                )
                self.assertEqual(
                    (
                        inspect_mock.call_count,
                        load_mock.call_count,
                        session_mock.call_count,
                    ),
                    admission_counts,
                )

            cold = run_trained_artifact_entrypoint(
                manifest_path,
                "power_policy",
                inputs,
                expected_package_sha256=inspected["package_sha256"],
                project_root=root,
            )
            np.testing.assert_array_equal(
                first["allocation_scores"],
                cold["allocation_scores"],
            )
            np.testing.assert_array_equal(
                second["allocation_scores"],
                cold["allocation_scores"],
            )

    def test_admitted_runtime_rejects_wrong_identity_and_invalid_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_single_package(root / "returned-model")
            inspected = inspect_trained_artifact(
                manifest_path,
                project_root=root,
            )
            with self.assertRaisesRegex(
                TrainedArtifactRuntimeError,
                "package identity mismatch",
            ):
                admit_trained_artifact_runtime(
                    manifest_path,
                    expected_package_sha256="0" * 64,
                    project_root=root,
                )

            admitted = admit_trained_artifact_runtime(
                manifest_path,
                expected_package_sha256=inspected["package_sha256"],
                project_root=root,
            )
            self.assertEqual(admitted.entrypoint_ids, ("power_policy",))
            self.assertEqual(
                admitted.entrypoint_session_evidence("power_policy"),
                {
                    "providers": ["CPUExecutionProvider"],
                    "provider_fallback_permitted": False,
                    "execution_mode": "ORT_SEQUENTIAL",
                    "intra_op_num_threads": 1,
                    "inter_op_num_threads": 1,
                    "graph_optimization_level": "ORT_DISABLE_ALL",
                    "enable_mem_pattern": False,
                    "enable_profiling": False,
                },
            )
            with self.assertRaisesRegex(
                TrainedArtifactRuntimeError,
                "unknown trained-artifact runtime entrypoint",
            ):
                admitted.entrypoint_session_evidence("not_declared")
            valid_inputs = {
                "channel_gain": np.ones((1, 2), dtype=np.float32),
                "noise_variance": np.full(
                    (1, 1), 0.2, dtype=np.float32
                ),
                "average_power_budget": np.ones(
                    (1, 1), dtype=np.float32
                ),
            }
            with self.assertRaisesRegex(
                TrainedArtifactRuntimeError,
                "no runtime entrypoint",
            ):
                admitted.run_entrypoint("not_declared", valid_inputs)
            with self.assertRaisesRegex(
                TrainedArtifactRuntimeError,
                "unexpected input",
            ):
                admitted.run_entrypoint(
                    "power_policy",
                    {
                        **valid_inputs,
                        "python_callback": np.ones(
                            (1,), dtype=np.float32
                        ),
                    },
                )
            with self.assertRaisesRegex(
                TrainedArtifactRuntimeError,
                "has dtype float64; expected float32",
            ):
                admitted.run_entrypoint(
                    "power_policy",
                    {
                        **valid_inputs,
                        "channel_gain": np.ones(
                            (1, 2), dtype=np.float64
                        ),
                    },
                )

    def test_admission_reconfines_inspected_component_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "returned-model"
            package.mkdir()
            manifest_path = package / "trained_artifact.yaml"
            manifest_path.write_text("schema_version: 2\n", encoding="utf-8")
            expected = "a" * 64
            inspection = {
                "schema_version": 2,
                "issues": [],
                "package_sha256": expected,
                "components": [
                    {
                        "id": "policy",
                        "path": str(root / "outside.onnx"),
                        "sha256": "b" * 64,
                    }
                ],
                "runtime": {
                    "available": True,
                    "backend": "onnxruntime",
                    "abi_version": 1,
                    "entrypoints": [
                        {
                            "id": "power_policy",
                            "component": "policy",
                            "inputs": [],
                            "outputs": [],
                        }
                    ],
                },
            }
            with patch(
                "noema_lab.core.trained_artifacts."
                "inspect_trained_artifact",
                return_value=inspection,
            ):
                with self.assertRaisesRegex(
                    TrainedArtifactRuntimeError,
                    "component path escapes the artifact package",
                ):
                    admit_trained_artifact_runtime(
                        manifest_path,
                        expected_package_sha256=expected,
                        project_root=root,
                    )

    def test_runtime_ready_is_distinct_from_publication_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "returned-model"
            manifest_path = _write_single_package(package)

            development = inspect_trained_artifact(
                manifest_path,
                project_root=Path(tmp),
                registry=build_registry(),
            )
            self.assertTrue(development["ready"], development["issues"])
            self.assertFalse(development["publication_ready"])
            self.assertEqual(development["publication_status"], "blocked")
            self.assertIn(
                "publication.selection_history is required",
                "; ".join(development["publication_issues"]),
            )

            _attach_publication_selection_history(manifest_path)
            publication = validate_trained_artifact_publication_readiness(
                manifest_path,
                project_root=Path(tmp),
                registry=build_registry(),
            )
            self.assertTrue(publication["ready"], publication["issues"])
            self.assertTrue(
                publication["publication_ready"],
                publication["publication_issues"],
            )
            self.assertEqual(publication["publication_status"], "ready")
            self.assertEqual(
                publication["publication_readiness"]["selected_candidate_id"],
                "candidate-a",
            )
            self.assertEqual(
                publication["runtime_component_set_sha256"],
                publication["publication_readiness"][
                    "runtime_component_set_sha256"
                ],
            )
            self.assertEqual(
                publication["runtime_identity_sha256"],
                publication["package_sha256"],
            )
            self.assertEqual(
                publication["artifact"]["manifest_file_sha256"],
                publication["artifact"]["sha256"],
            )

    def test_one_history_passes_core_readiness_and_paper_identity_freeze(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp:
            root = Path(tmp)
            package = root / "returned-model"
            manifest_path = _write_single_package(package)
            _attach_publication_selection_history(manifest_path)

            inspected = validate_trained_artifact_publication_readiness(
                manifest_path,
                project_root=root,
                registry=build_registry(),
            )
            package_file_sha256 = sha256_path(package)
            history_path = package / "provenance" / "model_selection_history.yaml"
            entry = {
                "path": package.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": package_file_sha256,
                "package_file_sha256": package_file_sha256,
                "runtime_component_set_sha256": inspected[
                    "runtime_component_set_sha256"
                ],
                "runtime_identity_sha256": inspected["runtime_identity_sha256"],
                "selection_history": {
                    "path": history_path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": _sha256(history_path),
                    "schema_version": 1,
                    "kind": "noema.model_selection_history",
                },
            }
            reporter = Reporter(strict=True)
            validate_publication_artifact_identity(
                entry,
                reporter,
                context="trained_artifacts[0]",
            )
            self.assertEqual(reporter.errors, [])

            wrong_domain = dict(entry)
            wrong_domain["runtime_component_set_sha256"] = inspected[
                "runtime_identity_sha256"
            ]
            mismatched = Reporter(strict=True)
            validate_publication_artifact_identity(
                wrong_domain,
                mismatched,
                context="trained_artifacts[0]",
            )
            self.assertTrue(
                any(
                    "runtime_component_set_sha256 disagrees" in issue
                    for issue in mismatched.errors
                )
            )
            self.assertTrue(
                any("does not bind" in issue for issue in mismatched.errors)
            )

            wrong_package = dict(entry)
            wrong_package["package_file_sha256"] = "f" * 64
            package_mismatch = Reporter(strict=True)
            validate_publication_artifact_identity(
                wrong_package,
                package_mismatch,
                context="trained_artifacts[0]",
            )
            self.assertTrue(
                any("disagrees with package bytes" in issue for issue in package_mismatch.errors)
            )

            wrong_runtime = dict(entry)
            wrong_runtime["runtime_identity_sha256"] = inspected[
                "runtime_component_set_sha256"
            ]
            runtime_mismatch = Reporter(strict=True)
            validate_publication_artifact_identity(
                wrong_runtime,
                runtime_mismatch,
                context="trained_artifacts[0]",
            )
            self.assertTrue(
                any(
                    "runtime_identity_sha256 disagrees" in issue
                    for issue in runtime_mismatch.errors
                )
            )

    def test_publication_selection_forbids_test_access_without_blocking_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "returned-model"
            manifest_path = _write_single_package(package)
            _attach_publication_selection_history(
                manifest_path,
                publication_test_accessed=True,
            )

            row = inspect_trained_artifact(
                manifest_path,
                project_root=Path(tmp),
                registry=build_registry(),
            )

            self.assertTrue(row["ready"], row["issues"])
            self.assertFalse(row["publication_ready"])
            self.assertIn(
                "publication-test access is forbidden during model selection",
                "; ".join(row["publication_issues"]),
            )
            with self.assertRaisesRegex(
                TrainedArtifactError,
                "publication-test access is forbidden",
            ):
                validate_trained_artifact_publication_readiness(
                    manifest_path,
                    project_root=Path(tmp),
                    registry=build_registry(),
                )

    def test_publication_selection_binds_selected_candidate_to_runtime_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "returned-model"
            manifest_path = _write_single_package(package)
            _attach_publication_selection_history(
                manifest_path,
                selected_runtime_component_set_sha256="f" * 64,
            )

            row = inspect_trained_artifact(
                manifest_path,
                project_root=Path(tmp),
                registry=build_registry(),
            )

            self.assertTrue(row["ready"], row["issues"])
            self.assertFalse(row["publication_ready"])
            self.assertIn(
                "does not bind the deployed runtime components",
                "; ".join(row["publication_issues"]),
            )

    def test_publication_selection_rejects_configuration_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "returned-model"
            manifest_path = _write_single_package(package)
            _attach_publication_selection_history(manifest_path)
            history_path = package / "provenance" / "model_selection_history.yaml"
            history = yaml.safe_load(history_path.read_text(encoding="utf-8"))
            history["candidates"][0]["configuration_identity_sha256"] = "f" * 64
            history_path.write_text(
                yaml.safe_dump(history, sort_keys=False), encoding="utf-8"
            )
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            history_sha = _sha256(history_path)
            manifest["publication"]["selection_history"]["sha256"] = history_sha
            for support in manifest["support_files"]:
                if support["role"] == "model_selection_history":
                    support["sha256"] = history_sha
            manifest_path.write_text(
                yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
            )

            row = inspect_trained_artifact(
                manifest_path,
                project_root=Path(tmp),
                registry=build_registry(),
            )
            self.assertTrue(row["ready"], row["issues"])
            self.assertFalse(row["publication_ready"])
            self.assertIn(
                "configuration.sha256 must match its support_files declaration",
                "; ".join(row["publication_issues"]),
            )

    def test_full_file_hashes_detect_component_and_contract_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "returned-model"
            manifest_path = _write_single_package(package)

            (package / "policy.onnx").write_bytes(
                (package / "policy.onnx").read_bytes() + b"tampered"
            )
            row = inspect_trained_artifact(manifest_path, project_root=Path(tmp))
            self.assertFalse(row["valid"])
            self.assertIn("SHA-256 does not match", "; ".join(row["issues"]))

            manifest_path = _write_single_package(package)
            contract = package / "slot_contract.yaml"
            contract.write_text(contract.read_text(encoding="utf-8") + "note: changed\n")
            row = inspect_trained_artifact(manifest_path, project_root=Path(tmp))
            self.assertFalse(row["valid"])
            self.assertIn("contract file SHA-256", "; ".join(row["issues"]))

    def test_declared_tensor_abi_must_match_onnx_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "returned-model"
            manifest_path = _write_single_package(package)
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            payload["runtime"]["entrypoints"][0]["inputs"][0]["dtype"] = "float64"
            manifest_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )

            row = inspect_trained_artifact(manifest_path, project_root=Path(tmp))

            self.assertFalse(row["valid"])
            self.assertIn("tensor ABI declares float64", "; ".join(row["issues"]))

    def test_missing_optional_runtime_is_valid_but_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "returned-model"
            manifest_path = _write_single_package(package)

            with patch(
                "noema_lab.core.trained_artifact_runtime.importlib.util.find_spec",
                return_value=None,
            ):
                row = inspect_trained_artifact(manifest_path, project_root=Path(tmp))

            self.assertTrue(row["valid"], row["issues"])
            self.assertFalse(row["ready"])
            self.assertEqual(row["status"], "unavailable")
            self.assertIn(
                "requires optional package",
                "; ".join(row["runtime"]["unavailable_reasons"]),
            )

    def test_grouped_encoder_decoder_application_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "returned-codec"
            manifest_path = _write_paired_package(package)

            row = inspect_trained_artifact(
                manifest_path,
                project_root=Path(tmp),
                registry=build_registry(),
            )
            self.assertTrue(row["ready"], row["issues"])
            self.assertEqual(row["application"]["mode"], "all_group_bindings")
            self.assertEqual(
                {item["role"] for item in row["compatible_operations"]},
                {"encoder", "decoder"},
            )

            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            payload["compatible_operations"][1]["role"] = "encoder"
            manifest_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            invalid = inspect_trained_artifact(
                manifest_path,
                project_root=Path(tmp),
                registry=build_registry(),
            )
            self.assertFalse(invalid["valid"])
            self.assertIn("duplicate role", "; ".join(invalid["issues"]))

    def test_operation_owned_binding_params_are_injected_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "returned-codec"
            manifest_path = _write_paired_package(package)

            row = inspect_trained_artifact(
                manifest_path,
                project_root=Path(tmp),
                registry=build_registry(),
            )
            self.assertTrue(row["ready"], row["issues"])
            bindings = {
                item["role"]: item for item in row["compatible_operations"]
            }
            self.assertEqual(
                bindings["encoder"]["params"],
                {
                    "runtime": "learned_artifact",
                    "artifact_manifest_path": "returned-codec/trained_artifact.yaml",
                    "artifact_entrypoint": "encoder",
                    "artifact_package_sha256": row["package_sha256"],
                },
            )
            self.assertEqual(
                bindings["decoder"]["params"],
                {
                    "runtime": "learned_artifact",
                    "artifact_manifest_path": "returned-codec/trained_artifact.yaml",
                    "artifact_entrypoint": "decoder",
                    "artifact_package_sha256": row["package_sha256"],
                },
            )

            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            payload["compatible_operations"][0]["params"] = {
                "runtime": "external_callable",
                "path": "/tmp/unverified.py",
                "callable": "run",
            }
            manifest_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            invalid = inspect_trained_artifact(
                manifest_path,
                project_root=Path(tmp),
                registry=build_registry(),
            )
            self.assertFalse(invalid["valid"])
            self.assertIn(
                "binding parameter runtime conflicts",
                "; ".join(invalid["issues"]),
            )
            self.assertEqual(
                invalid["compatible_operations"][0]["params"]["runtime"],
                "learned_artifact",
            )
            with self.assertRaisesRegex(
                TrainedArtifactError,
                "binding parameter runtime conflicts",
            ):
                import_external_trained_artifact_package(
                    Path(tmp) / "project",
                    package,
                )

    def test_operation_abi_validates_entrypoint_component_role_and_required_inputs(self):
        cases = {
            "entrypoint": lambda payload: (
                payload["runtime"]["entrypoints"][0].update({"id": "custom_encoder"}),
                payload["compatible_operations"][0].update(
                    {"runtime_entrypoint": "custom_encoder"}
                ),
            ),
            "component": lambda payload: (
                payload["components"][0].update({"id": "custom_encoder"}),
                payload["runtime"]["entrypoints"][0].update(
                    {"component": "custom_encoder"}
                ),
            ),
            "role": lambda payload: payload["components"][0].update(
                {"role": "custom_role"}
            ),
            "required_inputs": lambda payload: payload["compatible_operations"][
                0
            ].update({"required_inputs": []}),
        }
        expected_messages = {
            "entrypoint": "does not match operation ABI entrypoint",
            "component": "does not match operation ABI component",
            "role": "does not match operation ABI role",
            "required_inputs": "required_inputs do not match",
        }
        for case, mutate in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                manifest_path = _write_paired_package(Path(tmp) / "returned-codec")
                payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                mutate(payload)
                manifest_path.write_text(
                    yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
                )
                row = inspect_trained_artifact(
                    manifest_path,
                    project_root=Path(tmp),
                    registry=build_registry(),
                )
                self.assertFalse(row["valid"])
                self.assertIn(expected_messages[case], "; ".join(row["issues"]))

    def test_unknown_runtime_abi_is_rejected_by_inspection_and_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_single_package(root / "returned-model")
            original = inspect_trained_artifact(
                manifest_path,
                project_root=root,
                registry=build_registry(),
            )
            self.assertTrue(original["ready"], original["issues"])
            payload = yaml.safe_load(
                manifest_path.read_text(encoding="utf-8")
            )
            payload["runtime"]["abi_version"] = 999
            manifest_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )

            inspected = inspect_trained_artifact(
                manifest_path,
                project_root=root,
                registry=build_registry(),
            )
            self.assertFalse(inspected["valid"])
            self.assertFalse(inspected["ready"])
            self.assertFalse(inspected["runtime"]["available"])
            self.assertIn(
                "runtime.abi_version 999 is unsupported",
                "; ".join(inspected["issues"]),
            )
            validated = validate_trained_artifact_package(
                manifest_path,
                registry=build_registry(),
            )
            self.assertFalse(validated["ready"])
            with self.assertRaisesRegex(
                TrainedArtifactRuntimeError,
                "abi_version 999 is unsupported",
            ):
                run_trained_artifact_entrypoint(
                    manifest_path,
                    "power_policy",
                    {
                        "channel_gain": np.ones((1, 2), dtype=np.float32),
                        "noise_variance": np.full(
                            (1, 1), 0.2, dtype=np.float32
                        ),
                        "average_power_budget": np.ones(
                            (1, 1), dtype=np.float32
                        ),
                    },
                    expected_package_sha256=original["package_sha256"],
                    project_root=root,
                )

    def test_duplicate_manifest_keys_are_rejected_before_artifact_identity(self):
        for suffix in (".yaml", ".json"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest_path = _write_single_package(root / "returned-model")
                payload = yaml.safe_load(
                    manifest_path.read_text(encoding="utf-8")
                )
                if suffix == ".yaml":
                    manifest_path.write_text(
                        manifest_path.read_text(encoding="utf-8")
                        + "\nruntime:\n  abi_version: 1\n",
                        encoding="utf-8",
                    )
                else:
                    json_path = manifest_path.with_suffix(".json")
                    encoded = json.dumps(payload, sort_keys=True)
                    json_path.write_text(
                        encoded[:-1]
                        + ',"runtime":{"abi_version":1}}',
                        encoding="utf-8",
                    )
                    manifest_path = json_path

                with self.assertRaisesRegex(
                    TrainedArtifactError,
                    "Duplicate (YAML mapping|JSON object) key",
                ):
                    inspect_trained_artifact(
                        manifest_path,
                        project_root=root,
                        registry=build_registry(),
                    )

    def test_generic_importer_copies_only_declared_files_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            package = Path(tmp) / "external"
            _write_single_package(package)
            (package / "unlisted.py").write_text(
                "raise RuntimeError('must never be imported')\n", encoding="utf-8"
            )

            first = import_external_trained_artifact_package(
                root,
                package,
                operation="model.symbol_power_allocator",
                registry=build_registry(),
            )
            second = import_external_trained_artifact_package(
                root,
                package,
                operation="model.symbol_power_allocator",
                registry=build_registry(),
            )

            self.assertTrue(first["valid"], first["issues"])
            self.assertEqual(first["manifest_path"], second["manifest_path"])
            self.assertEqual(first["source_class"], "imported")
            self.assertEqual(
                first["source"]["imported_from_runtime_identity_sha256"],
                second["source"]["imported_from_runtime_identity_sha256"],
            )
            self.assertEqual(
                first["compatible_operations"][0]["params"][
                    "artifact_manifest_path"
                ],
                first["manifest_path"],
            )
            managed = root / first["manifest_path"]
            self.assertTrue((managed.parent / "policy.onnx").is_file())
            self.assertTrue((managed.parent / "slot_contract.yaml").is_file())
            self.assertFalse((managed.parent / "unlisted.py").exists())

    def test_directory_import_uses_nested_manifest_as_package_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "selected-directory"
            _write_single_package(source / "nested-package")

            row = import_external_trained_artifact_package(
                base / "project",
                source,
                registry=build_registry(),
            )

            self.assertTrue(row["valid"], row["issues"])
            managed_manifest = base / "project" / row["manifest_path"]
            self.assertTrue((managed_manifest.parent / "policy.onnx").is_file())
            self.assertTrue((managed_manifest.parent / "slot_contract.yaml").is_file())

    def test_managed_store_rejects_symlink_redirection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "external"
            _write_single_package(source)
            project = base / "project"
            outside = base / "outside"
            (project / ".noema").mkdir(parents=True)
            outside.mkdir()
            try:
                (project / ".noema" / "trained_artifacts").symlink_to(
                    outside,
                    target_is_directory=True,
                )
            except OSError as exc:  # pragma: no cover - platform permission gate
                self.skipTest("directory symlinks are unavailable: %s" % exc)

            with self.assertRaisesRegex(TrainedArtifactError, "symbolic links"):
                import_external_trained_artifact_package(
                    project,
                    source,
                    registry=build_registry(),
                )
            self.assertEqual(list(outside.iterdir()), [])

    def test_managed_identity_includes_tensor_abi_and_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            package_a = Path(tmp) / "package-a"
            package_b = Path(tmp) / "package-b"
            _write_single_package(package_a)
            manifest_b = _write_single_package(package_b)
            payload = yaml.safe_load(manifest_b.read_text(encoding="utf-8"))
            payload["runtime"]["entrypoints"][0]["inputs"][0]["semantic"] = (
                "normalized per-subcarrier channel power gain"
            )
            manifest_b.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )

            first = import_external_trained_artifact_package(root, package_a)
            second = import_external_trained_artifact_package(root, package_b)

            self.assertNotEqual(first["manifest_path"], second["manifest_path"])

    def test_non_json_and_cyclic_identity_fields_raise_controlled_error(self):
        mutations = {
            "timestamp": lambda payload: payload["application"].update(
                {"created_at": datetime(2026, 7, 14, tzinfo=timezone.utc)}
            ),
            "cycle": lambda payload: payload["application"].update(
                {"cycle": payload["application"]}
            ),
        }
        for case, mutate in mutations.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                manifest_path = _write_single_package(Path(tmp) / "returned-model")
                payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                mutate(payload)
                manifest_path.write_text(
                    yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
                )

                with self.assertRaisesRegex(
                    TrainedArtifactError,
                    "finite, acyclic, JSON-compatible",
                ):
                    inspect_trained_artifact(
                        manifest_path,
                        project_root=Path(tmp),
                        registry=build_registry(),
                    )

    def test_zip_path_traversal_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "malicious.noema-artifact"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.txt", b"escape")
                archive.writestr("trained_artifact.yaml", b"schema_version: 2\n")

            with self.assertRaisesRegex(TrainedArtifactError, "unsafe member path"):
                validate_trained_artifact_package(archive_path)
            self.assertFalse((Path(tmp) / "outside.txt").exists())

    def test_zip_duplicate_members_are_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "duplicate.noema-artifact"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("trained_artifact.yaml", b"schema_version: 2\n")
                    archive.writestr("trained_artifact.yaml", b"schema_version: 1\n")

            with self.assertRaisesRegex(TrainedArtifactError, "duplicate members"):
                validate_trained_artifact_package(archive_path)

    def test_zip_symbolic_link_member_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "symlink.noema-artifact"
            symlink = zipfile.ZipInfo("model-link.onnx")
            symlink.create_system = 3
            symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(symlink, b"outside-model.onnx")
                archive.writestr("trained_artifact.yaml", b"schema_version: 2\n")

            with self.assertRaisesRegex(TrainedArtifactError, "symbolic links"):
                validate_trained_artifact_package(archive_path)

    def test_zip_expanded_size_budget_is_enforced_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "oversized.noema-artifact"
            with zipfile.ZipFile(
                archive_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                archive.writestr("trained_artifact.yaml", b"schema_version: 2\n")
                archive.writestr("model.onnx", b"0123456789")

            with patch(
                "noema_lab.core.trained_artifacts."
                "MAX_IMPORTED_ARTIFACT_UNCOMPRESSED_BYTES",
                16,
            ):
                with self.assertRaisesRegex(TrainedArtifactError, "expands to"):
                    validate_trained_artifact_package(archive_path)

    def test_runtime_rejects_unknown_inputs_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "returned-model"
            manifest_path = _write_single_package(package)
            inspected = inspect_trained_artifact(
                manifest_path,
                project_root=Path(tmp),
            )

            with self.assertRaisesRegex(TrainedArtifactRuntimeError, "unexpected input"):
                run_trained_artifact_entrypoint(
                    manifest_path,
                    "power_policy",
                    {
                        "channel_gain": np.ones((1, 2), dtype=np.float32),
                        "noise_variance": np.full((1, 1), 0.2, dtype=np.float32),
                        "average_power_budget": np.ones((1, 1), dtype=np.float32),
                        "python_callback": np.ones((1,), dtype=np.float32),
                    },
                    expected_package_sha256=inspected["package_sha256"],
                    project_root=Path(tmp),
                )

    def test_runtime_rejects_package_substitution_at_the_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "returned-model"
            manifest_path = _write_single_package(package)
            inspected = inspect_trained_artifact(
                manifest_path,
                project_root=Path(tmp),
            )
            expected = inspected["package_sha256"]

            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            manifest["id"] = "substituted-package-at-same-path"
            manifest_path.write_text(
                yaml.safe_dump(manifest, sort_keys=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                TrainedArtifactRuntimeError,
                "package identity mismatch",
            ):
                run_trained_artifact_entrypoint(
                    manifest_path,
                    "power_policy",
                    {
                        "channel_gain": np.ones((1, 2), dtype=np.float32),
                        "noise_variance": np.full((1, 1), 0.2, dtype=np.float32),
                        "average_power_budget": np.ones((1, 1), dtype=np.float32),
                    },
                    expected_package_sha256=expected,
                    project_root=Path(tmp),
                )


def _csi_bindings(feedback_dimension: int) -> list[dict]:
    return [
        {
            "operation": "model.csi_feedback_encoder",
            "runtime_entrypoint": "encoder",
            "binding_group": "csi_feedback_codec",
            "role": "encoder",
            "required_inputs": ["csi"],
            "params": {"feedback_dimension": feedback_dimension},
        },
        {
            "operation": "model.csi_feedback_decoder",
            "runtime_entrypoint": "decoder",
            "binding_group": "csi_feedback_codec",
            "role": "decoder",
            "required_inputs": ["received_code"],
            "params": {"feedback_dimension": feedback_dimension},
        },
    ]


def _csi_recipe() -> dict:
    return {
        "steps": [
            {
                "id": "channel_state",
                "op": "wireless.miso_ofdm_csi",
                "params": {"tx_antennas": 8, "ofdm_fft_size": 32},
            },
            {
                "id": "feedback_encoder",
                "op": "model.csi_feedback_encoder",
                "params": {"feedback_dimension": 32},
            },
            {
                "id": "feedback_link",
                "op": "channel.csi_feedback_link",
                "params": {"mode": "uniform_quantized", "bits_per_latent": 4},
            },
            {
                "id": "feedback_decoder",
                "op": "model.csi_feedback_decoder",
                "params": {"feedback_dimension": 32},
            },
        ]
    }


def _write_single_package(package: Path) -> Path:
    package.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "kind": "noema.trainable_slot_contract@1",
        "id": "noema.slot.power-policy",
        "version": 1,
        "slot": {
            "inputs": ["channel_gain"],
            "outputs": ["allocation_scores"],
            "owner": "external trainer",
        },
    }
    contract_path = package / "slot_contract.yaml"
    contract_path.write_text(
        yaml.safe_dump(contract, sort_keys=False), encoding="utf-8"
    )
    model_path = package / "policy.onnx"
    _write_power_policy_onnx(model_path)
    manifest = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": "external.power-policy.example",
        "name": "External power policy",
        "contract": {
            "id": "noema.slot.power-policy",
            "version": 1,
            "path": "slot_contract.yaml",
            "sha256": canonical_json_sha256(contract),
            "file_sha256": _sha256(contract_path),
        },
        "components": [
            {
                "id": "policy",
                "role": "power_policy",
                "path": "policy.onnx",
                "format": "onnx",
                "sha256": _sha256(model_path),
            }
        ],
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 1,
            "entrypoints": [
                {
                    "id": "power_policy",
                    "component": "policy",
                    "inputs": [
                        {
                            "name": "channel_gain",
                            "dtype": "float32",
                            "shape": ["batch", 2],
                            "semantic": "per-subcarrier channel power gain",
                            "layout": "batch,subcarrier",
                        },
                        {
                            "name": "noise_variance",
                            "dtype": "float32",
                            "shape": ["batch", 1],
                            "semantic": "noise power",
                            "layout": "batch,scalar",
                        },
                        {
                            "name": "average_power_budget",
                            "dtype": "float32",
                            "shape": ["batch", 1],
                            "semantic": "average transmit-power budget",
                            "layout": "batch,scalar",
                        },
                    ],
                    "outputs": [
                        {
                            "name": "allocation_scores",
                            "dtype": "float32",
                            "shape": ["batch", 2],
                            "semantic": "unconstrained power-allocation scores",
                            "layout": "batch,subcarrier",
                        }
                    ],
                }
            ],
        },
        "application": {"mode": "single_binding"},
        "compatible_operations": [
            {
                "operation": "model.symbol_power_allocator",
                "runtime_entrypoint": "power_policy",
                "required_inputs": ["channel_state"],
                "params": {
                    "policy": "learned_artifact",
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": "power_policy",
                },
            }
        ],
        "training": {
            "architecture": "user-defined",
            "loss": "a completely external objective",
            "framework": "not used for compatibility",
        },
        "evaluation": {"reported_metric": 12.5},
    }
    manifest_path = package / "trained_artifact.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return manifest_path


def _write_paired_package(package: Path) -> Path:
    package.mkdir(parents=True, exist_ok=True)
    contract_path = package / "slot_contract.yaml"
    contract = {
        "schema_version": 1,
        "kind": "noema.trainable_slot_contract@1",
        "id": "noema.slot.deepjscc-pair",
        "version": 1,
    }
    contract_path.write_text(
        yaml.safe_dump(contract, sort_keys=False),
        encoding="utf-8",
    )
    encoder_path = package / "encoder.onnx"
    decoder_path = package / "decoder.onnx"
    image_shape = ["batch", 3, "height", "width"]
    symbol_shape = ["batch", "real_imag_channel", "symbol_height", "symbol_width"]
    _write_identity_onnx(
        encoder_path,
        input_name="images",
        output_name="symbols_ri",
        input_shape=image_shape,
        output_shape=symbol_shape,
    )
    _write_identity_onnx(
        decoder_path,
        input_name="symbols_ri",
        output_name="reconstruction",
        input_shape=symbol_shape,
        output_shape=image_shape,
    )

    def tensor(name: str, semantic: str, shape: list) -> dict:
        return {
            "name": name,
            "dtype": "float32",
            "shape": shape,
            "semantic": semantic,
        }

    manifest = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": "external.deepjscc-pair.example",
        "name": "External paired codec",
        "contract": {
            "id": "noema.slot.deepjscc-pair",
            "version": 1,
            "path": "slot_contract.yaml",
            "sha256": canonical_json_sha256(contract),
            "file_sha256": _sha256(contract_path),
        },
        "components": [
            {
                "id": "encoder",
                "role": "encoder",
                "path": "encoder.onnx",
                "format": "onnx",
                "sha256": _sha256(encoder_path),
            },
            {
                "id": "decoder",
                "role": "decoder",
                "path": "decoder.onnx",
                "format": "onnx",
                "sha256": _sha256(decoder_path),
            },
        ],
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 1,
            "entrypoints": [
                {
                    "id": "encoder",
                    "component": "encoder",
                    "inputs": [tensor("images", "normalized image tensor", image_shape)],
                    "outputs": [tensor("symbols_ri", "real/imaginary channel symbols", symbol_shape)],
                },
                {
                    "id": "decoder",
                    "component": "decoder",
                    "inputs": [tensor("symbols_ri", "real/imaginary received symbols", symbol_shape)],
                    "outputs": [tensor("reconstruction", "normalized image reconstruction", image_shape)],
                },
            ],
        },
        "application": {"mode": "all_group_bindings"},
        "compatible_operations": [
            {
                "operation": "model.deepjscc_external_encode",
                "runtime_entrypoint": "encoder",
                "binding_group": "deepjscc_sender_receiver",
                "role": "encoder",
                "preferred_step_id": "sender",
                "required_inputs": ["images"],
                "params": {},
            },
            {
                "operation": "model.deepjscc_external_decode",
                "runtime_entrypoint": "decoder",
                "binding_group": "deepjscc_sender_receiver",
                "role": "decoder",
                "preferred_step_id": "receiver",
                "required_inputs": ["symbols"],
                "params": {},
            },
        ],
    }
    manifest_path = package / "trained_artifact.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return manifest_path


def _write_phase_tracking_receiver_package(
    package: Path,
    *,
    legacy: bool = False,
) -> Path:
    package.mkdir(parents=True, exist_ok=True)
    contract_path = package / "training_contract.yaml"
    contract = {
        "schema_version": 1,
        "kind": "noema.trainable_slot_contract@1",
        "id": "qpsk_pilot_phase_tracking.test-contract",
        "version": 1,
    }
    contract_path.write_text(
        yaml.safe_dump(contract, sort_keys=False),
        encoding="utf-8",
    )
    component_path = package / "artifacts" / "phase_tracking_receiver.onnx"
    component_path.parent.mkdir(parents=True)
    _write_phase_tracking_receiver_onnx(component_path, legacy=legacy)

    def tensor(name: str, shape: list, semantic: str) -> dict:
        return {
            "name": name,
            "dtype": "float32",
            "shape": shape,
            "semantic": semantic,
        }

    input_name = "receiver_features" if legacy else "receiver_features_v3"
    input_width = 5 if legacy else 11
    outputs = (
        [
            tensor(
                "frame_bit_llr",
                ["packet", "frame_symbol", 2],
                "QPSK bit logits",
            )
        ]
        if legacy
        else [
            tensor(
                "residual_phase_rad",
                ["packet", "frame_symbol"],
                "full-circle residual phase relative to pilot smoothing",
            )
        ]
    )
    entrypoint = {
        "id": "phase_tracking_receiver",
        "component": "receiver",
        "inputs": [
            tensor(
                input_name,
                ["packet", "frame_symbol", input_width],
                (
                    "legacy received IQ and pilot context"
                    if legacy
                    else "smoother-corrected IQ and public receiver context v3"
                ),
            )
        ],
        "outputs": outputs,
    }
    manifest = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": "qpsk_pilot_phase_tracking.demodulator.phase_tracking_receiver",
        "name": "Learned phase-tracking receiver",
        "label": "Learned phase tracker · test",
        "contract": {
            "id": contract["id"],
            "version": contract["version"],
            "path": contract_path.name,
            "sha256": canonical_json_sha256(contract),
            "file_sha256": _sha256(contract_path),
        },
        "components": [
            {
                "id": "receiver",
                "role": "phase_tracking_receiver",
                "path": "artifacts/phase_tracking_receiver.onnx",
                "format": "onnx",
                "sha256": _sha256(component_path),
            }
        ],
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 1 if legacy else 2,
            "entrypoints": [entrypoint],
        },
        "application": {"mode": "single_binding"},
        "compatible_operations": [
            {
                "operation": "demodulation.phase_tracking_receiver_adapter",
                "runtime_entrypoint": "phase_tracking_receiver",
                "preferred_step_id": "demodulator",
                "required_inputs": ["rx_symbols", "pilot_context"],
                "params": {
                    "mode": "learned_artifact",
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": "phase_tracking_receiver",
                },
            }
        ],
        "source": {
            "project_manifest": "project_manifest.yaml",
            "training_template": "neural_receiver.phase_tracking_qpsk",
        },
        "training": {
            "framework": "torch",
            "architecture": "test temporal receiver",
        },
    }
    manifest_path = package / "trained_artifact.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path


def _write_phase_tracking_receiver_onnx(
    path: Path,
    *,
    legacy: bool = False,
) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except Exception as exc:  # pragma: no cover - optional dependency gate
        raise unittest.SkipTest("onnx is unavailable: %s" % exc) from exc
    input_name = "receiver_features" if legacy else "receiver_features_v3"
    input_width = 5 if legacy else 11
    nodes = (
        [
            helper.make_node(
                "Gather",
                [input_name, "bit_indices"],
                ["frame_bit_llr"],
                axis=2,
            )
        ]
        if legacy
        else [
                helper.make_node(
                    "Gather",
                    [input_name, "phase_index"],
                    ["phase_column"],
                    axis=2,
                ),
                helper.make_node(
                    "Squeeze",
                    ["phase_column", "squeeze_axis"],
                    ["phase_column_2d"],
                ),
                helper.make_node(
                    "Mul",
                    ["phase_column_2d", "zero"],
                    ["residual_phase_rad"],
                ),
            ]
    )
    outputs = (
        [
            helper.make_tensor_value_info(
                "frame_bit_llr",
                TensorProto.FLOAT,
                ["packet", "frame_symbol", 2],
            )
        ]
        if legacy
        else [
            helper.make_tensor_value_info(
                "residual_phase_rad",
                TensorProto.FLOAT,
                ["packet", "frame_symbol"],
            )
        ]
    )
    initializers = (
        [
            helper.make_tensor(
                "bit_indices",
                TensorProto.INT64,
                [2],
                [0, 1],
            )
        ]
        if legacy
        else [
            helper.make_tensor("phase_index", TensorProto.INT64, [1], [0]),
            helper.make_tensor("squeeze_axis", TensorProto.INT64, [1], [2]),
            helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "phase_tracking_receiver",
        [
            helper.make_tensor_value_info(
                input_name,
                TensorProto.FLOAT,
                ["packet", "frame_symbol", input_width],
            )
        ],
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.save_model(model, str(path))


def _write_identity_onnx(
    path: Path,
    *,
    input_name: str,
    output_name: str,
    input_shape=None,
    output_shape=None,
) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except Exception as exc:  # pragma: no cover - optional dependency gate
        raise unittest.SkipTest("onnx is unavailable: %s" % exc) from exc
    graph = helper.make_graph(
        [helper.make_node("Identity", [input_name], [output_name])],
        "returned_artifact_identity",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape or ["batch", 2])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape or input_shape or ["batch", 2])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.save_model(model, str(path))


def _write_power_policy_onnx(path: Path) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except Exception as exc:  # pragma: no cover - optional dependency gate
        raise unittest.SkipTest("onnx is unavailable: %s" % exc) from exc
    graph = helper.make_graph(
        [helper.make_node("Identity", ["channel_gain"], ["allocation_scores"])],
        "returned_power_policy",
        [
            helper.make_tensor_value_info("channel_gain", TensorProto.FLOAT, ["batch", 2]),
            helper.make_tensor_value_info("noise_variance", TensorProto.FLOAT, ["batch", 1]),
            helper.make_tensor_value_info("average_power_budget", TensorProto.FLOAT, ["batch", 1]),
        ],
        [helper.make_tensor_value_info("allocation_scores", TensorProto.FLOAT, ["batch", 2])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.save_model(model, str(path))


def _attach_publication_selection_history(
    manifest_path: Path,
    *,
    publication_test_accessed: bool = False,
    selected_runtime_component_set_sha256: str = "",
) -> None:
    package = manifest_path.parent
    initial = inspect_trained_artifact(manifest_path, project_root=package.parent)
    runtime_sha = str(
        initial["publication_readiness"]["runtime_component_set_sha256"]
    )
    provenance = package / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    search_program = provenance / "search.py"
    search_program.write_text(
        "# Frozen deterministic validation-only search program.\n",
        encoding="utf-8",
    )
    candidate_a = provenance / "candidate-a.yaml"
    candidate_a.write_text(
        yaml.safe_dump({"architecture": "small", "seed": 7}, sort_keys=True),
        encoding="utf-8",
    )
    candidate_b = provenance / "candidate-b.yaml"
    candidate_b.write_text(
        yaml.safe_dump({"architecture": "large", "seed": 11}, sort_keys=True),
        encoding="utf-8",
    )
    population_sha = "a" * 64
    history = {
        "schema_version": 1,
        "kind": "noema.model_selection_history",
        "all_candidates_disclosed": True,
        "publication_test_accessed": publication_test_accessed,
        "selection_population_role": "adaptation_validation",
        "selection_population": {
            "dataset_id": "dataset:validation",
            "version": "1",
            "split": "validation",
            "role": "adaptation_validation",
            "sha256": population_sha,
        },
        "objective": "mean validation reconstruction loss",
        "direction": "minimize",
        "selection_rule": {
            "method": "objective_extremum",
            "tie_breaker": "lexicographic_candidate_id",
            "tolerance": 0.0,
        },
        "selected_candidate_id": "candidate-a",
        "search_program": {
            "path": "provenance/search.py",
            "sha256": _sha256(search_program),
            "command": ["python", "provenance/search.py", "--frozen"],
        },
        "candidates": [
            {
                "id": "candidate-a",
                "configuration_path": "provenance/candidate-a.yaml",
                "configuration_identity_sha256": _sha256(candidate_a),
                "selection_population_sha256": population_sha,
                "publication_test_accessed": False,
                "status": "completed",
                "runtime_component_set_sha256": (
                    selected_runtime_component_set_sha256 or runtime_sha
                ),
                "objective_value": 0.125,
            },
            {
                "id": "candidate-b",
                "configuration_path": "provenance/candidate-b.yaml",
                "configuration_identity_sha256": _sha256(candidate_b),
                "selection_population_sha256": population_sha,
                "publication_test_accessed": False,
                "status": "resource_rejected",
                "failure_reason": "declared memory budget exceeded",
            },
        ],
    }
    history_path = provenance / "model_selection_history.yaml"
    history_path.write_text(
        yaml.safe_dump(history, sort_keys=False),
        encoding="utf-8",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["support_files"] = [
        {
            "role": "model_selection_history",
            "path": "provenance/model_selection_history.yaml",
            "sha256": _sha256(history_path),
        },
        {
            "role": "model_selection_search_program",
            "path": "provenance/search.py",
            "sha256": _sha256(search_program),
        },
        {
            "role": "model_selection_configuration",
            "path": "provenance/candidate-a.yaml",
            "sha256": _sha256(candidate_a),
        },
        {
            "role": "model_selection_configuration",
            "path": "provenance/candidate-b.yaml",
            "sha256": _sha256(candidate_b),
        },
    ]
    manifest["publication"] = {
        "selection_history": {
            "path": "provenance/model_selection_history.yaml",
            "sha256": _sha256(history_path),
        }
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
