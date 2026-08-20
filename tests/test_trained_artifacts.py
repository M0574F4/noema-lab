from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import yaml

from noema_lab.core.trained_artifacts import (
    TrainedArtifactError,
    _extract_safe_artifact_archive,
    discover_trained_artifacts,
    import_external_trained_artifact,
    inspect_trained_artifact,
)
from noema_lab.ops import build_registry


class TrainedArtifactTests(unittest.TestCase):
    def test_artifact_zip_rejects_portable_name_collisions_and_special_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collision = root / "collision.zip"
            with zipfile.ZipFile(collision, "w") as archive:
                archive.writestr("package/Model.bin", b"first")
                archive.writestr("package/model.bin", b"second")
            with self.assertRaisesRegex(TrainedArtifactError, "colliding member"):
                _extract_safe_artifact_archive(collision, root / "collision")

            unicode_collision = root / "unicode-collision.zip"
            with zipfile.ZipFile(unicode_collision, "w") as archive:
                archive.writestr(
                    "package/caf\N{LATIN SMALL LETTER E WITH ACUTE}.bin",
                    b"first",
                )
                archive.writestr(
                    "package/cafe\N{COMBINING ACUTE ACCENT}.bin",
                    b"second",
                )
            with self.assertRaisesRegex(TrainedArtifactError, "colliding member"):
                _extract_safe_artifact_archive(
                    unicode_collision,
                    root / "unicode-collision",
                )

            special = root / "special.zip"
            fifo = zipfile.ZipInfo("package/fifo")
            fifo.create_system = 3
            fifo.external_attr = 0o010644 << 16
            with zipfile.ZipFile(special, "w") as archive:
                archive.writestr(fifo, b"not a regular file")
            with self.assertRaisesRegex(TrainedArtifactError, "special-file"):
                _extract_safe_artifact_archive(special, root / "special")

            windows_ads = root / "windows-ads.zip"
            with zipfile.ZipFile(windows_ads, "w") as archive:
                archive.writestr("package/model.bin:metadata", b"unsafe")
            with self.assertRaisesRegex(TrainedArtifactError, "unsafe member path"):
                _extract_safe_artifact_archive(windows_ads, root / "windows-ads")

    def test_discovery_includes_returned_artifacts_from_workbench_training_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / ".noema" / "training_exports" / "ofdm_allocator"
            checkpoint = project / "checkpoints" / "allocator.npz"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"workbench-trained-allocator")
            expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (project / "trained_artifact.yaml").write_text(
                yaml.safe_dump(_manifest(expected_sha), sort_keys=False),
                encoding="utf-8",
            )

            rows = discover_trained_artifacts(
                root,
                registry=build_registry(),
                operation="model.symbol_power_allocator",
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["manifest_path"],
                ".noema/training_exports/ofdm_allocator/trained_artifact.yaml",
            )
            self.assertTrue(rows[0]["ready"])

    def test_discovery_returns_directly_applicable_hash_verified_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "differentiable_exports" / "allocator"
            checkpoint = project / "checkpoints" / "allocator.npz"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"safe-trained-allocator")
            expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            manifest = _manifest(expected_sha)
            manifest_path = project / "trained_artifact.yaml"
            manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

            rows = discover_trained_artifacts(
                root,
                registry=build_registry(),
                operation="model.symbol_power_allocator",
            )
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertTrue(row["ready"])
            self.assertEqual(row["id"], "example.tx_power.csi_allocator")
            self.assertEqual(row["artifact"]["actual_sha256"], expected_sha)
            binding = row["compatible_operations"][0]
            self.assertEqual(binding["required_inputs"], ["channel_state"])
            self.assertEqual(
                binding["params"]["checkpoint_path"],
                "differentiable_exports/allocator/checkpoints/allocator.npz",
            )
            self.assertNotIn("target_power", binding["params"])

            self.assertEqual(
                discover_trained_artifacts(root, registry=build_registry(), operation="model.jpeg_encode"),
                [],
            )

    def test_binding_cannot_redirect_away_from_verified_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "differentiable_exports" / "allocator"
            checkpoint = project / "checkpoints" / "allocator.npz"
            other = project / "checkpoints" / "other.npz"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"verified")
            other.write_bytes(b"not-the-verified-artifact")
            expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            manifest = _manifest(expected_sha)
            params = manifest["compatible_operations"][0]["params"]
            params["checkpoint_path"] = "checkpoints/other.npz"
            params["checkpoint_sha256"] = hashlib.sha256(other.read_bytes()).hexdigest()
            manifest_path = project / "trained_artifact.yaml"
            manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

            row = inspect_trained_artifact(
                manifest_path,
                project_root=root,
                registry=build_registry(),
            )
            self.assertFalse(row["ready"])
            issues = "; ".join(row["issues"])
            self.assertIn("does not reference the hash-verified artifact.path", issues)
            self.assertIn("checkpoint_sha256 does not match artifact.sha256", issues)

    def test_paired_application_metadata_and_bindings_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "differentiable_exports" / "deepjscc"
            checkpoint = project / "checkpoints" / "deepjscc.npz"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"safe-paired-deepjscc-weights")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            manifest_path = project / "trained_artifact.yaml"
            manifest_path.write_text(
                yaml.safe_dump(_paired_manifest(checkpoint_sha), sort_keys=False),
                encoding="utf-8",
            )

            row = inspect_trained_artifact(
                manifest_path,
                project_root=root,
                registry=build_registry(),
            )

            self.assertTrue(row["ready"], row["issues"])
            self.assertEqual(row["application"], {"mode": "all_group_bindings"})
            self.assertEqual(
                [binding["binding_group"] for binding in row["compatible_operations"]],
                ["deepjscc_sender_receiver", "deepjscc_sender_receiver"],
            )
            self.assertEqual(
                [binding["role"] for binding in row["compatible_operations"]],
                ["encoder", "decoder"],
            )
            self.assertEqual(
                [binding["preferred_step_id"] for binding in row["compatible_operations"]],
                ["sender", "receiver"],
            )
            for binding in row["compatible_operations"]:
                self.assertEqual(
                    binding["params"]["checkpoint_path"],
                    "differentiable_exports/deepjscc/checkpoints/deepjscc.npz",
                )

    def test_all_group_bindings_rejects_duplicate_preferred_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "differentiable_exports" / "deepjscc"
            checkpoint = project / "checkpoints" / "deepjscc.npz"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"safe-paired-deepjscc-weights")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            manifest = _paired_manifest(checkpoint_sha)
            manifest["compatible_operations"][1]["preferred_step_id"] = "sender"
            manifest_path = project / "trained_artifact.yaml"
            manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

            row = inspect_trained_artifact(
                manifest_path,
                project_root=root,
                registry=build_registry(),
            )

            self.assertFalse(row["ready"])
            self.assertIn("duplicate preferred_step_id", "; ".join(row["issues"]))

    def test_declared_support_file_is_hash_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "differentiable_exports" / "allocator"
            checkpoint = project / "checkpoints" / "allocator.npz"
            adapter = project / "adapter.py"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"safe-trained-allocator")
            adapter.write_text("def load(): return 1\n", encoding="utf-8")
            manifest = _manifest(hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            manifest["support_files"] = [
                {
                    "path": "adapter.py",
                    "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                    "role": "runtime_adapter",
                }
            ]
            manifest_path = project / "trained_artifact.yaml"
            manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

            ready = inspect_trained_artifact(manifest_path, project_root=root, registry=build_registry())
            self.assertTrue(ready["ready"])
            self.assertEqual(ready["support_files"][0]["path"], "differentiable_exports/allocator/adapter.py")

            adapter.write_text("def load(): return 2\n", encoding="utf-8")
            tampered = inspect_trained_artifact(manifest_path, project_root=root, registry=build_registry())
            self.assertFalse(tampered["ready"])
            self.assertIn("support_files[0]: SHA-256 does not match", "; ".join(tampered["issues"]))

    def test_external_checkpoint_import_validates_and_creates_managed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upload = root / "outside-training-output.npz"
            expected_sha = _write_test_deepset_checkpoint(upload)

            first = import_external_trained_artifact(
                root,
                upload,
                operation="model.symbol_power_allocator",
                original_filename="allocator seed 67.npz",
                label="My learned allocator",
                registry=build_registry(),
            )
            self.assertTrue(first["ready"])
            self.assertEqual(first["label"], "My learned allocator")
            self.assertEqual(first["artifact"]["sha256"], expected_sha)
            self.assertTrue(first["artifact"]["path"].startswith(".noema/trained_artifacts/imported/"))
            self.assertNotIn(str(upload), first["artifact"]["path"])
            binding = first["compatible_operations"][0]
            self.assertEqual(binding["params"]["policy"], "learned_checkpoint")
            self.assertEqual(binding["params"]["checkpoint_sha256"], expected_sha)
            self.assertNotIn("target_power", binding["params"])

            second = import_external_trained_artifact(
                root,
                upload,
                operation="model.symbol_power_allocator",
                original_filename="allocator seed 67.npz",
                label="My learned allocator",
                registry=build_registry(),
            )
            self.assertEqual(second["id"], first["id"])
            discovered = discover_trained_artifacts(root, registry=build_registry())
            self.assertEqual([row["id"] for row in discovered], [first["id"]])

    def test_external_checkpoint_import_rejects_npz_member_bomb_before_numpy_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upload = root / "too-many-members.npz"
            with zipfile.ZipFile(upload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for index in range(33):
                    archive.writestr("array_%02d.npy" % index, b"small")

            with self.assertRaisesRegex(TrainedArtifactError, "between 1 and 32 members"):
                import_external_trained_artifact(
                    root,
                    upload,
                    operation="model.symbol_power_allocator",
                    original_filename=upload.name,
                    registry=build_registry(),
                )
            self.assertEqual(discover_trained_artifacts(root, registry=build_registry()), [])


def _manifest(checkpoint_sha: str) -> dict:
    return {
        "schema_version": 1,
        "kind": "noema.trained_block_artifact",
        "id": "example.tx_power.csi_allocator",
        "name": "Example learned CSI allocator",
        "label": "Learned example allocator",
        "artifact": {
            "path": "checkpoints/allocator.npz",
            "sha256": checkpoint_sha,
            "format": "noema_csi_power_deepset_npz_v1",
        },
        "compatible_operations": [
            {
                "operation": "model.symbol_power_allocator",
                "label": "Learned example allocator",
                "description": "Frozen per-subcarrier policy",
                "required_inputs": ["channel_state"],
                "params": {
                    "policy": "learned_checkpoint",
                    "granularity": "per_subcarrier",
                    "budget_mode": "fixed_average",
                    "checkpoint_path": "checkpoints/allocator.npz",
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_format": "noema_csi_power_deepset_npz_v1",
                    "checkpoint_strict": True,
                },
            }
        ],
        "source": {"training_template": "example.template"},
    }


def _paired_manifest(checkpoint_sha: str) -> dict:
    checkpoint = {
        "runtime": "learned_checkpoint",
        "checkpoint_path": "checkpoints/deepjscc.npz",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_format": "noema_deepjscc_reference_cnn_npz_v1",
        "checkpoint_strict": True,
        "symbol_channels": 16,
    }
    return {
        "schema_version": 1,
        "kind": "noema.trained_block_artifact",
        "id": "example.deepjscc.paired",
        "name": "Example paired DeepJSCC",
        "application": {"mode": "all_group_bindings"},
        "artifact": {
            "path": "checkpoints/deepjscc.npz",
            "sha256": checkpoint_sha,
            "format": "noema_deepjscc_reference_cnn_npz_v1",
        },
        "compatible_operations": [
            {
                "operation": "model.deepjscc_external_encode",
                "binding_group": "deepjscc_sender_receiver",
                "role": "encoder",
                "preferred_step_id": "sender",
                "required_inputs": ["images"],
                "params": dict(checkpoint),
            },
            {
                "operation": "model.deepjscc_external_decode",
                "binding_group": "deepjscc_sender_receiver",
                "role": "decoder",
                "preferred_step_id": "receiver",
                "required_inputs": ["symbols"],
                "params": dict(checkpoint),
            },
        ],
    }


def _write_test_deepset_checkpoint(path: Path) -> str:
    metadata = {
        "schema_version": 1,
        "kind": "noema.csi_power_allocator_checkpoint",
        "format": "noema_csi_power_deepset_npz_v1",
        "input_contract": "log(max(gain*average_power/noise_variance,eps))",
        "output_contract": "euclidean_simplex_projection_times_fixed_sum_power",
        "activation": "relu",
        "hidden_dim": 2,
        "training": {
            "objective": "maximize_parallel_channel_shannon_spectral_efficiency",
            "supervised_labels_used": False,
            "water_filling_used_during_training": False,
        },
    }
    np.savez_compressed(
        path,
        feature_mean=np.asarray([0.0], dtype=np.float32),
        feature_scale=np.asarray([1.0], dtype=np.float32),
        phi_weight_0=np.asarray([[1.0], [-1.0]], dtype=np.float32),
        phi_bias_0=np.zeros(2, dtype=np.float32),
        phi_weight_1=np.eye(2, dtype=np.float32),
        phi_bias_1=np.zeros(2, dtype=np.float32),
        rho_weight_0=np.asarray(
            [[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, -1.0]],
            dtype=np.float32,
        ),
        rho_bias_0=np.zeros(2, dtype=np.float32),
        rho_weight_out=np.asarray([[1.0, -1.0]], dtype=np.float32),
        rho_bias_out=np.zeros(1, dtype=np.float32),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
