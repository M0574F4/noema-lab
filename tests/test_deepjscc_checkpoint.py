from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops.models.deepjscc_checkpoint import (
    CHECKPOINT_ARCHITECTURE,
    CHECKPOINT_FORMAT,
    CHECKPOINT_KIND,
    decode_deepjscc_symbols,
    encode_deepjscc_images,
    load_deepjscc_reference_checkpoint,
)
from noema_lab.ops.models.external import (
    DeepJsccExternalDecodeOperation,
    DeepJsccExternalEncodeOperation,
)
from noema_lab.ui.server import _learned_checkpoint_readiness


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class DeepJsccCheckpointTests(unittest.TestCase):
    def test_loader_enforces_hash_exact_keys_dtype_and_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_path = root / "valid.npz"
            valid_sha = _write_checkpoint(valid_path)
            checkpoint = load_deepjscc_reference_checkpoint(
                str(valid_path), valid_sha
            )
            self.assertEqual(checkpoint.symbol_channels, 2)
            self.assertEqual(checkpoint.metadata["format"], CHECKPOINT_FORMAT)

            with self.assertRaisesRegex(OperationError, "SHA-256 mismatch"):
                load_deepjscc_reference_checkpoint(str(valid_path), "0" * 64)

            extra_path = root / "extra.npz"
            extra_sha = _write_checkpoint(extra_path, extra={"untrusted": np.zeros(1)})
            with self.assertRaisesRegex(OperationError, "unexpected array"):
                load_deepjscc_reference_checkpoint(str(extra_path), extra_sha)

            shape_path = root / "bad_shape.npz"
            shape_sha = _write_checkpoint(
                shape_path,
                overrides={"encoder_0_weight": np.zeros((31, 3, 3, 3), dtype=np.float32)},
            )
            with self.assertRaisesRegex(OperationError, "encoder_0_weight must have shape"):
                load_deepjscc_reference_checkpoint(str(shape_path), shape_sha)

            dtype_path = root / "bad_dtype.npz"
            dtype_sha = _write_checkpoint(
                dtype_path,
                overrides={"decoder_2_bias": np.zeros(3, dtype=np.float64)},
            )
            with self.assertRaisesRegex(OperationError, "must have dtype float32"):
                load_deepjscc_reference_checkpoint(str(dtype_path), dtype_sha)

    def test_loader_rejects_duplicate_checkpoint_contract_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "duplicate-contract.npz"
            metadata_json = (
                '{"schema_version":1,'
                '"kind":"noema.deepjscc_checkpoint",'
                '"format":"attacker-selected-format",'
                '"format":"noema_deepjscc_reference_cnn_npz_v1",'
                '"architecture":"reference_cnn_v1",'
                '"input_channels":3,"output_channels":3,"symbol_channels":2}'
            )
            checkpoint_sha = _write_checkpoint(
                checkpoint_path,
                overrides={"metadata_json": metadata_json},
            )
            with self.assertRaisesRegex(
                OperationError, "Duplicate JSON object key `format`"
            ):
                load_deepjscc_reference_checkpoint(
                    str(checkpoint_path), checkpoint_sha
                )

    def test_schema_exposes_conditional_npz_and_portable_artifact_pickers(self):
        for operation in (DeepJsccExternalEncodeOperation(), DeepJsccExternalDecodeOperation()):
            description = operation.describe()
            properties = description["params_schema"]["properties"]
            self.assertEqual(
                properties["runtime"]["enum"],
                [
                    "training_interface",
                    "external_callable",
                    "learned_checkpoint",
                    "learned_artifact",
                ],
            )
            self.assertEqual(properties["runtime"]["default"], "training_interface")
            self.assertEqual(
                description["backends"]["differentiable_export"],
                ["torch"],
            )
            self.assertTrue(
                any(
                    row["runner"] == "differentiable_export"
                    and row["backend"] == "torch"
                    and row["implementation"].startswith("external_model_")
                    for row in description["materializations"]
                )
            )
            picker = properties["checkpoint_path"]["x-noema-ui"]
            self.assertEqual(picker["control"], "trained_artifact")
            self.assertIn(".npz", picker["accept"])
            portable_picker = properties["artifact_manifest_path"]["x-noema-ui"]
            self.assertEqual(portable_picker["control"], "trained_artifact")
            self.assertIn(".zip", portable_picker["accept"])
            for name in (
                "checkpoint_sha256",
                "checkpoint_format",
                "checkpoint_strict",
                "checkpoint_max_bytes",
                "symbol_channels",
            ):
                self.assertTrue(properties[name]["x-noema-ui"]["hidden"])

    def test_training_interface_is_export_only_and_external_callable_remains_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "images.npz"
            images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
            np.savez_compressed(image_path, images=images)
            artifact = Artifact("image.batch.numpy", image_path, {"shape": list(images.shape)})

            def context(runtime: str) -> OperationContext:
                return OperationContext(
                    recipe_name="deepjscc_runtime_contract",
                    step_id="sender",
                    params={"runtime": runtime},
                    inputs={"images": artifact},
                    run_dir=root,
                    step_dir=root / runtime,
                )

            with self.assertRaisesRegex(OperationError, "export-only typed interface"):
                DeepJsccExternalEncodeOperation().run(context("training_interface"))
            with self.assertRaisesRegex(RuntimeError, "requires params.callable"):
                DeepJsccExternalEncodeOperation().run(context("external_callable"))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_reference_cnn_safe_checkpoint_inference_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deepjscc.npz"
            checkpoint = load_deepjscc_reference_checkpoint(
                str(path), _write_checkpoint(path)
            )
            images = np.arange(2 * 8 * 8 * 3, dtype=np.uint8).reshape(2, 8, 8, 3)
            symbols = encode_deepjscc_images(checkpoint, images)
            self.assertEqual(symbols.shape, (2, 2, 2, 2))
            np.testing.assert_array_equal(symbols, np.zeros_like(symbols))

            decoded = decode_deepjscc_symbols(
                checkpoint,
                symbols,
                image_shape=images.shape,
            )
            self.assertEqual(decoded.shape, images.shape)
            # Zero decoder weights and biases end at sigmoid(0)=0.5.
            np.testing.assert_array_equal(decoded, np.full_like(images, 128))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_operations_run_managed_checkpoint_and_preserve_shape_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "deepjscc.npz"
            checkpoint_sha = _write_checkpoint(checkpoint_path)
            image_path = root / "images.npz"
            images = np.zeros((1, 7, 9, 3), dtype=np.uint8)
            image_metadata = {
                "shape": list(images.shape),
                "original_shapes": [[1, 7, 9, 3]],
                "dataset": "unit_test",
            }
            np.savez_compressed(
                image_path,
                images=images,
                metadata_json=json.dumps(image_metadata),
            )
            params = {
                "runtime": "learned_checkpoint",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_format": CHECKPOINT_FORMAT,
                "symbol_channels": 2,
            }
            encoded = DeepJsccExternalEncodeOperation().run(
                OperationContext(
                    recipe_name="deepjscc_checkpoint_test",
                    step_id="sender",
                    params=params,
                    inputs={
                        "images": Artifact("image.batch.numpy", image_path, image_metadata)
                    },
                    run_dir=root,
                    step_dir=root / "sender",
                )
            )
            symbol_artifact = encoded.outputs["symbols"]
            self.assertEqual(symbol_artifact.metadata["symbol_shape"], [1, 2, 2, 3])
            self.assertEqual(symbol_artifact.metadata["image_shape"], [1, 7, 9, 3])
            self.assertEqual(symbol_artifact.metadata["original_shapes"], [[1, 7, 9, 3]])
            self.assertEqual(symbol_artifact.metadata["adapter"], "learned_checkpoint")

            decoded = DeepJsccExternalDecodeOperation().run(
                OperationContext(
                    recipe_name="deepjscc_checkpoint_test",
                    step_id="receiver",
                    params=params,
                    inputs={"symbols": symbol_artifact},
                    run_dir=root,
                    step_dir=root / "receiver",
                )
            )
            with np.load(decoded.outputs["images"].path, allow_pickle=False) as payload:
                reconstructed = np.asarray(payload["images"])
            self.assertEqual(reconstructed.shape, images.shape)
            self.assertEqual(
                decoded.outputs["images"].metadata["checkpoint_sha256"],
                checkpoint_sha,
            )

            mismatched = dict(params, symbol_channels=3)
            with self.assertRaisesRegex(OperationError, "does not match checkpoint"):
                DeepJsccExternalEncodeOperation().run(
                    OperationContext(
                        recipe_name="deepjscc_checkpoint_test",
                        step_id="bad_sender",
                        params=mismatched,
                        inputs={
                            "images": Artifact(
                                "image.batch.numpy", image_path, image_metadata
                            )
                        },
                        run_dir=root,
                        step_dir=root / "bad_sender",
                    )
                )

    def test_benchmark_preflight_checks_paired_project_relative_checkpoint_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = (
                root
                / "differentiable_exports"
                / "deepjscc"
                / "checkpoints"
                / "deepjscc.npz"
            )
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_sha = _write_checkpoint(checkpoint_path)
            relative_checkpoint = str(checkpoint_path.relative_to(root))
            recipe_path = root / "recipes" / "learned_deepjscc.yaml"
            recipe_path.parent.mkdir(parents=True)
            shared_params = {
                "runtime": "learned_checkpoint",
                "checkpoint_path": relative_checkpoint,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_format": CHECKPOINT_FORMAT,
                "symbol_channels": 2,
            }
            recipe_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "name": "learned_deepjscc",
                        "metadata": {"training_performed": True},
                        "steps": [
                            {
                                "id": "sender",
                                "op": "model.deepjscc_external_encode",
                                "params": shared_params,
                            },
                            {
                                "id": "receiver",
                                "op": "model.deepjscc_external_decode",
                                "params": shared_params,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            readiness = _learned_checkpoint_readiness(recipe_path, root)
            self.assertTrue(readiness["valid"], readiness["blockers"])
            self.assertEqual(len(readiness["checkpoints"]), 1)
            self.assertEqual(
                readiness["checkpoints"][0]["steps"],
                ["sender", "receiver"],
            )
            self.assertEqual(readiness["actual_sha256"], checkpoint_sha)


def _write_checkpoint(
    path: Path,
    *,
    symbol_channels: int = 2,
    overrides=None,
    extra=None,
) -> str:
    arrays = {
        "encoder_0_weight": np.zeros((32, 3, 3, 3), dtype=np.float32),
        "encoder_0_bias": np.zeros((32,), dtype=np.float32),
        "encoder_2_weight": np.zeros(
            (2 * symbol_channels, 32, 3, 3), dtype=np.float32
        ),
        "encoder_2_bias": np.zeros((2 * symbol_channels,), dtype=np.float32),
        "decoder_0_weight": np.zeros(
            (2 * symbol_channels, 32, 4, 4), dtype=np.float32
        ),
        "decoder_0_bias": np.zeros((32,), dtype=np.float32),
        "decoder_2_weight": np.zeros((32, 3, 4, 4), dtype=np.float32),
        "decoder_2_bias": np.zeros((3,), dtype=np.float32),
        "metadata_json": json.dumps(
            {
                "schema_version": 1,
                "kind": CHECKPOINT_KIND,
                "format": CHECKPOINT_FORMAT,
                "architecture": CHECKPOINT_ARCHITECTURE,
                "input_channels": 3,
                "output_channels": 3,
                "symbol_channels": symbol_channels,
            },
            sort_keys=True,
        ),
    }
    arrays.update(overrides or {})
    arrays.update(extra or {})
    np.savez_compressed(path, **arrays)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
