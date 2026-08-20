from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import OperationContext
from noema_lab.ops.noise.image import RepresentationLatentNoiseOperation, SourceImagePerturbationOperation


class NoiseOperationTests(unittest.TestCase):
    def test_source_image_patch_mask_changes_pixels_and_preserves_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "images.npz"
            images = np.full((2, 16, 16, 3), 127, dtype=np.uint8)
            metadata = {"dataset": "unit", "shape": list(images.shape), "dtype": str(images.dtype)}
            np.savez_compressed(input_path, images=images, metadata_json=json.dumps(metadata))
            step_dir = root / "step"
            ctx = OperationContext(
                recipe_name="noise_test",
                step_id="source_noise",
                params={"mode": "patch_mask", "probability": 0.25, "patch_size": 4, "mask_value": 0, "seed": 7},
                inputs={"images": artifact("image.batch.numpy", input_path, metadata)},
                run_dir=root,
                step_dir=step_dir,
            )
            result = SourceImagePerturbationOperation().run(ctx)
            output = result.outputs["images"]
            self.assertEqual(output.kind, "image.batch.numpy")
            with np.load(output.path, allow_pickle=False) as payload:
                noised = payload["images"]
            self.assertEqual(noised.shape, images.shape)
            self.assertEqual(noised.dtype, images.dtype)
            self.assertGreater(result.metrics["source_perturbation.image.changed_fraction"], 0.0)
            self.assertIn("noise_history", output.metadata)

    def test_representation_latent_gaussian_noise_changes_latents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "latents.npz"
            latents = np.zeros((2, 3, 4, 4), dtype=np.float32)
            metadata = {"shape": list(latents.shape), "dtype": str(latents.dtype)}
            np.savez_compressed(input_path, latents=latents, metadata_json=json.dumps(metadata))
            ctx = OperationContext(
                recipe_name="noise_test",
                step_id="latent_noise",
                params={"mode": "gaussian", "sigma": 0.1, "seed": 13},
                inputs={"latents": artifact("semantic.latents.numpy", input_path, metadata)},
                run_dir=root,
                step_dir=root / "step",
            )
            result = RepresentationLatentNoiseOperation().run(ctx)
            output = result.outputs["latents"]
            self.assertEqual(output.kind, "semantic.latents.numpy")
            with np.load(output.path, allow_pickle=False) as payload:
                noised = payload["latents"]
            self.assertEqual(noised.shape, latents.shape)
            self.assertEqual(noised.dtype, np.float32)
            self.assertGreater(result.metrics["representation.latents.noise_mse"], 0.0)
            self.assertGreater(result.metrics["representation.latents.changed_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
