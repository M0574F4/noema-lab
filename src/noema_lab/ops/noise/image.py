from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.structured_input import decode_strict_json_object
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema

JsonDict = Dict[str, Any]


class SourceImagePerturbationOperation(Operation):
    id = "noise.source_image_perturbation"
    name = "Source-domain image perturbation"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    differentiability = {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": "Artifact-mode source perturbations are stochastic preprocessing blocks, not differentiable training modules.",
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": [],
    }
    equivalence = {"type": "statistical", "reason": "Stochastic perturbations are equivalent by declared distribution and seed policy."}
    formats = {"artifact": "npz", "tensor": "none"}
    params_schema = object_schema(
        {
            "mode": {
                "type": "string",
                "default": "none",
                "enum": ["none", "gaussian", "blur", "patch_mask", "salt_pepper"],
            },
            "sigma": {"type": "number", "default": 0.02, "minimum": 0.0},
            "probability": {"type": "number", "default": 0.02, "minimum": 0.0, "maximum": 1.0},
            "blur_radius": {"type": "number", "default": 1.5, "minimum": 0.0},
            "patch_size": {"type": "integer", "default": 32, "minimum": 1},
            "mask_value": {"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 255.0},
            "seed": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("images")
        images, metadata = _load_array_artifact(input_artifact.path, input_artifact.metadata, "images")
        mode = str(ctx.params.get("mode") or "none")
        seed = ctx.seed("source_image_perturbation")
        output, stats = _apply_image_noise(
            images,
            mode=mode,
            sigma=float(ctx.params.get("sigma", 0.02) or 0.0),
            probability=float(ctx.params.get("probability", 0.02) or 0.0),
            blur_radius=float(ctx.params.get("blur_radius", 1.5) or 0.0),
            patch_size=max(1, int(ctx.params.get("patch_size", 32) or 32)),
            mask_value=float(ctx.params.get("mask_value", 0.0) or 0.0),
            seed=seed,
        )
        output_metadata = dict(metadata)
        output_metadata.setdefault("noise_history", [])
        output_metadata["noise_history"] = list(output_metadata["noise_history"]) + [
            {"op": self.id, "mode": mode, "seed": seed, **stats}
        ]
        output_metadata.update({"source_perturbation_mode": mode, "dtype": str(output.dtype), "shape": list(output.shape)})
        path = ctx.output_path("images", ".npz")
        np.savez_compressed(path, images=output, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
            metrics={
                "source_perturbation.image.changed_fraction": stats["changed_fraction"],
                "source_perturbation.image.mse": stats["mse"],
            },
            metadata={"mode": mode, **stats},
        )


class RepresentationLatentNoiseOperation(Operation):
    id = "noise.representation_latents"
    name = "Representation-domain noise on continuous latents"
    input_kinds = {"latents": ["semantic.latents.numpy"]}
    output_kinds = {"latents": "semantic.latents.numpy"}
    differentiability = {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": "Artifact-mode latent perturbation is executed as NumPy preprocessing; differentiable training export needs a torch materialization.",
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": [],
    }
    equivalence = {"type": "statistical", "reason": "Stochastic latent perturbations are equivalent by declared distribution and seed policy."}
    formats = {"artifact": "npz", "tensor": "none"}
    params_schema = object_schema(
        {
            "mode": {"type": "string", "default": "none", "enum": ["none", "gaussian", "dropout"]},
            "sigma": {"type": "number", "default": 0.05, "minimum": 0.0},
            "probability": {"type": "number", "default": 0.05, "minimum": 0.0, "maximum": 1.0},
            "seed": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("latents")
        latents, metadata = _load_array_artifact(input_artifact.path, input_artifact.metadata, "latents")
        mode = str(ctx.params.get("mode") or "none")
        seed = ctx.seed("representation_latents")
        output, stats = _apply_latent_noise(
            latents.astype(np.float32, copy=False),
            mode=mode,
            sigma=float(ctx.params.get("sigma", 0.05) or 0.0),
            probability=float(ctx.params.get("probability", 0.05) or 0.0),
            seed=seed,
        )
        output_metadata = dict(metadata)
        output_metadata.setdefault("noise_history", [])
        output_metadata["noise_history"] = list(output_metadata["noise_history"]) + [
            {"op": self.id, "mode": mode, "seed": seed, **stats}
        ]
        output_metadata.update({"representation_noise_mode": mode, "dtype": str(output.dtype), "shape": list(output.shape)})
        path = ctx.output_path("latents", ".npz")
        np.savez_compressed(path, latents=output.astype(np.float32, copy=False), metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"latents": artifact("semantic.latents.numpy", path, output_metadata)},
            metrics={
                "representation.latents.changed_fraction": stats["changed_fraction"],
                "representation.latents.noise_mse": stats["mse"],
            },
            metadata={"mode": mode, **stats},
        )


def _load_array_artifact(path, fallback_metadata: JsonDict, array_name: str) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        array = payload[array_name]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Image noise artifact metadata_json",
                )
            )
    return array, metadata


def _apply_image_noise(
    images: np.ndarray,
    mode: str,
    sigma: float,
    probability: float,
    blur_radius: float,
    patch_size: int,
    mask_value: float,
    seed: int,
) -> Tuple[np.ndarray, JsonDict]:
    original = images.astype(np.float32, copy=False)
    if mode == "none":
        output = images.copy()
    else:
        rng = np.random.default_rng(seed)
        work = original.copy()
        if mode == "gaussian":
            work += rng.normal(0.0, float(sigma) * 255.0, size=work.shape).astype(np.float32)
        elif mode == "blur":
            work = _box_blur_batch(work, blur_radius)
        elif mode == "patch_mask":
            work = _patch_mask_batch(work, probability, patch_size, mask_value, rng)
        elif mode == "salt_pepper":
            work = _salt_pepper_batch(work, probability, rng)
        else:
            raise RuntimeError("Unsupported source image perturbation mode: %s" % mode)
        output = np.clip(np.rint(work), 0, 255).astype(images.dtype if np.issubdtype(images.dtype, np.integer) else np.float32)
    return output, _array_noise_stats(original, output.astype(np.float32, copy=False))


def _apply_latent_noise(latents: np.ndarray, mode: str, sigma: float, probability: float, seed: int) -> Tuple[np.ndarray, JsonDict]:
    original = latents.astype(np.float32, copy=False)
    if mode == "none":
        output = original.copy()
    else:
        rng = np.random.default_rng(seed)
        output = original.copy()
        if mode == "gaussian":
            output += rng.normal(0.0, sigma, size=output.shape).astype(np.float32)
        elif mode == "dropout":
            mask = rng.random(output.shape) < probability
            output[mask] = 0.0
        else:
            raise RuntimeError("Unsupported latent representation noise mode: %s" % mode)
    return output.astype(np.float32, copy=False), _array_noise_stats(original, output)


def _array_noise_stats(original: np.ndarray, output: np.ndarray) -> JsonDict:
    if original.size == 0:
        return {"changed_fraction": 0.0, "mse": 0.0}
    delta = output.astype(np.float32, copy=False) - original.astype(np.float32, copy=False)
    return {
        "changed_fraction": float(np.mean(np.abs(delta) > 1e-6)),
        "mse": float(np.mean(delta * delta)),
    }


def _box_blur_batch(images: np.ndarray, radius: float) -> np.ndarray:
    kernel = max(1, int(round(float(radius) * 2.0 + 1.0)))
    if kernel <= 1:
        return images.copy()
    work = images.astype(np.float32, copy=True)
    for axis in (1, 2):
        pad = [(0, 0)] * work.ndim
        pad[axis] = (kernel // 2, kernel - 1 - kernel // 2)
        padded = np.pad(work, pad, mode="edge")
        cumsum = np.cumsum(padded, axis=axis, dtype=np.float32)
        prefix_shape = list(cumsum.shape)
        prefix_shape[axis] = 1
        cumsum = np.concatenate([np.zeros(prefix_shape, dtype=np.float32), cumsum], axis=axis)
        high = [slice(None)] * cumsum.ndim
        low = [slice(None)] * cumsum.ndim
        high[axis] = slice(kernel, None)
        low[axis] = slice(0, -kernel)
        work = (cumsum[tuple(high)] - cumsum[tuple(low)]) / float(kernel)
    return work


def _patch_mask_batch(images: np.ndarray, probability: float, patch_size: int, mask_value: float, rng) -> np.ndarray:
    output = images.copy()
    if probability <= 0.0:
        return output
    count, height, width = output.shape[:3]
    channels = output.shape[3] if output.ndim >= 4 else 1
    patch_area = max(1, int(patch_size) * int(patch_size))
    image_area = max(1, int(height) * int(width))
    patches_per_image = int(np.ceil(float(probability) * float(image_area) / float(patch_area)))
    for index in range(count):
        for _ in range(max(1, patches_per_image)):
            top = int(rng.integers(0, max(1, height)))
            left = int(rng.integers(0, max(1, width)))
            bottom = min(height, top + int(patch_size))
            right = min(width, left + int(patch_size))
            if channels == 1 and output.ndim == 3:
                output[index, top:bottom, left:right] = mask_value
            else:
                output[index, top:bottom, left:right, :] = mask_value
    return output


def _salt_pepper_batch(images: np.ndarray, probability: float, rng) -> np.ndarray:
    output = images.copy()
    if probability <= 0.0:
        return output
    mask = rng.random(output.shape[:3]) < probability
    salt = rng.random(output.shape[:3]) < 0.5
    if output.ndim == 4:
        output[mask & salt, :] = 255.0
        output[mask & ~salt, :] = 0.0
    else:
        output[mask & salt] = 255.0
        output[mask & ~salt] = 0.0
    return output
