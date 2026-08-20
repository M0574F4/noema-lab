from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationResult,
    object_schema,
)
from noema_lab.core.sample_identity import content_sha256_item_ids

JsonDict = Dict[str, Any]


class LocalNpzImagesOperation(Operation):
    id = "source.local_npz_images"
    name = "Local NumPy image batch"
    output_kinds = {"images": "image.batch.numpy"}
    output_metadata_guarantees = {"images": ["original_shapes", "shape"]}
    params_schema = object_schema(
        {
            "path": {"type": "string"},
            "array": {"type": "string", "default": "images"},
            "dtype_policy": {
                "type": "string",
                "default": "strict_uint8",
                "enum": ["strict_uint8", "clip_round_uint8"],
                "description": (
                    "Reject non-uint8 input by default. clip_round_uint8 is an "
                    "explicit lossy conversion that is recorded in provenance."
                ),
            },
        },
        required=["path"],
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        source_path = Path(str(ctx.params["path"]))
        array_name = str(ctx.params.get("array", "images"))
        with np.load(str(source_path), allow_pickle=False) as payload:
            images = payload[array_name]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise OperationError("Expected image array [N,H,W,3], got %s" % (images.shape,))
        source_dtype = str(images.dtype)
        dtype_policy = str(ctx.params.get("dtype_policy") or "strict_uint8")
        coerced = False
        if images.dtype != np.uint8:
            if dtype_policy != "clip_round_uint8":
                raise OperationError(
                    "Local NPZ image array must use uint8; got %s. Set "
                    "dtype_policy=clip_round_uint8 to request an explicit lossy "
                    "round/clip conversion." % images.dtype
                )
            images = np.clip(np.rint(images), 0, 255).astype(np.uint8)
            coerced = True
        image_ids = content_sha256_item_ids(images)
        metadata = {
            "shape": list(images.shape),
            "original_shapes": [
                [
                    1,
                    int(images.shape[1]),
                    int(images.shape[2]),
                    int(images.shape[3]),
                ]
                for _index in range(int(images.shape[0]))
            ],
            "dtype": str(images.dtype),
            "source_dtype": source_dtype,
            "dtype_policy": dtype_policy,
            "dtype_coerced": coerced,
            "source": "local_npz",
            "source_path": str(source_path),
            "array": array_name,
            "source_item_count": int(images.shape[0]),
            "source_item_ids": image_ids,
            "source_item_id_source": "content_sha256",
            "image_ids": image_ids,
        }
        path = ctx.output_path("images", ".npz")
        np.savez_compressed(path, images=images, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"images": artifact("image.batch.numpy", path, metadata)},
            metadata={"image_count": int(images.shape[0]), "source_path": str(source_path)},
        )
