from __future__ import annotations

import json
import math
from typing import Any, Dict, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.structured_input import decode_strict_json_object
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema

JsonDict = Dict[str, Any]


class RepresentationIndexNoiseOperation(Operation):
    id = "noise.representation_indices"
    name = "Representation noise on discrete indices"
    input_kinds = {"indices": ["semantic.indices.numpy"]}
    output_kinds = {"indices": "semantic.indices.numpy"}
    params_schema = object_schema(
        {
            "mode": {
                "type": "string",
                "default": "none",
                "enum": ["none", "dropout", "random_replace", "burst"],
            },
            "probability": {"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 1.0},
            "replacement": {"type": "integer", "default": 0, "minimum": 0},
            "seed": {"type": "integer", "default": 0, "minimum": 0},
            "burst_length": {"type": "integer", "default": 64, "minimum": 1},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("indices")
        indices, metadata = _load_indices(input_artifact.path, input_artifact.metadata)
        mode = str(ctx.params.get("mode", "none"))
        probability = float(ctx.params.get("probability", 0.0))
        replacement = int(ctx.params.get("replacement", 0))
        seed = ctx.seed("representation_indices")
        burst_length = int(ctx.params.get("burst_length", 64))
        output, changed = _apply_noise(
            indices=indices,
            metadata=metadata,
            mode=mode,
            probability=probability,
            replacement=replacement,
            seed=seed,
            burst_length=burst_length,
        )
        noise_metadata = dict(metadata)
        noise_metadata.setdefault("noise_history", [])
        noise_metadata["noise_history"] = list(noise_metadata["noise_history"]) + [
            {
                "op": self.id,
                "mode": mode,
                "probability": probability,
                "changed_fraction": changed,
                "seed": seed,
            }
        ]
        path = ctx.output_path("indices", ".npz")
        np.savez_compressed(path, indices=output, metadata_json=json.dumps(noise_metadata))
        return OperationResult(
            outputs={"indices": artifact("semantic.indices.numpy", path, noise_metadata)},
            metrics={"representation.changed_fraction": changed},
            metadata={"mode": mode, "changed_fraction": changed},
        )


def _load_indices(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        indices = payload["indices"]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Semantic noise artifact metadata_json",
                )
            )
    return indices.astype(np.int64, copy=True), metadata


def _apply_noise(
    indices: np.ndarray,
    metadata: JsonDict,
    mode: str,
    probability: float,
    replacement: int,
    seed: int,
    burst_length: int,
) -> Tuple[np.ndarray, float]:
    if mode == "none" or probability <= 0.0:
        return indices.copy(), 0.0
    rng = np.random.RandomState(seed)
    output = indices.copy()
    original = output.copy()
    codebook_size = int(metadata.get("codebook_size") or max(int(output.max()) + 1, 1))
    flat = output.reshape(-1)
    if mode == "dropout":
        mask = rng.rand(flat.size) < probability
        flat[mask] = replacement % codebook_size
    elif mode == "random_replace":
        mask = rng.rand(flat.size) < probability
        flat[mask] = rng.randint(0, codebook_size, size=int(mask.sum()))
    elif mode == "burst":
        target = int(math.ceil(flat.size * probability))
        burst_count = int(math.ceil(float(target) / float(burst_length)))
        for _ in range(max(1, burst_count)):
            start = int(rng.randint(0, max(flat.size, 1)))
            end = min(flat.size, start + burst_length)
            flat[start:end] = replacement % codebook_size
    else:
        raise RuntimeError("Unknown representation noise mode: %s" % mode)
    changed = float(np.mean(output != original)) if output.size else 0.0
    return output, changed
