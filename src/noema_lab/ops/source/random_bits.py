from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np

from noema_lab.core import dataplane
from noema_lab.core.artifacts import artifact
from noema_lab.core.boundaries import validate_payload_bits
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema

JsonDict = Dict[str, Any]
MAX_RANDOM_BITS_PER_OUTPUT = 128 * 1024 * 1024
MAX_RANDOM_BITS_BATCH_SIZE = 65_536


class RandomBitsOperation(Operation):
    id = "source.random_bits"
    name = "Synthetic random bit source"
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    differentiability = {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": "Synthetic discrete source bits are generated as benchmark artifacts and do not participate in gradient flow.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": []}
    equivalence = {
        "type": "exact",
        "reason": "Given the same seed and bit count, random-bit source materializations must produce the same canonical uint8 bit vector.",
    }
    formats = {"artifact": "npz", "tensor": "none"}
    params_schema = object_schema(
        {
            "bit_count": {
                "type": "integer",
                "default": 4096,
                "minimum": 1,
                "maximum": MAX_RANDOM_BITS_PER_OUTPUT,
            },
            "batch_size": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": MAX_RANDOM_BITS_BATCH_SIZE,
            },
            "seed": {"type": "integer", "default": 0, "minimum": 0},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        bit_count = int(ctx.params.get("bit_count") or 4096)
        batch_size = int(ctx.params.get("batch_size") or 1)
        if bit_count < 1:
            raise RuntimeError("bit_count must be at least 1")
        if batch_size < 1:
            raise RuntimeError("batch_size must be at least 1")
        total_bits = bit_count * batch_size
        if total_bits > MAX_RANDOM_BITS_PER_OUTPUT:
            raise RuntimeError(
                "bit_count * batch_size must be at most %d bits"
                % MAX_RANDOM_BITS_PER_OUTPUT
            )
        seed = ctx.seed("random_bits")
        rng = np.random.RandomState(seed)
        raw_bits = rng.randint(0, 2, size=total_bits).astype(np.uint8, copy=False)
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        bits, metadata = validate_payload_bits(raw_bits, label=ctx.step_id, backend=backend)
        metadata.update(
            {
                "dataset": "synthetic_random_bits",
                "split": "fixed_seed",
                "source": "source.random_bits",
                "generator": "numpy.random.RandomState.randint",
                "seed": int(seed),
                "batch_size": int(batch_size),
                "bit_count_per_example": int(bit_count),
                "example_count": int(batch_size),
                "transport_block_size_bits": int(bit_count),
                "transport_block_count": int(batch_size),
                "capture_record_count": int(batch_size),
                "capture_record_shape": [int(bit_count)],
                "bit_count": int(bits.size),
                "payload_bit_count": int(bits.size),
                "bit_role": "payload",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
            }
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
        ctx.report_progress("Generated random bits", phase="data", status="completed", completed=1, total=1, percent=100.0, unit="batch")
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, metadata)},
            metrics={
                "source.bit_count": int(bits.size),
                "source.example_count": int(batch_size),
                "source.transport_block_size_bits": int(bit_count),
                "source.transport_block_count": int(batch_size),
                "channel.payload_bit_count": int(bits.size),
            },
            metadata=metadata,
        )
