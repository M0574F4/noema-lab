from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.structured_input import decode_strict_json_object
from noema_lab.core import dataplane
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationResult,
    object_schema,
)

JsonDict = Dict[str, Any]


class BitErrorRateOperation(Operation):
    id = "metrics.bit_error_rate"
    name = "Bit error rate"
    input_kinds = {
        "reference": [
            "channel.payload_bits.numpy",
            "channel.coded_bits.numpy",
            "channel.demod_bits.numpy",
            "channel.bits.numpy",
        ],
        "candidate": [
            "channel.payload_bits.numpy",
            "channel.coded_bits.numpy",
            "channel.demod_bits.numpy",
            "channel.bits.numpy",
        ],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema(
        {
            "label": {"type": "string", "default": "ber"},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        reference, reference_metadata = _load_bits(
            ctx.require_input("reference").path,
            ctx.require_input("reference").metadata,
        )
        candidate, candidate_metadata = _load_bits(
            ctx.require_input("candidate").path,
            ctx.require_input("candidate").metadata,
        )
        if int(reference.size) != int(candidate.size):
            raise OperationError(
                "Bit error rate requires equal bit counts: reference has %d bits "
                "but candidate has %d bits"
                % (int(reference.size), int(candidate.size))
            )
        compare_count = int(reference.size)
        if compare_count == 0:
            raise OperationError(
                "Bit error rate is undefined because no reference bits were provided"
            )
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        error_count, selected_backend = dataplane.bit_error_count(reference[:compare_count], candidate[:compare_count], backend)
        ber = float(error_count) / float(compare_count)
        report = {
            "schema_version": 1,
            "metric_family": "bit_error_rate",
            "label": str(ctx.params.get("label", "ber")),
            "ber": ber,
            "error_count": error_count,
            "compare_bit_count": int(compare_count),
            "reference_bit_count": int(reference.size),
            "candidate_bit_count": int(candidate.size),
            "length_delta": int(candidate.size) - int(reference.size),
            "reference_role": reference_metadata.get("bit_role"),
            "candidate_role": candidate_metadata.get("bit_role"),
            "data_plane_backend": selected_backend,
        }
        path = ctx.output_path("report", ".json")
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        label = str(ctx.params.get("label", "ber")).replace(" ", "_")
        return OperationResult(
            outputs={"report": artifact("metrics.report", path, report)},
            metrics={
                "channel.%s.ber" % label: ber,
                "channel.%s.error_count" % label: error_count,
                "channel.%s.compare_bit_count" % label: int(compare_count),
            },
            metadata={"label": label, "ber": ber, "data_plane_backend": selected_backend},
        )


class BlockErrorRateOperation(Operation):
    id = "metrics.block_error_rate"
    name = "Block error rate"
    input_kinds = {
        "reference": [
            "channel.payload_bits.numpy",
            "channel.coded_bits.numpy",
            "channel.demod_bits.numpy",
            "channel.bits.numpy",
        ],
        "candidate": [
            "channel.payload_bits.numpy",
            "channel.coded_bits.numpy",
            "channel.demod_bits.numpy",
            "channel.bits.numpy",
        ],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema(
        {
            "label": {"type": "string", "default": "bler"},
            "block_size": {"type": "integer", "default": 1024, "minimum": 1},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        reference, reference_metadata = _load_bits(
            ctx.require_input("reference").path,
            ctx.require_input("reference").metadata,
        )
        candidate, candidate_metadata = _load_bits(
            ctx.require_input("candidate").path,
            ctx.require_input("candidate").metadata,
        )
        if int(reference.size) == 0 or int(candidate.size) == 0:
            raise OperationError(
                "Block error rate is undefined when either bit array has no samples"
            )
        block_size = int(ctx.params.get("block_size") or 1024)
        if block_size < 1:
            raise RuntimeError("block_size must be at least 1")
        total_count = max(int(reference.size), int(candidate.size))
        block_count = int(np.ceil(float(total_count) / float(block_size))) if total_count else 0
        block_error_count = 0
        for block_index in range(block_count):
            start = block_index * block_size
            stop = min(start + block_size, total_count)
            ref_block = reference[start:min(stop, int(reference.size))]
            cand_block = candidate[start:min(stop, int(candidate.size))]
            length_mismatch = int(ref_block.size) != int(cand_block.size)
            compare_count = min(int(ref_block.size), int(cand_block.size))
            mismatch = bool(compare_count and np.any(ref_block[:compare_count] != cand_block[:compare_count]))
            if length_mismatch or mismatch:
                block_error_count += 1
        bler = float(block_error_count) / float(block_count) if block_count else 0.0
        label = str(ctx.params.get("label", "bler")).replace(" ", "_")
        report = {
            "schema_version": 1,
            "metric_family": "block_error_rate",
            "label": label,
            "bler": bler,
            "block_error_count": int(block_error_count),
            "block_count": int(block_count),
            "block_size": int(block_size),
            "reference_bit_count": int(reference.size),
            "candidate_bit_count": int(candidate.size),
            "length_delta": int(candidate.size) - int(reference.size),
            "reference_role": reference_metadata.get("bit_role"),
            "candidate_role": candidate_metadata.get("bit_role"),
        }
        path = ctx.output_path("report", ".json")
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={"report": artifact("metrics.report", path, report)},
            metrics={
                "channel.%s.bler" % label: bler,
                "channel.%s.block_error_count" % label: int(block_error_count),
                "channel.%s.block_count" % label: int(block_count),
                "channel.%s.block_size" % label: int(block_size),
            },
            metadata=report,
        )


def _load_bits(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        bits = payload["bits"]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Bit metric artifact metadata_json",
                )
            )
    canonical, _selected_backend = dataplane.require_canonical_bits(
        np.asarray(bits), "metrics bit artifact %s" % path, "python_numpy"
    )
    return canonical, metadata
