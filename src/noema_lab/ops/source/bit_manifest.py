from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from urllib.parse import urlsplit

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.boundaries import validate_payload_bits
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationResult,
    object_schema,
)
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.structured_input import decode_strict_yaml_or_json

JsonDict = Dict[str, Any]

BIT_MANIFEST_KIND = "noema.bit_payload_manifest"
BIT_MANIFEST_SCHEMA_VERSION = 1
BIT_PACKING = "packed_msb_first_hex_v1"
BIT_CONTENT_KIND = "noema.canonical_unpacked_bits"
BIT_CONTENT_SCHEMA_VERSION = 1
MAX_BIT_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_BIT_MANIFEST_ITEMS = 65_536
MAX_BIT_MANIFEST_OUTPUT_BITS = 128 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_RE = re.compile(r"^[0-9a-f]+$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "id",
    "version",
    "encoding",
    "ordered_item_ids",
    "splits",
    "items",
}
_ITEM_KEYS = {
    "item_id",
    "group_id",
    "bit_count",
    "packed_hex",
    "content_sha256",
}


class BitManifestSourceOperation(Operation):
    """Load hash-pinned, inline payload bits with publication-grade lineage."""

    id = "source.bit_manifest"
    name = "Frozen bit payload manifest source"
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    output_metadata_guarantees = {
        "bits": [
            "dataset_manifest_sha256",
            "source_operation_contract_sha256",
            "ordered_post_transform_sha256",
            "batch_tensor_sha256",
            "source_item_ids",
            "source_item_content_sha256",
            "source_group_ids",
            "source_split_ids",
            "source_items",
            "source_item_payload_bit_counts",
            "capture_record_count",
            "capture_record_shape",
        ]
    }
    differentiability = {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": (
            "Frozen discrete source bits are evidence artifacts and do not "
            "participate in gradient flow."
        ),
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": [],
    }
    equivalence = {
        "type": "exact",
        "reason": (
            "A hash-pinned manifest and ordered selection must materialize the "
            "same canonical unpacked uint8 vector on every host."
        ),
    }
    formats = {"artifact": "npz", "tensor": "none"}
    params_schema = object_schema(
        {
            "manifest_path": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Local .json/.yaml/.yml noema.bit_payload_manifest path. "
                    "URI schemes and symlinks are rejected."
                ),
            },
            "manifest_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
                "description": "Expected SHA-256 of the exact manifest file bytes.",
            },
            "item_ids": {
                "type": "array",
                "default": [],
                "items": {"type": "string", "minLength": 1},
                "maxItems": MAX_BIT_MANIFEST_ITEMS,
                "description": (
                    "Explicit ordered manifest item IDs. Mutually exclusive with selection."
                ),
            },
            "selection": {
                "type": "string",
                "default": "",
                "description": (
                    "Named manifest split. Mutually exclusive with item_ids."
                ),
            },
        },
        required=["manifest_path", "manifest_sha256"],
    )

    def validate_preflight(
        self,
        params: Mapping[str, Any],
        inputs: Mapping[str, str] | None = None,
    ) -> None:
        _validate_selector_params(params)
        _require_sha256(params.get("manifest_sha256"), "manifest_sha256")
        _manifest_path(str(params.get("manifest_path") or ""), require_file=False)

    def run(self, ctx: OperationContext) -> OperationResult:
        explicit_ids, selection = _validate_selector_params(ctx.params)
        expected_manifest_sha256 = _require_sha256(
            ctx.params.get("manifest_sha256"), "manifest_sha256"
        )
        manifest_path = _manifest_path(
            str(ctx.params.get("manifest_path") or ""), require_file=True
        )
        raw_manifest = _read_manifest_bytes(manifest_path)
        manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
        if manifest_sha256 != expected_manifest_sha256:
            raise OperationError(
                "Bit payload manifest SHA-256 mismatch: %s" % manifest_path
            )
        manifest = _decode_and_validate_manifest(manifest_path, raw_manifest)
        if hashlib.sha256(_read_manifest_bytes(manifest_path)).hexdigest() != manifest_sha256:
            raise OperationError("Bit payload manifest changed while loading")

        selected_records, selected_splits = _select_records(
            manifest,
            explicit_ids=explicit_ids,
            selection=selection,
        )
        bit_counts = [int(row["bit_count"]) for row in selected_records]
        if len(set(bit_counts)) != 1:
            raise OperationError(
                "Selected bit payload items must have one uniform bit_count; got %s"
                % ", ".join(str(value) for value in sorted(set(bit_counts)))
            )
        bit_count_per_item = int(bit_counts[0])
        total_bits = bit_count_per_item * len(selected_records)
        if total_bits > MAX_BIT_MANIFEST_OUTPUT_BITS:
            raise OperationError(
                "Selected bit payloads exceed the %d-bit output limit"
                % MAX_BIT_MANIFEST_OUTPUT_BITS
            )
        rows = [_unpack_record(row) for row in selected_records]
        raw_bits = np.concatenate(rows).astype(np.uint8, copy=False)
        bits, metadata = validate_payload_bits(
            raw_bits,
            label=ctx.step_id,
            backend="python_numpy",
        )

        manifest_id = str(manifest["id"])
        transform = {
            "name": "identity",
            "source_encoding": BIT_PACKING,
            "output_encoding": "unpacked_uint8_msb_first",
        }
        transform_sha256 = canonical_json_sha256(transform)
        source_items: List[JsonDict] = []
        ordered_items: List[JsonDict] = []
        for index, (record, split_id) in enumerate(
            zip(selected_records, selected_splits)
        ):
            item_id = str(record["item_id"])
            content_sha256 = str(record["content_sha256"])
            source_id = "%s:%s" % (manifest_id, item_id)
            source_item = {
                "order_index": index,
                "item_id": item_id,
                "sample_id": item_id,
                "source_id": source_id,
                "group_id": str(record["group_id"]),
                "split": split_id,
                "ancestry_ids": [source_id],
                "source_sha256": content_sha256,
                "transform": transform,
                "transform_fingerprint_sha256": transform_sha256,
                "post_transform_sha256": content_sha256,
                "bit_count": int(record["bit_count"]),
                "packed_byte_count": len(bytes.fromhex(str(record["packed_hex"]))),
            }
            source_items.append(source_item)
            ordered_items.append(
                {
                    "item_id": item_id,
                    "sample_id": item_id,
                    "content_sha256": content_sha256,
                    "post_transform_sha256": content_sha256,
                    "shape": [int(record["bit_count"])],
                    "dtype": "uint8",
                }
            )

        selected_ids = [str(row["item_id"]) for row in selected_records]
        selected_hashes = [str(row["content_sha256"]) for row in selected_records]
        selected_groups = [str(row["group_id"]) for row in selected_records]
        source_operation_contract = {
            "schema_version": 1,
            "operation": self.id,
            "manifest_id": manifest_id,
            "manifest_version": str(manifest["version"]),
            "manifest_sha256": manifest_sha256,
            "encoding": BIT_PACKING,
            "selection": selection or None,
            "item_ids": selected_ids,
            "uniform_bit_count": bit_count_per_item,
        }
        batch_tensor_sha256 = bit_payload_content_sha256(
            int(bits.size), _pack_bits_hex(bits)
        )
        metadata.update(
            {
                "dataset": manifest_id,
                "dataset_manifest_path": str(manifest_path),
                "dataset_manifest_sha256": manifest_sha256,
                "dataset_manifest_expected_sha256": expected_manifest_sha256,
                "dataset_manifest_version": str(manifest["version"]),
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "manifest_id": manifest_id,
                "manifest_version": str(manifest["version"]),
                "manifest_encoding": BIT_PACKING,
                "manifest_ordered_item_ids": list(manifest["ordered_item_ids"]),
                "selection": selection or None,
                "split": selection or (
                    selected_splits[0]
                    if len(set(selected_splits)) == 1
                    else None
                ),
                "source": self.id,
                "source_operation_contract": source_operation_contract,
                "source_operation_contract_sha256": canonical_json_sha256(
                    source_operation_contract
                ),
                "source_items": source_items,
                "source_item_ids": selected_ids,
                "sample_ids": selected_ids,
                "source_ids": [str(row["source_id"]) for row in source_items],
                "source_group_ids": selected_groups,
                "source_split_ids": selected_splits,
                "source_item_content_sha256": selected_hashes,
                "source_item_count": len(selected_records),
                "source_item_payload_bit_counts": bit_counts,
                "ordered_post_transform_items": ordered_items,
                "ordered_post_transform_sha256": canonical_json_sha256(
                    ordered_items
                ),
                "batch_tensor_sha256": batch_tensor_sha256,
                "content_hash_contract": {
                    "schema_version": BIT_CONTENT_SCHEMA_VERSION,
                    "kind": BIT_CONTENT_KIND,
                    "fields": ["bit_count", "packed_hex"],
                    "canonicalization": "canonical_json_sha256",
                },
                "batch_size": len(selected_records),
                "example_count": len(selected_records),
                "bit_count_per_example": bit_count_per_item,
                "transport_block_size_bits": bit_count_per_item,
                "transport_block_count": len(selected_records),
                "capture_record_count": len(selected_records),
                "capture_record_shape": [bit_count_per_item],
                "payload_bit_count": int(bits.size),
                "bit_role": "payload",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
            }
        )
        output_path = ctx.output_path("bits", ".npz")
        np.savez_compressed(
            output_path,
            bits=bits,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        ctx.report_progress(
            "Loaded frozen bit payload manifest",
            phase="data",
            status="completed",
            completed=len(selected_records),
            total=len(selected_records),
            percent=100.0,
            unit="source_items",
        )
        return OperationResult(
            outputs={
                "bits": artifact(
                    "channel.payload_bits.numpy", output_path, metadata
                )
            },
            metrics={
                "source.bit_count": int(bits.size),
                "source.example_count": len(selected_records),
                "source.transport_block_size_bits": bit_count_per_item,
                "source.transport_block_count": len(selected_records),
                "channel.payload_bit_count": int(bits.size),
            },
            metadata=metadata,
        )


def bit_payload_content_sha256(bit_count: int, packed_hex: str) -> str:
    """Hash the canonical bit-content identity used by manifest item records."""

    if type(bit_count) is not int or bit_count < 1:
        raise ValueError("bit_count must be a positive integer")
    if not isinstance(packed_hex, str) or _HEX_RE.fullmatch(packed_hex) is None:
        raise ValueError("packed_hex must be non-empty lowercase hexadecimal")
    return canonical_json_sha256(
        {
            "schema_version": BIT_CONTENT_SCHEMA_VERSION,
            "kind": BIT_CONTENT_KIND,
            "bit_count": bit_count,
            "packed_hex": packed_hex,
        }
    )


def _validate_selector_params(params: Mapping[str, Any]) -> Tuple[List[str], str]:
    raw_ids = params.get("item_ids") or []
    if not isinstance(raw_ids, list):
        raise OperationError("item_ids must be an ordered list")
    item_ids = [_identifier(value, "item_ids[%d]" % index) for index, value in enumerate(raw_ids)]
    if len({value.casefold() for value in item_ids}) != len(item_ids):
        raise OperationError("item_ids must not contain duplicates")
    raw_selection = params.get("selection") or ""
    if not isinstance(raw_selection, str):
        raise OperationError("selection must be a string")
    selection = raw_selection.strip()
    if selection != raw_selection:
        raise OperationError("selection may not contain surrounding whitespace")
    if selection:
        _identifier(selection, "selection")
    if bool(item_ids) == bool(selection):
        raise OperationError(
            "Specify exactly one of non-empty item_ids or selection"
        )
    return item_ids, selection


def _manifest_path(raw_path: str, *, require_file: bool) -> Path:
    if not isinstance(raw_path, str) or not raw_path or raw_path != raw_path.strip():
        raise OperationError("manifest_path must be a non-empty path without surrounding whitespace")
    if "\x00" in raw_path:
        raise OperationError("manifest_path contains a NUL byte")
    parsed = urlsplit(raw_path)
    if parsed.scheme or parsed.netloc or raw_path.startswith("//"):
        raise OperationError("manifest_path must be a local path without a URI scheme")
    candidate = Path(raw_path).expanduser()
    if candidate.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise OperationError("manifest_path must use a .json, .yaml, or .yml suffix")
    if not require_file:
        return candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise OperationError("Bit payload manifest is missing or unsafe: %s" % candidate)
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise OperationError("Cannot resolve bit payload manifest: %s" % candidate) from exc


def _read_manifest_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise OperationError("Bit payload manifest is missing or unsafe: %s" % path)
    try:
        size = path.stat().st_size
        if size < 1:
            raise OperationError("Bit payload manifest is empty: %s" % path)
        if size > MAX_BIT_MANIFEST_BYTES:
            raise OperationError(
                "Bit payload manifest exceeds the %d-byte limit"
                % MAX_BIT_MANIFEST_BYTES
            )
        raw = path.read_bytes()
    except OperationError:
        raise
    except OSError as exc:
        raise OperationError("Cannot read bit payload manifest: %s" % path) from exc
    if len(raw) != size:
        raise OperationError("Bit payload manifest changed while reading")
    return raw


def _decode_and_validate_manifest(path: Path, raw: bytes) -> JsonDict:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OperationError("Bit payload manifest must be UTF-8: %s" % path) from exc
    try:
        payload = decode_strict_yaml_or_json(
            text,
            input_format=path.suffix.lower().lstrip("."),
        )
    except (TypeError, ValueError) as exc:
        raise OperationError(
            "Cannot parse bit payload manifest %s: %s" % (path, exc)
        ) from exc
    if not isinstance(payload, Mapping):
        raise OperationError("Bit payload manifest must be a mapping")
    manifest = dict(payload)
    _require_exact_keys(manifest, _MANIFEST_KEYS, "Bit payload manifest")
    if manifest["schema_version"] != BIT_MANIFEST_SCHEMA_VERSION:
        raise OperationError("Bit payload manifest schema_version must be 1")
    if manifest["kind"] != BIT_MANIFEST_KIND:
        raise OperationError(
            "Bit payload manifest kind must be %s" % BIT_MANIFEST_KIND
        )
    manifest["id"] = _identifier(manifest["id"], "manifest id")
    version = manifest["version"]
    if not isinstance(version, str) or not version or version != version.strip():
        raise OperationError("Bit payload manifest version must be a non-empty string")
    if manifest["encoding"] != BIT_PACKING:
        raise OperationError(
            "Bit payload manifest encoding must be %s" % BIT_PACKING
        )

    raw_items = manifest["items"]
    if not isinstance(raw_items, list) or not raw_items:
        raise OperationError("Bit payload manifest items must be a non-empty list")
    if len(raw_items) > MAX_BIT_MANIFEST_ITEMS:
        raise OperationError(
            "Bit payload manifest contains more than %d items"
            % MAX_BIT_MANIFEST_ITEMS
        )
    items: List[JsonDict] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    total_bits = 0
    for index, raw_item in enumerate(raw_items):
        item = _validate_item(raw_item, index)
        folded_id = str(item["item_id"]).casefold()
        if folded_id in seen_ids:
            raise OperationError("Bit payload manifest contains duplicate item IDs")
        content_sha256 = str(item["content_sha256"])
        if content_sha256 in seen_hashes:
            raise OperationError(
                "Bit payload manifest contains duplicate payload content"
            )
        seen_ids.add(folded_id)
        seen_hashes.add(content_sha256)
        total_bits += int(item["bit_count"])
        if total_bits > MAX_BIT_MANIFEST_OUTPUT_BITS:
            raise OperationError(
                "Bit payload manifest exceeds the %d-bit content limit"
                % MAX_BIT_MANIFEST_OUTPUT_BITS
            )
        items.append(item)

    ordered_ids = _identifier_list(
        manifest["ordered_item_ids"], "ordered_item_ids", allow_empty=False
    )
    actual_order = [str(item["item_id"]) for item in items]
    if ordered_ids != actual_order:
        raise OperationError(
            "Bit payload manifest ordered_item_ids must exactly match items order"
        )
    splits = _validate_splits(manifest["splits"], actual_order)
    manifest["ordered_item_ids"] = ordered_ids
    manifest["splits"] = splits
    manifest["items"] = items
    return manifest


def _validate_item(raw_item: Any, index: int) -> JsonDict:
    if not isinstance(raw_item, Mapping):
        raise OperationError("Bit payload manifest item %d must be a mapping" % index)
    item = dict(raw_item)
    _require_exact_keys(item, _ITEM_KEYS, "Bit payload manifest item %d" % index)
    item_id = _identifier(item["item_id"], "item %d item_id" % index)
    group_id = _identifier(item["group_id"], "item %s group_id" % item_id)
    bit_count = item["bit_count"]
    if type(bit_count) is not int or bit_count < 1:
        raise OperationError("Bit payload manifest item %s bit_count must be a positive integer" % item_id)
    if bit_count > MAX_BIT_MANIFEST_OUTPUT_BITS:
        raise OperationError("Bit payload manifest item %s is too large" % item_id)
    packed_hex = item["packed_hex"]
    if not isinstance(packed_hex, str) or _HEX_RE.fullmatch(packed_hex) is None:
        raise OperationError(
            "Bit payload manifest item %s packed_hex must be non-empty lowercase hexadecimal"
            % item_id
        )
    expected_hex_length = 2 * ((bit_count + 7) // 8)
    if len(packed_hex) != expected_hex_length:
        raise OperationError(
            "Bit payload manifest item %s packed_hex length does not match bit_count"
            % item_id
        )
    packed = bytes.fromhex(packed_hex)
    remainder = bit_count % 8
    if remainder and packed[-1] & ((1 << (8 - remainder)) - 1):
        raise OperationError(
            "Bit payload manifest item %s has non-zero unused padding bits"
            % item_id
        )
    declared_sha256 = _require_sha256(
        item["content_sha256"], "item %s content_sha256" % item_id
    )
    actual_sha256 = bit_payload_content_sha256(bit_count, packed_hex)
    if declared_sha256 != actual_sha256:
        raise OperationError(
            "Bit payload manifest item %s content SHA-256 mismatch" % item_id
        )
    return {
        "item_id": item_id,
        "group_id": group_id,
        "bit_count": bit_count,
        "packed_hex": packed_hex,
        "content_sha256": declared_sha256,
    }


def _validate_splits(raw_splits: Any, ordered_ids: Sequence[str]) -> JsonDict:
    if not isinstance(raw_splits, Mapping) or not raw_splits:
        raise OperationError("Bit payload manifest splits must be a non-empty mapping")
    known = set(ordered_ids)
    order_index = {item_id: index for index, item_id in enumerate(ordered_ids)}
    seen_names: set[str] = set()
    assigned: Dict[str, str] = {}
    splits: JsonDict = {}
    for raw_name, raw_ids in raw_splits.items():
        name = _identifier(raw_name, "split name")
        if name.casefold() in seen_names:
            raise OperationError("Bit payload manifest contains duplicate split names")
        seen_names.add(name.casefold())
        split_ids = _identifier_list(raw_ids, "split %s" % name, allow_empty=False)
        unknown = [item_id for item_id in split_ids if item_id not in known]
        if unknown:
            raise OperationError(
                "Bit payload manifest split %s contains unknown item ID %s"
                % (name, unknown[0])
            )
        if split_ids != sorted(split_ids, key=order_index.__getitem__):
            raise OperationError(
                "Bit payload manifest split %s does not preserve manifest item order"
                % name
            )
        for item_id in split_ids:
            if item_id in assigned:
                raise OperationError(
                    "Bit payload manifest item %s appears in multiple splits"
                    % item_id
                )
            assigned[item_id] = name
        splits[name] = split_ids
    missing = [item_id for item_id in ordered_ids if item_id not in assigned]
    if missing:
        raise OperationError(
            "Bit payload manifest item %s is not assigned to a split" % missing[0]
        )
    return splits


def _select_records(
    manifest: Mapping[str, Any],
    *,
    explicit_ids: Sequence[str],
    selection: str,
) -> Tuple[List[JsonDict], List[str]]:
    ordered_ids = [str(value) for value in manifest["ordered_item_ids"]]
    if selection:
        splits = manifest["splits"]
        if selection not in splits:
            raise OperationError(
                "Bit payload manifest has no selection %s" % selection
            )
        selected_ids = [str(value) for value in splits[selection]]
    else:
        selected_ids = list(explicit_ids)
        known = set(ordered_ids)
        unknown = [item_id for item_id in selected_ids if item_id not in known]
        if unknown:
            raise OperationError(
                "Unknown bit payload manifest item ID: %s" % unknown[0]
            )
        order_index = {item_id: index for index, item_id in enumerate(ordered_ids)}
        if selected_ids != sorted(selected_ids, key=order_index.__getitem__):
            raise OperationError(
                "item_ids must preserve the manifest's declared item order"
            )
    by_id = {str(row["item_id"]): dict(row) for row in manifest["items"]}
    split_by_id = {
        str(item_id): str(split_id)
        for split_id, ids in manifest["splits"].items()
        for item_id in ids
    }
    return (
        [by_id[item_id] for item_id in selected_ids],
        [split_by_id[item_id] for item_id in selected_ids],
    )


def _unpack_record(record: Mapping[str, Any]) -> np.ndarray:
    packed = np.frombuffer(bytes.fromhex(str(record["packed_hex"])), dtype=np.uint8)
    return np.unpackbits(packed, bitorder="big")[: int(record["bit_count"])]


def _pack_bits_hex(bits: np.ndarray) -> str:
    return np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").tobytes().hex()


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise OperationError(
            "%s must match %s" % (field, _IDENTIFIER_RE.pattern)
        )
    return value


def _identifier_list(value: Any, field: str, *, allow_empty: bool) -> List[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise OperationError("%s must be a non-empty ordered list" % field)
    values = [_identifier(item, "%s[%d]" % (field, index)) for index, item in enumerate(value)]
    if len({item.casefold() for item in values}) != len(values):
        raise OperationError("%s must not contain duplicate IDs" % field)
    return values


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise OperationError("%s is missing field(s): %s" % (label, ", ".join(missing)))
    if unknown:
        raise OperationError("%s has unknown field(s): %s" % (label, ", ".join(unknown)))


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OperationError("%s must be a lowercase SHA-256" % field)
    return value
