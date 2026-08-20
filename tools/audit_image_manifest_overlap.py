#!/usr/bin/env python3
"""Audit exact and perceptual overlap between two Noema image manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
from PIL import Image

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.ops.source.image_dataset import (
    _center_crop,
    _image_identity_sha256,
    _load_image_manifest,
    _manifest_records_and_paths,
    _manifest_selection,
    _read_image_rgb,
    _resize_shorter_side,
)


def _difference_hash(image: np.ndarray) -> int:
    grayscale = Image.fromarray(image, mode="RGB").convert("L").resize(
        (9, 8),
        resample=Image.Resampling.LANCZOS,
    )
    values = np.asarray(grayscale, dtype=np.uint8)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.reshape(-1):
        result = (result << 1) | int(bit)
    return result


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _load_population(
    *,
    manifest_path: Path,
    dataset_id: str,
    split: str,
    resize_shorter_side: int,
    crop_size: int,
) -> Tuple[Mapping[str, Any], List[Dict[str, Any]]]:
    expected_manifest_sha256 = file_sha256(manifest_path)
    manifest = _load_image_manifest(
        manifest_path.resolve(),
        expected_dataset=dataset_id,
    )
    sample_ids = _manifest_selection(manifest, "", split)
    records, paths = _manifest_records_and_paths(
        manifest,
        manifest_path.resolve(),
        sample_ids,
    )
    rows: List[Dict[str, Any]] = []
    for sample_id, record, path in zip(sample_ids, records, paths):
        image = _center_crop(
            _resize_shorter_side(
                _read_image_rgb(path),
                resize_shorter_side,
            ),
            crop_size,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "file_sha256": str(record["sha256"]),
                "post_transform_sha256": _image_identity_sha256(image),
                "difference_hash": _difference_hash(image),
            }
        )
    identity = {
        "manifest_sha256": expected_manifest_sha256,
        "split": split,
        "resize_shorter_side": resize_shorter_side,
        "crop_size": crop_size,
        "ordered_sample_ids": sample_ids,
    }
    return identity, rows


def audit_overlap(
    *,
    left_manifest: Path,
    left_dataset_id: str,
    right_manifest: Path,
    right_dataset_id: str,
    split: str,
    resize_shorter_side: int,
    crop_size: int,
    perceptual_threshold: int,
) -> Dict[str, Any]:
    left_identity, left_rows = _load_population(
        manifest_path=left_manifest,
        dataset_id=left_dataset_id,
        split=split,
        resize_shorter_side=resize_shorter_side,
        crop_size=crop_size,
    )
    right_identity, right_rows = _load_population(
        manifest_path=right_manifest,
        dataset_id=right_dataset_id,
        split=split,
        resize_shorter_side=resize_shorter_side,
        crop_size=crop_size,
    )

    exact_file_pairs = [
        {
            "left_sample_id": left["sample_id"],
            "right_sample_id": right["sample_id"],
            "sha256": left["file_sha256"],
        }
        for left in left_rows
        for right in right_rows
        if left["file_sha256"] == right["file_sha256"]
    ]
    exact_transform_pairs = [
        {
            "left_sample_id": left["sample_id"],
            "right_sample_id": right["sample_id"],
            "sha256": left["post_transform_sha256"],
        }
        for left in left_rows
        for right in right_rows
        if left["post_transform_sha256"] == right["post_transform_sha256"]
    ]

    perceptual_pairs: List[Dict[str, Any]] = []
    minimum_hamming_distance = 64
    minimum_pair: Dict[str, Any] | None = None
    for left in left_rows:
        for right in right_rows:
            distance = _hamming(
                int(left["difference_hash"]),
                int(right["difference_hash"]),
            )
            if distance < minimum_hamming_distance:
                minimum_hamming_distance = distance
                minimum_pair = {
                    "left_sample_id": left["sample_id"],
                    "right_sample_id": right["sample_id"],
                    "hamming_distance": distance,
                }
            if distance <= perceptual_threshold:
                perceptual_pairs.append(
                    {
                        "left_sample_id": left["sample_id"],
                        "right_sample_id": right["sample_id"],
                        "hamming_distance": distance,
                    }
                )

    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "noema.image_manifest_overlap_audit",
        "status": "candidate_pre_freeze_check",
        "left": {
            **left_identity,
            "dataset_id": left_dataset_id,
            "sample_count": len(left_rows),
        },
        "right": {
            **right_identity,
            "dataset_id": right_dataset_id,
            "sample_count": len(right_rows),
        },
        "exact_file_overlap": {
            "count": len(exact_file_pairs),
            "pairs": exact_file_pairs,
        },
        "exact_post_transform_overlap": {
            "count": len(exact_transform_pairs),
            "pairs": exact_transform_pairs,
        },
        "difference_hash_audit": {
            "algorithm": "64-bit horizontal difference hash over transformed RGB",
            "threshold": perceptual_threshold,
            "candidate_pair_count": len(perceptual_pairs),
            "candidate_pairs": perceptual_pairs,
            "minimum_hamming_distance": minimum_hamming_distance,
            "minimum_pair": minimum_pair,
            "interpretation": (
                "A candidate pair requires human review; absence is not proof "
                "against every possible transformed or source-level duplicate."
            ),
        },
    }
    report["audit_sha256"] = canonical_json_sha256(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-manifest", required=True, type=Path)
    parser.add_argument("--left-dataset-id", required=True)
    parser.add_argument("--right-manifest", required=True, type=Path)
    parser.add_argument("--right-dataset-id", required=True)
    parser.add_argument("--split", default="publication_test")
    parser.add_argument("--resize-shorter-side", type=int, default=128)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--perceptual-threshold", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.perceptual_threshold < 0 or args.perceptual_threshold > 64:
        raise RuntimeError("perceptual threshold must be between 0 and 64")
    report = audit_overlap(
        left_manifest=args.left_manifest,
        left_dataset_id=args.left_dataset_id,
        right_manifest=args.right_manifest,
        right_dataset_id=args.right_dataset_id,
        split=args.split,
        resize_shorter_side=args.resize_shorter_side,
        crop_size=args.crop_size,
        perceptual_threshold=args.perceptual_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
