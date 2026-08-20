#!/usr/bin/env python3
"""Inventory the official BSDS500 train, validation, and test image splits.

The script verifies the source archive, records every image by SHA-256, and
writes a local retrieve-only manifest. It does not redistribute dataset bytes
or mark the dataset as publication-ready.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.ops.source.image_dataset import _load_image_manifest


BSDS500_ARCHIVE_SHA256 = (
    "97e49d31764f3912f0c4122707d53062ac9e783ba0f095e447a4d53c1a41af8e"
)
EXPECTED_COUNTS = {"train": 200, "val": 100, "test": 200}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path, split: str) -> List[Path]:
    directory = root / split
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("BSDS500 split is missing or unsafe: %s" % directory)
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if len(files) != EXPECTED_COUNTS[split]:
        raise RuntimeError(
            "BSDS500 %s requires %d JPEG images, found %d"
            % (split, EXPECTED_COUNTS[split], len(files))
        )
    return files


def _sample(split: str, path: Path) -> Dict[str, Any]:
    digest = _sha256(path)
    sample_id = "bsds500:%s:%s" % (split, path.stem)
    transform = {"name": "identity"}
    return {
        "sample_id": sample_id,
        "path": "%s/%s" % (split, path.name),
        "sha256": digest,
        "source_id": sample_id,
        "group_id": sample_id,
        "source_sha256": digest,
        "ancestry_ids": [sample_id],
        "transform": transform,
        "transform_fingerprint_sha256": canonical_json_sha256(transform),
    }


def _ids(rows: Iterable[Dict[str, Any]]) -> List[str]:
    return [str(row["sample_id"]) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--images-root",
        required=True,
        type=Path,
        help="Directory containing the official train/, val/, and test/ folders.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if _sha256(args.archive) != BSDS500_ARCHIVE_SHA256:
        raise RuntimeError("BSDS500 archive SHA-256 mismatch")
    rows_by_split = {
        split: [_sample(split, path) for path in _inventory(args.images_root, split)]
        for split in ("train", "val", "test")
    }
    output_parent = args.output.parent.resolve()
    try:
        relative_root = args.images_root.resolve().relative_to(output_parent)
    except ValueError as exc:
        raise RuntimeError(
            "Manifest output must be in an ancestor directory of the image root"
        ) from exc
    samples = [
        row
        for split in ("train", "val", "test")
        for row in rows_by_split[split]
    ]
    manifest = {
        "schema_version": 1,
        "kind": "noema.image_dataset_manifest",
        "id": "bsds500",
        "version": "bsds500-2013-official-splits",
        "root": relative_root.as_posix(),
        "source_page": (
            "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/"
            "grouping/resources.html"
        ),
        "license": {
            "license_id": (
                "Berkeley BSDS non-commercial research and educational use terms"
            ),
            "authoritative_source": "UC Berkeley Computer Vision Group",
            "terms_locator": (
                "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/"
            ),
            "redistribution": "not_explicitly_granted",
            "archive_disposition": "retrieve_only",
            "canonical_retrieval": (
                "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/"
                "grouping/BSR/BSR_bsds500.tgz"
            ),
            "archive_sha256": BSDS500_ARCHIVE_SHA256,
        },
        "publication_ready": False,
        "publication_blocker": (
            "Retrieve-only candidate: independent rights/privacy review and "
            "protocol freeze remain open."
        ),
        "splits": {
            "train": _ids(rows_by_split["train"]),
            "validation": _ids(rows_by_split["val"]),
            "training_all": _ids(rows_by_split["train"])
            + _ids(rows_by_split["val"]),
            "publication_test": _ids(rows_by_split["test"]),
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _load_image_manifest(args.output.resolve(), expected_dataset="bsds500")
    print("BSDS500 all-split manifest SHA-256: %s" % _sha256(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
