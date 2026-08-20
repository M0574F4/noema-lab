from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml

from noema_lab.core.recipes import Recipe, RecipeStep
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.training_plans import scenario_recipe_fingerprint

JsonDict = Dict[str, Any]


class ImageDataContractError(ValueError):
    pass


@dataclass(frozen=True)
class ImageFileRecord:
    image_id: str
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ImageFileDataPlan:
    recipe: Recipe
    recipe_sha256: str
    source_step: RecipeStep
    project_root: Path
    dataset_id: str
    dataset_dir: Path
    selected_image_ids: Tuple[str, ...]
    train_files: Tuple[ImageFileRecord, ...]
    validation_files: Tuple[ImageFileRecord, ...]
    crop_size: int
    repeat_count: int
    validation_count: int
    partition_policy_id: str
    manifest_path: str
    manifest_sha256: str
    manifest_split: str


def build_image_file_data_plan(
    recipe: Recipe,
    source_step: RecipeStep,
    *,
    project_root: Path,
) -> ImageFileDataPlan:
    """Resolve a recipe-selected image set into an immutable train/validation partition.

    Resolution is deliberately based on the generic file-facing parameters of an image
    source.  Dataset-specific catalogs remain the source operation's responsibility.
    """

    params = dict(source_step.params or {})
    root = Path(project_root).resolve()
    directory = Path(str(params.get("dataset_dir") or ".")).expanduser()
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.resolve()
    manifest_path_value = str(params.get("manifest_path") or "").strip()
    manifest_sha256 = str(params.get("manifest_sha256") or "").strip().lower()
    manifest_split = str(params.get("split") or "").strip()
    if manifest_path_value:
        from noema_lab.ops.source.image_dataset import (
            DEFAULT_KODAK_IMAGE_IDS,
            _load_image_manifest,
            _manifest_records_and_paths,
            _manifest_selection,
        )

        manifest_path = Path(manifest_path_value).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ImageDataContractError(
                "Image training manifest is missing or unsafe: %s" % manifest_path
            )
        manifest_path = manifest_path.resolve()
        observed_manifest_sha256 = _file_sha256(manifest_path)
        if (
            len(manifest_sha256) != 64
            or any(char not in "0123456789abcdef" for char in manifest_sha256)
            or observed_manifest_sha256 != manifest_sha256
        ):
            raise ImageDataContractError(
                "Image training manifest failed exact-file SHA-256 verification"
            )
        manifest = _load_image_manifest(
            manifest_path,
            expected_dataset=str(params.get("dataset") or "image_files"),
        )
        raw_image_ids = params.get("image_ids")
        if (
            str(raw_image_ids or "") == DEFAULT_KODAK_IMAGE_IDS
            and str(params.get("dataset") or "") != "kodak"
        ):
            raw_image_ids = ""
        image_ids = _manifest_selection(
            manifest,
            ",".join(_image_ids(raw_image_ids, allow_empty=True)),
            manifest_split,
        )
        manifest_records, manifest_paths = _manifest_records_and_paths(
            manifest,
            manifest_path,
            image_ids,
        )
        records = []
        seen_paths = set()
        for image_id, record, path in zip(
            image_ids, manifest_records, manifest_paths
        ):
            normalized = str(path)
            if normalized in seen_paths:
                raise ImageDataContractError(
                    "Image selection %s resolves more than once to %s"
                    % (source_step.id, path)
                )
            seen_paths.add(normalized)
            digest = _file_sha256(path)
            if digest != str(record.get("sha256") or ""):
                raise ImageDataContractError(
                    "Training image %s failed manifest SHA-256 verification"
                    % image_id
                )
            records.append(
                ImageFileRecord(
                    image_id=image_id,
                    path=path,
                    sha256=digest,
                    size_bytes=int(path.stat().st_size),
                )
            )
        directory = manifest_path.parent
    else:
        image_ids = _image_ids(params.get("image_ids"))
        records = []
        seen_paths = set()
        for image_id in image_ids:
            path = _resolve_image_path(directory, image_id)
            normalized = str(path)
            if normalized in seen_paths:
                raise ImageDataContractError(
                    "Image selection %s resolves more than once to %s"
                    % (source_step.id, path)
                )
            seen_paths.add(normalized)
            records.append(
                ImageFileRecord(
                    image_id=image_id,
                    path=path,
                    sha256=_file_sha256(path),
                    size_bytes=int(path.stat().st_size),
                )
            )
    if len(image_ids) < 2:
        raise ImageDataContractError(
            "File-backed image training requires at least two selected image IDs so "
            "train and validation are non-empty; %s selects %d."
            % (source_step.id, len(image_ids))
        )

    requested_validation_count = int(
        params.get("training_validation_count") or 0
    )
    if requested_validation_count:
        if requested_validation_count >= len(records):
            raise ImageDataContractError(
                "training_validation_count must leave at least one training image"
            )
        validation_count = requested_validation_count
        partition_policy_id = "ordered_recipe_selection_explicit_validation_count_v1"
    else:
        validation_count = max(1, len(records) // 5)
        partition_policy_id = "ordered_recipe_selection_last_fifth_validation_v1"
    train = tuple(records[:-validation_count])
    validation = tuple(records[-validation_count:])
    if not train or not validation:
        raise ImageDataContractError("Image train and validation partitions must both be non-empty")

    crop_size = int(params.get("crop_size") or 64)
    if crop_size < 4 or crop_size % 4:
        raise ImageDataContractError(
            "DeepJSCC image crop_size must be at least four and divisible by four; got %d"
            % crop_size
        )
    return ImageFileDataPlan(
        recipe=recipe,
        recipe_sha256=scenario_recipe_fingerprint(recipe),
        source_step=source_step,
        project_root=root,
        dataset_id=str(params.get("dataset") or "image_files"),
        dataset_dir=directory,
        selected_image_ids=tuple(image_ids),
        train_files=train,
        validation_files=validation,
        crop_size=crop_size,
        repeat_count=max(1, int(params.get("repeat_count") or 1)),
        validation_count=validation_count,
        partition_policy_id=partition_policy_id,
        manifest_path=manifest_path_value,
        manifest_sha256=manifest_sha256,
        manifest_split=manifest_split,
    )


def write_image_file_data_contract(
    plan: ImageFileDataPlan,
    out_dir: Path,
) -> JsonDict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset: JsonDict = {
        "id": plan.dataset_id,
        "source_step": plan.source_step.id,
        "source_operation": plan.source_step.op,
        "dataset_dir": str(plan.dataset_dir),
        "selected_image_ids": list(plan.selected_image_ids),
    }
    if plan.manifest_path:
        dataset["manifest"] = {
            "path": plan.manifest_path,
            "sha256": plan.manifest_sha256,
            "split": plan.manifest_split,
        }
    if plan.dataset_id == "kodak":
        from noema_lab.ops.source.kodak import KODAK_DATASET_PROVENANCE

        dataset["provenance"] = dict(KODAK_DATASET_PROVENANCE)
    contract: JsonDict = {
        "schema_version": 1,
        "kind": "noema.training_data_contract@1",
        "mode": "file_backed_live_differentiable",
        "source_recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
        },
        "ownership": {
            "source_selection": "noema_recipe",
            "partition_materialization": "noema_export",
            "file_integrity_validation": "noema_contract_and_external_trainer",
            "dataset_consumption": "external_researcher",
            "model_training": "external_researcher",
            "test_evaluation": "noema_ordinary_recipe_or_benchmark",
        },
        "dataset": dataset,
        "preprocessing": {
            "color": "rgb",
            "crop": "deterministic_center_crop_or_pad",
            "crop_size": plan.crop_size,
            "training_tensor": {
                "dtype": "float32",
                "layout": "NCHW",
                "domain": "normalized_[0,1]",
            },
            "repeat_count": plan.repeat_count,
        },
        "partition_policy": {
            "id": plan.partition_policy_id,
            "ordering": "recipe_image_ids",
            "validation_count": plan.validation_count,
            "selection_or_model_fitting_uses_test": False,
        },
        "splits": [
            _split_payload("train", "optimization", plan.train_files, plan.project_root),
            _split_payload(
                "validation",
                "checkpoint_selection_only",
                plan.validation_files,
                plan.project_root,
            ),
        ],
        "test_evaluation": {
            "owner": "noema_ordinary_recipe_or_benchmark",
            "included_in_training_bundle": False,
            "image_ids": [],
            "rule": (
                "Held-out test images must be selected by a separate ordinary artifact-bound "
                "Noema recipe and must not be exposed to train.py."
            ),
        },
    }
    path = out_dir / "data_contract.yaml"
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    return {
        "data_contract": contract,
        "data_contract_sha256": canonical_json_sha256(contract),
        "data_contract_file_sha256": _file_sha256(path),
        "capture_jobs": [],
        "files": ["data_contract.yaml"],
    }


def split_image_ids(contract: Mapping[str, Any], split: str) -> List[str]:
    for row in list(contract.get("splits") or []):
        if isinstance(row, Mapping) and str(row.get("id") or "") == split:
            return [
                str(item.get("image_id") or "")
                for item in list(row.get("files") or [])
                if isinstance(item, Mapping) and str(item.get("image_id") or "")
            ]
    return []


def _split_payload(
    split: str,
    use: str,
    records: Sequence[ImageFileRecord],
    project_root: Path,
) -> JsonDict:
    return {
        "id": split,
        "training_use": use,
        "image_count": len(records),
        "files": [_record_payload(item, project_root) for item in records],
    }


def _record_payload(record: ImageFileRecord, project_root: Path) -> JsonDict:
    payload: JsonDict = {
        "image_id": record.image_id,
        "resolved_path": str(record.path),
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
    }
    try:
        payload["project_relative_path"] = str(record.path.relative_to(project_root))
    except ValueError:
        pass
    return payload


def _image_ids(value: Any, *, allow_empty: bool = False) -> List[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [str(item).strip() for item in value if str(item).strip()]
    else:
        values = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not values and not allow_empty:
        raise ImageDataContractError("image_ids must explicitly select at least two image files")
    return values


def _resolve_image_path(directory: Path, image_id: str) -> Path:
    supplied = Path(image_id).expanduser()
    candidates: List[Path] = []
    if supplied.is_absolute():
        candidates.append(supplied)
    else:
        candidates.append(directory / supplied)
    if not supplied.suffix:
        candidates.extend(
            directory / (image_id + extension)
            for extension in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ImageDataContractError(
        "Selected image %r was not found beneath %s" % (image_id, directory)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
