from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_data_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(config.get("data") or {})
    configured = str(data.get("contract_path") or "").strip()
    if not configured:
        raise ValueError(
            "data.contract_path is required; train from a Noema export with an "
            "auditable file-backed data contract"
        )
    path = Path(configured).expanduser()
    if not path.is_file():
        raise FileNotFoundError("DeepJSCC data contract does not exist: %s" % path)
    expected_file_sha = str(data.get("contract_file_sha256") or "").strip()
    actual_file_sha = _file_sha256(path)
    if not expected_file_sha or actual_file_sha != expected_file_sha:
        raise ValueError("DeepJSCC data contract failed exact-file SHA-256 verification")
    payload = load_strict_yaml_or_json(path)
    if not isinstance(payload, dict):
        raise ValueError("DeepJSCC data contract must contain a mapping")
    if str(payload.get("kind") or "") != "noema.training_data_contract@1":
        raise ValueError("DeepJSCC data contract has an unsupported kind")
    if str(payload.get("mode") or "") != "file_backed_live_differentiable":
        raise ValueError("DeepJSCC data contract must use file_backed_live_differentiable mode")
    expected_semantic_sha = str(data.get("contract_sha256") or "").strip()
    if not expected_semantic_sha or _canonical_sha256(payload) != expected_semantic_sha:
        raise ValueError("DeepJSCC data contract failed semantic SHA-256 verification")
    split_ids = {
        str(row.get("id") or "")
        for row in list(payload.get("splits") or [])
        if isinstance(row, Mapping)
    }
    if split_ids != {"train", "validation"}:
        raise ValueError(
            "DeepJSCC training data contract must expose exactly train and validation; "
            "held-out test belongs to an ordinary Noema recipe"
        )
    test = dict(payload.get("test_evaluation") or {})
    if bool(test.get("included_in_training_bundle")) or list(test.get("image_ids") or []):
        raise ValueError("DeepJSCC train.py must not receive held-out test images")
    if bool(data.get("test_images_exposed_to_trainer", True)):
        raise ValueError("data.test_images_exposed_to_trainer must remain false")
    return payload


class DeepJSCCImageDataset(Dataset):
    def __init__(self, config: Mapping, split: str):
        data = dict(config.get("data") or {})
        self.split = str(split)
        contract = load_data_contract(config)
        split_row = next(
            (
                dict(row)
                for row in list(contract.get("splits") or [])
                if isinstance(row, Mapping) and str(row.get("id") or "") == self.split
            ),
            None,
        )
        if split_row is None:
            raise ValueError("Unknown DeepJSCC data split: %s" % self.split)
        records = [dict(item) for item in list(split_row.get("files") or [])]
        if not records:
            raise ValueError("DeepJSCC %s split is empty" % self.split)
        expected_ids = [str(item.get("image_id") or "") for item in records]
        configured_ids = [
            str(item) for item in list(data.get("%s_image_ids" % self.split) or [])
        ]
        if configured_ids != expected_ids:
            raise ValueError(
                "data.%s_image_ids disagrees with the hash-pinned data contract"
                % self.split
            )
        preprocessing = dict(contract.get("preprocessing") or {})
        self.crop_size = int(preprocessing.get("crop_size") or 0)
        if self.crop_size < 8 or self.crop_size % 8:
            raise ValueError("data contract crop_size must be a positive multiple of eight")
        self.repeat_count = max(1, int(preprocessing.get("repeat_count") or 1))
        self.paths: List[Path] = []
        for record in records:
            path = Path(str(record.get("resolved_path") or "")).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    "Contract image %s does not exist: %s"
                    % (record.get("image_id"), path)
                )
            expected_sha = str(record.get("sha256") or "")
            if not expected_sha or _file_sha256(path) != expected_sha:
                raise ValueError(
                    "Contract image %s failed SHA-256 verification"
                    % record.get("image_id")
                )
            self.paths.append(path)

    def __len__(self) -> int:
        return len(self.paths) * self.repeat_count

    def __getitem__(self, index: int):
        image = np.asarray(
            Image.open(self.paths[index % len(self.paths)]).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        image = self._crop_or_pad(image, index)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        return {"image": tensor}

    def _crop_or_pad(self, image: np.ndarray, index: int) -> np.ndarray:
        height, width, _channels = image.shape
        crop_height = min(self.crop_size, height)
        crop_width = min(self.crop_size, width)
        if self.split == "train" and self.repeat_count > 1:
            token = "%s:%d:%s" % (
                self.split,
                int(index),
                self.paths[index % len(self.paths)].name,
            )
            seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            top = int(rng.integers(0, max(height - crop_height, 0) + 1))
            left = int(rng.integers(0, max(width - crop_width, 0) + 1))
        else:
            top = max(0, (height - crop_height) // 2)
            left = max(0, (width - crop_width) // 2)
        cropped = image[top : top + crop_height, left : left + crop_width, :]
        if self.split == "train" and self.repeat_count > 1:
            transform = (int(index) // len(self.paths)) % 8
            if transform & 1:
                cropped = np.flip(cropped, axis=1)
            if transform & 2:
                cropped = np.flip(cropped, axis=0)
            if transform & 4:
                cropped = np.transpose(cropped, (1, 0, 2))
        crop_height, crop_width = cropped.shape[:2]
        if crop_height == self.crop_size and crop_width == self.crop_size:
            return np.ascontiguousarray(cropped)
        padded = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.float32)
        pad_top = (self.crop_size - crop_height) // 2
        pad_left = (self.crop_size - crop_width) // 2
        padded[pad_top : pad_top + crop_height, pad_left : pad_left + crop_width, :] = cropped
        return padded


def build_loader(config: Mapping, split: str) -> DataLoader:
    training = dict(config.get("training") or {})
    dataset = DeepJSCCImageDataset(config, split)
    return DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 8)),
        shuffle=split == "train",
        num_workers=int(training.get("num_workers", 0)),
        drop_last=False,
    )


def build_train_loader(config: Mapping) -> DataLoader:
    return build_loader(config, "train")


def build_validation_loader(config: Mapping) -> DataLoader:
    return build_loader(config, "validation")
