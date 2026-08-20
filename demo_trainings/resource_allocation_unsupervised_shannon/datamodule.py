from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


class ChannelGainCaptureDataset(Dataset):
    def __init__(self, gains: np.ndarray, capture_sha256: Sequence[str], capture_dirs: Sequence[Path]) -> None:
        values = np.asarray(gains, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError(f"channel_gains must have shape [sample, subcarrier], got {values.shape}")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("channel_gains contain non-finite or negative values")
        self.gains = values
        self.capture_sha256 = list(capture_sha256)
        self.capture_dirs = [Path(path) for path in capture_dirs]

    def __len__(self) -> int:
        return int(self.gains.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.gains[index].copy())


def load_capture_dataset(
    paths: str | Path | Sequence[str | Path],
    *,
    feature_tap: str = "channel_gains",
    expected_split: str | None = None,
) -> ChannelGainCaptureDataset:
    capture_dirs = _path_list(paths)
    if not capture_dirs:
        raise ValueError("At least one dataset-capture directory is required")
    chunks: list[np.ndarray] = []
    hashes: list[str] = []
    width: int | None = None
    for capture_dir in capture_dirs:
        schema_path = capture_dir / "schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError(f"Noema capture schema is missing: {schema_path}")
        schema = load_strict_yaml_or_json(schema_path)
        if schema.get("kind") != "noema.capture_dataset" or int(schema.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported capture schema in {schema_path}")
        split = str(schema.get("split") or "")
        if expected_split and split != expected_split:
            raise ValueError(f"Expected capture split {expected_split!r}, got {split!r} in {schema_path}")
        tap_schema = dict((schema.get("tap_schemas") or {}).get(feature_tap) or {})
        if not tap_schema:
            raise KeyError(f"Capture {capture_dir} does not declare tap {feature_tap!r}")
        record_shape = list(tap_schema.get("record_shape") or [])
        if len(record_shape) != 1 or int(record_shape[0]) < 1:
            raise ValueError(f"Tap {feature_tap!r} must contain one gain vector per record; got {record_shape}")
        if width is None:
            width = int(record_shape[0])
        elif width != int(record_shape[0]):
            raise ValueError("All captures must use the same subcarrier count")
        shard_records = list(schema.get("shards") or [])
        if not shard_records:
            raise ValueError(f"Capture has no shards: {capture_dir}")
        for record in shard_records:
            relative = record.get("path") if isinstance(record, dict) else record
            shard_path = capture_dir / str(relative)
            if not shard_path.is_file():
                raise FileNotFoundError(f"Capture shard is missing: {shard_path}")
            with np.load(str(shard_path), allow_pickle=False) as payload:
                if feature_tap not in payload.files:
                    raise KeyError(f"{shard_path} does not contain tap {feature_tap!r}")
                values = np.asarray(payload[feature_tap], dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != width:
                raise ValueError(f"Expected {feature_tap} [N,{width}] in {shard_path}, got {values.shape}")
            chunks.append(values)
        hashes.append(_sha256(schema_path))
    return ChannelGainCaptureDataset(np.concatenate(chunks, axis=0), hashes, capture_dirs)


def build_loader(
    dataset: ChannelGainCaptureDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        num_workers=max(0, int(num_workers)),
        generator=generator,
    )


def feature_statistics(
    dataset: ChannelGainCaptureDataset,
    *,
    reference_noise_variance: float,
    reference_average_power_budget: float,
    eps: float = 1e-12,
) -> tuple[float, float]:
    values = np.log(
        np.maximum(
            dataset.gains.astype(np.float64)
            * float(reference_average_power_budget)
            / float(reference_noise_variance),
            float(eps),
        )
    )
    return float(np.mean(values)), max(float(np.std(values)), 1e-6)


def _path_list(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        items: Iterable[str | Path] = [paths]
    else:
        items = paths
    result = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if "<" in text or ">" in text:
            raise ValueError(f"Replace capture placeholder before training: {text}")
        result.append(Path(text).expanduser())
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
