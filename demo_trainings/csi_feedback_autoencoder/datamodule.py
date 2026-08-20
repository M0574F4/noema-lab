from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


class CsiCaptureDataset(Dataset):
    def __init__(
        self,
        csi_ri: np.ndarray,
        *,
        split: str,
        capture_sha256: Sequence[str],
        capture_dirs: Sequence[Path],
    ) -> None:
        values = np.asarray(csi_ri, dtype=np.float32)
        if values.ndim != 4 or values.shape[0] < 1 or values.shape[1] != 2:
            raise ValueError(
                "CSI must have shape [sample,2,tx_antenna,subcarrier], got %s"
                % (values.shape,)
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("CSI capture contains non-finite values")
        self.csi_ri = np.ascontiguousarray(values)
        self.split = str(split)
        self.capture_sha256 = list(capture_sha256)
        self.capture_dirs = [Path(path) for path in capture_dirs]
        self.record_fingerprints = tuple(_record_sha256(row) for row in self.csi_ri)

    def __len__(self) -> int:
        return int(self.csi_ri.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.csi_ri[index].copy())


def load_capture_dataset(
    paths: str | Path | Sequence[str | Path],
    *,
    feature_tap: str = "true_csi",
    expected_split: str | None = None,
) -> CsiCaptureDataset:
    capture_dirs = _path_list(paths)
    if not capture_dirs:
        raise ValueError("At least one dataset-capture directory is required")
    chunks: list[np.ndarray] = []
    hashes: list[str] = []
    observed_split = ""
    expected_shape: tuple[int, int, int] | None = None
    for capture_dir in capture_dirs:
        schema_path = capture_dir / "schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError("Noema capture schema is missing: %s" % schema_path)
        schema = load_strict_yaml_or_json(schema_path)
        if (
            schema.get("kind") != "noema.capture_dataset"
            or int(schema.get("schema_version", 0)) != 1
        ):
            raise ValueError("Unsupported capture schema in %s" % schema_path)
        split = str(schema.get("split") or "")
        if expected_split and split != expected_split:
            raise ValueError(
                "Expected capture split %r, got %r in %s"
                % (expected_split, split, schema_path)
            )
        if observed_split and observed_split != split:
            raise ValueError(
                "All capture directories in one dataset must use the same split"
            )
        observed_split = split
        tap_schema = dict((schema.get("tap_schemas") or {}).get(feature_tap) or {})
        if not tap_schema:
            raise KeyError(
                "Capture %s does not declare tap %r" % (capture_dir, feature_tap)
            )
        record_shape = tuple(int(item) for item in tap_schema.get("record_shape") or [])
        if len(record_shape) != 3 or record_shape[0] != 2:
            raise ValueError(
                "Tap %r must store [2,tx_antenna,subcarrier] per record; got %s"
                % (feature_tap, record_shape)
            )
        if expected_shape is None:
            expected_shape = record_shape
        elif expected_shape != record_shape:
            raise ValueError("All CSI captures must have the same record shape")
        shard_records = list(schema.get("shards") or [])
        if not shard_records:
            raise ValueError("Capture has no shards: %s" % capture_dir)
        for record in shard_records:
            relative = record.get("path") if isinstance(record, dict) else record
            shard_path = capture_dir / str(relative)
            if not shard_path.is_file():
                raise FileNotFoundError("Capture shard is missing: %s" % shard_path)
            with np.load(str(shard_path), allow_pickle=False) as payload:
                if feature_tap not in payload.files:
                    raise KeyError(
                        "%s does not contain tap %r" % (shard_path, feature_tap)
                    )
                values = np.asarray(payload[feature_tap], dtype=np.float32)
            if values.ndim != 4 or tuple(values.shape[1:]) != expected_shape:
                raise ValueError(
                    "Expected %s [N,%s] in %s, got %s"
                    % (
                        feature_tap,
                        ",".join(map(str, expected_shape)),
                        shard_path,
                        values.shape,
                    )
                )
            chunks.append(values)
        hashes.append(_sha256(schema_path))
    return CsiCaptureDataset(
        np.concatenate(chunks, axis=0),
        split=observed_split,
        capture_sha256=hashes,
        capture_dirs=capture_dirs,
    )


def assert_disjoint_datasets(*datasets: CsiCaptureDataset) -> None:
    """Fail closed if deterministic capture splits share any realized CSI record."""

    for left_index, left in enumerate(datasets):
        left_set = set(left.record_fingerprints)
        if len(left_set) != len(left.record_fingerprints):
            raise ValueError(
                "Capture split %s contains duplicate CSI records" % left.split
            )
        for right in datasets[left_index + 1 :]:
            overlap = left_set.intersection(right.record_fingerprints)
            if overlap:
                raise ValueError(
                    "Capture splits %s and %s overlap in %d CSI records"
                    % (left.split, right.split, len(overlap))
                )


def build_loader(
    dataset: CsiCaptureDataset,
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


def load_data_contract(config: Mapping[str, Any]) -> Mapping[str, Any]:
    data = dict(config.get("data") or {})
    path = Path(str(data.get("contract_path") or "../data_contract.yaml"))
    if not path.is_file():
        raise FileNotFoundError("CSI training data contract is missing: %s" % path)
    payload = load_strict_yaml_or_json(path)
    if not isinstance(payload, dict):
        raise ValueError("data_contract.yaml must contain a mapping")
    _validate_csi_data_contract(
        payload,
        feature_tap=str(data.get("feature_tap") or "true_csi"),
    )
    expected_file_hash = str(data.get("contract_file_sha256") or "")
    if expected_file_hash and _sha256(path) != expected_file_hash:
        raise ValueError(
            "CSI data-contract file hash does not match the exported bundle"
        )
    expected_hash = str(data.get("contract_sha256") or "")
    if expected_hash and _canonical_sha256(payload) != expected_hash:
        raise ValueError(
            "CSI data-contract canonical hash does not match the exported bundle"
        )
    return payload


def _validate_csi_data_contract(
    payload: Mapping[str, Any],
    *,
    feature_tap: str,
) -> None:
    """Accept the legacy CSI contract and its neutral generic-tensor equivalent."""

    mode = str(payload.get("mode") or "")
    if mode == "captured_self_supervised_csi":
        return
    if mode != "captured_generic_tensors":
        raise ValueError(
            "CSI trainer requires captured_self_supervised_csi or "
            "captured_generic_tensors data"
        )
    signals = [
        dict(item)
        for item in payload.get("signals") or []
        if isinstance(item, Mapping)
    ]
    matches = [
        item
        for item in signals
        if str(item.get("tap_id") or "") == str(feature_tap)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Generic CSI data contract must declare exactly one signal for tap %r"
            % feature_tap
        )
    signal = matches[0]
    if not bool(signal.get("required")):
        raise ValueError(
            "Generic CSI data-contract signal %r must be required" % feature_tap
        )
    if str(signal.get("kind") or "") != "channel.miso_ofdm_csi.numpy":
        raise ValueError(
            "Generic CSI data-contract signal %r has incompatible kind %r"
            % (feature_tap, signal.get("kind"))
        )
    if not str(signal.get("reference") or "").endswith(".csi"):
        raise ValueError(
            "Generic CSI data-contract signal %r must reference a CSI output"
            % feature_tap
        )


def datasets_from_config(
    config: Mapping[str, Any],
    *,
    include_test: bool,
) -> dict[str, CsiCaptureDataset]:
    data = dict(config.get("data") or {})
    feature_tap = str(data.get("feature_tap") or "true_csi")
    datasets = {
        "train": load_capture_dataset(
            data.get("train_capture_dirs") or [],
            feature_tap=feature_tap,
            expected_split="train",
        ),
        "validation": load_capture_dataset(
            data.get("validation_capture_dirs") or [],
            feature_tap=feature_tap,
            expected_split="validation",
        ),
    }
    if include_test:
        datasets["test"] = load_capture_dataset(
            data.get("test_capture_dirs") or [],
            feature_tap=feature_tap,
            expected_split="test",
        )
    assert_disjoint_datasets(*datasets.values())
    return datasets


def materialize_data_contract_inventory(
    config: Mapping[str, Any],
    *,
    config_path: str | Path = "train_config.yaml",
) -> dict[str, Any]:
    """Bind every captured split to the exact immutable shard bytes it uses.

    Dataset capture cannot know these hashes when the training contract is
    exported because the shards do not exist yet.  The trainer therefore
    freezes the inventory after capture and before it writes a trained
    artifact.  This records provenance only; it does not expose the held-out
    test tensors to training.
    """

    updated = dict(config)
    data = dict(updated.get("data") or {})
    contract_path = Path(
        str(data.get("contract_path") or "../data_contract.yaml")
    ).expanduser()
    contract = load_strict_yaml_or_json(contract_path)
    if not isinstance(contract, dict):
        raise ValueError("data_contract.yaml must contain a mapping")
    splits = contract.get("splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError("CSI data contract must declare capture splits")

    split_rows: dict[str, dict[str, Any]] = {}
    for row in splits:
        if not isinstance(row, dict):
            raise ValueError("CSI data-contract split must be a mapping")
        split_id = str(row.get("id") or "").strip()
        if not split_id or split_id in split_rows:
            raise ValueError("CSI data-contract split IDs must be unique")
        split_rows[split_id] = row

    for split_id, row in split_rows.items():
        capture_dirs = _path_list(data.get("%s_capture_dirs" % split_id) or [])
        if not capture_dirs:
            raise ValueError("No capture directory is configured for %s" % split_id)
        files: list[dict[str, Any]] = []
        captured_samples = 0
        for capture_dir in capture_dirs:
            schema_path = capture_dir / "schema.json"
            if not schema_path.is_file():
                raise FileNotFoundError(
                    "Noema capture schema is missing: %s" % schema_path
                )
            schema = load_strict_yaml_or_json(schema_path)
            if (
                not isinstance(schema, dict)
                or str(schema.get("split") or "") != split_id
            ):
                raise ValueError(
                    "Capture %s does not belong to split %s" % (capture_dir, split_id)
                )
            captured_samples += int(schema.get("captured_samples") or 0)
            shards = schema.get("shards")
            if not isinstance(shards, list) or not shards:
                raise ValueError("Capture has no shards: %s" % capture_dir)
            for shard_index, shard in enumerate(shards):
                shard = shard if isinstance(shard, Mapping) else {"path": shard}
                relative = str(shard.get("path") or "").strip()
                if not relative:
                    raise ValueError(
                        "Capture %s shard %d has no path" % (capture_dir, shard_index)
                    )
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(
                        "Capture shard path must remain capture-relative: %s" % relative
                    )
                shard_path = capture_dir / relative_path
                if not shard_path.is_file():
                    raise FileNotFoundError("Capture shard is missing: %s" % shard_path)
                files.append(
                    {
                        "sample_id": "%s/%s/%s"
                        % (split_id, capture_dir.name, relative_path.as_posix()),
                        "sha256": _sha256(shard_path),
                        "path": relative_path.as_posix(),
                        "captured_samples": int(shard.get("captured_samples") or 0),
                        "sample_start": int(shard.get("sample_start") or 0),
                    }
                )
        if captured_samples != int(row.get("requested_samples") or 0):
            raise ValueError(
                "Capture split %s has %d records; expected %d"
                % (
                    split_id,
                    captured_samples,
                    int(row.get("requested_samples") or 0),
                )
            )
        row["captured_samples"] = captured_samples
        row["files"] = files

    temporary_contract = contract_path.with_suffix(contract_path.suffix + ".tmp")
    temporary_contract.write_text(
        yaml.safe_dump(contract, sort_keys=False),
        encoding="utf-8",
    )
    temporary_contract.replace(contract_path)
    data["contract_sha256"] = _canonical_sha256(contract)
    data["contract_file_sha256"] = _sha256(contract_path)
    updated["data"] = data

    resolved_config_path = Path(config_path).expanduser()
    temporary_config = resolved_config_path.with_suffix(
        resolved_config_path.suffix + ".tmp"
    )
    temporary_config.write_text(
        yaml.safe_dump(updated, sort_keys=False),
        encoding="utf-8",
    )
    temporary_config.replace(resolved_config_path)
    return updated


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
            raise ValueError("Replace capture placeholder before training: %s" % text)
        result.append(Path(text).expanduser())
    return result


def _record_sha256(record: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(record, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
