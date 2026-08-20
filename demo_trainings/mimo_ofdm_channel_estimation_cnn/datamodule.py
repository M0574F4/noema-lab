from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


@dataclass(frozen=True)
class ChannelDataset:
    pilot_ls_ri: np.ndarray
    pilot_mask: np.ndarray
    ls_estimate_ri: np.ndarray
    noise_variance: np.ndarray
    channel_truth_ri: np.ndarray
    record_sha256: tuple[str, ...]
    schema_sha256: tuple[str, ...]


def load_capture_dataset(
    directories: Sequence[str],
    *,
    pilot_tap: str,
    mask_tap: str,
    ls_tap: str,
    noise_tap: str,
    target_tap: str,
    expected_split: str,
) -> ChannelDataset:
    pilot_values: list[np.ndarray] = []
    mask_values: list[np.ndarray] = []
    ls_values: list[np.ndarray] = []
    noise_values: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    fingerprints: list[str] = []
    schema_hashes: list[str] = []
    for raw in directories:
        directory = Path(str(raw)).expanduser()
        schema_path = directory / "schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError(
                "Capture schema is missing for %s; capture the %s split first"
                % (directory, expected_split)
            )
        schema = load_strict_yaml_or_json(schema_path)
        if str(schema.get("kind") or "") != "noema.capture_dataset":
            raise ValueError("%s is not a Noema capture dataset" % directory)
        if str(schema.get("split") or "") != expected_split:
            raise ValueError(
                "Capture %s has split %s, expected %s"
                % (directory, schema.get("split"), expected_split)
            )
        available = set((schema.get("tap_schemas") or {}).keys())
        missing = [
            tap
            for tap in (pilot_tap, mask_tap, ls_tap, noise_tap, target_tap)
            if tap not in available
        ]
        if missing:
            raise ValueError("Capture %s is missing tap(s): %s" % (directory, ", ".join(missing)))
        for shard in schema.get("shards") or []:
            shard_path = directory / str(shard.get("path") or "")
            with np.load(str(shard_path), allow_pickle=False) as payload:
                pilot_complex = np.asarray(payload[pilot_tap])
                pilot_mask = np.asarray(payload[mask_tap], dtype=np.float32)
                ls_complex = np.asarray(payload[ls_tap])
                noise = np.asarray(payload[noise_tap], dtype=np.float32)
                truth = np.asarray(payload[target_tap])
            _validate_shapes(
                pilot_complex,
                pilot_mask,
                ls_complex,
                noise,
                truth,
                shard_path,
            )
            pilot_values.append(_real_imag(pilot_complex))
            mask_values.append(pilot_mask)
            ls_values.append(_real_imag(ls_complex))
            noise_values.append(noise.reshape(noise.shape[0], 1))
            targets.append(_real_imag(truth))
            fingerprints.extend(
                _record_sha256(
                    pilot_complex[index],
                    pilot_mask[index],
                    ls_complex[index],
                    noise[index],
                    truth[index],
                )
                for index in range(int(ls_complex.shape[0]))
            )
        schema_hashes.append(hashlib.sha256(schema_path.read_bytes()).hexdigest())
    if not ls_values:
        raise ValueError("No channel-estimation capture shards were found")
    return ChannelDataset(
        pilot_ls_ri=np.ascontiguousarray(np.concatenate(pilot_values, axis=0)),
        pilot_mask=np.ascontiguousarray(np.concatenate(mask_values, axis=0)),
        ls_estimate_ri=np.ascontiguousarray(np.concatenate(ls_values, axis=0)),
        noise_variance=np.ascontiguousarray(np.concatenate(noise_values, axis=0)),
        channel_truth_ri=np.ascontiguousarray(np.concatenate(targets, axis=0)),
        record_sha256=tuple(fingerprints),
        schema_sha256=tuple(schema_hashes),
    )


def split_integrity_report(datasets: Mapping[str, ChannelDataset]) -> dict[str, object]:
    sets = {name: set(dataset.record_sha256) for name, dataset in datasets.items()}
    problems = []
    for name, dataset in datasets.items():
        repeats = len(dataset.record_sha256) - len(sets[name])
        if repeats:
            problems.append("%s contains %d repeated channel record(s)" % (name, repeats))
    overlap = {}
    names = list(datasets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            key = "%s__%s" % (left, right)
            overlap[key] = len(sets[left].intersection(sets[right]))
            if overlap[key]:
                problems.append("%s share %d record(s)" % (key.replace("__", " and "), overlap[key]))
    if problems:
        raise ValueError("Dataset split-integrity check failed: %s" % "; ".join(problems))
    return {
        "status": "passed",
        "method": (
            "sha256(sparse pilot LS + pilot mask + interpolated LS + "
            "noise variance + channel truth)"
        ),
        "record_counts": {name: len(value.record_sha256) for name, value in datasets.items()},
        "record_sha256": {name: sorted(sets[name]) for name in names},
        "pairwise_overlap_record_counts": overlap,
    }


def build_loader(
    dataset: ChannelDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    values = TensorDataset(
        torch.from_numpy(dataset.pilot_ls_ri),
        torch.from_numpy(dataset.pilot_mask),
        torch.from_numpy(dataset.ls_estimate_ri),
        torch.from_numpy(dataset.noise_variance),
        torch.from_numpy(dataset.channel_truth_ri),
    )
    return DataLoader(
        values,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
    )


def fixed_prior_lmmse_estimate(
    dataset: ChannelDataset,
    *,
    assumed_tap_count: int = 4,
    channel_variance: float = 1.0,
) -> np.ndarray:
    """Return a practical LMMSE estimate with one fixed exponential PDP.

    The estimator deliberately receives no TDL-profile label or true
    covariance. It uses the same sparse pilot observations available to the
    learned runtime and a four-tap exponential prior for every realization.
    """

    pilot = _complex(dataset.pilot_ls_ri)
    mask = np.asarray(dataset.pilot_mask, dtype=np.float32)
    batch, rx, tx, subcarriers = pilot.shape
    tap_count = max(1, min(int(assumed_tap_count), int(subcarriers)))
    tap_power = np.exp(-np.arange(tap_count, dtype=np.float64))
    tap_power *= float(channel_variance) / max(float(np.sum(tap_power)), 1e-12)
    indices = np.arange(subcarriers, dtype=np.float64)
    differences = indices[:, None] - indices[None, :]
    tap_indices = np.arange(tap_count, dtype=np.float64)
    covariance = np.sum(
        tap_power[None, None, :]
        * np.exp(
            -2j
            * np.pi
            * differences[:, :, None]
            * tap_indices[None, None, :]
            / float(subcarriers)
        ),
        axis=2,
    )
    output = np.empty((batch, rx, tx, subcarriers), dtype=np.complex64)
    noise_values = np.asarray(dataset.noise_variance, dtype=np.float32).reshape(-1)
    for tx_index in range(tx):
        reference_mask = mask[0, tx_index] > 0.5
        if not np.all((mask[:, tx_index, :] > 0.5) == reference_mask[None, :]):
            raise ValueError("Fixed-prior LMMSE requires a consistent pilot mask")
        pilot_indices = np.flatnonzero(reference_mask)
        covariance_pp = covariance[np.ix_(pilot_indices, pilot_indices)]
        cross_covariance = covariance[:, pilot_indices]
        for noise_variance in np.unique(noise_values):
            selected = np.flatnonzero(
                np.isclose(noise_values, noise_variance, rtol=1e-6, atol=1e-9)
            )
            regularized = covariance_pp + float(noise_variance) * np.eye(
                pilot_indices.size,
                dtype=np.complex128,
            )
            weights = np.linalg.solve(
                regularized.T,
                cross_covariance.T,
            ).T
            output[selected, :, tx_index, :] = np.einsum(
                "kp,brp->brk",
                weights,
                pilot[selected, :, tx_index, :][:, :, pilot_indices],
                optimize=True,
            ).astype(np.complex64)
    return _real_imag(output)


def _validate_shapes(
    pilot_value: np.ndarray,
    pilot_mask: np.ndarray,
    ls_value: np.ndarray,
    noise: np.ndarray,
    truth: np.ndarray,
    path: Path,
) -> None:
    if (
        not np.iscomplexobj(pilot_value)
        or not np.iscomplexobj(ls_value)
        or not np.iscomplexobj(truth)
    ):
        raise ValueError(
            "%s must contain complex sparse-pilot, LS, and channel-truth tensors"
            % path
        )
    if ls_value.ndim != 4 or truth.shape != ls_value.shape:
        raise ValueError("%s requires aligned [record,rx,tx,subcarrier] tensors" % path)
    if pilot_value.shape != ls_value.shape:
        raise ValueError("%s sparse-pilot records do not align with LS records" % path)
    expected_mask = (
        int(ls_value.shape[0]),
        int(ls_value.shape[2]),
        int(ls_value.shape[3]),
    )
    if tuple(pilot_mask.shape) != expected_mask:
        raise ValueError(
            "%s pilot mask must have shape %s" % (path, expected_mask)
        )
    if np.any((pilot_mask != 0.0) & (pilot_mask != 1.0)):
        raise ValueError("%s pilot mask must be binary" % path)
    inactive = pilot_mask[:, None, :, :] == 0.0
    if np.any(np.abs(pilot_value[inactive.repeat(ls_value.shape[1], axis=1)]) > 1e-7):
        raise ValueError("%s sparse-pilot tensor must be zero outside the mask" % path)
    if noise.shape[0] != ls_value.shape[0]:
        raise ValueError("%s noise records do not align with channel records" % path)


def _real_imag(value: np.ndarray) -> np.ndarray:
    return np.stack([value.real, value.imag], axis=-1).astype(np.float32)


def _complex(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return (array[..., 0] + 1j * array[..., 1]).astype(np.complex64)


def _record_sha256(
    pilot_value: np.ndarray,
    pilot_mask: np.ndarray,
    ls_value: np.ndarray,
    noise: np.ndarray,
    truth: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for value in (pilot_value, pilot_mask, ls_value, noise, truth):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()
