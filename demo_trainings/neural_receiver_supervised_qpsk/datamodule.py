from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


@dataclass(frozen=True)
class ReceiverDataset:
    features: np.ndarray
    target_bits: np.ndarray
    capture_schema_sha256: tuple[str, ...]
    source_dirs: tuple[str, ...]
    packet_sha256: tuple[str, ...]
    snr_db: np.ndarray | None


def load_capture_dataset(
    directories: Sequence[str],
    *,
    feature_tap: str,
    target_tap: str,
    expected_split: str,
) -> ReceiverDataset:
    features = []
    targets = []
    schema_hashes = []
    sources = []
    packet_hashes = []
    snr_values = []
    snr_provenance_complete = True
    for raw in directories:
        directory = Path(str(raw)).expanduser()
        schema_path = directory / "schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError(
                "Capture schema is missing for %s; run the %s capture job in Noema first"
                % (directory, expected_split)
            )
        schema = load_strict_yaml_or_json(schema_path)
        if str(schema.get("kind") or "") != "noema.capture_dataset":
            raise ValueError("%s is not a Noema capture dataset" % directory)
        if str(schema.get("split") or "") != str(expected_split):
            raise ValueError(
                "Capture %s has split %s, expected %s"
                % (directory, schema.get("split"), expected_split)
            )
        tap_schemas = dict(schema.get("tap_schemas") or {})
        if feature_tap not in tap_schemas or target_tap not in tap_schemas:
            raise ValueError(
                "Capture %s does not contain required taps %s and %s"
                % (directory, feature_tap, target_tap)
            )
        packet_snr_db = _capture_packet_snr_db(schema)
        for shard in schema.get("shards") or []:
            shard_path = directory / str(shard.get("path") or "")
            with np.load(str(shard_path), allow_pickle=False) as payload:
                rx = np.asarray(payload[feature_tap])
                bits = np.asarray(payload[target_tap])
            packet_features, packet_targets = _packet_pairs(rx, bits, shard_path)
            features.append(packet_features)
            targets.append(packet_targets)
            packet_hashes.extend(
                _packet_sha256(rx[index], bits[index])
                for index in range(int(rx.shape[0]))
            )
            sample_start = int(shard.get("sample_start") or 0)
            if packet_snr_db is None:
                snr_provenance_complete = False
            else:
                packet_values = packet_snr_db[
                    sample_start : sample_start + int(rx.shape[0])
                ]
                if len(packet_values) != int(rx.shape[0]):
                    snr_provenance_complete = False
                else:
                    snr_values.append(
                        np.repeat(
                            np.asarray(packet_values, dtype=np.float64),
                            int(rx.shape[1]),
                        )
                    )
        schema_hashes.append(hashlib.sha256(schema_path.read_bytes()).hexdigest())
        sources.append(str(directory.resolve()))
    if not features:
        raise ValueError("No neural-receiver capture shards were found")
    feature_array = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    target_array = np.concatenate(targets, axis=0).astype(np.uint8, copy=False)
    if feature_array.shape[0] != target_array.shape[0]:
        raise ValueError("Neural-receiver features and targets have different sample counts")
    return ReceiverDataset(
        features=np.ascontiguousarray(feature_array),
        target_bits=np.ascontiguousarray(target_array),
        capture_schema_sha256=tuple(schema_hashes),
        source_dirs=tuple(sources),
        packet_sha256=tuple(packet_hashes),
        snr_db=(
            np.ascontiguousarray(np.concatenate(snr_values), dtype=np.float64)
            if snr_provenance_complete
            and snr_values
            and sum(int(value.size) for value in snr_values) == feature_array.shape[0]
            else None
        ),
    )


def split_integrity_report(
    datasets: dict[str, ReceiverDataset],
) -> dict[str, object]:
    """Reject repeated packets within or across independently declared splits."""

    split_sets = {
        name: set(dataset.packet_sha256) for name, dataset in datasets.items()
    }
    duplicate_counts = {
        name: len(dataset.packet_sha256) - len(split_sets[name])
        for name, dataset in datasets.items()
    }
    pairwise_overlap = {}
    names = list(datasets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pairwise_overlap["%s__%s" % (left, right)] = len(
                split_sets[left].intersection(split_sets[right])
            )

    issues = []
    for name, count in duplicate_counts.items():
        if count:
            issues.append("%s contains %d repeated packet(s)" % (name, count))
    for pair, count in pairwise_overlap.items():
        if count:
            issues.append("%s share %d packet(s)" % (pair.replace("__", " and "), count))
    if issues:
        raise ValueError(
            "Dataset split-integrity check failed before optimization: %s. "
            "Recapture every split with independent run seeds."
            % "; ".join(issues)
        )

    return {
        "status": "passed",
        "method": "sha256(canonical received-symbol packet + target-bit packet)",
        "packet_counts": {
            name: len(dataset.packet_sha256) for name, dataset in datasets.items()
        },
        "unique_packet_counts": {
            name: len(split_sets[name]) for name in datasets
        },
        "packet_set_sha256": {
            name: hashlib.sha256(
                "\n".join(sorted(split_sets[name])).encode("ascii")
            ).hexdigest()
            for name in datasets
        },
        "packet_sha256": {
            name: sorted(split_sets[name]) for name in datasets
        },
        "duplicate_packet_counts": duplicate_counts,
        "pairwise_overlap_packet_counts": pairwise_overlap,
    }


def held_out_split_integrity_report(
    dataset: ReceiverDataset,
    training_split_integrity: dict[str, object],
) -> dict[str, object]:
    """Verify a held-out split against fingerprints recorded before optimization."""

    test_report = split_integrity_report({"test": dataset})
    recorded = training_split_integrity.get("packet_sha256")
    if not isinstance(recorded, dict):
        raise ValueError(
            "The trained artifact lacks packet-level train/validation integrity evidence; "
            "retrain it with the current demonstration scaffold"
        )
    test_hashes = set(dataset.packet_sha256)
    overlap_counts = {}
    for split in ("train", "validation"):
        raw_hashes = recorded.get(split)
        if not isinstance(raw_hashes, list):
            raise ValueError(
                "The trained artifact lacks %s packet fingerprints; retrain it with "
                "the current demonstration scaffold" % split
            )
        overlap_counts["%s__test" % split] = len(
            set(str(value) for value in raw_hashes).intersection(test_hashes)
        )
    overlaps = [
        "%s share %d packet(s)" % (pair.replace("__", " and "), count)
        for pair, count in overlap_counts.items()
        if count
    ]
    if overlaps:
        raise ValueError(
            "Held-out split-integrity check failed before evaluation: %s. "
            "Recapture all splits with independent run seeds and retrain."
            % "; ".join(overlaps)
        )
    return {
        "status": "passed",
        "method": test_report["method"],
        "test_packet_count": len(dataset.packet_sha256),
        "test_unique_packet_count": len(test_hashes),
        "test_packet_set_sha256": test_report["packet_set_sha256"]["test"],
        "pairwise_overlap_packet_counts": overlap_counts,
    }


def build_loader(
    dataset: ReceiverDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    tensor_dataset = TensorDataset(
        torch.from_numpy(dataset.features),
        torch.from_numpy(dataset.target_bits),
    )
    return DataLoader(
        tensor_dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        num_workers=max(0, int(num_workers)),
        generator=generator,
    )


def _packet_pairs(
    rx_packets: np.ndarray,
    bit_packets: np.ndarray,
    source: Path,
) -> tuple[np.ndarray, np.ndarray]:
    rx = np.asarray(rx_packets)
    bits = np.asarray(bit_packets)
    if rx.ndim != 2 or bits.ndim != 2:
        raise ValueError(
            "Capture shard %s must store packet-major rx symbols and bits; got %s and %s"
            % (source, rx.shape, bits.shape)
        )
    if not np.iscomplexobj(rx):
        raise ValueError("Capture shard %s rx_symbols must be complex" % source)
    if rx.shape[0] != bits.shape[0] or bits.shape[1] != 2 * rx.shape[1]:
        raise ValueError(
            "Capture shard %s violates QPSK pairing: rx=%s target_bits=%s"
            % (source, rx.shape, bits.shape)
        )
    features = np.stack((rx.real, rx.imag), axis=-1).reshape(-1, 2)
    targets = bits.reshape(rx.shape[0], rx.shape[1], 2).reshape(-1, 2)
    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("Capture shard %s target bits are not canonical binary values" % source)
    return features.astype(np.float32, copy=False), targets.astype(np.uint8, copy=False)


def _capture_packet_snr_db(schema: dict) -> list[float] | None:
    values = []
    runs = schema.get("runs")
    if not isinstance(runs, list) or not runs:
        return None
    for run in runs:
        if not isinstance(run, dict):
            return None
        snr_db = _run_snr_db(run)
        if snr_db is None:
            return None
        captured_samples = int(run.get("captured_samples") or 0)
        if captured_samples < 1:
            return None
        values.extend([snr_db] * captured_samples)
    expected = int(schema.get("captured_samples") or len(values))
    return values if len(values) == expected else None


def _run_snr_db(run: dict) -> float | None:
    for field in ("sweep", "matrix_selection"):
        selection = run.get(field)
        if not isinstance(selection, dict):
            continue
        if "wireless_channel.snr_db" in selection:
            return float(selection["wireless_channel.snr_db"])
        candidates = [
            value
            for key, value in selection.items()
            if str(key).endswith(".snr_db")
        ]
        if len(candidates) == 1:
            return float(candidates[0])
    return None


def _packet_sha256(rx_packet: np.ndarray, bit_packet: np.ndarray) -> str:
    digest = hashlib.sha256(b"noema.neural_receiver.packet@1\0")
    for value in (rx_packet, bit_packet):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()
