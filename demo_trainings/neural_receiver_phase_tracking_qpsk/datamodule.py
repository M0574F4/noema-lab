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
class PhaseTrackingDataset:
    receiver_features: np.ndarray
    target_bits: np.ndarray
    data_mask: np.ndarray
    phase_truth: np.ndarray | None
    residual_phase_target: np.ndarray
    residual_phase_mask: np.ndarray
    capture_schema_sha256: tuple[str, ...]
    source_dirs: tuple[str, ...]
    packet_sha256: tuple[str, ...]
    snr_db: np.ndarray | None


def load_capture_dataset(
    directories: Sequence[str],
    *,
    feature_tap: str,
    pilot_context_tap: str,
    target_tap: str,
    phase_truth_tap: str = "",
    feature_reference: str = "",
    pilot_context_reference: str = "",
    target_reference: str = "",
    phase_truth_reference: str = "",
    pilot_smoothing_neighbors: int = 5,
    expected_split: str,
) -> PhaseTrackingDataset:
    features = []
    targets = []
    masks = []
    phase_values = []
    residual_phase_targets = []
    residual_phase_masks = []
    phase_available: bool | None = None
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
        feature_key = _resolve_tap_id(
            tap_schemas,
            configured_id=feature_tap,
            reference=feature_reference,
            label="received symbols",
            required=True,
            directory=directory,
        )
        pilot_context_key = _resolve_tap_id(
            tap_schemas,
            configured_id=pilot_context_tap,
            reference=pilot_context_reference,
            label="public pilot context",
            required=True,
            directory=directory,
        )
        target_key = _resolve_tap_id(
            tap_schemas,
            configured_id=target_tap,
            reference=target_reference,
            label="target bits",
            required=True,
            directory=directory,
        )
        phase_key = _resolve_tap_id(
            tap_schemas,
            configured_id=phase_truth_tap,
            reference=phase_truth_reference,
            label="phase-truth diagnostic",
            required=False,
            directory=directory,
        )
        has_phase = phase_key is not None
        if phase_available is None:
            phase_available = has_phase
        elif phase_available != has_phase:
            raise ValueError(
                "Phase-truth diagnostic must be present in every capture directory or none"
            )

        packet_snr_db = _capture_packet_snr_db(schema)
        for shard in schema.get("shards") or []:
            shard_path = directory / str(shard.get("path") or "")
            with np.load(str(shard_path), allow_pickle=False) as payload:
                rx = np.asarray(payload[feature_key])
                pilots = np.asarray(payload[pilot_context_key])
                bits = np.asarray(payload[target_key])
                phase = (
                    np.asarray(payload[phase_key])
                    if has_phase
                    else None
                )
            (
                packet_features,
                packet_targets,
                packet_mask,
                packet_phase,
                packet_residual_phase,
                packet_residual_mask,
            ) = _packet_records(
                rx,
                pilots,
                bits,
                phase,
                shard_path,
                pilot_smoothing_neighbors=pilot_smoothing_neighbors,
            )
            features.append(packet_features)
            targets.append(packet_targets)
            masks.append(packet_mask)
            if packet_phase is not None:
                phase_values.append(packet_phase)
            residual_phase_targets.append(packet_residual_phase)
            residual_phase_masks.append(packet_residual_mask)
            packet_hashes.extend(
                _packet_sha256(rx[index], pilots[index], bits[index])
                for index in range(int(rx.shape[0]))
            )
            sample_start = int(shard.get("sample_start") or 0)
            if packet_snr_db is None:
                snr_provenance_complete = False
            else:
                values = packet_snr_db[
                    sample_start : sample_start + int(rx.shape[0])
                ]
                if len(values) != int(rx.shape[0]):
                    snr_provenance_complete = False
                else:
                    snr_values.extend(values)
        schema_hashes.append(hashlib.sha256(schema_path.read_bytes()).hexdigest())
        sources.append(str(directory.resolve()))

    if not features:
        raise ValueError("No phase-tracking receiver capture shards were found")
    feature_array = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    target_array = np.concatenate(targets, axis=0).astype(np.uint8, copy=False)
    mask_array = np.concatenate(masks, axis=0).astype(bool, copy=False)
    if feature_array.shape[:2] != target_array.shape[:2]:
        raise ValueError("Receiver features and reconstructed frame targets are misaligned")
    return PhaseTrackingDataset(
        receiver_features=np.ascontiguousarray(feature_array),
        target_bits=np.ascontiguousarray(target_array),
        data_mask=np.ascontiguousarray(mask_array),
        phase_truth=(
            np.ascontiguousarray(
                np.concatenate(phase_values, axis=0),
                dtype=np.float32,
            )
            if phase_available and phase_values
            else None
        ),
        residual_phase_target=np.ascontiguousarray(
            np.concatenate(residual_phase_targets, axis=0),
            dtype=np.float32,
        ),
        residual_phase_mask=np.ascontiguousarray(
            np.concatenate(residual_phase_masks, axis=0),
            dtype=bool,
        ),
        capture_schema_sha256=tuple(schema_hashes),
        source_dirs=tuple(sources),
        packet_sha256=tuple(packet_hashes),
        snr_db=(
            np.ascontiguousarray(np.asarray(snr_values), dtype=np.float64)
            if snr_provenance_complete
            and len(snr_values) == int(feature_array.shape[0])
            else None
        ),
    )


def _resolve_tap_id(
    tap_schemas: dict,
    *,
    configured_id: str,
    reference: str,
    label: str,
    required: bool,
    directory: Path,
) -> str | None:
    """Resolve a captured tensor independently of its user-chosen tap ID.

    Capture tap IDs are labels chosen in Workbench and are not part of the
    operation ABI.  The graph reference (for example
    ``modulator.pilot_context``) is the stable identity across exports.
    """

    configured = str(configured_id or "").strip()
    expected_reference = str(reference or "").strip()
    if configured in tap_schemas:
        row = tap_schemas.get(configured)
        captured_reference = (
            str((row or {}).get("from") or (row or {}).get("reference") or "")
            if isinstance(row, dict)
            else ""
        )
        if not expected_reference or not captured_reference:
            return configured
        if captured_reference == expected_reference:
            return configured

    matches = []
    if expected_reference:
        for tap_id, raw_schema in tap_schemas.items():
            if not isinstance(raw_schema, dict):
                continue
            captured_reference = str(
                raw_schema.get("from") or raw_schema.get("reference") or ""
            )
            if captured_reference == expected_reference:
                matches.append(str(tap_id))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "Capture %s contains multiple taps for %s (%s): %s"
            % (directory, label, expected_reference, ", ".join(sorted(matches)))
        )
    if not required:
        return None
    identity = expected_reference or configured or label
    available = ", ".join(sorted(str(item) for item in tap_schemas)) or "none"
    raise ValueError(
        "Capture %s does not contain required %s %s; available tap IDs: %s"
        % (directory, label, identity, available)
    )


def build_loader(
    dataset: PhaseTrackingDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if dataset.snr_db is None:
        snr_db = np.full(
            (dataset.receiver_features.shape[0],),
            np.nan,
            dtype=np.float32,
        )
    else:
        snr_db = np.asarray(dataset.snr_db, dtype=np.float32)
    tensor_dataset = TensorDataset(
        torch.from_numpy(dataset.receiver_features),
        torch.from_numpy(dataset.target_bits),
        torch.from_numpy(dataset.data_mask),
        torch.from_numpy(dataset.residual_phase_target),
        torch.from_numpy(dataset.residual_phase_mask),
        torch.from_numpy(snr_db),
    )
    return DataLoader(
        tensor_dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        num_workers=max(0, int(num_workers)),
        generator=generator,
    )


def split_integrity_report(
    datasets: dict[str, PhaseTrackingDataset],
) -> dict[str, object]:
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
    issues = [
        "%s contains %d repeated packet(s)" % (name, count)
        for name, count in duplicate_counts.items()
        if count
    ]
    issues.extend(
        "%s share %d packet(s)" % (pair.replace("__", " and "), count)
        for pair, count in pairwise_overlap.items()
        if count
    )
    if issues:
        raise ValueError(
            "Dataset split-integrity check failed before optimization: %s. "
            "Recapture every split with independent run seeds."
            % "; ".join(issues)
        )
    return {
        "status": "passed",
        "method": "sha256(received frame + public pilot context + target data bits)",
        "packet_counts": {
            name: len(dataset.packet_sha256) for name, dataset in datasets.items()
        },
        "packet_sha256": {
            name: sorted(split_sets[name]) for name in datasets
        },
        "duplicate_packet_counts": duplicate_counts,
        "pairwise_overlap_packet_counts": pairwise_overlap,
    }


def held_out_split_integrity_report(
    dataset: PhaseTrackingDataset,
    training_split_integrity: dict[str, object],
) -> dict[str, object]:
    recorded = training_split_integrity.get("packet_sha256")
    if not isinstance(recorded, dict):
        raise ValueError(
            "The artifact lacks packet-level train/validation integrity evidence"
        )
    test_hashes = set(dataset.packet_sha256)
    overlap_counts = {}
    for split in ("train", "validation"):
        raw_hashes = recorded.get(split)
        if not isinstance(raw_hashes, list):
            raise ValueError("The artifact lacks %s packet fingerprints" % split)
        overlap_counts["%s__test" % split] = len(
            set(str(value) for value in raw_hashes).intersection(test_hashes)
        )
    if any(overlap_counts.values()):
        raise ValueError(
            "Held-out split-integrity check failed: %s"
            % ", ".join(
                "%s=%d" % (key, value)
                for key, value in overlap_counts.items()
                if value
            )
        )
    return {
        "status": "passed",
        "method": "sha256(received frame + public pilot context + target data bits)",
        "test_packet_count": len(dataset.packet_sha256),
        "test_unique_packet_count": len(test_hashes),
        "pairwise_overlap_packet_counts": overlap_counts,
    }


def _packet_records(
    rx_packets: np.ndarray,
    pilot_packets: np.ndarray,
    bit_packets: np.ndarray,
    phase_packets: np.ndarray | None,
    source: Path,
    *,
    pilot_smoothing_neighbors: int = 5,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
]:
    rx = np.asarray(rx_packets)
    pilots = np.asarray(pilot_packets)
    bits = np.asarray(bit_packets)
    if rx.ndim != 2 or pilots.ndim != 3 or bits.ndim != 2:
        raise ValueError(
            "Capture shard %s must store rx [packet,symbol], pilot context "
            "[packet,symbol,3], and bits [packet,bit]; got %s, %s, and %s"
            % (source, rx.shape, pilots.shape, bits.shape)
        )
    if not np.iscomplexobj(rx):
        raise ValueError("Capture shard %s rx_symbols must be complex" % source)
    if pilots.shape[:2] != rx.shape or pilots.shape[-1] != 3:
        raise ValueError(
            "Capture shard %s pilot context does not align with rx symbols"
            % source
        )
    if bits.shape[0] != rx.shape[0]:
        raise ValueError("Capture shard %s bit packets do not align" % source)
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Capture shard %s target bits are not canonical binary values" % source)

    pilot_mask = pilots[..., 0] > 0.5
    data_mask = ~pilot_mask
    data_counts = np.sum(data_mask, axis=1)
    if np.any(2 * data_counts != bits.shape[1]):
        raise ValueError(
            "Capture shard %s violates QPSK data/pilot framing: data symbols=%s bits=%d"
            % (source, data_counts.tolist(), bits.shape[1])
        )
    targets = np.zeros((*rx.shape, 2), dtype=np.uint8)
    for packet_index in range(int(rx.shape[0])):
        targets[packet_index, data_mask[packet_index], :] = bits[
            packet_index
        ].reshape(-1, 2)

    known = pilots[..., 1].astype(np.float32) + 1j * pilots[..., 2].astype(
        np.float32
    )
    observation = np.where(
        pilot_mask,
        rx * np.conj(known),
        0.0 + 0.0j,
    )
    smoothing_phase = _pilot_smoothed_phase(
        observation,
        pilot_mask,
        neighbors=pilot_smoothing_neighbors,
    )
    smoothing_phasor = np.exp(1j * smoothing_phase).astype(
        np.complex64,
        copy=False,
    )
    corrected = (
        rx * np.conj(smoothing_phasor)
    ).astype(np.complex64, copy=False)
    pilot_innovation = np.zeros(rx.shape, dtype=np.complex64)
    raw_innovation = observation[pilot_mask] * np.conj(
        smoothing_phasor[pilot_mask]
    )
    pilot_innovation[pilot_mask] = raw_innovation / np.maximum(
        np.abs(raw_innovation),
        1e-12,
    )
    raw_fourth_power = -(corrected.astype(np.complex64, copy=False) ** 4)
    qpsk_fourth_power = raw_fourth_power / np.maximum(
        np.abs(raw_fourth_power),
        1e-12,
    )
    receiver_features = np.stack(
        (
            corrected.real,
            corrected.imag,
            rx.real,
            rx.imag,
            pilot_mask.astype(np.float32),
            pilot_innovation.real,
            pilot_innovation.imag,
            smoothing_phasor.real,
            smoothing_phasor.imag,
            qpsk_fourth_power.real,
            qpsk_fourth_power.imag,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)

    phase = None
    if phase_packets is not None:
        phase = np.asarray(phase_packets, dtype=np.float32)
        if phase.shape != rx.shape:
            raise ValueError(
                "Capture shard %s phase truth does not align with rx symbols"
                % source
            )
        residual_phase_target = np.angle(
            np.exp(1j * (phase - smoothing_phase))
        ).astype(np.float32, copy=False)
        residual_phase_mask = np.ones(rx.shape, dtype=bool)
    else:
        residual_phase_target = np.zeros(rx.shape, dtype=np.float32)
        residual_phase_mask = np.zeros(rx.shape, dtype=bool)
    return (
        receiver_features,
        targets,
        data_mask,
        phase,
        residual_phase_target,
        residual_phase_mask,
    )


def _pilot_interpolation_phase(
    pilot_observation: np.ndarray,
    pilot_mask: np.ndarray,
) -> np.ndarray:
    result = np.zeros(pilot_mask.shape, dtype=np.float64)
    sample_axis = np.arange(pilot_mask.shape[1], dtype=np.float64)
    for packet_index in range(pilot_mask.shape[0]):
        indices = np.flatnonzero(pilot_mask[packet_index])
        if indices.size < 2:
            raise ValueError("Every phase-tracking packet requires at least two pilots")
        phases = np.unwrap(np.angle(pilot_observation[packet_index, indices]))
        result[packet_index] = np.interp(
            sample_axis,
            indices.astype(np.float64),
            phases.astype(np.float64),
        )
    return result


def _pilot_smoothed_phase(
    pilot_observation: np.ndarray,
    pilot_mask: np.ndarray,
    *,
    neighbors: int = 5,
) -> np.ndarray:
    """Match the runtime's local-linear pilot smoother exactly.

    A local line is fitted at every pilot from its nearest pilot observations;
    the denoised pilot estimates are then interpolated across the packet.  The
    complete packet is available, so future public pilots may be used.
    """

    result = np.zeros(pilot_mask.shape, dtype=np.float64)
    sample_axis = np.arange(pilot_mask.shape[1], dtype=np.float64)
    requested = max(2, int(neighbors))
    weight_cache: dict[tuple[int, ...], np.ndarray] = {}
    for packet_index in range(pilot_mask.shape[0]):
        indices = np.flatnonzero(pilot_mask[packet_index])
        if indices.size < 2:
            raise ValueError("Every phase-tracking packet requires at least two pilots")
        count = min(requested, int(indices.size))
        phases = np.unwrap(np.angle(pilot_observation[packet_index, indices]))
        pilot_axis = indices.astype(np.float64)
        cache_key = tuple(int(value) for value in indices)
        weights = weight_cache.get(cache_key)
        if weights is None:
            pilot_weights = np.zeros(
                (indices.size, indices.size),
                dtype=np.float64,
            )
            for pilot_position, center in enumerate(pilot_axis):
                distances = np.abs(pilot_axis - center)
                selected = np.lexsort((indices, distances))[:count]
                centered = pilot_axis[selected] - center
                design = np.stack((centered, np.ones_like(centered)), axis=1)
                predictor = np.asarray([0.0, 1.0]) @ np.linalg.pinv(design)
                pilot_weights[pilot_position, selected] = predictor
            interpolation_weights = np.zeros(
                (sample_axis.size, indices.size),
                dtype=np.float64,
            )
            for symbol_position, position in enumerate(sample_axis):
                if position <= pilot_axis[0]:
                    interpolation_weights[symbol_position, 0] = 1.0
                    continue
                if position >= pilot_axis[-1]:
                    interpolation_weights[symbol_position, -1] = 1.0
                    continue
                right = int(np.searchsorted(pilot_axis, position, side="right"))
                left = right - 1
                fraction = (position - pilot_axis[left]) / (
                    pilot_axis[right] - pilot_axis[left]
                )
                interpolation_weights[symbol_position, left] = 1.0 - fraction
                interpolation_weights[symbol_position, right] = fraction
            weights = interpolation_weights @ pilot_weights
            weight_cache[cache_key] = weights
        result[packet_index] = weights @ phases.astype(np.float64)
    return result


def _capture_packet_snr_db(schema: dict) -> list[float] | None:
    values = []
    runs = schema.get("runs")
    if not isinstance(runs, list) or not runs:
        return None
    for run in runs:
        if not isinstance(run, dict):
            return None
        snr = _run_snr_db(run)
        count = int(run.get("captured_samples") or 0)
        if snr is None or count < 1:
            return None
        values.extend([snr] * count)
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


def _packet_sha256(*values: np.ndarray) -> str:
    digest = hashlib.sha256(b"noema.phase_tracking_receiver.packet@1\0")
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()
