from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


@dataclass(frozen=True)
class CaptureEvidence:
    schema_sha256: tuple[str, ...]
    source_dirs: tuple[str, ...]
    paired_state_sha256: tuple[str, ...]
    trajectory_cluster_ids: tuple[str, ...]
    trajectory_cluster_method: str


class DelayedCurrentCsiDataset(Dataset):
    """Aligned causal CSI history/current outcome rows from one trajectory."""

    def __init__(
        self,
        csi_history: np.ndarray,
        current_gains: np.ndarray,
        evidence: CaptureEvidence,
    ) -> None:
        history = _valid_csi_history(csi_history, "causal CSI history")
        current = _valid_gain_matrix(current_gains, "current CSI")
        if (
            history.shape[0] != current.shape[0]
            or history.shape[2] != current.shape[1]
        ):
            raise ValueError(
                "CSI history and current CSI must align on sample/subcarrier "
                "axes; got %s and %s" % (history.shape, current.shape)
            )
        if len(evidence.paired_state_sha256) != history.shape[0]:
            raise ValueError("paired-state evidence does not align with CSI rows")
        self.csi_history = history
        self.latest_observed_gains = np.maximum(
            np.sum(np.square(history[:, -1]), axis=-1),
            1e-12,
        ).astype(np.float32, copy=False)
        # Compatibility alias for classical delayed-CSI baselines. Learned
        # policies receive csi_history from __getitem__.
        self.delayed_gains = self.latest_observed_gains
        self.current_gains = current
        self.capture_schema_sha256 = evidence.schema_sha256
        self.source_dirs = evidence.source_dirs
        self.paired_state_sha256 = evidence.paired_state_sha256
        self.trajectory_cluster_ids = evidence.trajectory_cluster_ids
        self.trajectory_cluster_method = evidence.trajectory_cluster_method

    def __len__(self) -> int:
        return int(self.csi_history.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.csi_history[index].copy()),
            torch.from_numpy(self.current_gains[index].copy()),
        )


def load_capture_dataset(
    paths: str | Path | Sequence[str | Path],
    *,
    delayed_csi_tap: str = "delayed_csi",
    current_csi_tap: str = "current_csi",
    expected_split: str | None = None,
) -> DelayedCurrentCsiDataset:
    capture_dirs = _path_list(paths)
    if not capture_dirs:
        raise ValueError("At least one dataset-capture directory is required")

    history_chunks: list[np.ndarray] = []
    current_chunks: list[np.ndarray] = []
    state_hashes: list[str] = []
    schema_hashes: list[str] = []
    trajectory_cluster_ids: list[str] = []
    trajectory_cluster_methods: list[str] = []
    width: int | None = None
    history_length: int | None = None
    for capture_index, capture_dir in enumerate(capture_dirs):
        schema_path = capture_dir / "schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError(
                "Noema capture schema is missing: %s" % schema_path
            )
        schema = load_strict_yaml_or_json(schema_path)
        if (
            not isinstance(schema, dict)
            or schema.get("kind") != "noema.capture_dataset"
            or int(schema.get("schema_version", 0)) != 1
        ):
            raise ValueError("Unsupported capture schema in %s" % schema_path)
        split = str(schema.get("split") or "")
        if expected_split and split != expected_split:
            raise ValueError(
                "Expected capture split %r, got %r in %s"
                % (expected_split, split, schema_path)
            )
        tap_schemas = schema.get("tap_schemas") or {}
        if not isinstance(tap_schemas, dict):
            raise ValueError("Capture tap_schemas must be a mapping")
        directory_state_start = len(state_hashes)
        delayed_shape = _tap_record_shape(
            tap_schemas,
            delayed_csi_tap,
            capture_dir,
        )
        current_shape = _tap_record_shape(
            tap_schemas,
            current_csi_tap,
            capture_dir,
        )
        if len(delayed_shape) != 3 or int(delayed_shape[2]) != 2:
            raise ValueError(
                "CSI-history tap %r must have record shape [history,subcarrier,2]; "
                "got %s in %s"
                % (delayed_csi_tap, delayed_shape, capture_dir)
            )
        if len(current_shape) != 1:
            raise ValueError(
                "Current-CSI tap %r must have record shape [subcarrier]; got "
                "%s in %s" % (current_csi_tap, current_shape, capture_dir)
            )
        if int(delayed_shape[1]) != int(current_shape[0]):
            raise ValueError(
                "CSI-history/current tap subcarrier widths differ in %s"
                % capture_dir
            )
        if width is None:
            width = int(delayed_shape[1])
            history_length = int(delayed_shape[0])
        elif width != int(delayed_shape[1]):
            raise ValueError("All captures must use the same subcarrier count")
        elif history_length != int(delayed_shape[0]):
            raise ValueError("All captures must use the same CSI history length")

        shards = list(schema.get("shards") or [])
        if not shards:
            raise ValueError("Capture has no shards: %s" % capture_dir)
        for shard in shards:
            relative = shard.get("path") if isinstance(shard, dict) else shard
            shard_path = capture_dir / str(relative)
            if not shard_path.is_file():
                raise FileNotFoundError(
                    "Capture shard is missing: %s" % shard_path
                )
            with np.load(str(shard_path), allow_pickle=False) as payload:
                missing = [
                    tap
                    for tap in (delayed_csi_tap, current_csi_tap)
                    if tap not in payload.files
                ]
                if missing:
                    raise KeyError(
                        "%s is missing tap(s) %s"
                        % (shard_path, ", ".join(missing))
                    )
                delayed = np.asarray(
                    payload[delayed_csi_tap],
                    dtype=np.float32,
                )
                current = np.asarray(
                    payload[current_csi_tap],
                    dtype=np.float32,
                )
            expected_width = int(width or 0)
            expected_history = int(history_length or 0)
            if (
                delayed.ndim != 4
                or current.ndim != 2
                or delayed.shape[0] != current.shape[0]
                or delayed.shape[1:] != (
                    expected_history,
                    expected_width,
                    2,
                )
                or current.shape[1] != expected_width
            ):
                raise ValueError(
                    "Aligned taps in %s must have shapes [N,%d,%d,2] and "
                    "[N,%d]; got %s and %s"
                    % (
                        shard_path,
                        expected_history,
                        expected_width,
                        expected_width,
                        delayed.shape,
                        current.shape,
                    )
                )
            delayed = _valid_csi_history(
                delayed, "causal CSI history"
            )
            current = _valid_gain_matrix(current, "current CSI")
            history_chunks.append(delayed)
            current_chunks.append(current)
            state_hashes.extend(
                _paired_state_sha256(delayed[index], current[index])
                for index in range(delayed.shape[0])
            )
        schema_hashes.append(_sha256(schema_path))
        stored_directory_rows = len(state_hashes) - directory_state_start
        declared_directory_rows = int(
            schema.get("captured_samples") or stored_directory_rows
        )
        if declared_directory_rows != stored_directory_rows:
            raise ValueError(
                "Capture schema reports %d rows, but its shards contain %d in %s"
                % (
                    declared_directory_rows,
                    stored_directory_rows,
                    capture_dir,
                )
            )
        directory_cluster_ids, cluster_method = _capture_trajectory_clusters(
            schema,
            capture_index=capture_index,
            expected_count=stored_directory_rows,
        )
        trajectory_cluster_ids.extend(directory_cluster_ids)
        trajectory_cluster_methods.append(cluster_method)

    delayed_values = np.concatenate(history_chunks, axis=0)
    current_values = np.concatenate(current_chunks, axis=0)
    if len(trajectory_cluster_ids) != int(delayed_values.shape[0]):
        raise ValueError(
            "Capture trajectory-cluster metadata describes %d rows, but the "
            "stored taps contain %d"
            % (len(trajectory_cluster_ids), delayed_values.shape[0])
        )
    distinct_cluster_methods = sorted(set(trajectory_cluster_methods))
    return DelayedCurrentCsiDataset(
        delayed_values,
        current_values,
        CaptureEvidence(
            schema_sha256=tuple(schema_hashes),
            source_dirs=tuple(str(path) for path in capture_dirs),
            paired_state_sha256=tuple(state_hashes),
            trajectory_cluster_ids=tuple(trajectory_cluster_ids),
            trajectory_cluster_method=(
                distinct_cluster_methods[0]
                if len(distinct_cluster_methods) == 1
                else "mixed:" + ",".join(distinct_cluster_methods)
            ),
        ),
    )


def paired_cluster_evidence(
    candidate_goodput: np.ndarray,
    baseline_goodput_by_method: Mapping[str, np.ndarray],
    *,
    cluster_ids: Sequence[str],
    cluster_method: str,
    operating_points: Sequence[Mapping[str, float]],
    minimum_relative_improvement: float = 0.005,
    maximum_relative_point_regression: float = 0.002,
    confidence_level: float = 0.95,
    minimum_cluster_count: int = 30,
) -> dict[str, object]:
    """Evaluate a candidate with paired trajectory-cluster uncertainty.

    Inputs have shape ``[operating_point, captured_state]``. Rows from multiple
    OFDM times on the same independently generated TDL block share a cluster;
    they are therefore averaged before estimating uncertainty.
    """

    candidate = _goodput_matrix(candidate_goodput, "candidate goodput")
    if len(cluster_ids) != candidate.shape[1]:
        raise ValueError(
            "cluster_ids must contain one trajectory identity per captured state"
        )
    if len(operating_points) != candidate.shape[0]:
        raise ValueError(
            "operating_points must contain one entry per goodput matrix row"
        )
    if not baseline_goodput_by_method:
        raise ValueError("At least one deployable baseline is required")
    minimum_relative_improvement = _fraction(
        minimum_relative_improvement,
        "minimum_relative_improvement",
    )
    maximum_relative_point_regression = _fraction(
        maximum_relative_point_regression,
        "maximum_relative_point_regression",
    )
    confidence_level = float(confidence_level)
    if not math.isfinite(confidence_level) or not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1")
    minimum_cluster_count = max(2, int(minimum_cluster_count))

    baseline_matrices = {
        str(method): _goodput_matrix(values, "%s goodput" % method)
        for method, values in baseline_goodput_by_method.items()
    }
    wrong_shapes = {
        method: values.shape
        for method, values in baseline_matrices.items()
        if values.shape != candidate.shape
    }
    if wrong_shapes:
        raise ValueError(
            "Baseline goodput matrices must match candidate shape %s; got %s"
            % (candidate.shape, wrong_shapes)
        )

    cluster_indices: dict[str, list[int]] = defaultdict(list)
    for index, cluster_id in enumerate(cluster_ids):
        cluster_indices[str(cluster_id)].append(index)
    ordered_clusters = sorted(cluster_indices)
    baseline_cluster_means = {
        method: _cluster_means(values, cluster_indices, ordered_clusters)
        for method, values in baseline_matrices.items()
    }
    baseline_scores = {
        method: float(np.mean(values))
        for method, values in baseline_cluster_means.items()
    }
    strongest_method = max(
        sorted(baseline_scores),
        key=lambda method: baseline_scores[method],
    )
    strongest = baseline_matrices[strongest_method]
    strongest_score = baseline_scores[strongest_method]
    candidate_clusters = _cluster_means(
        candidate,
        cluster_indices,
        ordered_clusters,
    )
    candidate_score = float(np.mean(candidate_clusters))
    cluster_differences = candidate_clusters - baseline_cluster_means[
        strongest_method
    ]
    absolute_improvement = float(np.mean(cluster_differences))
    relative_improvement = absolute_improvement / max(
        abs(strongest_score),
        1e-12,
    )
    cluster_count = int(cluster_differences.size)
    standard_error = (
        float(np.std(cluster_differences, ddof=1) / math.sqrt(cluster_count))
        if cluster_count > 1
        else float("inf")
    )
    critical_value = NormalDist().inv_cdf(
        0.5 + confidence_level / 2.0
    )
    ci_lower = absolute_improvement - critical_value * standard_error
    ci_upper = absolute_improvement + critical_value * standard_error

    point_rows = []
    point_checks_passed = True
    for point_index, point in enumerate(operating_points):
        candidate_point = float(np.mean(candidate[point_index]))
        baseline_point = float(np.mean(strongest[point_index]))
        difference = candidate_point - baseline_point
        relative_difference = difference / max(abs(baseline_point), 1e-12)
        passed = relative_difference >= -maximum_relative_point_regression
        point_checks_passed = point_checks_passed and passed
        point_rows.append(
            {
                "average_power_budget": float(
                    point["average_power_budget"]
                ),
                "noise_variance": float(point["noise_variance"]),
                "candidate_expected_goodput_bps_hz": candidate_point,
                "baseline_expected_goodput_bps_hz": baseline_point,
                "paired_difference_bps_hz": difference,
                "relative_difference": relative_difference,
                "passed": passed,
            }
        )

    cluster_metadata_available = not str(cluster_method).startswith(
        "independent_row_fallback"
    ) and not str(cluster_method).startswith("mixed:")
    checks = {
        "practical_relative_improvement": (
            relative_improvement >= minimum_relative_improvement
        ),
        "paired_cluster_ci_lower_bound_positive": ci_lower > 0.0,
        "minimum_independent_trajectory_clusters": (
            cluster_count >= minimum_cluster_count
        ),
        "trajectory_cluster_metadata_available": cluster_metadata_available,
        "bounded_operating_point_regression": point_checks_passed,
    }
    reasons = []
    if not checks["practical_relative_improvement"]:
        reasons.append(
            "aggregate relative improvement %.4g is below the required %.4g"
            % (relative_improvement, minimum_relative_improvement)
        )
    if not checks["paired_cluster_ci_lower_bound_positive"]:
        reasons.append(
            "paired trajectory-cluster confidence interval includes zero"
        )
    if not checks["minimum_independent_trajectory_clusters"]:
        reasons.append(
            "only %d independent trajectory clusters are available; %d are required"
            % (cluster_count, minimum_cluster_count)
        )
    if not checks["trajectory_cluster_metadata_available"]:
        reasons.append(
            "capture metadata does not identify independent TDL trajectory clusters"
        )
    if not checks["bounded_operating_point_regression"]:
        reasons.append(
            "candidate exceeds the allowed relative regression at one or more "
            "validation operating points"
        )
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "status": "passed" if passed else "insufficient_evidence",
        "reasons": reasons,
        "strongest_deployable_baseline": {
            "method_id": strongest_method,
            "expected_goodput_bps_hz": strongest_score,
            "all_baseline_expected_goodput_bps_hz": baseline_scores,
        },
        "candidate_expected_goodput_bps_hz": candidate_score,
        "absolute_improvement_bps_hz": absolute_improvement,
        "relative_improvement": relative_improvement,
        "criteria": {
            "minimum_relative_improvement": minimum_relative_improvement,
            "maximum_relative_point_regression": (
                maximum_relative_point_regression
            ),
            "confidence_level": confidence_level,
            "minimum_cluster_count": minimum_cluster_count,
        },
        "paired_cluster_confidence_interval": {
            "method": (
                "normal confidence interval over equal-weight means of paired "
                "independent TDL trajectory clusters"
            ),
            "trajectory_cluster_method": str(cluster_method),
            "cluster_count": cluster_count,
            "estimate_bps_hz": absolute_improvement,
            "standard_error_bps_hz": standard_error,
            "lower_bps_hz": ci_lower,
            "upper_bps_hz": ci_upper,
        },
        "checks": checks,
        "operating_points": point_rows,
    }


def _capture_trajectory_clusters(
    schema: Mapping[str, object],
    *,
    capture_index: int,
    expected_count: int,
) -> tuple[list[str], str]:
    """Recover independent TDL-block identities from capture provenance."""

    if expected_count < 1:
        raise ValueError("Capture schema must report a positive captured_samples")
    runs = schema.get("runs")
    if not isinstance(runs, list) or not runs:
        return (
            [
                "capture%d:independent_row:%d" % (capture_index, row)
                for row in range(expected_count)
            ],
            "independent_row_fallback_missing_run_metadata",
        )
    cluster_ids: list[str] = []
    for run_index, raw_run in enumerate(runs):
        if not isinstance(raw_run, Mapping):
            break
        run_count = int(raw_run.get("captured_samples") or 0)
        channel_distribution = raw_run.get("channel_distribution")
        steps = (
            channel_distribution.get("steps")
            if isinstance(channel_distribution, Mapping)
            else None
        )
        channel_state = next(
            (
                step
                for step in steps or []
                if isinstance(step, Mapping)
                and str(step.get("id") or "") == "channel_state"
            ),
            None,
        )
        metadata = (
            channel_state.get("metadata")
            if isinstance(channel_state, Mapping)
            else None
        )
        block_count = int(
            metadata.get("ofdm_block_count") or 0
        ) if isinstance(metadata, Mapping) else 0
        if run_count < 1 or block_count < 1:
            break
        cluster_ids.extend(
            "capture%d:run%d:tdl_block%d"
            % (capture_index, run_index, row % block_count)
            for row in range(run_count)
        )
    if len(cluster_ids) == expected_count:
        return (
            cluster_ids,
            "time_major_ofdm_symbol_rows_clustered_by_independent_tdl_block",
        )
    return (
        [
            "capture%d:independent_row:%d" % (capture_index, row)
            for row in range(expected_count)
        ],
        "independent_row_fallback_incomplete_trajectory_metadata",
    )


def _goodput_matrix(value: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[0] < 1
        or result.shape[1] < 1
        or not np.all(np.isfinite(result))
    ):
        raise ValueError("%s must be a finite [operating_point,state] array" % label)
    return result


def _cluster_means(
    values: np.ndarray,
    cluster_indices: Mapping[str, Sequence[int]],
    ordered_clusters: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [
            np.mean(values[:, cluster_indices[cluster_id]])
            for cluster_id in ordered_clusters
        ],
        dtype=np.float64,
    )


def _fraction(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError("%s must be a finite fraction in [0,1)" % label)
    return result


def build_loader(
    dataset: DelayedCurrentCsiDataset,
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
    dataset: DelayedCurrentCsiDataset,
    *,
    reference_noise_variance: float,
    reference_average_power_budget: float,
    eps: float = 1e-12,
) -> tuple[float, float]:
    history = dataset.csi_history.astype(np.float64)
    gains = np.sum(np.square(history), axis=-1)
    values = np.log(
        np.maximum(
            gains,
            float(eps),
        )
    )
    return float(np.mean(values)), max(float(np.std(values)), 1e-6)


def split_integrity_report(
    train: DelayedCurrentCsiDataset,
    validation: DelayedCurrentCsiDataset,
) -> dict[str, object]:
    return _integrity_report(
        {"train": train, "validation": validation},
        context="before optimization",
    )


def held_out_split_integrity_report(
    test: DelayedCurrentCsiDataset,
    training_report: dict[str, object],
) -> dict[str, object]:
    recorded = training_report.get("paired_state_sha256")
    if not isinstance(recorded, dict):
        raise ValueError(
            "The trained artifact lacks train/validation split fingerprints"
        )
    test_set = set(test.paired_state_sha256)
    duplicate_count = len(test.paired_state_sha256) - len(test_set)
    overlaps = {}
    for split in ("train", "validation"):
        values = recorded.get(split)
        if not isinstance(values, list):
            raise ValueError(
                "The trained artifact lacks %s split fingerprints" % split
            )
        overlaps["%s__test" % split] = len(set(values).intersection(test_set))
    issues = []
    if duplicate_count:
        issues.append("test contains %d repeated state pair(s)" % duplicate_count)
    issues.extend(
        "%s share %d state pair(s)" % (name.replace("__", " and "), count)
        for name, count in overlaps.items()
        if count
    )
    if issues:
        raise ValueError(
            "Held-out split-integrity check failed: %s" % "; ".join(issues)
        )
    return {
        "status": "passed",
        "method": "sha256(causal complex CSI history + aligned current gain row)",
        "test_state_count": len(test.paired_state_sha256),
        "test_unique_state_count": len(test_set),
        "pairwise_overlap_state_counts": overlaps,
    }


def _integrity_report(
    datasets: dict[str, DelayedCurrentCsiDataset],
    *,
    context: str,
) -> dict[str, object]:
    sets = {
        split: set(dataset.paired_state_sha256)
        for split, dataset in datasets.items()
    }
    duplicates = {
        split: len(dataset.paired_state_sha256) - len(sets[split])
        for split, dataset in datasets.items()
    }
    names = list(datasets)
    overlaps = {
        "%s__%s" % (names[left], names[right]): len(
            sets[names[left]].intersection(sets[names[right]])
        )
        for left in range(len(names))
        for right in range(left + 1, len(names))
    }
    issues = [
        "%s contains %d repeated state pair(s)" % (split, count)
        for split, count in duplicates.items()
        if count
    ]
    issues.extend(
        "%s share %d state pair(s)" % (name.replace("__", " and "), count)
        for name, count in overlaps.items()
        if count
    )
    if issues:
        raise ValueError(
            "Dataset split-integrity check failed %s: %s. "
            "Recapture every split with independent channel and estimation-error seeds."
            % (context, "; ".join(issues))
        )
    return {
        "status": "passed",
        "method": "sha256(causal complex CSI history + aligned current gain row)",
        "state_counts": {
            split: len(dataset.paired_state_sha256)
            for split, dataset in datasets.items()
        },
        "paired_state_sha256": {
            split: sorted(values) for split, values in sets.items()
        },
        "duplicate_state_counts": duplicates,
        "pairwise_overlap_state_counts": overlaps,
    }


def _tap_record_shape(
    schemas: dict,
    tap_id: str,
    capture_dir: Path,
) -> tuple[int, ...]:
    raw = schemas.get(tap_id)
    if not isinstance(raw, dict):
        raise KeyError("Capture %s does not declare tap %r" % (capture_dir, tap_id))
    shape = list(raw.get("record_shape") or [])
    if not shape or any(int(value) < 1 for value in shape):
        raise ValueError(
            "Tap %r has an invalid record shape %s"
            % (tap_id, shape)
        )
    return tuple(int(value) for value in shape)


def _valid_csi_history(value: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if (
        result.ndim != 4
        or result.shape[0] < 1
        or result.shape[1] < 1
        or result.shape[2] < 1
        or result.shape[3] != 2
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(
            "%s must have finite shape [sample,history,subcarrier,2], got %s"
            % (label, result.shape)
        )
    return np.ascontiguousarray(result)


def _valid_gain_matrix(value: np.ndarray, label: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError("%s must have shape [sample,subcarrier]" % label)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("%s contains non-finite or negative values" % label)
    return result


def _paired_state_sha256(
    delayed: np.ndarray,
    current: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for label, value in (("delayed", delayed), ("current", current)):
        array = np.ascontiguousarray(value, dtype="<f4")
        digest.update(label.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _path_list(
    paths: str | Path | Sequence[str | Path],
) -> list[Path]:
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
            raise ValueError(
                "Replace capture placeholder before training: %s" % text
            )
        result.append(Path(text).expanduser())
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
