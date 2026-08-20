from __future__ import annotations

import json
import math
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.boundaries import validate_channel_bits, validate_channel_symbols
from noema_lab.core.capture_layout import remove_capture_record_metadata
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationResult,
    object_schema,
)
from noema_lab.core.structured_input import decode_strict_json_object


JsonDict = Dict[str, Any]


def _load_state(path) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        if "h_freq" not in payload.files:
            raise OperationError("OFDM channel-state artifact is missing h_freq")
        h_freq = np.asarray(payload["h_freq"], dtype=np.complex64)
        metadata = (
            decode_strict_json_object(
                str(payload["metadata_json"]),
                label="OFDM channel-state metadata_json",
            )
            if "metadata_json" in payload.files
            else {}
        )
    if h_freq.ndim != 3 or any(int(size) < 1 for size in h_freq.shape):
        raise OperationError(
            "OFDM channel state must have shape [block, ofdm_symbol, subcarrier]"
        )
    if not bool(np.all(np.isfinite(h_freq))):
        raise OperationError("OFDM channel state must contain only finite values")
    return h_freq, dict(metadata)


def _load_power(path) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        if "power" not in payload.files:
            raise OperationError("OFDM allocation artifact is missing power")
        power = np.asarray(payload["power"], dtype=np.float64)
        metadata = (
            decode_strict_json_object(
                str(payload["metadata_json"]),
                label="OFDM power-allocation metadata_json",
            )
            if "metadata_json" in payload.files
            else {}
        )
    if power.ndim != 2 or any(int(size) < 1 for size in power.shape):
        raise OperationError(
            "OFDM allocation must have shape [state, subcarrier]"
        )
    if not bool(np.all(np.isfinite(power))):
        raise OperationError("OFDM allocation must contain only finite values")
    if bool(np.any(power < 0.0)):
        raise OperationError("OFDM allocation must be nonnegative")
    return power, dict(metadata)


def _load_delivery_bits(input_artifact: Any, label: str) -> Tuple[np.ndarray, JsonDict]:
    """Load one canonical bit artifact without trusting wrapper-only metadata."""

    with np.load(str(input_artifact.path), allow_pickle=False) as payload:
        if "bits" not in payload.files:
            raise OperationError("%s artifact is missing bits" % label)
        bits, _boundary_metadata = validate_channel_bits(
            payload["bits"], label=label
        )
        embedded = (
            decode_strict_json_object(
                str(payload["metadata_json"]),
                label="%s metadata_json" % label,
            )
            if "metadata_json" in payload.files
            else {}
        )
    fallback = dict(input_artifact.metadata or {})
    for key in (
        "nr_profile_sha256",
        "nr_transport_blocks",
        "nr_transport_block_crc_status",
    ):
        if key in fallback and key in embedded and fallback[key] != embedded[key]:
            raise OperationError(
                "%s wrapper metadata disagrees with embedded %s" % (label, key)
            )
    metadata = dict(fallback)
    metadata.update(embedded)
    declared_count = metadata.get("bit_count")
    if declared_count is not None and (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or int(declared_count) != int(bits.size)
    ):
        raise OperationError(
            "%s bit_count metadata does not match its bit array" % label
        )
    return bits, metadata


def _load_delivery_symbols(
    input_artifact: Any, label: str
) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(input_artifact.path), allow_pickle=False) as payload:
        if "symbols" not in payload.files:
            raise OperationError("%s artifact is missing symbols" % label)
        symbols, _boundary_metadata = validate_channel_symbols(
            payload["symbols"], label=label, representation="complex"
        )
        embedded = (
            decode_strict_json_object(
                str(payload["metadata_json"]),
                label="%s metadata_json" % label,
            )
            if "metadata_json" in payload.files
            else {}
        )
    fallback = dict(input_artifact.metadata or {})
    for key in ("nr_profile_sha256", "nr_transport_blocks"):
        if key in fallback and key in embedded and fallback[key] != embedded[key]:
            raise OperationError(
                "%s wrapper metadata disagrees with embedded %s" % (label, key)
            )
    metadata = dict(fallback)
    metadata.update(embedded)
    declared_count = metadata.get("symbol_count")
    if declared_count is not None and (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or int(declared_count) != int(symbols.size)
    ):
        raise OperationError(
            "%s symbol_count metadata does not match its symbol array" % label
        )
    return symbols, metadata


def _load_delivery_report(input_artifact: Any) -> JsonDict:
    try:
        report = decode_strict_json_object(
            input_artifact.path.read_text(encoding="utf-8"),
            label="NR LDPC decoder report",
        )
    except (OSError, ValueError) as exc:
        raise OperationError("Could not read the NR LDPC decoder report") from exc
    wrapper = dict(input_artifact.metadata or {})
    for key in (
        "profile_sha256",
        "transport_block_count",
        "transport_block_crc_status",
        "transport_block_crc_failure_count",
    ):
        if key in wrapper and wrapper[key] != report.get(key):
            raise OperationError(
                "NR LDPC decoder report wrapper disagrees with %s" % key
            )
    return report


def _strict_metadata_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OperationError("%s must be an integer" % label)
    result = int(value)
    if result < minimum:
        raise OperationError("%s must be at least %d" % (label, minimum))
    return result


def _assert_bound_metadata(
    metadata: Mapping[str, Any],
    canonical: Mapping[str, Any],
    label: str,
) -> None:
    for key, expected in canonical.items():
        if key not in metadata:
            raise OperationError("%s is missing encoder-bound %s" % (label, key))
        if metadata[key] != expected:
            raise OperationError(
                "%s encoder-bound %s does not match the NR LDPC profile"
                % (label, key)
            )


def _validated_nr_delivery_blocks(
    blocks: Any,
) -> Tuple[list[JsonDict], int, int]:
    if not isinstance(blocks, list) or not blocks:
        raise OperationError(
            "NR LDPC delivery requires non-empty nr_transport_blocks metadata"
        )
    validated: list[JsonDict] = []
    next_coded_offset = 0
    valid_payload_total = 0
    previous_source_item: int | None = None
    for index, raw_block in enumerate(blocks):
        if not isinstance(raw_block, Mapping):
            raise OperationError(
                "NR LDPC transport block %d must be an object" % index
            )
        block = dict(raw_block)
        source_item_index = _strict_metadata_int(
            block.get("source_item_index"),
            "NR LDPC transport block %d source_item_index" % index,
        )
        if (
            (previous_source_item is None and source_item_index != 0)
            or (
                previous_source_item is not None
                and source_item_index
                not in {previous_source_item, previous_source_item + 1}
            )
        ):
            raise OperationError(
                "NR LDPC source-item indices must be ordered and contiguous"
            )
        previous_source_item = source_item_index
        valid_count = _strict_metadata_int(
            block.get("valid_payload_bit_count"),
            "NR LDPC transport block %d valid_payload_bit_count" % index,
            minimum=1,
        )
        encoder_input_count = _strict_metadata_int(
            block.get("encoder_input_bit_count"),
            "NR LDPC transport block %d encoder_input_bit_count" % index,
            minimum=valid_count,
        )
        tail_padding = _strict_metadata_int(
            block.get("tail_zero_padding_bit_count"),
            "NR LDPC transport block %d tail_zero_padding_bit_count" % index,
        )
        if tail_padding != encoder_input_count - valid_count:
            raise OperationError(
                "NR LDPC transport block %d payload/tail accounting is inconsistent"
                % index
            )
        coded_offset = _strict_metadata_int(
            block.get("coded_offset"),
            "NR LDPC transport block %d coded_offset" % index,
        )
        coded_count = _strict_metadata_int(
            block.get("num_coded_bits"),
            "NR LDPC transport block %d num_coded_bits" % index,
            minimum=1,
        )
        if coded_offset != next_coded_offset:
            raise OperationError(
                "NR LDPC coded offsets must form one contiguous partition"
            )
        bits_per_symbol = _strict_metadata_int(
            block.get("num_bits_per_symbol"),
            "NR LDPC transport block %d num_bits_per_symbol" % index,
            minimum=1,
        )
        layers = _strict_metadata_int(
            block.get("num_layers"),
            "NR LDPC transport block %d num_layers" % index,
            minimum=1,
        )
        if bits_per_symbol != 2 or layers != 1:
            raise OperationError(
                "NR LDPC OFDM delivery v1 requires single-layer QPSK transport"
            )
        if coded_count % bits_per_symbol:
            raise OperationError(
                "NR LDPC transport block %d coded bits do not occupy whole QPSK data REs"
                % index
            )
        rate_matched_lengths = block.get("rate_matched_codeword_lengths")
        if rate_matched_lengths is not None:
            if (
                not isinstance(rate_matched_lengths, list)
                or not rate_matched_lengths
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or int(value) <= 0
                    for value in rate_matched_lengths
                )
                or sum(int(value) for value in rate_matched_lengths)
                != coded_count
            ):
                raise OperationError(
                    "NR LDPC transport block %d rate-matched lengths do not match num_coded_bits"
                    % index
                )
        validated.append(block)
        valid_payload_total += valid_count
        next_coded_offset += coded_count
    return validated, valid_payload_total, next_coded_offset


def _finite_metadata_float(
    metadata: Mapping[str, Any],
    key: str,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if key not in metadata:
        raise OperationError("%s metadata is missing %s" % (label, key))
    try:
        result = float(metadata[key])
    except (TypeError, ValueError) as exc:
        raise OperationError("%s metadata %s must be numeric" % (label, key)) from exc
    if not math.isfinite(result):
        raise OperationError("%s metadata %s must be finite" % (label, key))
    if positive and result <= 0.0:
        raise OperationError("%s metadata %s must be positive" % (label, key))
    if nonnegative and result < 0.0:
        raise OperationError(
            "%s metadata %s must be nonnegative" % (label, key)
        )
    return result


def _complex_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.complex128).reshape(-1)
    right_values = np.asarray(right, dtype=np.complex128).reshape(-1)
    denominator = math.sqrt(
        float(np.vdot(left_values, left_values).real)
        * float(np.vdot(right_values, right_values).real)
    )
    if denominator <= 1e-30:
        return 0.0
    return float(abs(np.vdot(left_values, right_values)) / denominator)


def _gain_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denominator = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    if denominator <= 1e-30:
        return 0.0
    return float(np.dot(x, y) / denominator)


class OfdmDelayedCsiOperation(Operation):
    """Derive aligned transmitter-visible and current CSI from one TDL path."""

    id = "wireless.ofdm_delayed_csi"
    name = "Delayed/noisy OFDM transmitter CSI"
    input_kinds = {"state": ["channel.ofdm_channel_state.numpy"]}
    output_kinds = {
        "actual_state": "channel.ofdm_channel_state.numpy",
        "transmitter_csi": "channel.ofdm_channel_state.numpy",
    }
    differentiability = {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": (
            "This operation freezes a causal pairing between old transmitter CSI "
            "and the later channel state applied by the physical link."
        ),
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": [],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "causal_same_trajectory_csi_aging",
            "status": "implemented",
        },
        {
            "runner": "dataset_capture",
            "backend": "numpy",
            "implementation": "causal_same_trajectory_csi_aging_capture",
            "status": "implemented",
        },
    ]
    formats = {"artifact": "npz", "tensor": "none"}
    params_schema = object_schema(
        {
            "feedback_delay_ofdm_symbols": {
                "type": "integer",
                "default": 8,
                "minimum": 0,
                "description": (
                    "CSI age in OFDM symbols. Old and current states are sliced "
                    "causally from the same Sionna TDL trajectory."
                ),
                "x-noema-ui": {"label": "CSI feedback delay (OFDM symbols)"},
            },
            "allocation_ofdm_symbols": {
                "type": "integer",
                "default": 24,
                "minimum": 1,
                "description": (
                    "Number of aligned old/current OFDM symbols retained per TDL block."
                ),
                "x-noema-ui": {"label": "Allocation interval (OFDM symbols)"},
            },
            "csi_history_length": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": (
                    "Number of consecutive causal complex-CSI snapshots exposed "
                    "to a history-aware transmitter policy. The newest snapshot "
                    "is still feedback_delay_ofdm_symbols older than its paired "
                    "current channel state."
                ),
                "x-noema-ui": {"label": "Causal CSI history length"},
            },
            "capture_temporal_stride": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": (
                    "Dataset-capture downsampling stride along each TDL "
                    "trajectory. Runtime allocation and scoring retain every "
                    "aligned state."
                ),
                "x-noema-ui": {
                    "label": "Capture temporal stride",
                    "allow_sweep": False,
                },
            },
            "add_estimation_noise": {
                "type": "boolean",
                "default": True,
                "description": "Add independent complex Gaussian CSI-estimation error.",
            },
            "csi_estimation_snr_db": {
                "type": "number",
                "default": 20.0,
                "description": (
                    "Mean old-channel power divided by complex CSI-estimation-error "
                    "variance, in dB."
                ),
                "x-noema-ui": {"label": "CSI estimation SNR (dB)"},
            },
            "seed": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Optional explicit estimation-error seed. Channel aging itself "
                    "comes from the upstream TDL trajectory."
                ),
            },
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        h_freq, source_metadata = _load_state(ctx.require_input("state").path)
        ctx.raise_if_cancelled()
        delay = int(ctx.params.get("feedback_delay_ofdm_symbols") or 0)
        history_length = max(
            1, int(ctx.params.get("csi_history_length") or 1)
        )
        capture_temporal_stride = max(
            1, int(ctx.params.get("capture_temporal_stride") or 1)
        )
        requested = int(
            ctx.params.get("allocation_ofdm_symbols")
            or max(
                1,
                int(h_freq.shape[1]) - delay - history_length + 1,
            )
        )
        available = (
            int(h_freq.shape[1]) - delay - history_length + 1
        )
        if available < 1:
            raise OperationError(
                "CSI history length %d and feedback delay %d leave no later "
                "OFDM symbol in a %d-symbol trajectory"
                % (history_length, delay, int(h_freq.shape[1]))
            )
        if requested > available:
            raise OperationError(
                "allocation_ofdm_symbols=%d exceeds the %d causal old/current pairs "
                "available with csi_history_length=%d and "
                "feedback_delay_ofdm_symbols=%d"
                % (requested, available, history_length, delay)
            )

        # Each current state j is paired with a window [j, ..., j+H-1].
        # The window's newest sample is exactly `delay` symbols older than the
        # current state at j+H-1+delay.  No sample at or after the current
        # state is ever forwarded to the transmitter.
        history_true = np.stack(
            [
                h_freq[:, offset : offset + requested, :]
                for offset in range(history_length)
            ],
            axis=2,
        ).astype(np.complex64, copy=True)
        old_true = history_true[:, :, -1, :]
        actual_start = history_length - 1 + delay
        actual = h_freq[
            :, actual_start : actual_start + requested, :
        ].astype(
            np.complex64, copy=True
        )
        observed_history = history_true.copy()
        estimation_snr_db = float(ctx.params.get("csi_estimation_snr_db") or 0.0)
        if not math.isfinite(estimation_snr_db):
            raise OperationError("csi_estimation_snr_db must be finite")
        estimation_error_variance = 0.0
        estimation_error_seed = int(ctx.seed("ofdm_delayed_csi"))
        if bool(ctx.params.get("add_estimation_noise", True)):
            old_power = float(np.mean(np.abs(history_true) ** 2))
            estimation_error_variance = old_power / (
                10.0 ** (estimation_snr_db / 10.0)
            )
            rng = np.random.RandomState(estimation_error_seed)
            sigma = math.sqrt(max(estimation_error_variance, 0.0) / 2.0)
            error = sigma * (
                rng.standard_normal(observed_history.shape)
                + 1j * rng.standard_normal(observed_history.shape)
            )
            observed_history = (
                observed_history + error.astype(np.complex64)
            ).astype(np.complex64, copy=False)
        observed = observed_history[:, :, -1, :]

        actual_gains = np.maximum(np.abs(actual) ** 2, 1e-12).astype(np.float32)
        observed_gains = np.maximum(np.abs(observed) ** 2, 1e-12).astype(
            np.float32
        )
        mismatch_nmse = float(
            np.mean(np.abs(observed.astype(np.complex128) - actual) ** 2)
            / max(float(np.mean(np.abs(actual) ** 2)), 1e-30)
        )
        complex_correlation = _complex_correlation(observed, actual)
        gain_correlation = _gain_correlation(observed_gains, actual_gains)
        subcarrier_spacing_hz = (
            float(source_metadata.get("subcarrier_spacing_khz") or 15.0) * 1e3
        )
        nominal_symbol_duration_s = 1.0 / max(subcarrier_spacing_hz, 1e-30)
        feedback_delay_s = float(delay) * nominal_symbol_duration_s
        snapshot_count = int(actual.shape[0] * actual.shape[1])
        capture_time_indices = np.arange(
            0,
            requested,
            capture_temporal_stride,
            dtype=np.int64,
        )
        capture_record_count = int(
            actual.shape[0] * capture_time_indices.size
        )

        common = dict(source_metadata)
        remove_capture_record_metadata(common)
        common.update(
            {
                "scenario_kind": "temporally_correlated_ofdm_with_delayed_transmitter_csi",
                "feedback_delay_ofdm_symbols": delay,
                "feedback_delay_seconds_nominal_without_cp": feedback_delay_s,
                "csi_history_length": history_length,
                "csi_history_spacing_ofdm_symbols": 1,
                "csi_history_order": "oldest_to_newest",
                "csi_history_newest_to_current_delay_ofdm_symbols": delay,
                "csi_history_contains_current_or_future_state": False,
                "allocation_ofdm_symbols": requested,
                "source_trajectory_ofdm_symbols": int(h_freq.shape[1]),
                "trajectory_pairing": "same_sionna_tdl_block_causal_slice",
                "trajectory_pairing_leaks_future_csi": False,
                "csi_estimation_noise_enabled": bool(
                    ctx.params.get("add_estimation_noise", True)
                ),
                "csi_estimation_snr_db": estimation_snr_db,
                "csi_estimation_error_variance": estimation_error_variance,
                "csi_estimation_error_seed": estimation_error_seed,
                "observed_actual_complex_correlation": complex_correlation,
                "observed_actual_gain_correlation": gain_correlation,
                "observed_actual_complex_nmse": mismatch_nmse,
                "ofdm_block_count": int(actual.shape[0]),
                "num_ofdm_symbols": requested,
                "ofdm_resource_element_capacity": int(np.prod(actual.shape)),
                "snapshot_count": snapshot_count,
                "capture_record_axis": 0,
                "capture_record_unit": "aligned_old_current_ofdm_symbol_state",
                "capture_record_count": capture_record_count,
                "capture_temporal_stride": capture_temporal_stride,
                "capture_selected_allocation_symbol_indices": (
                    capture_time_indices.tolist()
                ),
                "capture_selected_symbol_count_per_tdl_block": int(
                    capture_time_indices.size
                ),
                "capture_record_order": (
                    "ofdm_symbol_then_independent_tdl_block"
                ),
                "array": "gains",
            }
        )
        actual_metadata = {
            **common,
            "csi_role": "actual_current_channel_state",
            "csi_representation": "current_complex_frequency_response",
            "transmitter_visible": False,
            "channel_application_state": True,
        }
        observed_metadata = {
            **common,
            "csi_role": "delayed_noisy_transmitter_observation",
            "csi_representation": (
                "causal_delayed_noisy_complex_frequency_response_history"
                if history_length > 1
                else "delayed_noisy_complex_frequency_response"
            ),
            "transmitter_visible": True,
            "channel_application_state": False,
            "transmitter_csi_assumption": "delayed_noisy",
            "runtime_feature_array": (
                "csi_history" if history_length > 1 else "gains"
            ),
            "runtime_feature_layout": (
                "block,allocation_symbol,history,subcarrier,iq"
                if history_length > 1
                else "block,allocation_symbol,subcarrier"
            ),
        }

        actual_path = ctx.output_path("actual_state", ".npz")
        observed_path = ctx.output_path("transmitter_csi", ".npz")
        ctx.raise_if_cancelled()
        # Capture interleaves independent TDL blocks before advancing the
        # within-block time index.  A prefix-limited capture therefore sees
        # broad trajectory diversity instead of many adjacent correlated
        # symbols from only a few blocks.  Runtime consumers continue to use
        # h_freq (and the canonical block-major gains array).
        actual_capture_gains = np.transpose(
            actual_gains[:, capture_time_indices, :], (1, 0, 2)
        ).reshape(-1, actual.shape[-1])
        observed_capture_gains = np.transpose(
            observed_gains[:, capture_time_indices, :], (1, 0, 2)
        ).reshape(-1, observed.shape[-1])
        observed_history_iq = np.stack(
            [observed_history.real, observed_history.imag],
            axis=-1,
        ).astype(np.float32, copy=False)
        observed_capture_history = np.transpose(
            observed_history_iq[:, capture_time_indices, :, :, :],
            (1, 0, 2, 3, 4),
        ).reshape(
            -1,
            history_length,
            observed.shape[-1],
            2,
        )
        actual_metadata["array"] = "capture_gains"
        actual_metadata["capture_record_shape"] = [
            int(actual.shape[-1])
        ]
        observed_metadata["array"] = (
            "capture_csi_history"
            if history_length > 1
            else "capture_gains"
        )
        observed_metadata["capture_record_shape"] = (
            [history_length, int(observed.shape[-1]), 2]
            if history_length > 1
            else [int(observed.shape[-1])]
        )
        np.savez_compressed(
            actual_path,
            h_freq=actual,
            gains=actual_gains.reshape(-1, actual.shape[-1]),
            capture_gains=actual_capture_gains,
            metadata_json=json.dumps(actual_metadata, sort_keys=True),
        )
        np.savez_compressed(
            observed_path,
            h_freq=observed,
            gains=observed_gains.reshape(-1, observed.shape[-1]),
            capture_gains=observed_capture_gains,
            csi_history=observed_history_iq,
            capture_csi_history=observed_capture_history,
            metadata_json=json.dumps(observed_metadata, sort_keys=True),
        )
        metrics = {
            "resource.csi.feedback_delay_ofdm_symbols": delay,
            "resource.csi.feedback_delay_seconds": feedback_delay_s,
            "resource.csi.observed_actual_complex_correlation": complex_correlation,
            "resource.csi.observed_actual_gain_correlation": gain_correlation,
            "resource.csi.observed_actual_complex_nmse": mismatch_nmse,
            "resource.csi.estimation_error_variance": estimation_error_variance,
        }
        return OperationResult(
            outputs={
                "actual_state": artifact(
                    "channel.ofdm_channel_state.numpy",
                    actual_path,
                    actual_metadata,
                ),
                "transmitter_csi": artifact(
                    "channel.ofdm_channel_state.numpy",
                    observed_path,
                    observed_metadata,
                ),
            },
            metrics=metrics,
            metadata={
                "scenario_kind": common["scenario_kind"],
                "trajectory_pairing": common["trajectory_pairing"],
                "metrics": metrics,
            },
        )


def _normal_approximation_bler(
    capacity_bps_hz: np.ndarray,
    dispersion: np.ndarray,
    *,
    blocklength: int,
    target_rate_bps_hz: float,
    third_order: bool,
) -> np.ndarray:
    capacity = np.asarray(capacity_bps_hz, dtype=np.float64)
    variance = np.maximum(np.asarray(dispersion, dtype=np.float64), 1e-12)
    correction = (
        math.log2(float(blocklength)) / (2.0 * float(blocklength))
        if third_order
        else 0.0
    )
    z = (
        (capacity - float(target_rate_bps_hz) + correction)
        * math.sqrt(float(blocklength))
        / np.sqrt(variance)
    )
    return np.asarray(
        [0.5 * math.erfc(float(value) / math.sqrt(2.0)) for value in z],
        dtype=np.float64,
    )


def _reliability_preview(
    actual_gains: np.ndarray,
    observed_gains: np.ndarray,
    power: np.ndarray,
    noise_variance: float,
    state_metadata: Mapping[str, Any],
    allocation_metadata: Mapping[str, Any],
) -> JsonDict:
    count = min(int(actual_gains.shape[0]), 16)
    snapshots = []
    for index in range(count):
        actual = np.maximum(actual_gains[index], 1e-12)
        observed = np.maximum(observed_gains[index], 1e-12)
        snapshots.append(
            {
                "index": index,
                "channel_gain": actual.tolist(),
                "observed_channel_gain": observed.tolist(),
                "unit_power_snr_db": (
                    10.0 * np.log10(actual / noise_variance)
                ).tolist(),
                "observed_unit_power_snr_db": (
                    10.0 * np.log10(observed / noise_variance)
                ).tolist(),
                "inverse_unit_snr": (noise_variance / actual).tolist(),
                "allocated_power": power[index].tolist(),
            }
        )
    return {
        "schema_version": 2,
        "scenario_kind": "temporally_correlated_ofdm_with_delayed_transmitter_csi",
        "allocation_axis": "subcarrier",
        "allocation_granularity": "per_subcarrier_per_ofdm_symbol_state",
        "snapshot_axis": "aligned_old_current_ofdm_symbol_state",
        "snapshot_count": int(actual_gains.shape[0]),
        "preview_snapshot_count": count,
        "channel_count": int(actual_gains.shape[1]),
        "noise_variance": noise_variance,
        "reference_snr_db": 10.0
        * math.log10(
            max(
                float(state_metadata.get("average_power_budget") or 0.0),
                1e-30,
            )
            / noise_variance
        ),
        "total_power": float(allocation_metadata.get("total_power") or 0.0),
        "policy": str(allocation_metadata.get("policy") or "power_allocator"),
        "actual_trace_label": "Current channel",
        "observed_trace_label": "Delayed/noisy transmitter CSI",
        "feedback_delay_ofdm_symbols": int(
            state_metadata.get("feedback_delay_ofdm_symbols") or 0
        ),
        "snapshots": snapshots,
    }


class OfdmFiniteBlocklengthReliabilityMetricsOperation(Operation):
    id = "metrics.ofdm_finite_blocklength_allocation"
    name = "Finite-blocklength delayed-CSI allocation metrics"
    input_kinds = {
        "actual_state": ["channel.ofdm_channel_state.numpy"],
        "transmitter_csi": ["channel.ofdm_channel_state.numpy"],
        "allocation": ["channel.power_allocation.numpy"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema(
        {
            "blocklength_channel_uses": {
                "type": "integer",
                "default": 128,
                "minimum": 16,
                "description": (
                    "Short-packet blocklength used by the parallel-channel normal approximation."
                ),
            },
            "target_rate_bps_hz": {
                "type": "number",
                "default": 2.0,
                "minimum": 0.0,
                "description": "Fixed payload rate per complex channel use.",
            },
            "include_third_order_term": {
                "type": "boolean",
                "default": True,
                "description": "Include log2(n)/(2n) in the normal approximation.",
            },
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        actual_h, actual_metadata = _load_state(
            ctx.require_input("actual_state").path
        )
        observed_h, observed_metadata = _load_state(
            ctx.require_input("transmitter_csi").path
        )
        power, allocation_metadata = _load_power(
            ctx.require_input("allocation").path
        )
        ctx.raise_if_cancelled()
        actual_gains = np.maximum(np.abs(actual_h) ** 2, 1e-12).reshape(
            -1, actual_h.shape[-1]
        )
        observed_gains = np.maximum(np.abs(observed_h) ** 2, 1e-12).reshape(
            -1, observed_h.shape[-1]
        )
        if actual_gains.shape != observed_gains.shape:
            raise OperationError(
                "Actual and transmitter-visible CSI must contain aligned state rows"
            )
        if power.shape != actual_gains.shape:
            raise OperationError(
                "Allocation shape %s does not match aligned current CSI shape %s"
                % (power.shape, actual_gains.shape)
            )
        noise_variance = _finite_metadata_float(
            actual_metadata,
            "noise_variance",
            "actual current CSI",
            positive=True,
        )
        observed_noise = _finite_metadata_float(
            observed_metadata,
            "noise_variance",
            "transmitter CSI",
            positive=True,
        )
        if not math.isclose(
            noise_variance, observed_noise, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise OperationError(
                "Actual and transmitter CSI disagree on channel noise variance"
            )
        average_power = _finite_metadata_float(
            actual_metadata,
            "average_power_budget",
            "actual current CSI",
            nonnegative=True,
        )
        total_power = average_power * float(actual_gains.shape[1])
        row_power = np.sum(power, axis=1)
        power_error = np.abs(row_power - total_power)
        tolerance = max(1e-6, 1e-7 * max(1.0, abs(total_power)))
        if bool(np.any(power_error > tolerance)):
            index = int(np.argmax(power_error))
            raise OperationError(
                "Allocation violates the sum-power budget at state %d: %.12g "
                "versus %.12g"
                % (index, float(row_power[index]), total_power)
            )

        blocklength = int(
            ctx.params.get("blocklength_channel_uses") or 128
        )
        target_rate = float(ctx.params.get("target_rate_bps_hz") or 0.0)
        if not math.isfinite(target_rate) or target_rate < 0.0:
            raise OperationError("target_rate_bps_hz must be finite and nonnegative")
        snr = actual_gains * power / noise_variance
        per_tone_capacity = np.log2(1.0 + snr)
        state_capacity = np.mean(per_tone_capacity, axis=1)
        log2e = math.log2(math.e)
        per_tone_dispersion = (
            1.0 - np.power(1.0 + snr, -2.0)
        ) * (log2e**2)
        state_dispersion = np.mean(per_tone_dispersion, axis=1)
        predicted_bler = _normal_approximation_bler(
            state_capacity,
            state_dispersion,
            blocklength=blocklength,
            target_rate_bps_hz=target_rate,
            third_order=bool(
                ctx.params.get("include_third_order_term", True)
            ),
        )
        state_goodput = target_rate * (1.0 - predicted_bler)
        mean_goodput = float(np.mean(state_goodput))
        mean_bler = float(np.mean(predicted_bler))
        nominal_snr_db = 10.0 * math.log10(
            max(average_power, 1e-30) / noise_variance
        )
        observed_actual_complex_correlation = _complex_correlation(
            observed_h, actual_h
        )
        observed_actual_gain_correlation = _gain_correlation(
            observed_gains, actual_gains
        )
        csi_nmse = float(
            np.mean(np.abs(observed_h.astype(np.complex128) - actual_h) ** 2)
            / max(float(np.mean(np.abs(actual_h) ** 2)), 1e-30)
        )
        metrics = {
            "resource.finite_blocklength.expected_goodput_bps_hz": mean_goodput,
            "resource.finite_blocklength.predicted_bler": mean_bler,
            "resource.finite_blocklength.mean_capacity_bps_hz": float(
                np.mean(state_capacity)
            ),
            "resource.finite_blocklength.p05_goodput_bps_hz": float(
                np.percentile(state_goodput, 5.0)
            ),
            "resource.finite_blocklength.blocklength_channel_uses": blocklength,
            "resource.finite_blocklength.target_rate_bps_hz": target_rate,
            "resource.nominal_snr_db": nominal_snr_db,
            "resource.average_transmit_power_budget": average_power,
            "resource.csi.feedback_delay_ofdm_symbols": int(
                actual_metadata.get("feedback_delay_ofdm_symbols") or 0
            ),
            "resource.csi.feedback_delay_seconds": float(
                actual_metadata.get(
                    "feedback_delay_seconds_nominal_without_cp"
                )
                or 0.0
            ),
            "resource.csi.observed_actual_complex_correlation": (
                observed_actual_complex_correlation
            ),
            "resource.csi.observed_actual_gain_correlation": (
                observed_actual_gain_correlation
            ),
            "resource.csi.observed_actual_complex_nmse": csi_nmse,
            "resource.power_constraint.max_abs_error": float(
                np.max(power_error)
            ),
            "resource.power_constraint.max_relative_error": float(
                np.max(power_error) / max(abs(total_power), 1e-30)
            ),
            "resource.power_constraint.max_negative_violation": 0.0,
            "channel.noise_variance": noise_variance,
            "task.score": mean_goodput,
        }
        preview = _reliability_preview(
            actual_gains,
            observed_gains,
            power,
            noise_variance,
            actual_metadata,
            allocation_metadata,
        )
        rows = [
            {
                "state": int(index),
                "capacity_bps_hz": float(state_capacity[index]),
                "dispersion": float(state_dispersion[index]),
                "predicted_bler": float(predicted_bler[index]),
                "expected_goodput_bps_hz": float(state_goodput[index]),
            }
            for index in range(min(16, int(state_capacity.size)))
        ]
        report = {
            "schema_version": 1,
            "metric_family": "resource_allocation_finite_blocklength",
            "rows": rows,
            "metrics": metrics,
            "metadata": {
                "objective": (
                    "maximize_expected_finite_blocklength_goodput_under_delayed_noisy_csi"
                ),
                "normal_approximation": (
                    "epsilon=Q((C-R+log2(n)/(2n))*sqrt(n/V))"
                    if bool(ctx.params.get("include_third_order_term", True))
                    else "epsilon=Q((C-R)*sqrt(n/V))"
                ),
                "capacity_definition": (
                    "mean_subcarrier(log2(1+actual_gain*allocated_power/noise_variance))"
                ),
                "dispersion_definition": (
                    "mean_subcarrier((1-(1+snr)^-2)*(log2(e))^2)"
                ),
                "water_filling_role": (
                    "water_filling_on_delayed_csi_is_a_mismatched_baseline_not_an_oracle"
                ),
                "current_csi_available_to_allocator": False,
                "actual_state_metadata": actual_metadata,
                "transmitter_csi_metadata": observed_metadata,
                "allocation_metadata": allocation_metadata,
                "resource_allocation_preview": preview,
            },
            "resource_allocation_preview": preview,
        }
        report_path = ctx.output_path("report", ".json")
        ctx.raise_if_cancelled()
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return OperationResult(
            outputs={
                "report": artifact(
                    "metrics.report", report_path, report
                )
            },
            metrics=metrics,
            metadata=report,
        )


class NrLdpcOfdmDeliveryMetricsOperation(Operation):
    """CRC-gated, all-attempt NR-LDPC goodput over occupied QPSK data REs."""

    id = "metrics.nr_ldpc_ofdm_delivery"
    name = "NR LDPC OFDM delivery metrics (v1)"
    metric_profile = "noema.nr_ldpc_ofdm_delivery.v1"
    input_kinds = {
        "reference_payload": [
            "channel.payload_bits.numpy",
            "channel.bits.numpy",
        ],
        "decoded_payload": [
            "channel.payload_bits.numpy",
            "channel.bits.numpy",
        ],
        "decoder_report": ["metrics.report"],
        "tx_symbols": ["channel.symbols.complex_numpy"],
        "actual_state": ["channel.ofdm_channel_state.numpy"],
        "allocation": ["channel.power_allocation.numpy"],
    }
    optional_input_kinds = {
        "rx_symbols": ["channel.rx_symbols.complex_numpy"],
    }
    output_kinds = {"report": "metrics.report"}
    differentiability = {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": (
            "CRC decisions and bit-exact payload delivery are discrete evidence "
            "boundaries."
        ),
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": [],
    }
    equivalence = {
        "type": "exact",
        "reason": (
            "The v1 score is a deterministic count of CRC-gated payload bits and "
            "occupied QPSK data resource elements."
        ),
    }
    formats = {"artifact": "json", "tensor": "none"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        from noema_lab.ops.channel.nr_ldpc import _profile_sha256

        reference_bits, reference_metadata = _load_delivery_bits(
            ctx.require_input("reference_payload"),
            "reference payload",
        )
        decoded_bits, decoded_metadata = _load_delivery_bits(
            ctx.require_input("decoded_payload"),
            "decoded payload",
        )
        decoder_report = _load_delivery_report(
            ctx.require_input("decoder_report")
        )
        tx_symbols, tx_metadata = _load_delivery_symbols(
            ctx.require_input("tx_symbols"),
            "allocated transmit symbols",
        )
        actual_artifact = ctx.require_input("actual_state")
        actual_h, actual_metadata = _load_state(actual_artifact.path)
        allocation_artifact = ctx.require_input("allocation")
        power, allocation_metadata = _load_power(allocation_artifact.path)
        ctx.raise_if_cancelled()

        blocks, valid_payload_total, coded_bit_total = (
            _validated_nr_delivery_blocks(tx_metadata.get("nr_transport_blocks"))
        )
        backend_versions = tx_metadata.get("nr_backend_versions")
        if not isinstance(backend_versions, Mapping) or not backend_versions:
            raise OperationError(
                "Allocated transmit symbols are missing NR LDPC backend versions"
            )
        decoder_iterations = _strict_metadata_int(
            tx_metadata.get("nr_decoder_num_bp_iter"),
            "NR LDPC decoder iteration count",
            minimum=1,
        )
        profile_sha256 = tx_metadata.get("nr_profile_sha256")
        expected_profile_sha256 = _profile_sha256(
            blocks,
            decoder_iterations,
            backend_versions,
        )
        if (
            not isinstance(profile_sha256, str)
            or not profile_sha256
            or profile_sha256 != expected_profile_sha256
        ):
            raise OperationError(
                "Allocated transmit symbols carry an invalid NR LDPC profile hash"
            )
        canonical_binding = {
            "nr_profile_sha256": profile_sha256,
            "nr_transport_blocks": blocks,
            "nr_decoder_num_bp_iter": decoder_iterations,
            "nr_backend_versions": dict(backend_versions),
        }
        for metadata, label in (
            (tx_metadata, "allocated transmit symbols"),
            (decoded_metadata, "decoded payload"),
            (actual_metadata, "actual channel state"),
            (allocation_metadata, "power allocation"),
        ):
            if metadata is not tx_metadata:
                _assert_bound_metadata(metadata, canonical_binding, label)
            if _strict_metadata_int(
                metadata.get("nr_transport_block_count"),
                "%s nr_transport_block_count" % label,
                minimum=1,
            ) != len(blocks):
                raise OperationError(
                    "%s transport-block count does not match the NR LDPC records"
                    % label.capitalize()
                )
        for key in canonical_binding:
            fallback = dict(actual_artifact.metadata or {})
            if key in fallback and fallback[key] != actual_metadata[key]:
                raise OperationError(
                    "Actual channel-state wrapper metadata disagrees with embedded %s"
                    % key
                )
            fallback = dict(allocation_artifact.metadata or {})
            if key in fallback and fallback[key] != allocation_metadata[key]:
                raise OperationError(
                    "Power-allocation wrapper metadata disagrees with embedded %s"
                    % key
                )

        if valid_payload_total != int(reference_bits.size):
            raise OperationError(
                "NR LDPC valid-payload sum %d does not match reference payload size %d"
                % (valid_payload_total, int(reference_bits.size))
            )
        if int(decoded_bits.size) != valid_payload_total:
            raise OperationError(
                "Decoded payload size does not match the NR LDPC valid-payload sum"
            )
        for metadata, label in (
            (tx_metadata, "allocated transmit symbols"),
            (decoded_metadata, "decoded payload"),
        ):
            declared_input = metadata.get("channel_code_input_bit_count")
            if declared_input is None or _strict_metadata_int(
                declared_input,
                "%s channel_code_input_bit_count" % label,
            ) != valid_payload_total:
                raise OperationError(
                    "%s payload-bit accounting does not match the NR LDPC blocks"
                    % label
                )
            declared_output = metadata.get("channel_code_output_bit_count")
            if declared_output is None or _strict_metadata_int(
                declared_output,
                "%s channel_code_output_bit_count" % label,
            ) != coded_bit_total:
                raise OperationError(
                    "%s coded-bit accounting does not match the NR LDPC blocks"
                    % label
                )

        if str(tx_metadata.get("modulation") or "").lower() != "qpsk":
            raise OperationError(
                "NR LDPC OFDM delivery v1 requires QPSK transmit symbols"
            )
        if _strict_metadata_int(
            tx_metadata.get("bits_per_symbol"),
            "allocated transmit-symbol bits_per_symbol",
            minimum=1,
        ) != 2:
            raise OperationError(
                "NR LDPC OFDM delivery v1 requires two coded bits per data RE"
            )
        if str(tx_metadata.get("transport_mode") or "fixed_modulation") != (
            "fixed_modulation"
        ):
            raise OperationError(
                "NR LDPC OFDM delivery v1 does not permit adaptive bit-loading"
            )
        occupied_data_re = int(tx_symbols.size)
        expected_occupied_data_re = coded_bit_total // 2
        if occupied_data_re != expected_occupied_data_re:
            raise OperationError(
                "Occupied QPSK data RE count %d does not match %d NR coded bits"
                % (occupied_data_re, coded_bit_total)
            )
        padded_bit_count = tx_metadata.get("padded_bit_count")
        if padded_bit_count is not None and _strict_metadata_int(
            padded_bit_count,
            "allocated transmit-symbol padded_bit_count",
        ) != coded_bit_total:
            raise OperationError(
                "QPSK modulation padding is inconsistent with the NR coded stream"
            )

        block_count, ofdm_symbol_count, subcarrier_count = (
            int(actual_h.shape[0]),
            int(actual_h.shape[1]),
            int(actual_h.shape[2]),
        )
        state_grid_capacity = block_count * ofdm_symbol_count * subcarrier_count
        expected_power_shape = (
            block_count * ofdm_symbol_count,
            subcarrier_count,
        )
        if power.shape != expected_power_shape:
            raise OperationError(
                "Power-allocation shape %s does not match actual-state grid %s"
                % (power.shape, expected_power_shape)
            )
        if occupied_data_re > state_grid_capacity:
            raise OperationError(
                "Actual OFDM state does not cover every occupied QPSK data RE"
            )
        if actual_metadata.get("channel_application_state") is not True:
            raise OperationError(
                "actual_state is not marked as the channel-application state"
            )
        if actual_metadata.get("transmitter_visible") is not False:
            raise OperationError(
                "actual_state must not be transmitter-visible in this delayed-CSI score"
            )
        state_seed = _strict_metadata_int(
            actual_metadata.get("channel_state_seed"),
            "actual-state channel_state_seed",
        )
        for metadata, label in (
            (allocation_metadata, "power allocation"),
            (tx_metadata, "allocated transmit symbols"),
            (decoded_metadata, "decoded payload"),
        ):
            if _strict_metadata_int(
                metadata.get("channel_state_seed"),
                "%s channel_state_seed" % label,
            ) != state_seed:
                raise OperationError(
                    "%s is bound to a different OFDM channel-state seed"
                    % label.capitalize()
                )
        declared_capacity = actual_metadata.get("ofdm_resource_element_capacity")
        if declared_capacity is not None and _strict_metadata_int(
            declared_capacity,
            "actual-state ofdm_resource_element_capacity",
            minimum=1,
        ) != state_grid_capacity:
            raise OperationError(
                "Actual-state resource-element capacity metadata is inconsistent"
            )
        declared_snapshots = allocation_metadata.get("snapshot_count")
        if declared_snapshots is not None and _strict_metadata_int(
            declared_snapshots,
            "power-allocation snapshot_count",
            minimum=1,
        ) != expected_power_shape[0]:
            raise OperationError(
                "Power-allocation snapshot count does not match the actual state"
            )
        declared_subcarriers = allocation_metadata.get("subcarrier_count")
        if declared_subcarriers is not None and _strict_metadata_int(
            declared_subcarriers,
            "power-allocation subcarrier_count",
            minimum=1,
        ) != subcarrier_count:
            raise OperationError(
                "Power-allocation subcarrier count does not match the actual state"
            )

        noise_variance = _finite_metadata_float(
            actual_metadata,
            "noise_variance",
            "actual channel state",
            positive=True,
        )
        for metadata, label in (
            (allocation_metadata, "power allocation"),
            (tx_metadata, "allocated transmit symbols"),
            (decoded_metadata, "decoded payload"),
        ):
            bound_noise = _finite_metadata_float(
                metadata, "noise_variance", label, positive=True
            )
            if not math.isclose(
                noise_variance, bound_noise, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise OperationError(
                    "%s noise_variance does not match the actual channel state"
                    % label.capitalize()
                )
        average_power_budget = _finite_metadata_float(
            actual_metadata,
            "average_power_budget",
            "actual channel state",
            nonnegative=True,
        )
        total_power_budget = average_power_budget * float(subcarrier_count)
        allocation_target = _finite_metadata_float(
            allocation_metadata,
            "target_power",
            "power allocation",
            nonnegative=True,
        )
        allocation_total = _finite_metadata_float(
            allocation_metadata,
            "total_power",
            "power allocation",
            nonnegative=True,
        )
        if not math.isclose(
            allocation_target,
            average_power_budget,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ) or not math.isclose(
            allocation_total,
            total_power_budget,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise OperationError(
                "Power-allocation budget metadata does not match the actual state"
            )
        tx_selected_power = _finite_metadata_float(
            tx_metadata,
            "power_allocator_selected_power",
            "allocated transmit symbols",
            nonnegative=True,
        )
        if not math.isclose(
            tx_selected_power,
            average_power_budget,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise OperationError(
                "Allocated transmit-symbol power budget does not match the actual state"
            )
        row_power = np.sum(power, axis=1)
        power_error = np.abs(row_power - total_power_budget)
        tolerance = max(1e-6, 1e-7 * max(1.0, abs(total_power_budget)))
        if bool(np.any(power_error > tolerance)):
            row = int(np.argmax(power_error))
            raise OperationError(
                "Power allocation violates the sum-power budget at state %d: %.12g versus %.12g"
                % (row, float(row_power[row]), total_power_budget)
            )
        scheduled_power = power.reshape(-1)[:occupied_data_re]
        realized_tx_power = np.abs(tx_symbols.astype(np.complex128)) ** 2
        symbol_power_tolerance = max(
            1e-7,
            2e-5 * max(1.0, float(np.max(scheduled_power, initial=0.0))),
        )
        if not np.allclose(
            realized_tx_power,
            scheduled_power,
            rtol=2e-5,
            atol=symbol_power_tolerance,
        ):
            raise OperationError(
                "Allocated transmit-symbol powers do not match the bound power map"
            )

        status_values = decoder_report.get("transport_block_crc_status")
        if (
            not isinstance(status_values, list)
            or len(status_values) != len(blocks)
            or any(not isinstance(value, bool) for value in status_values)
        ):
            raise OperationError(
                "NR decoder CRC status count does not match the transport blocks"
            )
        statuses = [bool(value) for value in status_values]
        if decoder_report.get("profile_sha256") != profile_sha256:
            raise OperationError(
                "NR decoder report profile hash does not match the coded stream"
            )
        if _strict_metadata_int(
            decoder_report.get("transport_block_count"),
            "NR decoder report transport_block_count",
            minimum=1,
        ) != len(blocks):
            raise OperationError(
                "NR decoder report transport-block count is inconsistent"
            )
        failure_indices = [
            index for index, passed in enumerate(statuses) if not passed
        ]
        if _strict_metadata_int(
            decoder_report.get("transport_block_crc_failure_count"),
            "NR decoder report CRC failure count",
        ) != len(failure_indices):
            raise OperationError(
                "NR decoder report CRC failure count is inconsistent"
            )
        raw_failure_indices = decoder_report.get("failed_transport_block_indices")
        if raw_failure_indices != failure_indices:
            raise OperationError(
                "NR decoder report failed-block indices are inconsistent"
            )
        if _strict_metadata_int(
            decoder_report.get("decoded_bit_count"),
            "NR decoder report decoded_bit_count",
        ) != int(decoded_bits.size):
            raise OperationError(
                "NR decoder report decoded-bit count is inconsistent"
            )
        expected_bler = float(len(failure_indices)) / float(len(blocks))
        try:
            reported_bler = float(decoder_report["transport_block_error_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OperationError(
                "NR decoder report transport_block_error_rate must be numeric"
            ) from exc
        if not math.isfinite(reported_bler) or not math.isclose(
            reported_bler, expected_bler, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise OperationError(
                "NR decoder report transport-block error rate is inconsistent"
            )
        decoded_statuses = decoded_metadata.get(
            "nr_transport_block_crc_status"
        )
        if decoded_statuses != statuses:
            raise OperationError(
                "Decoded payload CRC statuses do not match the decoder report"
            )

        simulator_executed_uses: int | None = None
        simulator_grid_padding: int | None = None
        if "rx_symbols" in ctx.inputs:
            rx_symbols, rx_metadata = _load_delivery_symbols(
                ctx.inputs["rx_symbols"], "received symbols"
            )
            _assert_bound_metadata(rx_metadata, canonical_binding, "received symbols")
            if _strict_metadata_int(
                rx_metadata.get("channel_state_seed"),
                "received-symbol channel_state_seed",
            ) != state_seed:
                raise OperationError(
                    "Received symbols are bound to a different OFDM channel-state seed"
                )
            if int(rx_symbols.size) != occupied_data_re:
                raise OperationError(
                    "Received-symbol count does not match occupied transmitted data REs"
                )
            rx_noise = _finite_metadata_float(
                rx_metadata, "noise_variance", "received symbols", positive=True
            )
            if not math.isclose(
                rx_noise, noise_variance, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise OperationError(
                    "Received-symbol noise variance does not match the actual state"
                )
            simulator_executed_uses = _strict_metadata_int(
                rx_metadata.get("channel_use_count"),
                "received-symbol simulator channel_use_count",
                minimum=1,
            )
            simulator_grid_padding = _strict_metadata_int(
                rx_metadata.get("grid_padding_symbol_count"),
                "received-symbol grid_padding_symbol_count",
            )
            payload_symbol_count = _strict_metadata_int(
                rx_metadata.get("payload_symbol_count"),
                "received-symbol payload_symbol_count",
                minimum=1,
            )
            if (
                payload_symbol_count != occupied_data_re
                or simulator_executed_uses != state_grid_capacity
                or simulator_grid_padding
                != simulator_executed_uses - occupied_data_re
            ):
                raise OperationError(
                    "Received-symbol occupied/executed/grid-padding accounting is inconsistent"
                )

        rows: list[JsonDict] = []
        payload_offset = 0
        delivered_payload_bits = 0
        for index, (block, crc_passed) in enumerate(zip(blocks, statuses)):
            valid_count = int(block["valid_payload_bit_count"])
            coded_offset = int(block["coded_offset"])
            coded_count = int(block["num_coded_bits"])
            reference_row = reference_bits[
                payload_offset : payload_offset + valid_count
            ]
            decoded_row = decoded_bits[
                payload_offset : payload_offset + valid_count
            ]
            mismatch_count = int(np.count_nonzero(reference_row != decoded_row))
            if crc_passed and mismatch_count:
                raise OperationError(
                    "NR transport block %d passed CRC but its decoded payload mismatches the reference"
                    % index
                )
            delivered = valid_count if crc_passed else 0
            block_data_re = coded_count // 2
            delivered_payload_bits += delivered
            rows.append(
                {
                    "transport_block_index": index,
                    "source_item_index": int(block["source_item_index"]),
                    "payload_bit_offset": payload_offset,
                    "valid_payload_bit_count": valid_count,
                    "encoder_input_bit_count": int(
                        block["encoder_input_bit_count"]
                    ),
                    "tail_zero_padding_bit_count": int(
                        block["tail_zero_padding_bit_count"]
                    ),
                    "effective_tb_size_bits": block.get(
                        "effective_tb_size_bits"
                    ),
                    "tb_quantization_padding_bits": block.get(
                        "tb_quantization_padding_bits"
                    ),
                    "target_coderate": block.get("target_coderate"),
                    "effective_valid_payload_coderate": block.get(
                        "effective_valid_payload_coderate"
                    ),
                    "num_code_blocks": block.get("num_code_blocks"),
                    "tb_crc_length_bits": block.get("tb_crc_length_bits"),
                    "cb_crc_length_bits_per_code_block": block.get(
                        "cb_crc_length_bits_per_code_block"
                    ),
                    "total_logical_crc_bits": block.get(
                        "total_logical_crc_bits"
                    ),
                    "base_graph": block.get("base_graph"),
                    "lifting_size": block.get("lifting_size"),
                    "lifting_set_index": block.get("lifting_set_index"),
                    "ldpc_information_size": block.get(
                        "ldpc_information_size"
                    ),
                    "ldpc_mother_code_bits": block.get(
                        "ldpc_mother_code_bits"
                    ),
                    "rate_matched_codeword_lengths": block.get(
                        "rate_matched_codeword_lengths"
                    ),
                    "coded_bit_offset": coded_offset,
                    "coded_bit_count": coded_count,
                    "qpsk_data_resource_element_offset": coded_offset // 2,
                    "occupied_qpsk_data_resource_element_count": block_data_re,
                    "crc_passed": crc_passed,
                    "payload_mismatch_bit_count": mismatch_count,
                    "delivered_payload_bit_count": delivered,
                    "delivery_success": crc_passed,
                    "goodput_bits_per_occupied_qpsk_data_re": (
                        float(delivered) / float(block_data_re)
                    ),
                }
            )
            payload_offset += valid_count

        goodput = float(delivered_payload_bits) / float(occupied_data_re)
        delivered_tb_count = len(blocks) - len(failure_indices)
        metrics: JsonDict = {
            "channel.nr_ldpc_ofdm.all_attempt_goodput_bits_per_occupied_data_resource_element": goodput,
            "channel.nr_ldpc_ofdm.transport_block_error_rate": expected_bler,
            "channel.nr_ldpc_ofdm.transport_block_crc_failure_count": len(
                failure_indices
            ),
            "channel.nr_ldpc_ofdm.attempted_transport_block_count": len(blocks),
            "channel.nr_ldpc_ofdm.delivered_transport_block_count": delivered_tb_count,
            "channel.nr_ldpc_ofdm.attempted_payload_bit_count": valid_payload_total,
            "channel.nr_ldpc_ofdm.delivered_payload_bit_count": delivered_payload_bits,
            "channel.nr_ldpc_ofdm.occupied_data_resource_element_count": occupied_data_re,
            "channel.nr_ldpc.goodput_bits_per_occupied_qpsk_data_re": goodput,
            "channel.achieved_payload_goodput_bits_per_resource_element": goodput,
            "channel.nr_ldpc.transport_block_error_rate": expected_bler,
            "channel.nr_ldpc.transport_block_crc_failure_count": len(
                failure_indices
            ),
            "channel.nr_ldpc.attempted_transport_block_count": len(blocks),
            "channel.nr_ldpc.delivered_transport_block_count": delivered_tb_count,
            "channel.nr_ldpc.attempted_payload_bit_count": valid_payload_total,
            "channel.nr_ldpc.delivered_payload_bit_count": delivered_payload_bits,
            "channel.occupied_qpsk_data_resource_element_count": occupied_data_re,
            "channel.ofdm_state_grid_resource_element_capacity": state_grid_capacity,
            "resource.power_constraint.max_abs_error": float(
                np.max(power_error)
            ),
            "resource.power_constraint.max_negative_violation": 0.0,
            "channel.noise_variance": noise_variance,
            "task.score": goodput,
        }
        if simulator_executed_uses is not None:
            metrics.update(
                {
                    "channel.nr_ldpc_ofdm.simulator_executed_resource_element_count": (
                        simulator_executed_uses
                    ),
                    "channel.nr_ldpc_ofdm.simulator_grid_padding_resource_element_count": (
                        simulator_grid_padding
                    ),
                    "channel.simulator_executed_channel_use_count": (
                        simulator_executed_uses
                    ),
                    "channel.simulator_grid_padding_channel_use_count": (
                        simulator_grid_padding
                    ),
                }
            )
        report: JsonDict = {
            "schema_version": 1,
            "metric_profile": self.metric_profile,
            "metric_family": "crc_gated_nr_ldpc_ofdm_delivery",
            "primary_metric": (
                "channel.nr_ldpc_ofdm.all_attempt_goodput_bits_per_occupied_data_resource_element"
            ),
            "denominator_policy": (
                "occupied_transmitted_single_layer_qpsk_data_resource_elements; "
                "all_attempted_transport_blocks_included; simulator_grid_padding_excluded"
            ),
            "rows": rows,
            "metrics": metrics,
            "metadata": {
                "crc_status_authoritative": True,
                "crc_failed_transport_block_delivery_bits": 0,
                "crc_pass_payload_mismatch_policy": "raise_operation_error",
                "nr_profile_sha256": profile_sha256,
                "nr_decoder_num_bp_iter": decoder_iterations,
                "nr_backend_versions": dict(backend_versions),
                "transport_block_count": len(blocks),
                "failed_transport_block_indices": failure_indices,
                "occupied_qpsk_data_resource_element_count": occupied_data_re,
                "actual_state_grid_resource_element_capacity": state_grid_capacity,
                "simulator_executed_channel_use_count": simulator_executed_uses,
                "simulator_grid_padding_channel_use_count": simulator_grid_padding,
                "noise_variance": noise_variance,
                "average_power_budget": average_power_budget,
                "total_power_budget_per_ofdm_symbol": total_power_budget,
                "allocation_policy": allocation_metadata.get("policy"),
                "power_constraint_tolerance": tolerance,
            },
        }
        report_path = ctx.output_path("report", ".json")
        ctx.raise_if_cancelled()
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return OperationResult(
            outputs={"report": artifact("metrics.report", report_path, report)},
            metrics=metrics,
            metadata=report,
        )
