from __future__ import annotations

"""A fixed, auditable NR-framed PUSCH domain anchor.

This operation deliberately implements one narrow Sionna-supported PUSCH
coordinate.  It is a domain-reference workload, not an NR conformance claim or
an absolute link-budget model.  Keeping the physical coordinate fixed in code
makes accidental changes to PRB, DMRS, CP, MCS, receiver, and accounting
semantics change the operation source identity instead of looking like an
ordinary parameter sweep.
"""

import importlib.util
import json
import math
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.boundaries import validate_payload_bits
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationResult,
    object_schema,
)
from noema_lab.core.reproducibility import (
    canonical_json_sha256,
    installed_dependency_version,
)
from noema_lab.core.structured_input import decode_strict_json_object


JsonDict = Dict[str, Any]

ANCHOR_ID = "nr-pusch-domain-anchor-v1"
REPORT_KIND = "noema.nr_pusch_domain_anchor_result"
REPORT_SCHEMA_VERSION = 1
SEED_MODULUS = 2**31 - 1
CHANNEL_SEED_OFFSET = 100_000
RESERVED_IMPAIRMENT_SEED_OFFSET = 200_000
NOISE_SEED_OFFSET = 300_000
MAX_MASTER_SEED = SEED_MODULUS - 1 - NOISE_SEED_OFFSET

# Fixed carrier and allocation coordinate.
SUBCARRIER_SPACING_HZ = 15_000.0
CARRIER_RESOURCE_BLOCKS = 25
PUSCH_RESOURCE_BLOCKS = 10
SUBCARRIERS_PER_RESOURCE_BLOCK = 12
PUSCH_SUBCARRIERS = PUSCH_RESOURCE_BLOCKS * SUBCARRIERS_PER_RESOURCE_BLOCK
OUTER_FFT_SIZE = 512
OUTER_SAMPLE_RATE_HZ = OUTER_FFT_SIZE * SUBCARRIER_SPACING_HZ
CARRIER_ACTIVE_SUBCARRIERS = (
    CARRIER_RESOURCE_BLOCKS * SUBCARRIERS_PER_RESOURCE_BLOCK
)
CARRIER_LEFT_GUARD_SUBCARRIERS = (
    OUTER_FFT_SIZE - CARRIER_ACTIVE_SUBCARRIERS
) // 2
PUSCH_START_RESOURCE_BLOCK = 7
PUSCH_OUTER_START = (
    CARRIER_LEFT_GUARD_SUBCARRIERS
    + PUSCH_START_RESOURCE_BLOCK * SUBCARRIERS_PER_RESOURCE_BLOCK
)
PUSCH_OUTER_STOP = PUSCH_OUTER_START + PUSCH_SUBCARRIERS

SLOT_OFDM_SYMBOLS = 14
NORMAL_CP_SAMPLES = (40, 36, 36, 36, 36, 36, 36, 40, 36, 36, 36, 36, 36, 36)
SLOT_SAMPLE_COUNT = SLOT_OFDM_SYMBOLS * OUTER_FFT_SIZE + sum(NORMAL_CP_SAMPLES)
SLOT_DURATION_S = SLOT_SAMPLE_COUNT / OUTER_SAMPLE_RATE_HZ

MCS_TABLE = 1
MCS_INDEX = 7
MCS_TARGET_CODERATE_X1024 = 526
MCS_TARGET_CODERATE = MCS_TARGET_CODERATE_X1024 / 1024.0
MODULATION = "QPSK"
BITS_PER_SYMBOL = 2
TRANSPORT_BLOCK_SIZE_BITS = 1608
CODED_BITS = 3120
DATA_RESOURCE_ELEMENTS = 1560
DMRS_RESOURCE_ELEMENTS = 120
ALLOCATED_RESOURCE_ELEMENTS = 1680
CP_TIME_BANDWIDTH_RESOURCE_UNITS = 1800

TDL_MODEL = "C300"
TDL_DELAY_SPREAD_S = 300e-9
TDL_MAXIMUM_PATH_DELAY_S = 2595e-9
CARRIER_FREQUENCY_HZ = 3.5e9
MOBILITY_KMH = 120.0
MOBILITY_MPS = MOBILITY_KMH / 3.6
CHANNEL_L_MIN = -6
CHANNEL_L_MAX = 26
DECODER_BP_ITERATIONS = 20

_ANCHOR_LOCK = threading.RLock()


def _fixed_configuration() -> JsonDict:
    return {
        "scope": {
            "label": "normalized_siso_nr_framed_pusch_domain_reference",
            "nr_conformance_claim": False,
            "deployed_throughput_claim": False,
            "absolute_link_budget": False,
            "channel_normalized": True,
        },
        "carrier": {
            "subcarrier_spacing_hz": SUBCARRIER_SPACING_HZ,
            "carrier_resource_blocks": CARRIER_RESOURCE_BLOCKS,
            "carrier_active_subcarriers": CARRIER_ACTIVE_SUBCARRIERS,
            "outer_fft_size": OUTER_FFT_SIZE,
            "sample_rate_hz": OUTER_SAMPLE_RATE_HZ,
            "left_guard_subcarriers": CARRIER_LEFT_GUARD_SUBCARRIERS,
            "right_guard_subcarriers": CARRIER_LEFT_GUARD_SUBCARRIERS,
        },
        "pusch": {
            "resource_blocks": PUSCH_RESOURCE_BLOCKS,
            "subcarriers": PUSCH_SUBCARRIERS,
            "start_resource_block_within_carrier_grid": PUSCH_START_RESOURCE_BLOCK,
            "outer_subcarrier_start_inclusive": PUSCH_OUTER_START,
            "outer_subcarrier_stop_exclusive": PUSCH_OUTER_STOP,
            "mapping_type": "A",
            "symbol_allocation": [0, SLOT_OFDM_SYMBOLS],
            "num_layers": 1,
            "num_antenna_ports": 1,
            "precoding": "non-codebook",
            "transform_precoding": False,
            "n_rnti": 1,
            "n_id": 1,
            "n_cell_id": 1,
        },
        "cyclic_prefix": {
            "type": "normal",
            "samples_per_symbol": list(NORMAL_CP_SAMPLES),
            "slot_sample_count": SLOT_SAMPLE_COUNT,
            "slot_duration_s": SLOT_DURATION_S,
        },
        "dmrs": {
            "config_type": 1,
            "type_a_position": 2,
            "additional_position": 1,
            "length": 1,
            "port_set": [0],
            "num_cdm_groups_without_data": 1,
            "symbol_indices_zero_based": [2, 11],
        },
        "transport": {
            "channel_type": "PUSCH",
            "mcs_table": MCS_TABLE,
            "mcs_index": MCS_INDEX,
            "modulation": MODULATION,
            "bits_per_symbol": BITS_PER_SYMBOL,
            "target_coderate_x1024": MCS_TARGET_CODERATE_X1024,
            "target_coderate": MCS_TARGET_CODERATE,
            "transport_block_size_bits": TRANSPORT_BLOCK_SIZE_BITS,
            "coded_bits": CODED_BITS,
            "decoder_bp_iterations": DECODER_BP_ITERATIONS,
            "crc_authoritative": True,
        },
        "resource_accounting": {
            "data_resource_elements": DATA_RESOURCE_ELEMENTS,
            "dmrs_resource_elements": DMRS_RESOURCE_ELEMENTS,
            "allocated_resource_elements": ALLOCATED_RESOURCE_ELEMENTS,
            "cp_time_bandwidth_resource_units": CP_TIME_BANDWIDTH_RESOURCE_UNITS,
            "allocation_bandwidth_hz": PUSCH_SUBCARRIERS * SUBCARRIER_SPACING_HZ,
        },
        "channel": {
            "model": TDL_MODEL,
            "delay_spread_s": TDL_DELAY_SPREAD_S,
            "maximum_path_delay_s": TDL_MAXIMUM_PATH_DELAY_S,
            "carrier_frequency_hz": CARRIER_FREQUENCY_HZ,
            "mobility_kmh": MOBILITY_KMH,
            "mobility_mps": MOBILITY_MPS,
            "normalize_channel": True,
            "l_min": CHANNEL_L_MIN,
            "l_max": CHANNEL_L_MAX,
            "discrete_impulse_response_support_extent_samples": (
                CHANNEL_L_MAX - CHANNEL_L_MIN
            ),
            "discrete_impulse_response_tap_count": (
                CHANNEL_L_MAX - CHANNEL_L_MIN + 1
            ),
            "shortest_cyclic_prefix_samples": min(NORMAL_CP_SAMPLES),
            "tap_count_strictly_less_than_shortest_cyclic_prefix": (
                CHANNEL_L_MAX - CHANNEL_L_MIN + 1
                < min(NORMAL_CP_SAMPLES)
            ),
        },
        "receiver": {
            "channel_estimator": "PUSCHLSChannelEstimator_linear_interpolation",
            "detector": "LMMSE_bit_maxlog",
            "decoder_bp_iterations": DECODER_BP_ITERATIONS,
            "perfect_receiver_csi": False,
        },
        "randomness": {
            "channel_seed_offset": CHANNEL_SEED_OFFSET,
            "reserved_impairment_seed_offset": (
                RESERVED_IMPAIRMENT_SEED_OFFSET
            ),
            "noise_seed_offset": NOISE_SEED_OFFSET,
            "noise_pairing_across_ebno": (
                "same standardized complex-noise draw, rescaled by variance"
            ),
        },
    }


FIXED_CONFIGURATION_SHA256 = canonical_json_sha256(_fixed_configuration())


def _sionna_available() -> bool:
    version = installed_dependency_version("sionna") or ""
    major = version.split(".", 1)[0]
    return (
        importlib.util.find_spec("sionna") is not None
        and importlib.util.find_spec("torch") is not None
        and major.isdigit()
        and int(major) >= 2
    )


def _availability() -> JsonDict:
    if _sionna_available():
        return {
            "available": True,
            "extra": "wireless",
            "implementation": (
                "Sionna PUSCH frequency grid plus explicit 512-point normal-CP "
                "time-domain TDL-C300 wrapper"
            ),
            "backend_versions": {
                "sionna": installed_dependency_version("sionna"),
                "torch": installed_dependency_version("torch"),
            },
            "nr_conformance_claim": False,
        }
    return {
        "available": False,
        "optional": True,
        "extra": "wireless",
        "missing": ["sionna>=2", "torch"],
        "reason": (
            'Install with `python -m pip install "noema-lab[wireless]"` or '
            "`uv sync --extra wireless` to execute the PUSCH domain anchor"
        ),
        "nr_conformance_claim": False,
    }


def _load_payload_bits(path: Path) -> Tuple[np.ndarray, JsonDict]:
    try:
        with np.load(str(path), allow_pickle=False) as payload:
            if "bits" not in payload.files:
                raise OperationError("PUSCH anchor payload artifact is missing bits")
            bits, boundary = validate_payload_bits(
                payload["bits"], label="nr_pusch_anchor.payload"
            )
            metadata = (
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="PUSCH anchor payload metadata_json",
                )
                if "metadata_json" in payload.files
                else {}
            )
    except OperationError:
        raise
    except Exception as exc:
        raise OperationError("Could not load PUSCH anchor payload bits") from exc
    if int(bits.size) != TRANSPORT_BLOCK_SIZE_BITS:
        raise OperationError(
            "PUSCH anchor requires exactly %d payload bits, got %d"
            % (TRANSPORT_BLOCK_SIZE_BITS, int(bits.size))
        )
    return np.ascontiguousarray(bits.reshape(-1), dtype=np.uint8), {
        **metadata,
        **boundary,
    }


def _source_binding(metadata: Mapping[str, Any]) -> JsonDict:
    keys = (
        "dataset_manifest_sha256",
        "ordered_post_transform_sha256",
        "batch_tensor_sha256",
        "source_item_ids",
        "source_item_content_sha256",
        "source_group_ids",
        "source_split_ids",
        "source_item_payload_bit_counts",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def _backend_versions(torch: Any) -> JsonDict:
    sionna_version = str(installed_dependency_version("sionna") or "").strip()
    torch_version = str(
        installed_dependency_version("torch")
        or getattr(torch, "__version__", "")
    ).strip()
    if not sionna_version or not torch_version:
        raise OperationError("Could not identify Sionna/PyTorch runtime versions")
    return {"sionna": sionna_version, "torch": torch_version}


def _expected_noise_variance(ebno_db: float) -> float:
    # Unit-energy PUSCH data and DMRS occupy 1,680 frequency-domain REs.
    # Normal CP adds 120 expected RE-equivalent energy units, so the declared
    # equal-power coordinate has E_b = 1,800 / 1,608.
    energy_per_information_bit = (
        CP_TIME_BANDWIDTH_RESOURCE_UNITS / TRANSPORT_BLOCK_SIZE_BITS
    )
    return float(energy_per_information_bit / (10.0 ** (ebno_db / 10.0)))


def validate_nr_pusch_anchor_report(report: Mapping[str, Any]) -> None:
    """Fail closed on framing, accounting, and CRC-gating mutations."""

    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise OperationError("PUSCH anchor report schema_version changed")
    if report.get("kind") != REPORT_KIND:
        raise OperationError("PUSCH anchor report kind changed")
    if report.get("anchor_id") != ANCHOR_ID:
        raise OperationError("PUSCH anchor report identity changed")
    if report.get("fixed_configuration_sha256") != FIXED_CONFIGURATION_SHA256:
        raise OperationError("PUSCH anchor fixed-configuration hash changed")
    if report.get("fixed_configuration") != _fixed_configuration():
        raise OperationError("PUSCH anchor fixed configuration changed")

    seeds = report.get("seeds")
    if not isinstance(seeds, Mapping):
        raise OperationError("PUSCH anchor report is missing seeds")
    for key in ("master_seed", "channel_seed", "noise_seed"):
        value = seeds.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise OperationError("PUSCH anchor %s must be an integer" % key)
        if not 0 <= value < SEED_MODULUS:
            raise OperationError("PUSCH anchor %s is outside the RNG range" % key)
    master_seed = int(seeds["master_seed"])
    if master_seed > MAX_MASTER_SEED:
        raise OperationError("PUSCH anchor master seed cannot fit stream offsets")
    if seeds["channel_seed"] != master_seed + CHANNEL_SEED_OFFSET:
        raise OperationError("PUSCH anchor channel seed offset changed")
    if seeds["noise_seed"] != master_seed + NOISE_SEED_OFFSET:
        raise OperationError("PUSCH anchor noise seed offset changed")

    coordinate = report.get("coordinate")
    if not isinstance(coordinate, Mapping):
        raise OperationError("PUSCH anchor report is missing its coordinate")
    ebno_db = float(coordinate.get("ebno_db"))
    noise_variance = float(coordinate.get("complex_noise_variance"))
    if not math.isfinite(ebno_db) or not math.isfinite(noise_variance):
        raise OperationError("PUSCH anchor coordinate must be finite")
    if noise_variance <= 0.0 or not math.isclose(
        noise_variance,
        _expected_noise_variance(ebno_db),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise OperationError("PUSCH anchor Eb/N0-to-noise mapping changed")

    outcome = report.get("outcome")
    if not isinstance(outcome, Mapping) or not isinstance(
        outcome.get("transport_block_crc_pass"), bool
    ):
        raise OperationError("PUSCH anchor report is missing its CRC outcome")
    crc_pass = bool(outcome["transport_block_crc_pass"])
    delivered_bits = TRANSPORT_BLOCK_SIZE_BITS if crc_pass else 0
    expected = {
        "attempted_transport_block_count": 1,
        "delivered_transport_block_count": int(crc_pass),
        "transport_block_crc_failure_count": int(not crc_pass),
        "attempted_payload_bit_count": TRANSPORT_BLOCK_SIZE_BITS,
        "delivered_payload_bit_count": delivered_bits,
        "data_resource_element_count": DATA_RESOURCE_ELEMENTS,
        "dmrs_resource_element_count": DMRS_RESOURCE_ELEMENTS,
        "allocated_resource_element_count": ALLOCATED_RESOURCE_ELEMENTS,
        "cp_time_bandwidth_resource_unit_count": (
            CP_TIME_BANDWIDTH_RESOURCE_UNITS
        ),
    }
    for key, value in expected.items():
        if outcome.get(key) != value:
            raise OperationError("PUSCH anchor outcome %s changed" % key)

    ratio_expectations = {
        "transport_block_error_rate": float(not crc_pass),
        "goodput_bits_per_data_resource_element": (
            delivered_bits / DATA_RESOURCE_ELEMENTS
        ),
        "goodput_bits_per_allocated_resource_element": (
            delivered_bits / ALLOCATED_RESOURCE_ELEMENTS
        ),
        "goodput_bits_per_cp_time_bandwidth_resource_unit": (
            delivered_bits / CP_TIME_BANDWIDTH_RESOURCE_UNITS
        ),
        "slot_throughput_bps": delivered_bits / SLOT_DURATION_S,
    }
    for key, value in ratio_expectations.items():
        observed = outcome.get(key)
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise OperationError("PUSCH anchor outcome %s must be numeric" % key)
        if not math.isclose(float(observed), value, rel_tol=1e-12, abs_tol=1e-12):
            raise OperationError("PUSCH anchor CRC-gated %s changed" % key)

    energy = report.get("energy")
    if not isinstance(energy, Mapping):
        raise OperationError("PUSCH anchor report is missing energy accounting")
    for key in (
        "data_frequency_grid_energy",
        "dmrs_frequency_grid_energy",
        "allocated_frequency_grid_energy",
        "transmitted_waveform_energy",
        "energy_per_attempted_payload_bit",
    ):
        value = energy.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OperationError("PUSCH anchor energy %s must be numeric" % key)
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise OperationError("PUSCH anchor energy %s must be positive" % key)
    if not math.isclose(
        float(energy["allocated_frequency_grid_energy"]),
        float(energy["data_frequency_grid_energy"])
        + float(energy["dmrs_frequency_grid_energy"]),
        rel_tol=1e-7,
        abs_tol=1e-6,
    ):
        raise OperationError("PUSCH anchor data/DMRS energy partition changed")
    expected_delivered_energy = (
        float(energy["transmitted_waveform_energy"]) / delivered_bits
        if delivered_bits
        else None
    )
    observed_delivered_energy = energy.get("energy_per_delivered_payload_bit")
    if expected_delivered_energy is None:
        if observed_delivered_energy is not None:
            raise OperationError(
                "PUSCH anchor failed CRC must not define delivered-bit energy"
            )
    elif not isinstance(observed_delivered_energy, (int, float)) or not math.isclose(
        float(observed_delivered_energy),
        expected_delivered_energy,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise OperationError("PUSCH anchor delivered-bit energy changed")

    backend = report.get("backend_versions")
    if not isinstance(backend, Mapping) or not all(
        str(backend.get(key) or "").strip() for key in ("sionna", "torch")
    ):
        raise OperationError("PUSCH anchor backend identity is incomplete")


class NrPuschDomainAnchorOperation(Operation):
    """Execute one fixed equal-power NR-framed PUSCH slot."""

    id = "wireless.nr_pusch_domain_anchor_v1"
    name = "Fixed equal-power NR-framed PUSCH domain anchor"
    input_kinds = {"bits": ["channel.payload_bits.numpy"]}
    output_kinds = {
        "decoded_bits": "channel.payload_bits.numpy",
        "report": "metrics.report",
    }
    differentiability = {
        "framework": "sionna",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": (
            "The operation is a frozen evidence-producing PUSCH link, not a "
            "trainable materialization."
        ),
    }
    backends = {
        "benchmark_run": ["sionna"],
        "dataset_capture": [],
        "differentiable_export": [],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "sionna",
            "implementation": "fixed_nr_pusch_domain_anchor_v1",
            "status": "implemented",
        }
    ]
    equivalence = {
        "type": "exact",
        "reason": (
            "For one pinned Sionna/PyTorch runtime on CPU, payload and explicit "
            "channel/noise seeds deterministically identify the slot outcome."
        ),
    }
    formats = {"artifact": "npz+json", "tensor": "none"}
    params_schema = object_schema(
        {
            "ebno_db": {
                "type": "number",
                "minimum": -20.0,
                "maximum": 40.0,
                "description": (
                    "Normalized information-bit Eb/N0. The conversion includes "
                    "DMRS and normal-CP time-bandwidth overhead."
                ),
            },
            "seed": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_MASTER_SEED,
                "description": (
                    "Master slot seed; channel and noise streams use the fixed "
                    "+100000 and +300000 study offsets."
                ),
            },
        },
        required=["ebno_db", "seed"],
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _availability()
        payload["fixed_configuration"] = _fixed_configuration()
        payload["fixed_configuration_sha256"] = FIXED_CONFIGURATION_SHA256
        return payload

    def validate_preflight(
        self,
        params: Mapping[str, Any],
        inputs: Mapping[str, str] | None = None,
    ) -> None:
        ebno_db = params.get("ebno_db")
        seed = params.get("seed")
        if isinstance(ebno_db, bool) or not isinstance(ebno_db, (int, float)):
            raise OperationError("PUSCH anchor ebno_db must be numeric")
        if not math.isfinite(float(ebno_db)):
            raise OperationError("PUSCH anchor ebno_db must be finite")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise OperationError("PUSCH anchor seed must be an integer")
        if seed < 0 or seed > MAX_MASTER_SEED:
            raise OperationError(
                "PUSCH anchor seed must be in [0, %d]" % MAX_MASTER_SEED
            )

    def run(self, ctx: OperationContext) -> OperationResult:
        if not _sionna_available():
            raise OperationError(str(_availability()["reason"]))
        bits, source_metadata = _load_payload_bits(
            ctx.require_input("bits").path
        )
        ebno_db = float(ctx.params["ebno_db"])
        master_seed = int(ctx.params["seed"])
        channel_seed = master_seed + CHANNEL_SEED_OFFSET
        noise_seed = master_seed + NOISE_SEED_OFFSET
        noise_variance = _expected_noise_variance(ebno_db)
        ctx.raise_if_cancelled()

        with _ANCHOR_LOCK:
            try:
                import torch  # type: ignore
                from sionna.phy import config as sionna_config  # type: ignore
                from sionna.phy.channel import AWGN, TimeChannel  # type: ignore
                from sionna.phy.channel.tr38901 import TDL  # type: ignore
                from sionna.phy.nr import (  # type: ignore
                    CarrierConfig,
                    PUSCHConfig,
                    PUSCHDMRSConfig,
                    PUSCHReceiver,
                    PUSCHTransmitter,
                    TBConfig,
                )
                from sionna.phy.ofdm import (  # type: ignore
                    OFDMDemodulator,
                    OFDMModulator,
                )

                carrier = CarrierConfig(
                    n_cell_id=1,
                    cyclic_prefix="normal",
                    subcarrier_spacing=15,
                    n_size_grid=CARRIER_RESOURCE_BLOCKS,
                    slot_number=0,
                    frame_number=0,
                )
                dmrs = PUSCHDMRSConfig(
                    config_type=1,
                    type_a_position=2,
                    additional_position=1,
                    length=1,
                    dmrs_port_set=[0],
                    num_cdm_groups_without_data=1,
                )
                tb = TBConfig(
                    mcs_index=MCS_INDEX,
                    mcs_table=MCS_TABLE,
                    channel_type="PUSCH",
                    n_id=1,
                )
                pusch = PUSCHConfig(
                    carrier_config=carrier,
                    pusch_dmrs_config=dmrs,
                    tb_config=tb,
                    n_size_bwp=PUSCH_RESOURCE_BLOCKS,
                    n_start_bwp=PUSCH_START_RESOURCE_BLOCK,
                    num_layers=1,
                    num_antenna_ports=1,
                    mapping_type="A",
                    symbol_allocation=[0, SLOT_OFDM_SYMBOLS],
                    n_rnti=1,
                    precoding="non-codebook",
                    transform_precoding=False,
                )
                transmitter = PUSCHTransmitter(
                    pusch,
                    return_bits=False,
                    output_domain="freq",
                )
                receiver = PUSCHReceiver(
                    transmitter,
                    channel_estimator=None,
                    return_tb_crc_status=True,
                    input_domain="freq",
                )

                input_tensor = torch.as_tensor(
                    bits.reshape(1, 1, TRANSPORT_BLOCK_SIZE_BITS),
                    dtype=torch.float32,
                )
                pusch_grid = transmitter(input_tensor)
                outer_grid = torch.zeros(
                    (1, 1, 1, SLOT_OFDM_SYMBOLS, OUTER_FFT_SIZE),
                    dtype=torch.complex64,
                )
                outer_grid[..., PUSCH_OUTER_START:PUSCH_OUTER_STOP] = pusch_grid

                cp = np.asarray(NORMAL_CP_SAMPLES, dtype=np.int32)
                modulator = OFDMModulator(cp)
                demodulator = OFDMDemodulator(
                    fft_size=OUTER_FFT_SIZE,
                    l_min=CHANNEL_L_MIN,
                    cyclic_prefix_length=cp,
                )
                tx_waveform = modulator(outer_grid)
                if int(tx_waveform.shape[-1]) != SLOT_SAMPLE_COUNT:
                    raise OperationError("PUSCH anchor slot sample count changed")

                channel_model = TDL(
                    model=TDL_MODEL,
                    delay_spread=TDL_DELAY_SPREAD_S,
                    carrier_frequency=CARRIER_FREQUENCY_HZ,
                    min_speed=MOBILITY_MPS,
                    max_speed=MOBILITY_MPS,
                    num_rx_ant=1,
                    num_tx_ant=1,
                )
                channel = TimeChannel(
                    channel_model,
                    bandwidth=OUTER_SAMPLE_RATE_HZ,
                    num_time_samples=SLOT_SAMPLE_COUNT,
                    maximum_delay_spread=TDL_MAXIMUM_PATH_DELAY_S,
                    l_min=CHANNEL_L_MIN,
                    normalize_channel=True,
                    return_channel=True,
                )
                if channel.l_max != CHANNEL_L_MAX:
                    raise OperationError("PUSCH anchor channel l_max changed")
                if channel.l_max - channel.l_min + 1 != 33:
                    raise OperationError(
                        "PUSCH anchor discrete impulse-response tap count changed"
                    )
                if channel.l_max - channel.l_min + 1 >= min(NORMAL_CP_SAMPLES):
                    raise OperationError(
                        "PUSCH anchor channel taps no longer fit the shortest CP"
                    )

                sionna_config.seed = channel_seed
                rx_signal, _h_time = channel(tx_waveform, None)
                sionna_config.seed = noise_seed
                rx_waveform = AWGN()(rx_signal, noise_variance)
                rx_outer_grid = demodulator(rx_waveform)
                rx_pusch_grid = rx_outer_grid[
                    ..., PUSCH_OUTER_START:PUSCH_OUTER_STOP
                ]
                decoded_tensor, crc_tensor = receiver(
                    rx_pusch_grid, noise_variance
                )

                decoded = (
                    decoded_tensor.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(np.uint8, copy=False)
                )
                crc_pass = bool(
                    crc_tensor.detach().cpu().numpy().reshape(-1)[0]
                )
                grid_type = transmitter.resource_grid.build_type_grid()
                data_mask = grid_type == 0
                dmrs_mask = grid_type == 1
                data_energy = float(
                    torch.sum(torch.abs(pusch_grid[0][data_mask]) ** 2).item()
                )
                dmrs_energy = float(
                    torch.sum(torch.abs(pusch_grid[0][dmrs_mask]) ** 2).item()
                )
                allocated_energy = float(
                    torch.sum(torch.abs(pusch_grid) ** 2).item()
                )
                waveform_energy = float(
                    torch.sum(torch.abs(tx_waveform) ** 2).item()
                )
            except OperationError:
                raise
            except Exception as exc:
                raise OperationError(
                    "Sionna NR-framed PUSCH anchor execution failed: %s" % exc
                ) from exc

        ctx.raise_if_cancelled()
        if int(decoded.size) != TRANSPORT_BLOCK_SIZE_BITS:
            raise OperationError("PUSCH anchor decoded TB size changed")
        bit_error_count = int(np.count_nonzero(decoded != bits))
        bit_error_rate = bit_error_count / TRANSPORT_BLOCK_SIZE_BITS
        delivered_bits = TRANSPORT_BLOCK_SIZE_BITS if crc_pass else 0
        backend_versions = _backend_versions(torch)
        outcome = {
            "transport_block_crc_pass": crc_pass,
            "attempted_transport_block_count": 1,
            "delivered_transport_block_count": int(crc_pass),
            "transport_block_crc_failure_count": int(not crc_pass),
            "transport_block_error_rate": float(not crc_pass),
            "attempted_payload_bit_count": TRANSPORT_BLOCK_SIZE_BITS,
            "delivered_payload_bit_count": delivered_bits,
            "decoded_bit_error_count": bit_error_count,
            "decoded_bit_error_rate": bit_error_rate,
            "data_resource_element_count": DATA_RESOURCE_ELEMENTS,
            "dmrs_resource_element_count": DMRS_RESOURCE_ELEMENTS,
            "allocated_resource_element_count": ALLOCATED_RESOURCE_ELEMENTS,
            "cp_time_bandwidth_resource_unit_count": (
                CP_TIME_BANDWIDTH_RESOURCE_UNITS
            ),
            "goodput_bits_per_data_resource_element": (
                delivered_bits / DATA_RESOURCE_ELEMENTS
            ),
            "goodput_bits_per_allocated_resource_element": (
                delivered_bits / ALLOCATED_RESOURCE_ELEMENTS
            ),
            "goodput_bits_per_cp_time_bandwidth_resource_unit": (
                delivered_bits / CP_TIME_BANDWIDTH_RESOURCE_UNITS
            ),
            "slot_throughput_bps": delivered_bits / SLOT_DURATION_S,
        }
        energy = {
            "unit": "normalized_complex_sample_energy",
            "data_frequency_grid_energy": data_energy,
            "dmrs_frequency_grid_energy": dmrs_energy,
            "allocated_frequency_grid_energy": allocated_energy,
            "transmitted_waveform_energy": waveform_energy,
            "energy_per_attempted_payload_bit": (
                waveform_energy / TRANSPORT_BLOCK_SIZE_BITS
            ),
            "energy_per_delivered_payload_bit": (
                waveform_energy / delivered_bits if delivered_bits else None
            ),
            "aggregate_ratio_instruction": (
                "sum transmitted_waveform_energy divided by sum "
                "delivered_payload_bit_count; never average success-only ratios"
            ),
        }
        report: JsonDict = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "anchor_id": ANCHOR_ID,
            "status": "development_or_candidate_observation_not_conformance",
            "fixed_configuration": _fixed_configuration(),
            "fixed_configuration_sha256": FIXED_CONFIGURATION_SHA256,
            "coordinate": {
                "ebno_db": ebno_db,
                "complex_noise_variance": noise_variance,
                "energy_per_information_bit_definition": (
                    "1800 CP-and-DMRS-inclusive normalized resource units / "
                    "1608 attempted payload bits"
                ),
            },
            "seeds": {
                "master_seed": master_seed,
                "channel_seed": channel_seed,
                "noise_seed": noise_seed,
                "derivation": (
                    "channel_seed=master_seed+100000; "
                    "noise_seed=master_seed+300000"
                ),
            },
            "source_binding": _source_binding(source_metadata),
            "backend_versions": backend_versions,
            "outcome": outcome,
            "energy": energy,
            "claim_boundary": {
                "nr_conformance_claim": False,
                "deployed_throughput_claim": False,
                "absolute_link_budget": False,
                "independent_execution": False,
                "external_domain_review_complete": False,
            },
        }
        validate_nr_pusch_anchor_report(report)

        decoded_metadata = {
            **source_metadata,
            "boundary_contract": "channel.payload_bits",
            "bit_count": TRANSPORT_BLOCK_SIZE_BITS,
            "dtype": "uint8",
            "shape": [TRANSPORT_BLOCK_SIZE_BITS],
            "receiver_output": "best_effort_bits_with_separate_authoritative_tb_crc",
            "transport_block_crc_pass": crc_pass,
            "fixed_configuration_sha256": FIXED_CONFIGURATION_SHA256,
        }
        decoded_path = ctx.output_path("decoded_bits", ".npz")
        report_path = ctx.output_path("report", ".json")
        np.savez_compressed(
            decoded_path,
            bits=decoded,
            metadata_json=json.dumps(decoded_metadata, sort_keys=True),
        )
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        prefix = "channel.nr_pusch_anchor"
        metrics = {
            f"{prefix}.transport_block_error_rate": outcome[
                "transport_block_error_rate"
            ],
            f"{prefix}.transport_block_crc_failure_count": outcome[
                "transport_block_crc_failure_count"
            ],
            f"{prefix}.attempted_transport_block_count": 1,
            f"{prefix}.delivered_transport_block_count": outcome[
                "delivered_transport_block_count"
            ],
            f"{prefix}.attempted_payload_bit_count": TRANSPORT_BLOCK_SIZE_BITS,
            f"{prefix}.delivered_payload_bit_count": delivered_bits,
            f"{prefix}.decoded_bit_error_rate": bit_error_rate,
            f"{prefix}.data_resource_element_count": DATA_RESOURCE_ELEMENTS,
            f"{prefix}.dmrs_resource_element_count": DMRS_RESOURCE_ELEMENTS,
            f"{prefix}.allocated_resource_element_count": (
                ALLOCATED_RESOURCE_ELEMENTS
            ),
            f"{prefix}.cp_time_bandwidth_resource_unit_count": (
                CP_TIME_BANDWIDTH_RESOURCE_UNITS
            ),
            f"{prefix}.goodput_bits_per_data_resource_element": outcome[
                "goodput_bits_per_data_resource_element"
            ],
            f"{prefix}.goodput_bits_per_allocated_resource_element": outcome[
                "goodput_bits_per_allocated_resource_element"
            ],
            f"{prefix}.goodput_bits_per_cp_time_bandwidth_resource_unit": outcome[
                "goodput_bits_per_cp_time_bandwidth_resource_unit"
            ],
            f"{prefix}.slot_throughput_bps": outcome["slot_throughput_bps"],
            f"{prefix}.transmitted_waveform_energy": waveform_energy,
            f"{prefix}.ebno_db": ebno_db,
            f"{prefix}.complex_noise_variance": noise_variance,
        }
        return OperationResult(
            outputs={
                "decoded_bits": artifact(
                    "channel.payload_bits.numpy",
                    decoded_path,
                    decoded_metadata,
                ),
                "report": artifact(
                    "metrics.report",
                    report_path,
                    {
                        "kind": REPORT_KIND,
                        "anchor_id": ANCHOR_ID,
                        "fixed_configuration_sha256": (
                            FIXED_CONFIGURATION_SHA256
                        ),
                        "transport_block_crc_pass": crc_pass,
                    },
                ),
            },
            metrics=metrics,
            metadata={
                "anchor_id": ANCHOR_ID,
                "fixed_configuration_sha256": FIXED_CONFIGURATION_SHA256,
                "transport_block_crc_pass": crc_pass,
                "nr_conformance_claim": False,
                "metrics": metrics,
            },
        )
