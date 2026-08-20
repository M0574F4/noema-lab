from __future__ import annotations

import importlib.util
import json
import math
import threading
from functools import lru_cache
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.structured_input import decode_strict_json_object
from noema_lab.core.boundaries import validate_channel_bits
from noema_lab.core.capture_layout import (
    CaptureRecordLayout,
    CaptureRecordLayoutError,
    explicit_capture_record_layout,
    set_uniform_capture_record_layout,
)
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


JsonDict = Dict[str, Any]

NR_LDPC_STANDARD = {
    "name": "3GPP TS 38.212",
    "version": "18.8.0",
    "release": "18",
    "published": "2026-02",
    "authoritative_locator": (
        "https://www.etsi.org/deliver/etsi_ts/138200_138299/138212/"
        "18.08.00_60/ts_138212v180800p.pdf"
    ),
    "archive_locator": (
        "https://www.3gpp.org/ftp/Specs/archive/38_series/38.212/"
        "38212-i80.zip"
    ),
}

_NR_LOCK = threading.RLock()
NR_LDPC_MIN_TARGET_CODERATE = 0.2
NR_LDPC_MAX_TARGET_CODERATE = 0.925
NR_LDPC_DEFAULT_BP_ITERATIONS = 20


def _validated_capture_layout(
    metadata: Mapping[str, Any],
    element_count: int,
    label: str,
) -> CaptureRecordLayout | None:
    try:
        return explicit_capture_record_layout(
            metadata,
            element_count,
            label=label,
        )
    except CaptureRecordLayoutError as exc:
        raise OperationError(str(exc)) from exc


def _rewrite_uniform_capture_layout(
    metadata: JsonDict,
    layout: CaptureRecordLayout | None,
    element_count: int,
    label: str,
) -> None:
    try:
        set_uniform_capture_record_layout(
            metadata,
            layout,
            element_count,
            label=label,
        )
    except CaptureRecordLayoutError as exc:
        raise OperationError(str(exc)) from exc


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
            "implementation": "sionna.phy.nr.TBEncoder/TBDecoder",
            "backend_versions": {
                "sionna": installed_dependency_version("sionna"),
                "torch": installed_dependency_version("torch"),
            },
            "standard": dict(NR_LDPC_STANDARD),
        }
    return {
        "available": False,
        "optional": True,
        "extra": "wireless",
        "missing": ["sionna>=2.0.1", "torch>=2.9.1"],
        "reason": (
            'Install with `python -m pip install "noema-lab[wireless]"` in an '
            "installed environment, or `uv sync --extra wireless` in a source checkout, "
            "to execute the 3GPP NR LDPC transport-block path"
        ),
        "standard": dict(NR_LDPC_STANDARD),
    }


def _require_sionna():
    if not _sionna_available():
        raise OperationError(
            'Install with `python -m pip install "noema-lab[wireless]"` in an '
            "installed environment, or `uv sync --extra wireless` in a source checkout, "
            "to execute channel.nr_ldpc_encoder/channel.nr_ldpc_decoder"
        )
    try:
        import torch  # type: ignore
        from sionna.phy.nr import TBDecoder, TBEncoder  # type: ignore
        from sionna.phy.nr.utils import calculate_tb_size  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the optional wheel
        raise OperationError(
            "The installed Sionna wireless extra does not expose the 5G NR "
            "transport-block encoder/decoder"
        ) from exc
    return torch, TBEncoder, TBDecoder, calculate_tb_size


def _nr_backend_versions(torch) -> JsonDict:
    """Return the exact installed backend versions bound to an encoded stream."""

    sionna_version = installed_dependency_version("sionna")
    torch_version = installed_dependency_version("torch") or str(
        getattr(torch, "__version__", "")
    ).strip()
    if not sionna_version or not torch_version:
        raise OperationError(
            "Could not identify installed Sionna and PyTorch versions for NR LDPC"
        )
    return {
        "sionna": str(sionna_version),
        "torch": str(torch_version),
    }


def _bound_backend_versions(value: Any) -> JsonDict:
    if not isinstance(value, Mapping) or set(value) not in (
        {"sionna", "torch"},
        {"sionna", "tensorflow"},
    ):
        raise OperationError(
            "NR LDPC decoder requires exact encoder-bound Sionna/PyTorch versions "
            "or a legacy Sionna/TensorFlow provenance record"
        )
    framework = "torch" if "torch" in value else "tensorflow"
    versions = {
        "sionna": str(value.get("sionna") or "").strip(),
        framework: str(value.get(framework) or "").strip(),
    }
    if not all(versions.values()):
        raise OperationError(
            "NR LDPC encoder-bound backend versions must be non-empty"
        )
    return versions


def _profile_sha256(
    blocks: Any,
    decoder_num_bp_iter: int,
    backend_versions: Mapping[str, Any],
) -> str:
    return canonical_json_sha256(
        {
            "standard": NR_LDPC_STANDARD,
            "blocks": blocks,
            "decoder_num_bp_iter": int(decoder_num_bp_iter),
            "backend_versions": dict(backend_versions),
        }
    )


def _target_coderate(value: Any) -> float:
    if isinstance(value, bool):
        raise OperationError("NR LDPC target_coderate must be a number")
    try:
        coderate = float(value)
    except (TypeError, ValueError) as exc:
        raise OperationError("NR LDPC target_coderate must be a number") from exc
    if not math.isfinite(coderate) or not (
        NR_LDPC_MIN_TARGET_CODERATE
        <= coderate
        <= NR_LDPC_MAX_TARGET_CODERATE
    ):
        raise OperationError(
            "NR LDPC target_coderate must be between %.3f and %.3f; "
            "the upper bound is the preflighted Sionna limit"
            % (NR_LDPC_MIN_TARGET_CODERATE, NR_LDPC_MAX_TARGET_CODERATE)
        )
    return coderate


def _decoder_iterations(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OperationError("NR LDPC encoder-bound num_bp_iter must be an integer")
    iterations = int(value)
    if iterations < 1 or iterations > 200:
        raise OperationError(
            "NR LDPC encoder-bound num_bp_iter must be between 1 and 200"
        )
    return iterations


def _load_bits(path, fallback_metadata: Mapping[str, Any]) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        bits = payload["bits"]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="NR-LDPC bit artifact metadata_json",
                )
            )
    canonical, _ = validate_channel_bits(bits, label="3GPP NR LDPC input")
    _validated_capture_layout(
        metadata,
        int(canonical.size),
        "3GPP NR LDPC bit artifact %s" % path,
    )
    return canonical.astype(np.uint8, copy=False), metadata


def _load_llr(path, fallback_metadata: Mapping[str, Any]) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        if "llr" not in payload:
            raise OperationError("NR LDPC decoder input does not contain an llr array")
        values = np.asarray(payload["llr"])
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="NR-LDPC symbol artifact metadata_json",
                )
            )
    if values.dtype.kind not in {"f", "i", "u"}:
        raise OperationError("NR LDPC decoder LLRs must be numeric")
    llr = values.astype(np.float32, copy=False).reshape(-1)
    if not np.all(np.isfinite(llr)):
        raise OperationError("NR LDPC decoder LLRs must be finite")
    _validated_capture_layout(
        metadata,
        int(llr.size),
        "3GPP NR LDPC LLR count/layout for artifact %s" % path,
    )
    return llr, metadata


def _load_symbols(path, fallback_metadata: Mapping[str, Any]) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        if "symbols" not in payload:
            raise OperationError("Resource accounting symbol input has no symbols array")
        symbols = np.asarray(payload["symbols"]).reshape(-1)
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="NR-LDPC LLR artifact metadata_json",
                )
            )
    if symbols.dtype.kind != "c" or not np.all(np.isfinite(symbols)):
        raise OperationError("Resource accounting symbols must be finite complex values")
    _validated_capture_layout(
        metadata,
        int(symbols.size),
        "Resource-accounting symbol artifact %s" % path,
    )
    return symbols.astype(np.complex64, copy=False), metadata


def _source_counts(
    metadata: Mapping[str, Any],
    total: int,
    capture_layout: CaptureRecordLayout | None = None,
) -> list[int]:
    declared_totals: list[str] = []
    for key in (
        "source_item_framed_bit_counts",
        "source_item_payload_bit_counts",
    ):
        if key not in metadata or metadata.get(key) is None:
            continue
        raw = metadata.get(key)
        if (
            not isinstance(raw, list)
            or not raw
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in raw
            )
        ):
            raise OperationError("%s must contain integer counts" % key)
        counts = [int(value) for value in raw]
        if any(value <= 0 for value in counts):
            raise OperationError("%s must contain positive counts" % key)
        if sum(counts) != int(total):
            declared_totals.append("%s=%d" % (key, sum(counts)))
            continue
        if capture_layout is not None and (
            len(counts) != capture_layout.count
            or any(
                value != capture_layout.elements_per_record
                for value in counts
            )
        ):
            raise OperationError(
                "%s conflicts with the explicit capture-record layout" % key
            )
        return counts
    if declared_totals:
        raise OperationError(
            "No source-item bit-count declaration matches the current %d-bit "
            "NR LDPC input (%s)"
            % (int(total), ", ".join(declared_totals))
        )
    if capture_layout is not None:
        return [
            int(capture_layout.elements_per_record)
        ] * int(capture_layout.count)
    return [int(total)] if total else []


def _source_pixel_counts(metadata: Mapping[str, Any]) -> list[int]:
    declared_counts = None
    if "source_item_pixel_counts" in metadata:
        declared_counts = _positive_integer_list(
            metadata["source_item_pixel_counts"],
            "source_item_pixel_counts",
        )

    shape_counts = None
    if "original_shapes" in metadata:
        raw_shapes = metadata["original_shapes"]
        if not isinstance(raw_shapes, list) or not raw_shapes:
            raise OperationError(
                "original_shapes must be a non-empty list of [1,H,W,C] shapes"
            )
        shape_counts = []
        for index, raw_shape in enumerate(raw_shapes):
            dimensions = _positive_shape(
                raw_shape,
                "original_shapes[%d]" % index,
            )
            if dimensions[0] != 1:
                raise OperationError(
                    "original_shapes[%d] must describe one source item; "
                    "its first dimension must be 1" % index
                )
            shape_counts.append(dimensions[1] * dimensions[2])

    batch_shape_counts = None
    if "original_shape" in metadata:
        dimensions = _positive_shape(
            metadata["original_shape"],
            "original_shape",
        )
        batch_shape_counts = [dimensions[1] * dimensions[2]] * dimensions[0]

    declarations = [
        (name, values)
        for name, values in (
            ("source_item_pixel_counts", declared_counts),
            ("original_shapes", shape_counts),
            ("original_shape", batch_shape_counts),
        )
        if values is not None
    ]
    if declarations:
        expected_count = len(declarations[0][1])
        for name, values in declarations[1:]:
            if len(values) != expected_count:
                raise OperationError(
                    "%s declares %d source items, but %s declares %d"
                    % (
                        declarations[0][0],
                        expected_count,
                        name,
                        len(values),
                    )
                )
        if declared_counts is not None and shape_counts is not None:
            if declared_counts != shape_counts:
                raise OperationError(
                    "source_item_pixel_counts disagrees with original_shapes"
                )
        if declared_counts is not None:
            return declared_counts
        if shape_counts is not None:
            return shape_counts
        return batch_shape_counts or []

    if "shape" in metadata:
        dimensions = _positive_shape(metadata["shape"], "shape")
        return [dimensions[1] * dimensions[2]] * dimensions[0]
    return []


def _positive_integer_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise OperationError("%s must be a non-empty list of positive integers" % label)
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        raise OperationError("%s must contain only positive integers" % label)
    return [int(item) for item in value]


def _positive_shape(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise OperationError("%s must be a four-integer [N,H,W,C] list" % label)
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
        for dimension in value
    ):
        raise OperationError("%s dimensions must be positive integers" % label)
    return [int(dimension) for dimension in value]


def _coded_length(target_input_bits: int, target_coderate: float, quantum: int) -> int:
    requested = int(math.ceil(float(target_input_bits) / float(target_coderate)))
    return int(math.ceil(float(requested) / float(quantum)) * quantum)


def _transport_block_configuration(
    valid_bits: int,
    target_coderate: float,
    num_bits_per_symbol: int,
    num_layers: int,
) -> Tuple[int, int]:
    """Choose a quantized Sionna TB target that cannot truncate valid bits."""

    target_coderate = _target_coderate(target_coderate)
    _torch, _encoder, _decoder, calculate_tb_size = _require_sionna()
    quantum = int(num_bits_per_symbol * num_layers)
    candidate = max(24, int(math.ceil(float(valid_bits) / 8.0) * 8))
    for _ in range(1024):
        coded = _coded_length(candidate, target_coderate, quantum)
        try:
            config = calculate_tb_size(
                modulation_order=num_bits_per_symbol,
                target_coderate=target_coderate,
                target_tb_size=candidate,
                num_coded_bits=coded,
                num_layers=num_layers,
                verbose=False,
            )
        except AssertionError as exc:
            raise OperationError(
                "Sionna rejected the preflighted NR transport-block configuration"
            ) from exc
        effective_tb_size = int(np.asarray(config[0]).reshape(-1)[0])
        if effective_tb_size >= valid_bits:
            return candidate, coded
        candidate += 8
    raise OperationError(
        "Could not select a standards-quantized NR transport block without "
        "truncating valid payload bits"
    )


@lru_cache(maxsize=128)
def _nr_codec(
    target_tb_size: int,
    num_coded_bits: int,
    target_coderate: float,
    num_bits_per_symbol: int,
    num_layers: int,
    n_rnti: int,
    n_id: int,
    channel_type: str,
    codeword_index: int,
    num_bp_iter: int,
):
    target_coderate = _target_coderate(target_coderate)
    num_bp_iter = _decoder_iterations(num_bp_iter)
    _torch, TBEncoder, TBDecoder, _calculate = _require_sionna()
    try:
        encoder = TBEncoder(
            target_tb_size=int(target_tb_size),
            num_coded_bits=int(num_coded_bits),
            target_coderate=float(target_coderate),
            num_bits_per_symbol=int(num_bits_per_symbol),
            num_layers=int(num_layers),
            n_rnti=int(n_rnti),
            n_id=int(n_id),
            channel_type=str(channel_type),
            codeword_index=int(codeword_index),
            use_scrambler=True,
            verbose=False,
            precision="single",
            device="cpu",
        )
        decoder = TBDecoder(
            encoder,
            num_bp_iter=int(num_bp_iter),
            precision="single",
            device="cpu",
        )
    except AssertionError as exc:
        raise OperationError(
            "Sionna rejected the bound NR LDPC encoder/decoder configuration"
        ) from exc
    return encoder, decoder


def _block_record(
    encoder,
    *,
    source_item_index: int,
    valid_payload_bits: int,
    target_input_bits: int,
    coded_offset: int,
    num_coded_bits: int,
    target_coderate: float,
    num_bits_per_symbol: int,
    num_layers: int,
    n_rnti: int,
    n_id: int,
    channel_type: str,
    codeword_index: int,
) -> JsonDict:
    ldpc = encoder.ldpc_encoder
    tb_size = int(encoder.tb_size)
    tb_crc_length = int(getattr(encoder, "_tb_crc_length", 0))
    cb_crc_length = int(getattr(encoder, "_cb_crc_length", 0))
    num_cbs = int(encoder.num_cbs)
    cw_lengths = [int(value) for value in np.asarray(encoder.cw_lengths).reshape(-1)]
    return {
        "source_item_index": int(source_item_index),
        "valid_payload_bit_count": int(valid_payload_bits),
        "encoder_input_bit_count": int(target_input_bits),
        "tail_zero_padding_bit_count": int(target_input_bits - valid_payload_bits),
        "effective_tb_size_bits": tb_size,
        "tb_quantization_padding_bits": int(tb_size - target_input_bits),
        "coded_offset": int(coded_offset),
        "num_coded_bits": int(num_coded_bits),
        "target_coderate": float(target_coderate),
        "effective_valid_payload_coderate": (
            float(valid_payload_bits) / float(num_coded_bits)
        ),
        "num_code_blocks": num_cbs,
        "tb_crc_length_bits": tb_crc_length,
        "cb_crc_length_bits_per_code_block": cb_crc_length,
        "total_logical_crc_bits": int(
            tb_crc_length + cb_crc_length * num_cbs
        ),
        "rate_matched_codeword_lengths": cw_lengths,
        "base_graph": str(getattr(ldpc, "_bg", "unknown")),
        "lifting_size": int(getattr(ldpc, "z", 0)),
        "lifting_set_index": int(getattr(ldpc, "_i_ls", -1)),
        "ldpc_information_size": int(getattr(ldpc, "k_ldpc", 0)),
        "ldpc_mother_code_bits": int(getattr(ldpc, "n_ldpc", 0)),
        "num_bits_per_symbol": int(num_bits_per_symbol),
        "num_layers": int(num_layers),
        "n_rnti": int(n_rnti),
        "n_id": int(n_id),
        "channel_type": str(channel_type),
        "codeword_index": int(codeword_index),
        "scrambling_enabled": True,
    }


class NrLdpcEncoderOperation(Operation):
    id = "channel.nr_ldpc_encoder"
    name = "3GPP NR transport-block LDPC encoder"
    input_kinds = {
        "bits": [
            "channel.payload_bits.numpy",
            "channel.framed_bits.numpy",
            "channel.bits.numpy",
        ]
    }
    output_kinds = {"coded_bits": "channel.coded_bits.numpy"}
    backends = {
        "benchmark_run": ["sionna"],
        "dataset_capture": ["sionna"],
        "differentiable_export": [],
    }
    differentiability = {
        "framework": "torch",
        "gradient": "stop",
        "trainable_params": False,
        "exportable": False,
        "reason": "CRC attachment, segmentation, rate matching, and hard bit coding are discrete.",
    }
    equivalence = {
        "type": "behavioral",
        "reason": (
            "Publication use requires independent TS 38.212 vectors or a second "
            "standards implementation; internal round trips alone are not conformance."
        ),
    }
    params_schema = object_schema(
        {
            "target_coderate": {
                "type": "number",
                "default": 0.5,
                "minimum": NR_LDPC_MIN_TARGET_CODERATE,
                "maximum": NR_LDPC_MAX_TARGET_CODERATE,
            },
            "transport_block_size_bits": {
                "type": "integer",
                "default": 16000,
                "minimum": 24,
                "maximum": 100000,
            },
            "num_bits_per_symbol": {
                "type": "integer",
                "default": 2,
                "enum": [1, 2, 4, 6, 8],
            },
            "num_layers": {"type": "integer", "default": 1, "minimum": 1, "maximum": 8},
            "n_rnti": {"type": "integer", "default": 1, "minimum": 0, "maximum": 65535},
            "n_id": {"type": "integer", "default": 1, "minimum": 0, "maximum": 1023},
            "channel_type": {
                "type": "string",
                "default": "PUSCH",
                "enum": ["PUSCH", "PDSCH"],
            },
            "codeword_index": {"type": "integer", "default": 0, "enum": [0, 1]},
            "num_bp_iter": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _availability()
        payload["standard"] = dict(NR_LDPC_STANDARD)
        return payload

    def validate_preflight(
        self,
        params: Mapping[str, Any],
        inputs: Mapping[str, str] | None = None,
    ) -> None:
        _target_coderate(params.get("target_coderate", 0.5))
        _decoder_iterations(
            params.get("num_bp_iter", NR_LDPC_DEFAULT_BP_ITERATIONS)
        )
        channel_type = str(params.get("channel_type") or "PUSCH")
        codeword_index = int(params.get("codeword_index", 0))
        if channel_type == "PUSCH" and codeword_index != 0:
            raise OperationError(
                "PUSCH NR LDPC transport blocks require codeword_index=0"
            )

    def run(self, ctx: OperationContext) -> OperationResult:
        target_coderate = _target_coderate(
            ctx.params.get("target_coderate", 0.5)
        )
        torch, _TBEncoder, _TBDecoder, _calculate = _require_sionna()
        backend_versions = _nr_backend_versions(torch)
        input_artifact = ctx.require_input("bits")
        bits, metadata = _load_bits(input_artifact.path, input_artifact.metadata)
        if not bits.size:
            raise OperationError("NR LDPC encoder requires at least one payload bit")
        capture_layout = _validated_capture_layout(
            metadata,
            int(bits.size),
            "NR LDPC encoder input at %s" % ctx.step_id,
        )

        max_tb_bits = int(ctx.params.get("transport_block_size_bits", 16000))
        num_bits_per_symbol = int(ctx.params.get("num_bits_per_symbol", 2))
        num_layers = int(ctx.params.get("num_layers", 1))
        n_rnti = int(ctx.params.get("n_rnti", 1))
        n_id = int(ctx.params.get("n_id", 1))
        channel_type = str(ctx.params.get("channel_type", "PUSCH"))
        codeword_index = int(ctx.params.get("codeword_index", 0))
        num_bp_iter = _decoder_iterations(
            ctx.params.get("num_bp_iter", NR_LDPC_DEFAULT_BP_ITERATIONS)
        )
        if channel_type == "PUSCH" and codeword_index != 0:
            raise OperationError("PUSCH NR LDPC transport blocks require codeword_index=0")

        source_counts = _source_counts(
            metadata,
            int(bits.size),
            capture_layout,
        )
        coded_rows: list[np.ndarray] = []
        blocks: list[JsonDict] = []
        source_item_coded_counts: list[int] = []
        input_offset = 0
        coded_offset = 0
        with _NR_LOCK:
            for source_index, source_count in enumerate(source_counts):
                source_coded = 0
                source_offset = 0
                while source_offset < source_count:
                    valid_count = min(max_tb_bits, source_count - source_offset)
                    target_input, coded_count = _transport_block_configuration(
                        valid_count,
                        target_coderate,
                        num_bits_per_symbol,
                        num_layers,
                    )
                    encoder, _decoder = _nr_codec(
                        target_input,
                        coded_count,
                        target_coderate,
                        num_bits_per_symbol,
                        num_layers,
                        n_rnti,
                        n_id,
                        channel_type,
                        codeword_index,
                        num_bp_iter,
                    )
                    row = np.zeros((target_input,), dtype=np.float32)
                    start = input_offset + source_offset
                    row[:valid_count] = bits[start : start + valid_count]
                    try:
                        coded_tensor = encoder(
                            torch.as_tensor(row[None, :], dtype=torch.float32)
                        )
                        coded = coded_tensor.detach().cpu().numpy().reshape(-1)
                    except AssertionError as exc:
                        raise OperationError(
                            "Sionna asserted while encoding the bound NR transport block"
                        ) from exc
                    if int(coded.size) != coded_count:
                        raise OperationError(
                            "Sionna NR encoder returned %d bits, expected %d"
                            % (int(coded.size), coded_count)
                        )
                    if not np.all((coded == 0) | (coded == 1)):
                        raise OperationError("Sionna NR encoder returned non-binary values")
                    coded_rows.append(coded.astype(np.uint8, copy=False))
                    blocks.append(
                        _block_record(
                            encoder,
                            source_item_index=source_index,
                            valid_payload_bits=valid_count,
                            target_input_bits=target_input,
                            coded_offset=coded_offset,
                            num_coded_bits=coded_count,
                            target_coderate=target_coderate,
                            num_bits_per_symbol=num_bits_per_symbol,
                            num_layers=num_layers,
                            n_rnti=n_rnti,
                            n_id=n_id,
                            channel_type=channel_type,
                            codeword_index=codeword_index,
                        )
                    )
                    source_offset += valid_count
                    coded_offset += coded_count
                    source_coded += coded_count
                input_offset += source_count
                source_item_coded_counts.append(source_coded)

        coded_bits = np.concatenate(coded_rows).astype(np.uint8, copy=False)
        total_crc = int(sum(block["total_logical_crc_bits"] for block in blocks))
        total_tb = len(blocks)
        total_cb = int(sum(block["num_code_blocks"] for block in blocks))
        effective_rate = float(bits.size) / float(coded_bits.size)
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "array": "bits",
                "arrays": {
                    "bits": {
                        "dtype": str(coded_bits.dtype),
                        "shape": [int(coded_bits.size)],
                    }
                },
                "bit_count": int(coded_bits.size),
                "byte_count": int(math.ceil(float(coded_bits.size) / 8.0)),
                "coded_bit_count": int(coded_bits.size),
                "dtype": str(coded_bits.dtype),
                "shape": [int(coded_bits.size)],
                "storage": "unpacked_uint8",
                "bit_role": "coded",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "coding_scheme": "3gpp_nr_ldpc",
                "protected_digital_baseline": (
                    "sionna_nr_transport_block_coder_qpsk_link_abstraction"
                ),
                "physical_link_claim_scope": "transport_block_coding_only",
                "nr_phy_features_not_modeled": [
                    "mcs_table_or_index",
                    "prb_allocation",
                    "resource_grid",
                    "layer_mapping",
                    "dmrs",
                    "control_channels",
                    "nr_waveform",
                ],
                "channel_code_input_bit_count": int(bits.size),
                "channel_code_output_bit_count": int(coded_bits.size),
                "code_rate": effective_rate,
                "target_coderate": target_coderate,
                "source_item_coded_bit_counts": source_item_coded_counts,
                "nr_transport_blocks": blocks,
                "nr_transport_block_count": total_tb,
                "nr_code_block_count": total_cb,
                "nr_logical_crc_bit_count": total_crc,
                "nr_decoder_num_bp_iter": num_bp_iter,
                "nr_standard": dict(NR_LDPC_STANDARD),
                "nr_backend": "sionna.phy.nr.TBEncoder",
                "nr_backend_versions": backend_versions,
            }
        )
        output_metadata["nr_profile_sha256"] = _profile_sha256(
            blocks,
            num_bp_iter,
            backend_versions,
        )
        if capture_layout is not None and (
            len(source_item_coded_counts) != capture_layout.count
            or len(set(source_item_coded_counts)) != 1
        ):
            raise OperationError(
                "NR LDPC encoder could not preserve equal explicit capture records"
            )
        _rewrite_uniform_capture_layout(
            output_metadata,
            capture_layout,
            int(coded_bits.size),
            "NR LDPC encoder output at %s" % ctx.step_id,
        )
        output_metadata.setdefault("coding_history", [])
        output_metadata["coding_history"] = list(output_metadata["coding_history"]) + [
            {
                "scheme": "3gpp_nr_ldpc",
                "standard_version": "18.8.0",
                "input_bit_count": int(bits.size),
                "output_bit_count": int(coded_bits.size),
                "code_rate": effective_rate,
                "transport_block_count": total_tb,
                "code_block_count": total_cb,
            }
        ]
        path = ctx.output_path("coded_bits", ".npz")
        np.savez_compressed(
            path,
            bits=coded_bits,
            metadata_json=json.dumps(output_metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={
                "coded_bits": artifact(
                    "channel.coded_bits.numpy", path, output_metadata
                )
            },
            metrics={
                "channel.code_rate": effective_rate,
                "channel.target_code_rate": target_coderate,
                "channel.channel_code_input_bit_count": int(bits.size),
                "channel.coded_bit_count": int(coded_bits.size),
                "channel.fec_overhead_bit_count": int(coded_bits.size - bits.size),
                "channel.nr_ldpc.transport_block_count": total_tb,
                "channel.nr_ldpc.code_block_count": total_cb,
                "channel.nr_ldpc.logical_crc_bit_count": total_crc,
                "channel.nr_ldpc.standard_profile": 1,
            },
            metadata={
                "scheme": "3gpp_nr_ldpc",
                "standard": dict(NR_LDPC_STANDARD),
                "transport_block_count": total_tb,
                "code_block_count": total_cb,
                "profile_sha256": output_metadata["nr_profile_sha256"],
                "backend_versions": backend_versions,
            },
        )


class NrLdpcDecoderOperation(Operation):
    id = "channel.nr_ldpc_decoder"
    name = "3GPP NR transport-block LDPC soft decoder"
    input_kinds = {"llr": ["channel.llr.numpy"]}
    output_kinds = {
        "bits": "channel.payload_bits.numpy",
        "report": "metrics.report",
    }
    backends = {
        "benchmark_run": ["sionna"],
        "dataset_capture": ["sionna"],
        "differentiable_export": [],
    }
    differentiability = {
        "framework": "torch",
        "gradient": "stop",
        "trainable_params": False,
        "exportable": False,
        "reason": "The publication receiver returns hard bits and CRC decisions.",
    }
    params_schema = object_schema(
        {
            "on_tb_crc_failure": {
                "type": "string",
                "default": "zero_fill",
                "enum": ["zero_fill", "keep_estimate", "raise"],
            },
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _availability()
        payload["standard"] = dict(NR_LDPC_STANDARD)
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        if "num_bp_iter" in ctx.params:
            raise OperationError(
                "NR LDPC decoder num_bp_iter overrides are forbidden; the encoder-bound "
                "iteration count is part of the profile identity"
            )
        torch, _TBEncoder, _TBDecoder, _calculate = _require_sionna()
        runtime_backend_versions = _nr_backend_versions(torch)
        llr_artifact = ctx.require_input("llr")
        llr, metadata = _load_llr(llr_artifact.path, llr_artifact.metadata)
        capture_layout = _validated_capture_layout(
            metadata,
            int(llr.size),
            "NR LDPC decoder input at %s" % ctx.step_id,
        )
        blocks = metadata.get("nr_transport_blocks")
        if not isinstance(blocks, list) or not blocks:
            raise OperationError(
                "NR LDPC decoder requires encoder-bound nr_transport_blocks metadata"
            )
        bound_profile = metadata.get("nr_profile_sha256")
        bound_iterations = _decoder_iterations(
            metadata.get(
                "nr_decoder_num_bp_iter", NR_LDPC_DEFAULT_BP_ITERATIONS
            )
        )
        bound_backend_versions = _bound_backend_versions(
            metadata.get("nr_backend_versions")
        )
        expected_profile = _profile_sha256(
            blocks,
            bound_iterations,
            bound_backend_versions,
        )
        if not isinstance(bound_profile, str) or bound_profile != expected_profile:
            raise OperationError(
                "NR LDPC block/profile metadata does not match its encoder-bound identity"
            )
        if bound_backend_versions != runtime_backend_versions:
            raise OperationError(
                "NR LDPC encoder-bound backend versions do not match the decode runtime"
            )
        expected_coded = int(
            sum(int(block.get("num_coded_bits") or 0) for block in blocks)
        )
        if expected_coded != int(llr.size):
            raise OperationError(
                "NR LDPC LLR count %d does not match the bound coded-bit count %d"
                % (int(llr.size), expected_coded)
            )
        if capture_layout is not None:
            raw_coded_counts = metadata.get("source_item_coded_bit_counts")
            if (
                not isinstance(raw_coded_counts, list)
                or len(raw_coded_counts) != capture_layout.count
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or int(value) != capture_layout.elements_per_record
                    for value in raw_coded_counts
                )
            ):
                raise OperationError(
                    "NR LDPC coded-bit accounting conflicts with the explicit "
                    "capture-record layout"
                )
        failure_policy = str(ctx.params.get("on_tb_crc_failure", "zero_fill"))
        num_bp_iter = bound_iterations
        decoded_rows: list[np.ndarray] = []
        crc_status: list[bool] = []
        failure_indices: list[int] = []
        next_coded_offset = 0
        previous_source_item: int | None = None
        with _NR_LOCK:
            for block_index, block in enumerate(blocks):
                target_input = int(block["encoder_input_bit_count"])
                coded_count = int(block["num_coded_bits"])
                target_rate = float(block["target_coderate"])
                bps = int(block["num_bits_per_symbol"])
                layers = int(block["num_layers"])
                n_rnti = int(block["n_rnti"])
                n_id = int(block["n_id"])
                channel_type = str(block["channel_type"])
                codeword_index = int(block["codeword_index"])
                start = int(block["coded_offset"])
                source_item_index = int(block["source_item_index"])
                if start != next_coded_offset:
                    raise OperationError(
                        "NR LDPC block offsets must form one contiguous coded-bit partition"
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
                encoder, decoder = _nr_codec(
                    target_input,
                    coded_count,
                    target_rate,
                    bps,
                    layers,
                    n_rnti,
                    n_id,
                    channel_type,
                    codeword_index,
                    num_bp_iter,
                )
                # Noema demappers use positive LLR for bit zero. Sionna's
                # TBDecoder consumes logits with positive sign for bit one.
                logits = -llr[start : start + coded_count]
                try:
                    decoded_tensor, status_tensor = decoder(
                        torch.as_tensor(logits[None, :], dtype=torch.float32)
                    )
                except AssertionError as exc:
                    raise OperationError(
                        "Sionna asserted while decoding the bound NR transport block"
                    ) from exc
                estimate = (
                    decoded_tensor.detach().cpu().numpy().reshape(-1).astype(np.uint8)
                )
                valid_count = int(block["valid_payload_bit_count"])
                estimate = estimate[:valid_count]
                ok = bool(status_tensor.detach().cpu().reshape(-1)[0].item())
                crc_status.append(ok)
                if not ok:
                    failure_indices.append(block_index)
                    if failure_policy == "raise":
                        raise OperationError(
                            "NR transport block %d failed its TB CRC" % block_index
                        )
                    if failure_policy == "zero_fill":
                        estimate = np.zeros((valid_count,), dtype=np.uint8)
                decoded_rows.append(estimate)
                next_coded_offset += coded_count

        decoded = np.concatenate(decoded_rows).astype(np.uint8, copy=False)
        expected_payload = int(metadata.get("channel_code_input_bit_count") or decoded.size)
        if int(decoded.size) != expected_payload:
            raise OperationError(
                "NR decoder reconstructed %d payload bits, expected %d"
                % (int(decoded.size), expected_payload)
            )
        if capture_layout is not None:
            source_payload_counts = [0] * capture_layout.count
            for block in blocks:
                source_index = int(block["source_item_index"])
                if source_index < 0 or source_index >= capture_layout.count:
                    raise OperationError(
                        "NR LDPC block source-item index exceeds the explicit "
                        "capture-record layout"
                    )
                source_payload_counts[source_index] += int(
                    block["valid_payload_bit_count"]
                )
            if (
                any(value <= 0 for value in source_payload_counts)
                or len(set(source_payload_counts)) != 1
                or sum(source_payload_counts) != int(decoded.size)
            ):
                raise OperationError(
                    "NR LDPC decoder could not reconstruct equal explicit capture "
                    "records"
                )
        failure_count = len(failure_indices)
        block_count = len(blocks)
        report: JsonDict = {
            "schema_version": 1,
            "standard": dict(NR_LDPC_STANDARD),
            "backend": "sionna.phy.nr.TBDecoder",
            "backend_versions": bound_backend_versions,
            "profile_sha256": metadata.get("nr_profile_sha256"),
            "transport_block_count": block_count,
            "transport_block_crc_status": crc_status,
            "failed_transport_block_indices": failure_indices,
            "transport_block_crc_failure_count": failure_count,
            "transport_block_error_rate": float(failure_count) / float(block_count),
            "failure_policy": failure_policy,
            "num_bp_iter": num_bp_iter,
            "decoded_bit_count": int(decoded.size),
        }
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "bit_count": int(decoded.size),
                "bit_role": "payload",
                "decoder": "3gpp_nr_ldpc",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "nr_transport_block_crc_status": crc_status,
                "nr_transport_block_crc_failure_count": failure_count,
                "nr_transport_block_failure_policy": failure_policy,
            }
        )
        _rewrite_uniform_capture_layout(
            output_metadata,
            capture_layout,
            int(decoded.size),
            "NR LDPC decoder output at %s" % ctx.step_id,
        )
        bits_path = ctx.output_path("bits", ".npz")
        report_path = ctx.output_path("report", ".json")
        np.savez_compressed(
            bits_path,
            bits=decoded,
            metadata_json=json.dumps(output_metadata, sort_keys=True),
        )
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return OperationResult(
            outputs={
                "bits": artifact("channel.payload_bits.numpy", bits_path, output_metadata),
                "report": artifact("metrics.report", report_path, report),
            },
            metrics={
                "channel.decoded_bit_count": int(decoded.size),
                "channel.nr_ldpc.transport_block_crc_failure_count": failure_count,
                "channel.nr_ldpc.transport_block_error_rate": report[
                    "transport_block_error_rate"
                ],
                "channel.nr_ldpc.decode_success": int(failure_count == 0),
            },
            metadata=report,
        )


class CommunicationResourceAccountingOperation(Operation):
    """Emit non-conflated native, wrapper, framing, FEC, and modeled-link counts."""

    id = "channel.communication_resource_accounting"
    name = "Communication resource accounting boundary"
    accounting_profile = "noema.communication_resources.v1"
    input_kinds = {
        "payload_bits": ["channel.payload_bits.numpy", "channel.bits.numpy"],
        "framed_bits": ["channel.payload_bits.numpy", "channel.bits.numpy"],
        "coded_bits": ["channel.coded_bits.numpy", "channel.bits.numpy"],
        "symbols": ["channel.symbols.complex_numpy"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema(
        {
            "pilot_symbol_count": {"type": "integer", "default": 0, "minimum": 0},
            "physical_header_symbol_count": {"type": "integer", "default": 0, "minimum": 0},
            "total_channel_use_count": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        payload, payload_meta = _load_bits(
            ctx.require_input("payload_bits").path,
            ctx.require_input("payload_bits").metadata,
        )
        framed, framed_meta = _load_bits(
            ctx.require_input("framed_bits").path,
            ctx.require_input("framed_bits").metadata,
        )
        coded, coded_meta = _load_bits(
            ctx.require_input("coded_bits").path,
            ctx.require_input("coded_bits").metadata,
        )
        symbols, symbol_meta = _load_symbols(
            ctx.require_input("symbols").path,
            ctx.require_input("symbols").metadata,
        )
        native_bits = payload_meta.get("native_codec_bit_count")
        if not isinstance(native_bits, int) or native_bits < 0:
            raise OperationError(
                "Publication resource accounting requires native_codec_bit_count "
                "from the source codec; safe-wrapper size is not native rate"
            )
        native_bits = int(native_bits)
        payload_bits = int(payload.size)
        framed_bits = int(framed.size)
        coded_bits = int(coded.size)
        if native_bits > payload_bits:
            raise OperationError("native codec bits exceed serialized payload bits")
        if framed_bits < payload_bits:
            raise OperationError("framed bit count is smaller than serialized payload")
        if coded_bits < framed_bits:
            raise OperationError("FEC output is smaller than its framed input")
        bits_per_symbol = int(symbol_meta.get("bits_per_symbol") or 0)
        if bits_per_symbol <= 0:
            raise OperationError("Modulator metadata does not bind bits_per_symbol")
        modulation_capacity_bits = int(symbols.size) * bits_per_symbol
        if modulation_capacity_bits < coded_bits:
            raise OperationError("Modulation symbol capacity is smaller than coded bits")

        wrapper_bits = payload_bits - native_bits
        framing_bits = framed_bits - payload_bits
        fec_bits = coded_bits - framed_bits
        modulation_padding_bits = modulation_capacity_bits - coded_bits
        pilot_symbols = int(ctx.params.get("pilot_symbol_count", 0))
        physical_header_symbols = int(
            ctx.params.get("physical_header_symbol_count", 0)
        )
        occupied_symbols = int(symbols.size) + pilot_symbols + physical_header_symbols
        declared_total = int(ctx.params.get("total_channel_use_count", 0))
        total_uses = declared_total or occupied_symbols
        if total_uses < occupied_symbols:
            raise OperationError(
                "total_channel_use_count is smaller than data, pilot, and physical-header symbols"
            )
        grid_padding_symbols = total_uses - occupied_symbols
        report: JsonDict = {
            "schema_version": 1,
            "accounting_profile": self.accounting_profile,
            "native_codec_bit_count": native_bits,
            "safe_serialization_wrapper_bit_count": wrapper_bits,
            "wire_protocol_overhead_bit_count": wrapper_bits,
            "serialized_payload_bit_count": payload_bits,
            "serialized_payload_format": payload_meta.get(
                "serialized_payload_format"
            )
            or payload_meta.get("payload_format"),
            "framing_header_crc_padding_bit_count": framing_bits,
            "framed_bit_count": framed_bits,
            "fec_rate_matching_overhead_bit_count": fec_bits,
            "coded_bit_count": coded_bits,
            "modulation_padding_bit_count": modulation_padding_bits,
            "modulator_capacity_bit_count": modulation_capacity_bits,
            "data_symbol_count": int(symbols.size),
            "pilot_symbol_count": pilot_symbols,
            "physical_header_symbol_count": physical_header_symbols,
            "grid_padding_symbol_count": grid_padding_symbols,
            "total_channel_use_count": total_uses,
            "bits_per_symbol": bits_per_symbol,
            "codec": payload_meta.get("codec"),
            "payload_format": payload_meta.get("payload_format"),
            "packet_protocol": framed_meta.get("packet_protocol"),
            "channel_code": coded_meta.get("coding_scheme"),
            "modulation": symbol_meta.get("modulation"),
        }
        pixel_counts = _source_pixel_counts(payload_meta)
        pixel_count = int(sum(pixel_counts)) if pixel_counts else None
        declared_pixel_count = payload_meta.get("pixel_count")
        if isinstance(declared_pixel_count, int) and declared_pixel_count > 0:
            if pixel_count is not None and pixel_count != declared_pixel_count:
                raise OperationError(
                    "Declared pixel_count disagrees with source-item image shapes"
                )
            pixel_count = int(declared_pixel_count)
        if isinstance(pixel_count, int) and pixel_count > 0:
            report.update(
                {
                    "pixel_count": pixel_count,
                    "native_codec_bpp": float(native_bits) / float(pixel_count),
                    "serialized_payload_bpp": float(payload_bits) / float(pixel_count),
                    "framed_bpp": float(framed_bits) / float(pixel_count),
                    "coded_bpp": float(coded_bits) / float(pixel_count),
                    "padded_bpp": float(modulation_capacity_bits)
                    / float(pixel_count),
                    "channel_uses_per_pixel": float(total_uses) / float(pixel_count),
                }
            )
        raw_coded_counts = coded_meta.get("source_item_coded_bit_counts")
        if isinstance(raw_coded_counts, list):
            source_coded_counts = [int(value) for value in raw_coded_counts]
        else:
            source_coded_counts = [coded_bits]
        if (
            not source_coded_counts
            or any(value <= 0 for value in source_coded_counts)
            or sum(source_coded_counts) != coded_bits
        ):
            raise OperationError(
                "Per-source coded-bit counts must form the complete coded stream"
            )
        if pixel_counts and len(pixel_counts) != len(source_coded_counts):
            raise OperationError(
                "Per-source pixel and coded-bit counts have different item counts"
            )
        source_data_symbols: list[int] = []
        bit_offset = 0
        for count in source_coded_counts:
            first_symbol = bit_offset // bits_per_symbol
            end_offset = bit_offset + count
            last_symbol_exclusive = int(
                math.ceil(float(end_offset) / float(bits_per_symbol))
            )
            source_data_symbols.append(last_symbol_exclusive - first_symbol)
            bit_offset = end_offset
        shared_overhead_symbols = (
            pilot_symbols + physical_header_symbols + grid_padding_symbols
        )
        source_total_uses = [
            value + shared_overhead_symbols for value in source_data_symbols
        ]
        report.update(
            {
                "source_item_coded_bit_counts": source_coded_counts,
                "source_item_data_symbol_counts": source_data_symbols,
                "source_item_total_channel_use_counts": source_total_uses,
                "shared_symbol_overhead_attribution": "charge_all_to_each_source_item",
                "source_item_use_counts_are_additive": bool(
                    shared_overhead_symbols == 0
                    and sum(source_data_symbols) == int(symbols.size)
                ),
            }
        )
        if pixel_counts:
            source_uses_per_pixel = [
                float(uses) / float(pixels)
                for uses, pixels in zip(source_total_uses, pixel_counts)
            ]
            report.update(
                {
                    "source_item_pixel_counts": pixel_counts,
                    "source_item_channel_uses_per_pixel": source_uses_per_pixel,
                    "max_source_item_uses_per_pixel": max(source_uses_per_pixel),
                }
            )
        report["accounting_identity_sha256"] = canonical_json_sha256(report)
        path = ctx.output_path("report", ".json")
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metrics: JsonDict = {
            "codec.native_bit_count": native_bits,
            "codec.safe_serialization_wrapper_bit_count": wrapper_bits,
            "channel.payload_bit_count": payload_bits,
            "channel.framing_overhead_bit_count": framing_bits,
            "channel.framed_bit_count": framed_bits,
            "channel.fec_overhead_bit_count": fec_bits,
            "channel.coded_bit_count": coded_bits,
            "channel.modulation_padding_bit_count": modulation_padding_bits,
            "channel.transmitted_bit_count": modulation_capacity_bits,
            "channel.data_symbol_count": int(symbols.size),
            "channel.pilot_symbol_count": pilot_symbols,
            "channel.physical_header_symbol_count": physical_header_symbols,
            "channel.grid_padding_symbol_count": grid_padding_symbols,
            "channel.channel_use_count": total_uses,
            "channel.total_channel_use_count": total_uses,
        }
        if "native_codec_bpp" in report:
            metrics.update(
                {
                    "rate.native_codec_bpp": report["native_codec_bpp"],
                    "rate.serialized_payload_bpp": report["serialized_payload_bpp"],
                    "rate.framed_bpp": report["framed_bpp"],
                    "rate.coded_bpp": report["coded_bpp"],
                    "rate.padded_bpp": report["padded_bpp"],
                    "channel.uses_per_pixel": report["channel_uses_per_pixel"],
                }
            )
        if "max_source_item_uses_per_pixel" in report:
            metrics["channel.max_source_item_uses_per_pixel"] = report[
                "max_source_item_uses_per_pixel"
            ]
        return OperationResult(
            outputs={"report": artifact("metrics.report", path, report)},
            metrics=metrics,
            metadata=report,
        )


class CommunicationResourceAccountingV2Operation(
    CommunicationResourceAccountingOperation
):
    """Strict semantic-kind accounting boundary.

    The v1 operation remains available for historical recipes whose fixed-point
    checkpoints expose ``channel.bits.numpy``.  This version rejects generic
    bits and requires distinct payload, framed, and coded artifacts.
    """

    id = "channel.communication_resource_accounting.v2"
    name = "Communication resource accounting boundary (strict kinds v2)"
    accounting_profile = "noema.communication_resources.v2"
    input_kinds = {
        "payload_bits": ["channel.payload_bits.numpy"],
        "framed_bits": ["channel.framed_bits.numpy"],
        "coded_bits": ["channel.coded_bits.numpy"],
        "symbols": ["channel.symbols.complex_numpy"],
    }
