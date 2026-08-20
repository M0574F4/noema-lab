from __future__ import annotations

"""Controlled automatic-modulation-recognition operations.

The source deliberately keeps waveform frames and class truth in separate
artifacts. That separation preserves the system model while allowing a later
training plan to capture either value for any researcher-defined purpose.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact, file_sha256
from noema_lab.core.structured_input import decode_strict_json_object
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationResult, object_schema


JsonDict = Dict[str, Any]
MODULATION_CLASSES = ("bpsk", "qpsk", "qam16")
MODULATION_FRAME_KIND = "ai_phy.modulation_frames.numpy"
MODULATION_LABEL_KIND = "ai_phy.modulation_labels.numpy"
MODULATION_OBSERVATION_KIND = "ai_phy.modulation_iq_frames.numpy"
MODULATION_PREDICTION_KIND = "ai_phy.modulation_predictions.numpy"
CONFUSION_MATRIX_KIND = "metrics.confusion_matrix.numpy"


def _constellations() -> Dict[str, np.ndarray]:
    return {
        "bpsk": np.asarray([-1.0, 1.0], dtype=np.complex64),
        "qpsk": (
            np.asarray([-1.0 - 1.0j, -1.0 + 1.0j, 1.0 - 1.0j, 1.0 + 1.0j], dtype=np.complex64)
            / np.sqrt(2.0)
        ).astype(np.complex64),
        "qam16": (
            np.asarray(
                [complex(i, q) for i in (-3.0, -1.0, 1.0, 3.0) for q in (-3.0, -1.0, 1.0, 3.0)],
                dtype=np.complex64,
            )
            / np.sqrt(10.0)
        ).astype(np.complex64),
    }


def _save_npz(path: Path, metadata: Mapping[str, Any], **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays, metadata_json=json.dumps(dict(metadata), sort_keys=True))


def _load_npz(path: Path) -> Tuple[Dict[str, np.ndarray], JsonDict]:
    try:
        with np.load(str(path), allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files if name != "metadata_json"}
            metadata = (
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Modulation artifact metadata_json",
                )
                if "metadata_json" in payload
                else {}
            )
    except Exception as exc:
        raise OperationError("Could not read modulation NPZ artifact %s: %s" % (path, exc)) from exc
    return arrays, dict(metadata or {})


class ModulationFrameSourceOperation(Operation):
    id = "source.modulation_frames"
    name = "Balanced modulation-frame source"
    output_kinds = {
        "frames": MODULATION_FRAME_KIND,
        "labels": MODULATION_LABEL_KIND,
    }
    differentiability = {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": "The seeded symbol and label generator is benchmark data, not a learned component.",
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": [],
    }
    formats = {"artifact": "npz", "tensor": "numpy.ndarray"}
    params_schema = object_schema(
        {
            "frame_count": {"type": "integer", "default": 96, "minimum": 3},
            "symbols_per_frame": {"type": "integer", "default": 128, "minimum": 16},
            "seed": {"type": "integer", "default": 23, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        frame_count = int(ctx.params.get("frame_count", 96))
        symbols_per_frame = int(ctx.params.get("symbols_per_frame", 128))
        if frame_count % len(MODULATION_CLASSES) != 0:
            raise OperationError(
                "Balanced modulation source frame_count must be divisible by %d" % len(MODULATION_CLASSES)
            )
        seed = ctx.seed("modulation_frames", default=23)
        rng = np.random.RandomState(seed)
        each = frame_count // len(MODULATION_CLASSES)
        labels = np.repeat(np.arange(len(MODULATION_CLASSES), dtype=np.int64), each)
        frames = np.empty((frame_count, symbols_per_frame), dtype=np.complex64)
        constellations = _constellations()
        for class_id, class_name in enumerate(MODULATION_CLASSES):
            points = constellations[class_name]
            positions = np.flatnonzero(labels == class_id)
            indices = rng.randint(0, len(points), size=(len(positions), symbols_per_frame))
            frames[positions] = points[indices]
        order = rng.permutation(frame_count)
        frames = frames[order]
        labels = labels[order]
        common = {
            "dataset": "synthetic_modulation_iq",
            "version": "synthetic-modulation-iq-blind-carrier-v1",
            "split": "fixed_seed",
            "frame_count": frame_count,
            "symbols_per_frame": symbols_per_frame,
            "class_names": list(MODULATION_CLASSES),
            "class_counts": {name: each for name in MODULATION_CLASSES},
            "balanced": True,
            "average_symbol_power": 1.0,
            "seed": int(seed),
            "capture_record_axis": 0,
        }
        frame_metadata = dict(common)
        frame_metadata.update(
            {
                "array": "symbols",
                "axes": ["frame", "symbol"],
                "truth_separated": True,
            }
        )
        label_metadata = dict(common)
        label_metadata.update(
            {
                "array": "class_ids",
                "axes": ["frame"],
                "label_encoding": "zero_based_class_id",
            }
        )
        frames_path = ctx.output_path("frames", ".npz")
        labels_path = ctx.output_path("labels", ".npz")
        _save_npz(frames_path, frame_metadata, symbols=frames)
        _save_npz(labels_path, label_metadata, class_ids=labels)
        return OperationResult(
            outputs={
                "frames": artifact(MODULATION_FRAME_KIND, frames_path, frame_metadata),
                "labels": artifact(MODULATION_LABEL_KIND, labels_path, label_metadata),
            },
            metrics={
                "modulation_recognition.frame_count": frame_count,
                "modulation_recognition.symbols_per_frame": symbols_per_frame,
            },
            metadata={"balanced": True, "class_names": list(MODULATION_CLASSES)},
        )


class ModulationAwgnObservationOperation(Operation):
    id = "wireless.modulation_awgn_observation"
    name = "Modulation frames with carrier uncertainty and AWGN"
    input_kinds = {"frames": [MODULATION_FRAME_KIND]}
    output_kinds = {"observation": MODULATION_OBSERVATION_KIND}
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "Complex AWGN can be reproduced by a differentiable tensor materialization.",
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": ["torch"],
    }
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema(
        {
            "snr_db_min": {"type": "number", "default": 8.0},
            "snr_db_max": {"type": "number", "default": 8.0},
            "carrier_phase_min_rad": {"type": "number", "default": 0.0},
            "carrier_phase_max_rad": {"type": "number", "default": 0.0},
            "frequency_offset_min_cycles_per_symbol": {
                "type": "number",
                "default": 0.0,
            },
            "frequency_offset_max_cycles_per_symbol": {
                "type": "number",
                "default": 0.0,
            },
            "seed": {"type": "integer", "default": 23001, "minimum": 0},
        }
    )

    def validate_preflight(
        self,
        params: Mapping[str, Any],
        inputs: Mapping[str, str] | None = None,
    ) -> None:
        snr_min = float(params.get("snr_db_min", 8.0))
        snr_max = float(params.get("snr_db_max", 8.0))
        if snr_max < snr_min:
            raise OperationError(
                "snr_db_max must be greater than or equal to snr_db_min"
            )
        phase_min = float(params.get("carrier_phase_min_rad", 0.0))
        phase_max = float(params.get("carrier_phase_max_rad", 0.0))
        if phase_max < phase_min:
            raise OperationError(
                "carrier_phase_max_rad must be greater than or equal to "
                "carrier_phase_min_rad"
            )
        frequency_min = float(
            params.get("frequency_offset_min_cycles_per_symbol", 0.0)
        )
        frequency_max = float(
            params.get("frequency_offset_max_cycles_per_symbol", 0.0)
        )
        if frequency_max < frequency_min:
            raise OperationError(
                "frequency_offset_max_cycles_per_symbol must be greater than "
                "or equal to frequency_offset_min_cycles_per_symbol"
            )

    def run(self, ctx: OperationContext) -> OperationResult:
        arrays, source_metadata = _load_npz(ctx.require_input("frames").path)
        if "symbols" not in arrays:
            raise OperationError("Modulation-frame artifact is missing `symbols`")
        symbols = np.asarray(arrays["symbols"], dtype=np.complex64)
        if symbols.ndim != 2:
            raise OperationError("Modulation frames must have shape [frame, symbol]")
        snr_min = float(ctx.params.get("snr_db_min", 8.0))
        snr_max = float(ctx.params.get("snr_db_max", 8.0))
        if not math.isfinite(snr_min) or not math.isfinite(snr_max) or snr_max < snr_min:
            raise OperationError("snr_db_max must be finite and greater than or equal to snr_db_min")
        phase_min = float(ctx.params.get("carrier_phase_min_rad", 0.0))
        phase_max = float(ctx.params.get("carrier_phase_max_rad", 0.0))
        frequency_min = float(
            ctx.params.get("frequency_offset_min_cycles_per_symbol", 0.0)
        )
        frequency_max = float(
            ctx.params.get("frequency_offset_max_cycles_per_symbol", 0.0)
        )
        if (
            not all(
                math.isfinite(value)
                for value in (phase_min, phase_max, frequency_min, frequency_max)
            )
            or phase_max < phase_min
            or frequency_max < frequency_min
        ):
            raise OperationError(
                "Carrier phase and frequency-offset ranges must be finite and ordered"
            )
        seed = ctx.seed("modulation_awgn", default=23001)
        rng = np.random.RandomState(seed)
        frame_count = int(symbols.shape[0])
        if snr_min == snr_max:
            snr_db = np.full((frame_count,), snr_min, dtype=np.float32)
        else:
            snr_db = rng.uniform(snr_min, snr_max, size=frame_count).astype(np.float32)
        if phase_min == phase_max:
            carrier_phase = np.full((frame_count,), phase_min, dtype=np.float32)
        else:
            carrier_phase = rng.uniform(
                phase_min, phase_max, size=frame_count
            ).astype(np.float32)
        if frequency_min == frequency_max:
            frequency_offset = np.full(
                (frame_count,), frequency_min, dtype=np.float32
            )
        else:
            frequency_offset = rng.uniform(
                frequency_min, frequency_max, size=frame_count
            ).astype(np.float32)
        symbol_index = np.arange(symbols.shape[1], dtype=np.float32).reshape(1, -1)
        carrier_rotation = np.exp(
            1j
            * (
                carrier_phase.reshape(-1, 1)
                + 2.0
                * np.pi
                * frequency_offset.reshape(-1, 1)
                * symbol_index
            )
        ).astype(np.complex64)
        impaired = (symbols * carrier_rotation).astype(np.complex64)
        noise_variance = np.power(10.0, -snr_db / 10.0).astype(np.float32)
        scale = np.sqrt(noise_variance / 2.0).reshape(-1, 1)
        noise = scale * (
            rng.standard_normal(symbols.shape).astype(np.float32)
            + 1j * rng.standard_normal(symbols.shape).astype(np.float32)
        )
        received = (impaired + noise).astype(np.complex64)
        iq_ri = np.stack([received.real, received.imag], axis=-1).astype(np.float32)
        metadata = {
            "dataset": source_metadata.get("dataset", "synthetic_modulation_iq"),
            "version": source_metadata.get(
                "version", "synthetic-modulation-iq-blind-carrier-v1"
            ),
            "array": "iq_ri",
            "axes": ["frame", "symbol", "iq"],
            "capture_record_axis": 0,
            "frame_count": frame_count,
            "symbols_per_frame": int(symbols.shape[1]),
            "class_names": list(MODULATION_CLASSES),
            "observation_model": (
                "unit_power_symbols_with_unknown_per_frame_carrier_phase_and_"
                "frequency_offset_plus_complex_awgn"
            ),
            "snr_db_min": float(np.min(snr_db)),
            "snr_db_max": float(np.max(snr_db)),
            "snr_db_mean": float(np.mean(snr_db)),
            "snr_db_per_frame": [float(item) for item in snr_db],
            "noise_variance_per_frame": [float(item) for item in noise_variance],
            "carrier_phase_offset_rad_per_frame": [
                float(item) for item in carrier_phase
            ],
            "timing_offset_symbols": 0.0,
            "frequency_offset_cycles_per_symbol_per_frame": [
                float(item) for item in frequency_offset
            ],
            "seed": int(seed),
        }
        path = ctx.output_path("observation", ".npz")
        _save_npz(path, metadata, iq_ri=iq_ri)
        return OperationResult(
            outputs={"observation": artifact(MODULATION_OBSERVATION_KIND, path, metadata)},
            metrics={
                "channel.snr_db": float(np.mean(snr_db)),
                "channel.snr_db_min": float(np.min(snr_db)),
                "channel.snr_db_max": float(np.max(snr_db)),
                "channel.carrier.initial_phase_abs_mean_rad": float(
                    np.mean(np.abs(carrier_phase))
                ),
                "channel.carrier.cfo_abs_mean_cycles_per_symbol": float(
                    np.mean(np.abs(frequency_offset))
                ),
                "modulation_recognition.frame_count": frame_count,
            },
            metadata=metadata,
        )


class ModulationClassifierAdapterOperation(Operation):
    id = "model.modulation_classifier_adapter"
    name = "Modulation classifier"
    input_kinds = {"observation": [MODULATION_OBSERVATION_KIND]}
    output_kinds = {"prediction": MODULATION_PREDICTION_KIND}
    differentiability = {
        "framework": "torch",
        "gradient": "surrogate",
        "trainable_params": True,
        "exportable": True,
        "reason": "The classifier slot accepts a portable learned artifact while classical modes remain reproducible baselines.",
    }
    backends = {
        "benchmark_run": ["numpy", "onnxruntime"],
        "dataset_capture": ["numpy", "onnxruntime"],
        "differentiable_export": ["torch"],
    }
    equivalence = {
        "type": "behavioral",
        "reason": "Classifiers are compared through the fixed class vocabulary and task metrics.",
    }
    formats = {"artifact": "npz", "tensor": "torch.Tensor", "checkpoint": "onnx"}
    trained_artifact_abi = {
        "component_id": "classifier",
        "component_role": "modulation_classifier",
        "entrypoint_id": "modulation_classifier",
        "required_operation_inputs": ["observation"],
        "inputs": {"iq_ri": {"dtype": "float32", "shape": ["batch", "sample", 2]}},
        "outputs": {"class_logits": {"dtype": "float32", "shape": ["batch", 3]}},
        "binding_params": {
            "mode": "learned_artifact",
            "artifact_manifest_path": "trained_artifact.yaml",
            "artifact_entrypoint": "modulation_classifier",
        },
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "blind_differential_cumulant",
            "status": "implemented",
            "parameter_bindings": {"mode": "classical_cumulant"},
        },
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "awgn_mixture_likelihood",
            "status": "implemented",
            "parameter_bindings": {"mode": "classical_likelihood"},
        },
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "minimum_constellation_evm",
            "status": "implemented",
            "parameter_bindings": {"mode": "classical_evm"},
        },
        {
            "runner": "benchmark_run",
            "backend": "onnxruntime",
            "implementation": "portable_trained_artifact_runtime",
            "status": "implemented",
            "parameter_bindings": {"mode": "learned_artifact"},
        },
        {
            "runner": "dataset_capture",
            "backend": "numpy",
            "implementation": "blind_differential_cumulant",
            "status": "implemented",
            "parameter_bindings": {"mode": "classical_cumulant"},
        },
        {
            "runner": "dataset_capture",
            "backend": "numpy",
            "implementation": "awgn_mixture_likelihood",
            "status": "implemented",
            "parameter_bindings": {"mode": "classical_likelihood"},
        },
        {
            "runner": "dataset_capture",
            "backend": "onnxruntime",
            "implementation": "portable_trained_artifact_runtime",
            "status": "implemented",
            "parameter_bindings": {"mode": "learned_artifact"},
        },
        {
            "runner": "differentiable_export",
            "backend": "torch",
            "implementation": "modulation_classification_training_endpoint",
            "status": "implemented",
        },
    ]
    params_schema = object_schema(
        {
            "mode": {
                "type": "string",
                "default": "classical_cumulant",
                "enum": [
                    "classical_cumulant",
                    "classical_likelihood",
                    "classical_evm",
                    "learned_artifact",
                ],
            },
            "artifact_manifest_path": {
                "type": "string",
                "default": "",
                "description": "Schema-v2 trained artifact implementing the modulation-classifier ABI.",
                "x-noema-ui": {
                    "control": "trained_artifact",
                    "label": "Trained artifact",
                    "accept": ".zip,.noema-artifact,.yaml,.yml,.json,application/octet-stream",
                    "visible_when": {"mode": "learned_artifact"},
                    "derived_params": [
                        "mode",
                        "artifact_manifest_path",
                        "artifact_entrypoint",
                        "artifact_package_sha256",
                    ],
                },
            },
            "artifact_entrypoint": {
                "type": "string",
                "default": "modulation_classifier",
                "x-noema-ui": {"hidden": True},
            },
            "artifact_package_sha256": {
                "type": "string",
                "default": "",
                "x-noema-ui": {"hidden": True},
            },
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        arrays, metadata = _load_npz(ctx.require_input("observation").path)
        if "iq_ri" not in arrays:
            raise OperationError("Modulation observation is missing `iq_ri`")
        iq_ri = np.asarray(arrays["iq_ri"], dtype=np.float32)
        if iq_ri.ndim != 3 or iq_ri.shape[-1] != 2:
            raise OperationError("Modulation observation iq_ri must have shape [frame, symbol, 2]")
        if not np.all(np.isfinite(iq_ri)):
            raise OperationError("Modulation observation contains non-finite values")
        received = iq_ri[..., 0] + 1j * iq_ri[..., 1]
        oracle_received = _remove_known_carrier_impairment(received, metadata)
        likelihood, oracle_evm = _classical_scores(oracle_received, metadata)
        _, blind_evm = _classical_scores(received, metadata)
        cumulant_scores, cumulant_features = _blind_cumulant_scores(received)
        mode = str(ctx.params.get("mode") or "classical_cumulant")
        checkpoint_sha = None
        if mode == "classical_cumulant":
            class_logits = cumulant_scores
            selected_backend = "numpy"
        elif mode == "classical_likelihood":
            class_logits = likelihood
            selected_backend = "numpy"
        elif mode == "classical_evm":
            class_logits = -np.square(blind_evm).astype(np.float32)
            selected_backend = "numpy"
        elif mode == "learned_artifact":
            manifest_value = str(ctx.params.get("artifact_manifest_path") or "").strip()
            entrypoint = str(ctx.params.get("artifact_entrypoint") or "modulation_classifier").strip()
            if not manifest_value:
                raise OperationError("learned_artifact modulation classifier requires params.artifact_manifest_path")
            manifest_path = Path(manifest_value).expanduser()
            if not manifest_path.is_file():
                raise OperationError("Modulation-classifier trained-artifact manifest does not exist: %s" % manifest_path)
            try:
                from noema_lab.core.trained_artifact_runtime import run_trained_artifact_entrypoint

                outputs = run_trained_artifact_entrypoint(
                    manifest_path,
                    entrypoint,
                    {"iq_ri": iq_ri},
                    expected_package_sha256=str(
                        ctx.params.get("artifact_package_sha256") or ""
                    ),
                )
            except Exception as exc:
                raise OperationError("Modulation-classifier trained-artifact inference failed: %s" % exc) from exc
            if "class_logits" not in outputs:
                raise OperationError("Modulation-classifier trained artifact did not return `class_logits`")
            class_logits = np.asarray(outputs["class_logits"], dtype=np.float32)
            expected = (int(iq_ri.shape[0]), len(MODULATION_CLASSES))
            if tuple(class_logits.shape) != expected:
                raise OperationError(
                    "Modulation-classifier class_logits must have shape %s, got %s"
                    % (expected, tuple(class_logits.shape))
                )
            if not np.all(np.isfinite(class_logits)):
                raise OperationError("Modulation-classifier class_logits contains non-finite values")
            checkpoint_sha = file_sha256(manifest_path)
            selected_backend = "onnxruntime"
        else:
            raise OperationError("Unknown modulation classifier mode: %s" % mode)
        predictions = np.argmax(class_logits, axis=1).astype(np.int64)
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "array": "class_ids",
                "class_names": list(MODULATION_CLASSES),
                "classifier_mode": mode,
                "data_plane_backend": selected_backend,
                "capture_record_axis": 0,
            }
        )
        if checkpoint_sha:
            output_metadata["checkpoint_sha256"] = checkpoint_sha
            output_metadata["artifact_manifest_sha256"] = checkpoint_sha
        path = ctx.output_path("prediction", ".npz")
        _save_npz(
            path,
            output_metadata,
            class_ids=predictions,
            class_logits=np.asarray(class_logits, dtype=np.float32),
            likelihood_scores=likelihood,
            blind_cumulant_scores=cumulant_scores,
            blind_cumulant_features=cumulant_features,
            evm_rms=blind_evm,
            oracle_compensated_evm_rms=oracle_evm,
        )
        return OperationResult(
            outputs={"prediction": artifact(MODULATION_PREDICTION_KIND, path, output_metadata)},
            metrics={"modulation_recognition.predicted_frame_count": int(predictions.size)},
            metadata={
                "mode": mode,
                "class_names": list(MODULATION_CLASSES),
                "checkpoint_sha256": checkpoint_sha,
                "artifact_manifest_sha256": checkpoint_sha,
                "data_plane_backend": selected_backend,
            },
        )


class ModulationClassificationMetricsOperation(Operation):
    id = "metrics.modulation_classification"
    name = "Modulation-recognition metrics"
    input_kinds = {
        "truth": [MODULATION_LABEL_KIND],
        "prediction": [MODULATION_PREDICTION_KIND],
    }
    output_kinds = {
        "report": "metrics.report",
        "confusion_matrix": CONFUSION_MATRIX_KIND,
    }
    differentiability = {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": "Discrete evaluation metrics are terminal evidence.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": []}
    formats = {"artifact": "json|npz", "tensor": "numpy.ndarray"}
    params_schema = object_schema(
        {"snr_bin_width_db": {"type": "number", "default": 2.0, "exclusiveMinimum": 0.0}}
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        truth_arrays, truth_metadata = _load_npz(ctx.require_input("truth").path)
        pred_arrays, pred_metadata = _load_npz(ctx.require_input("prediction").path)
        if "class_ids" not in truth_arrays or "class_ids" not in pred_arrays:
            raise OperationError("Modulation truth and prediction artifacts must contain `class_ids`")
        truth = np.asarray(truth_arrays["class_ids"], dtype=np.int64)
        prediction = np.asarray(pred_arrays["class_ids"], dtype=np.int64)
        if truth.ndim != 1 or prediction.ndim != 1 or truth.shape != prediction.shape:
            raise OperationError("Modulation truth and prediction must be equal-length one-dimensional class IDs")
        class_count = len(MODULATION_CLASSES)
        if truth.size == 0:
            raise OperationError("Modulation classification requires at least one frame")
        if np.any(truth < 0) or np.any(truth >= class_count) or np.any(prediction < 0) or np.any(prediction >= class_count):
            raise OperationError("Modulation class IDs must be in [0, %d]" % (class_count - 1))
        if list(truth_metadata.get("class_names") or MODULATION_CLASSES) != list(MODULATION_CLASSES):
            raise OperationError("Modulation truth class vocabulary does not match the operation contract")
        if list(pred_metadata.get("class_names") or MODULATION_CLASSES) != list(MODULATION_CLASSES):
            raise OperationError("Modulation prediction class vocabulary does not match the operation contract")
        confusion = np.zeros((class_count, class_count), dtype=np.int64)
        np.add.at(confusion, (truth, prediction), 1)
        support = confusion.sum(axis=1)
        predicted_count = confusion.sum(axis=0)
        diagonal = np.diag(confusion).astype(np.float64)
        recall = np.divide(diagonal, support, out=np.zeros(class_count), where=support > 0)
        precision = np.divide(diagonal, predicted_count, out=np.zeros(class_count), where=predicted_count > 0)
        f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros(class_count), where=(precision + recall) > 0)
        normalized = np.divide(
            confusion.astype(np.float64), support.reshape(-1, 1), out=np.zeros_like(confusion, dtype=np.float64), where=support.reshape(-1, 1) > 0
        ).astype(np.float32)
        accuracy = float(np.mean(truth == prediction))
        balanced_accuracy = float(np.mean(recall))
        macro_f1 = float(np.mean(f1))
        metrics = {
            "modulation_recognition.accuracy": accuracy,
            "modulation_recognition.balanced_accuracy": balanced_accuracy,
            "modulation_recognition.macro_f1": macro_f1,
            "classification.balanced_accuracy": balanced_accuracy,
            "task.accuracy": accuracy,
            "task.score": macro_f1,
        }
        snr_values = pred_metadata.get("snr_db_per_frame")
        accuracy_by_snr = []
        if isinstance(snr_values, list) and len(snr_values) == int(truth.size):
            snr = np.asarray(snr_values, dtype=np.float64)
            metrics["channel.snr_db"] = float(np.mean(snr))
            width = float(ctx.params.get("snr_bin_width_db", 2.0))
            bins = np.floor(snr / width) * width
            for lower in sorted(set(float(item) for item in bins)):
                mask = bins == lower
                accuracy_by_snr.append(
                    {
                        "snr_db_lower": lower,
                        "snr_db_upper": lower + width,
                        "frame_count": int(np.count_nonzero(mask)),
                        "accuracy": float(np.mean(truth[mask] == prediction[mask])),
                    }
                )
        confusion_metadata = {
            "class_names": list(MODULATION_CLASSES),
            "orientation": "rows=true_class, columns=predicted_class",
            "support": [int(item) for item in support],
            "counts": confusion.tolist(),
            "row_normalized": normalized.tolist(),
            "per_class": [
                {
                    "class_id": index,
                    "class_name": MODULATION_CLASSES[index],
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index in range(class_count)
            ],
        }
        report_payload = {
            "metrics": metrics,
            "metadata": {
                "task": "automatic_modulation_recognition",
                "frame_count": int(truth.size),
                "classifier_mode": pred_metadata.get("classifier_mode"),
                "confusion_matrix": confusion_metadata,
                "accuracy_by_snr": accuracy_by_snr,
            },
        }
        report_path = ctx.output_path("report", ".json")
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        matrix_path = ctx.output_path("confusion_matrix", ".npz")
        _save_npz(
            matrix_path,
            confusion_metadata,
            counts=confusion,
            row_normalized=normalized,
            class_names=np.asarray(MODULATION_CLASSES),
        )
        return OperationResult(
            outputs={
                "report": artifact("metrics.report", report_path, report_payload["metadata"]),
                "confusion_matrix": artifact(CONFUSION_MATRIX_KIND, matrix_path, confusion_metadata),
            },
            metrics=metrics,
            metadata=report_payload["metadata"],
        )


def _classical_scores(received: np.ndarray, metadata: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    frame_count = int(received.shape[0])
    raw_variance = metadata.get("noise_variance_per_frame")
    if isinstance(raw_variance, list) and len(raw_variance) == frame_count:
        variance = np.maximum(np.asarray(raw_variance, dtype=np.float64), 1e-8)
    else:
        snr = float(metadata.get("snr_db_mean", 8.0))
        variance = np.full((frame_count,), max(10.0 ** (-snr / 10.0), 1e-8), dtype=np.float64)
    likelihood = np.empty((frame_count, len(MODULATION_CLASSES)), dtype=np.float32)
    evm = np.empty_like(likelihood)
    for class_id, class_name in enumerate(MODULATION_CLASSES):
        points = _constellations()[class_name]
        distances = np.abs(received[:, :, None] - points.reshape(1, 1, -1)) ** 2
        nearest = np.min(distances, axis=2)
        evm[:, class_id] = np.sqrt(np.mean(nearest, axis=1)).astype(np.float32)
        scaled = -distances.astype(np.float64) / variance.reshape(-1, 1, 1)
        maximum = np.max(scaled, axis=2, keepdims=True)
        log_mixture = maximum[..., 0] + np.log(np.sum(np.exp(scaled - maximum), axis=2)) - math.log(len(points))
        likelihood[:, class_id] = np.mean(log_mixture, axis=1).astype(np.float32)
    return likelihood, evm


def _remove_known_carrier_impairment(
    received: np.ndarray,
    metadata: Mapping[str, Any],
) -> np.ndarray:
    """Diagnostic oracle synchronization using simulator-only nuisance truth."""

    frame_count, symbol_count = received.shape
    raw_phase = metadata.get("carrier_phase_offset_rad_per_frame")
    raw_frequency = metadata.get(
        "frequency_offset_cycles_per_symbol_per_frame"
    )
    if not (
        isinstance(raw_phase, list)
        and len(raw_phase) == frame_count
        and isinstance(raw_frequency, list)
        and len(raw_frequency) == frame_count
    ):
        return np.asarray(received, dtype=np.complex64)
    phase = np.asarray(raw_phase, dtype=np.float64).reshape(-1, 1)
    frequency = np.asarray(raw_frequency, dtype=np.float64).reshape(-1, 1)
    symbol_index = np.arange(symbol_count, dtype=np.float64).reshape(1, -1)
    correction = np.exp(
        -1j * (phase + 2.0 * np.pi * frequency * symbol_index)
    )
    return np.asarray(received * correction, dtype=np.complex64)


def _blind_cumulant_scores(
    received: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """I/Q-only phase/CFO-invariant modulation features and prototype scores."""

    power = np.abs(received) ** 2
    energy = np.maximum(np.mean(power, axis=1, keepdims=True), 1e-8)
    normalized = received / np.sqrt(energy)
    normalized_power = np.abs(normalized) ** 2
    differential = normalized[:, 1:] * np.conj(normalized[:, :-1])
    differential /= np.maximum(np.abs(differential), 1e-8)
    features = np.stack(
        [
            np.abs(np.mean(np.square(differential), axis=1)),
            np.abs(np.mean(np.power(differential, 4), axis=1)),
            np.mean(np.square(normalized_power), axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    prototypes = np.asarray(
        [_blind_cumulant_prototype(_constellations()[name]) for name in MODULATION_CLASSES],
        dtype=np.float32,
    )
    scales = np.asarray([0.22, 0.24, 0.18], dtype=np.float32)
    scores = -np.sum(
        np.square(
            (features[:, None, :] - prototypes[None, :, :])
            / scales.reshape(1, 1, -1)
        ),
        axis=2,
    )
    return scores.astype(np.float32), features


def _blind_cumulant_prototype(points: np.ndarray) -> np.ndarray:
    unit = points / np.maximum(np.abs(points), 1e-8)
    average_power = float(np.mean(np.abs(points) ** 2))
    return np.asarray(
        [
            float(np.abs(np.mean(np.square(unit))) ** 2),
            float(np.abs(np.mean(np.power(unit, 4))) ** 2),
            float(np.mean(np.abs(points) ** 4) / max(average_power**2, 1e-8)),
        ],
        dtype=np.float32,
    )
