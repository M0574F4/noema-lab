from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from noema_lab.core.artifacts import artifact, file_sha256
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationResult,
    object_schema,
)


JsonDict = Dict[str, Any]


def _artifact_params(*, default_mode: str, modes: list[str], entrypoint: str) -> JsonDict:
    return object_schema(
        {
            "mode": {
                "type": "string",
                "default": default_mode,
                "enum": modes,
                "title": "Comparison method",
            },
            "artifact_manifest_path": {
                "type": "string",
                "default": "",
                "title": "Trained artifact",
                "x-noema-ui": {
                    "control": "trained_artifact",
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
                "default": entrypoint,
                "x-noema-ui": {"hidden": True},
            },
            "artifact_package_sha256": {
                "type": "string",
                "default": "",
                "x-noema-ui": {"hidden": True},
            },
        }
    )


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], JsonDict]:
    if not path.is_file():
        raise OperationError("Required NumPy artifact does not exist: %s" % path)
    with np.load(str(path), allow_pickle=False) as payload:
        arrays = {
            name: np.asarray(payload[name])
            for name in payload.files
            if name != "metadata_json"
        }
        metadata: JsonDict = {}
        if "metadata_json" in payload.files:
            metadata = dict(json.loads(str(payload["metadata_json"].item())))
    return arrays, metadata


def _run_artifact(
    ctx: OperationContext,
    *,
    default_entrypoint: str,
    inputs: Mapping[str, np.ndarray],
    label: str,
) -> tuple[Mapping[str, np.ndarray], str]:
    manifest_value = str(ctx.params.get("artifact_manifest_path") or "").strip()
    if not manifest_value:
        raise OperationError("%s requires params.artifact_manifest_path" % label)
    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.is_file():
        raise OperationError("%s manifest does not exist: %s" % (label, manifest_path))
    try:
        from noema_lab.core.trained_artifact_runtime import run_trained_artifact_entrypoint

        outputs = run_trained_artifact_entrypoint(
            manifest_path,
            str(ctx.params.get("artifact_entrypoint") or default_entrypoint),
            dict(inputs),
            expected_package_sha256=str(
                ctx.params.get("artifact_package_sha256") or ""
            ),
        )
    except Exception as exc:
        raise OperationError("%s inference failed: %s" % (label, exc)) from exc
    return outputs, file_sha256(manifest_path)


def _project_simplex(values: np.ndarray, total: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ordered = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - float(total)
    indices = np.arange(1, values.shape[1] + 1, dtype=np.float64)[None, :]
    active = ordered - cumulative / indices > 0.0
    rho = np.maximum(np.sum(active, axis=1), 1)
    theta = cumulative[np.arange(values.shape[0]), rho - 1] / rho
    return np.maximum(values - theta[:, None], 0.0).astype(np.float32)


def _water_fill(gains: np.ndarray, noise: float, total: float) -> np.ndarray:
    inverse = float(noise) / np.maximum(np.asarray(gains, dtype=np.float64), 1e-9)
    lo = np.min(inverse, axis=1)
    hi = np.max(inverse, axis=1) + float(total)
    for _ in range(60):
        level = 0.5 * (lo + hi)
        allocated = np.maximum(level[:, None] - inverse, 0.0)
        too_much = np.sum(allocated, axis=1) > float(total)
        hi = np.where(too_much, level, hi)
        lo = np.where(too_much, lo, level)
    return _project_simplex(np.maximum(lo[:, None] - inverse, 0.0), total)


def _isac_oracle(features: np.ndarray, total_power: float) -> np.ndarray:
    comm = np.maximum(features[..., 0], 1e-8).astype(np.float64)
    sensing = np.maximum(features[..., 1], 1e-8).astype(np.float64)
    noise = np.maximum(features[:, :1, 2], 1e-8).astype(np.float64)
    alpha = np.clip(features[:, :1, 3], 0.0, 1.0).astype(np.float64)
    power = np.full(comm.shape, float(total_power) / comm.shape[1], dtype=np.float64)
    for iteration in range(160):
        sensing_sum = np.sum(power * sensing, axis=1, keepdims=True)
        gradient = (
            (1.0 - alpha)
            * comm
            / (math.log(2.0) * comm.shape[1] * (noise + power * comm))
            + alpha
            * sensing
            / (math.log(2.0) * (noise + sensing_sum))
        )
        power = _project_simplex(power + (0.25 / math.sqrt(iteration + 1.0)) * gradient, total_power)
    return np.asarray(power, dtype=np.float32)


class IsacOfdmScenarioOperation(Operation):
    id = "source.isac_ofdm_scenario"
    name = "Synthetic ISAC OFDM allocation scenario"
    output_kinds = {"problem": "isac.ofdm_allocation_problem.numpy"}
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": ["torch"],
    }
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "synthetic_frequency_selective_isac", "status": "implemented"},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "synthetic_frequency_selective_isac", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "captured_isac_allocation_contract", "status": "implemented"},
    ]
    params_schema = object_schema(
        {
            "example_count": {"type": "integer", "default": 64, "minimum": 1},
            "subcarriers": {"type": "integer", "default": 12, "minimum": 4},
            "snr_db": {"type": "number", "default": 10.0},
            "sensing_weight": {"type": "number", "default": 0.4, "minimum": 0.0, "maximum": 1.0},
            "total_power": {"type": "number", "default": 1.0, "exclusiveMinimum": 0.0},
            "seed": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        count = int(ctx.params.get("example_count") or 64)
        carriers = int(ctx.params.get("subcarriers") or 12)
        snr_db = float(ctx.params.get("snr_db") or 0.0)
        alpha = float(ctx.params.get("sensing_weight") or 0.0)
        total_power = float(ctx.params.get("total_power") or 1.0)
        rng = np.random.RandomState(ctx.seed("isac_ofdm_scenario"))
        common = rng.lognormal(mean=0.0, sigma=0.35, size=(count, 1))
        comm = rng.gamma(shape=1.5, scale=0.8, size=(count, carriers)) * common
        delay = rng.uniform(0.08, 0.35, size=(count, 1))
        phase = rng.uniform(-math.pi, math.pi, size=(count, 1))
        grid = np.arange(carriers, dtype=np.float64)[None, :]
        sensing = 0.15 + np.abs(
            np.cos(2.0 * math.pi * delay * grid + phase)
        ) ** 2
        sensing *= rng.lognormal(mean=0.0, sigma=0.2, size=(count, 1))
        noise = 10.0 ** (-snr_db / 10.0)
        features = np.stack(
            [
                comm,
                sensing,
                np.full_like(comm, noise),
                np.full_like(comm, alpha),
            ],
            axis=-1,
        ).astype(np.float32)
        metadata = {
            "array": "features",
            "scenario": "synthetic_frequency_selective_isac_ofdm_v1",
            "snr_db": snr_db,
            "sensing_weight": alpha,
            "total_power": total_power,
            "subcarriers": carriers,
            "example_count": count,
            "seed": int(ctx.seed("isac_ofdm_scenario")),
        }
        path = ctx.output_path("problem", ".npz")
        np.savez_compressed(path, features=features, metadata_json=json.dumps(metadata, sort_keys=True))
        return OperationResult(
            outputs={"problem": artifact(self.output_kinds["problem"], path, metadata)},
            metrics={"channel.snr_db": snr_db, "isac.sensing_weight": alpha},
            metadata=metadata,
        )


class IsacOfdmAllocatorOperation(Operation):
    id = "model.isac_ofdm_allocator_adapter"
    name = "ISAC OFDM allocation adapter"
    input_kinds = {"problem": ["isac.ofdm_allocation_problem.numpy"]}
    output_kinds = {"decision": "isac.ofdm_power_allocation.numpy"}
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": True,
        "exportable": True,
        "reason": "The allocation boundary supports a portable power-simplex policy artifact.",
    }
    backends = {
        "benchmark_run": ["numpy", "onnxruntime"],
        "dataset_capture": ["numpy", "onnxruntime"],
        "differentiable_export": ["torch"],
    }
    trained_artifact_abi = {
        "component_id": "allocator",
        "component_role": "joint_isac_ofdm_power_allocator",
        "entrypoint_id": "isac_allocator",
        "required_operation_inputs": ["problem"],
        "inputs": {"features": {"dtype": "float32", "shape": ["batch", "subcarrier", 4]}},
        "outputs": {"power": {"dtype": "float32", "shape": ["batch", "subcarrier"]}},
        "binding_params": {
            "mode": "learned_artifact",
            "artifact_manifest_path": "trained_artifact.yaml",
            "artifact_entrypoint": "isac_allocator",
        },
    }
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "equal_power", "status": "implemented", "parameter_bindings": {"mode": "equal_power"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "communication_water_filling", "status": "implemented", "parameter_bindings": {"mode": "communications_water_filling"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "scalarized_projected_gradient_oracle", "status": "implemented", "parameter_bindings": {"mode": "scalarized_reference"}},
        {"runner": "benchmark_run", "backend": "onnxruntime", "implementation": "portable_trained_artifact_runtime", "status": "implemented", "parameter_bindings": {"mode": "learned_artifact"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "reference_allocators", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "label_free_scalarized_utility", "status": "implemented"},
    ]
    params_schema = _artifact_params(
        default_mode="equal_power",
        modes=["equal_power", "communications_water_filling", "scalarized_reference", "learned_artifact"],
        entrypoint="isac_allocator",
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        arrays, metadata = _load_npz(ctx.require_input("problem").path)
        features = np.asarray(arrays.get("features"), dtype=np.float32)
        if features.ndim != 3 or features.shape[2] != 4:
            raise OperationError("ISAC features must have shape [batch, subcarrier, 4]")
        total_power = float(metadata.get("total_power") or 1.0)
        mode = str(ctx.params.get("mode") or "equal_power")
        checkpoint_sha = ""
        if mode == "equal_power":
            power = np.full(features.shape[:2], total_power / features.shape[1], dtype=np.float32)
        elif mode == "communications_water_filling":
            power = _water_fill(features[..., 0], float(features[0, 0, 2]), total_power)
        elif mode == "scalarized_reference":
            power = _isac_oracle(features, total_power)
        elif mode == "learned_artifact":
            outputs, checkpoint_sha = _run_artifact(
                ctx,
                default_entrypoint="isac_allocator",
                inputs={"features": features},
                label="ISAC allocator artifact",
            )
            if "power" not in outputs:
                raise OperationError("ISAC allocator artifact did not return `power`")
            power = np.asarray(outputs["power"], dtype=np.float32)
        else:
            raise OperationError("Unsupported ISAC allocator mode `%s`" % mode)
        if power.shape != features.shape[:2] or not np.all(np.isfinite(power)):
            raise OperationError("ISAC power output has an invalid shape or non-finite values")
        if np.any(power < -1e-6):
            raise OperationError("ISAC power output must be non-negative")
        sums = np.sum(np.maximum(power, 0.0), axis=1, keepdims=True)
        if np.any(sums <= 1e-9):
            raise OperationError("ISAC power output must allocate positive total power")
        power = np.maximum(power, 0.0) * (total_power / sums)
        out_metadata = {**metadata, "array": "power", "allocator_mode": mode}
        if checkpoint_sha:
            out_metadata["artifact_manifest_sha256"] = checkpoint_sha
        path = ctx.output_path("decision", ".npz")
        np.savez_compressed(path, power=power.astype(np.float32), metadata_json=json.dumps(out_metadata, sort_keys=True))
        return OperationResult(
            outputs={"decision": artifact(self.output_kinds["decision"], path, out_metadata)},
            metadata=out_metadata,
        )


class IsacOfdmMetricsOperation(Operation):
    id = "metrics.isac_ofdm"
    name = "ISAC communication/sensing allocation metrics"
    input_kinds = {
        "problem": ["isac.ofdm_allocation_problem.numpy"],
        "decision": ["isac.ofdm_power_allocation.numpy"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        problem, metadata = _load_npz(ctx.require_input("problem").path)
        decision, decision_metadata = _load_npz(ctx.require_input("decision").path)
        features = np.asarray(problem["features"], dtype=np.float64)
        power = np.asarray(decision["power"], dtype=np.float64)
        comm = features[..., 0]
        sensing = features[..., 1]
        noise = np.maximum(features[:, 0, 2], 1e-12)
        alpha = np.clip(features[:, 0, 3], 0.0, 1.0)
        rate = np.sum(np.log2(1.0 + power * comm / noise[:, None]), axis=1)
        sensing_linear = np.sum(power * sensing, axis=1) / noise
        sensing_info = np.log2(1.0 + sensing_linear)
        utility = (1.0 - alpha) * rate / comm.shape[1] + alpha * sensing_info
        metrics = {
            "isac.communication_rate_bps_hz": float(np.mean(rate)),
            "isac.sensing_snr_db": float(10.0 * np.log10(max(float(np.mean(sensing_linear)), 1e-12))),
            "isac.scalarized_utility": float(np.mean(utility)),
            "isac.sensing_weight": float(np.mean(alpha)),
            "channel.snr_db": float(metadata.get("snr_db") or 0.0),
            "task.score": float(np.mean(utility / (1.0 + utility))),
        }
        report = {
            "schema_version": 1,
            "task": "joint_isac_resource_allocation",
            "method": str(decision_metadata.get("allocator_mode") or ""),
            "metrics": metrics,
        }
        path = ctx.output_path("report", ".json")
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return OperationResult(outputs={"report": artifact("metrics.report", path)}, metrics=metrics, metadata=report)


def _near_field_steering(
    ranges_m: np.ndarray,
    angles_deg: np.ndarray,
    *,
    antennas: int,
    carrier_frequency_ghz: float,
) -> np.ndarray:
    wavelength = 299_792_458.0 / (float(carrier_frequency_ghz) * 1e9)
    x = (np.arange(antennas, dtype=np.float64) - (antennas - 1) / 2.0) * wavelength / 2.0
    angle = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))[:, None]
    radius = np.asarray(ranges_m, dtype=np.float64)[:, None]
    target_x = radius * np.sin(angle)
    target_y = radius * np.cos(angle)
    distance = np.sqrt((target_x - x[None, :]) ** 2 + target_y**2)
    relative = distance - radius
    response = (radius / np.maximum(distance, 1e-9)) * np.exp(
        -1j * 2.0 * math.pi * relative / wavelength
    )
    return (response / np.sqrt(antennas)).astype(np.complex64)


class NearFieldXlMimoScenarioOperation(Operation):
    id = "source.near_field_xl_mimo_scenario"
    name = "Synthetic near-field XL-MIMO pilot scenario"
    output_kinds = {
        "problem": "near_field.array_observation.numpy",
        "truth": "near_field.range_angle_truth.numpy",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch"]}
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "spherical_wave_coherent_pilot", "status": "implemented"},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "spherical_wave_coherent_pilot", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "captured_range_angle_supervision", "status": "implemented"},
    ]
    params_schema = object_schema(
        {
            "example_count": {"type": "integer", "default": 64, "minimum": 1},
            "antennas": {"type": "integer", "default": 32, "minimum": 8},
            "carrier_frequency_ghz": {"type": "number", "default": 28.0, "exclusiveMinimum": 0.0},
            "range_min_m": {"type": "number", "default": 0.5, "exclusiveMinimum": 0.0},
            "range_max_m": {"type": "number", "default": 5.0, "exclusiveMinimum": 0.0},
            "angle_min_deg": {"type": "number", "default": -55.0},
            "angle_max_deg": {"type": "number", "default": 55.0},
            "snr_db": {"type": "number", "default": 15.0},
            "seed": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        count = int(ctx.params.get("example_count") or 64)
        antennas = int(ctx.params.get("antennas") or 32)
        carrier = float(ctx.params.get("carrier_frequency_ghz") or 28.0)
        range_min = float(ctx.params.get("range_min_m") or 0.5)
        range_max = float(ctx.params.get("range_max_m") or 5.0)
        angle_min = float(ctx.params.get("angle_min_deg") or -55.0)
        angle_max = float(ctx.params.get("angle_max_deg") or 55.0)
        if range_max <= range_min or angle_max <= angle_min:
            raise OperationError("Near-field scenario bounds must be strictly ordered")
        snr_db = float(ctx.params.get("snr_db") or 0.0)
        rng = np.random.RandomState(ctx.seed("near_field_xl_mimo"))
        ranges = rng.uniform(range_min, range_max, size=count).astype(np.float32)
        angles = rng.uniform(angle_min, angle_max, size=count).astype(np.float32)
        clean = _near_field_steering(ranges, angles, antennas=antennas, carrier_frequency_ghz=carrier)
        noise_std = math.sqrt(10.0 ** (-snr_db / 10.0) / (2.0 * antennas))
        noise = noise_std * (
            rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)
        )
        observation = clean + noise.astype(np.complex64)
        reference = observation[:, :1]
        observation *= np.conj(reference) / np.maximum(np.abs(reference), 1e-9)
        truth = np.stack([ranges, angles], axis=1).astype(np.float32)
        metadata = {
            "scenario": "synthetic_spherical_wave_coherent_pilot_v1",
            "snr_db": snr_db,
            "antennas": antennas,
            "carrier_frequency_ghz": carrier,
            "range_min_m": range_min,
            "range_max_m": range_max,
            "angle_min_deg": angle_min,
            "angle_max_deg": angle_max,
            "example_count": count,
            "seed": int(ctx.seed("near_field_xl_mimo")),
        }
        problem_path = ctx.output_path("problem", ".npz")
        np.savez_compressed(
            problem_path,
            array_observation=observation.astype(np.complex64),
            metadata_json=json.dumps({**metadata, "array": "array_observation"}, sort_keys=True),
        )
        truth_path = ctx.output_path("truth", ".npz")
        np.savez_compressed(
            truth_path,
            range_angle=truth,
            metadata_json=json.dumps({**metadata, "array": "range_angle"}, sort_keys=True),
        )
        return OperationResult(
            outputs={
                "problem": artifact(self.output_kinds["problem"], problem_path, {**metadata, "array": "array_observation"}),
                "truth": artifact(self.output_kinds["truth"], truth_path, {**metadata, "array": "range_angle"}),
            },
            metrics={"channel.snr_db": snr_db},
            metadata=metadata,
        )


def _polar_search(observation: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    count, antennas = observation.shape
    ranges = np.linspace(float(metadata["range_min_m"]), float(metadata["range_max_m"]), 18)
    angles = np.linspace(float(metadata["angle_min_deg"]), float(metadata["angle_max_deg"]), 45)
    candidates = np.array([(r, a) for r in ranges for a in angles], dtype=np.float32)
    steering = _near_field_steering(
        candidates[:, 0],
        candidates[:, 1],
        antennas=antennas,
        carrier_frequency_ghz=float(metadata["carrier_frequency_ghz"]),
    )
    scores = np.abs(observation @ np.conj(steering.T)) ** 2
    return candidates[np.argmax(scores, axis=1)].reshape(count, 2)


def _far_field_search(observation: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    count, antennas = observation.shape
    angles = np.linspace(float(metadata["angle_min_deg"]), float(metadata["angle_max_deg"]), 181)
    far_ranges = np.full(angles.shape, 1e6, dtype=np.float32)
    steering = _near_field_steering(
        far_ranges,
        angles,
        antennas=antennas,
        carrier_frequency_ghz=float(metadata["carrier_frequency_ghz"]),
    )
    scores = np.abs(observation @ np.conj(steering.T)) ** 2
    selected_angles = angles[np.argmax(scores, axis=1)]
    midpoint = 0.5 * (float(metadata["range_min_m"]) + float(metadata["range_max_m"]))
    return np.stack([np.full(count, midpoint), selected_angles], axis=1).astype(np.float32)


class NearFieldEstimatorAdapterOperation(Operation):
    id = "model.near_field_estimator_adapter"
    name = "Near-field range-angle estimator adapter"
    input_kinds = {"problem": ["near_field.array_observation.numpy"]}
    optional_input_kinds = {"truth": ["near_field.range_angle_truth.numpy"]}
    output_kinds = {"estimate": "near_field.range_angle_estimate.numpy"}
    differentiability = {"framework": "torch", "gradient": "surrogate", "trainable_params": True, "exportable": True, "reason": "The coherent array observation is exposed through a portable range-angle estimator ABI."}
    backends = {"benchmark_run": ["numpy", "onnxruntime"], "dataset_capture": ["numpy", "onnxruntime"], "differentiable_export": ["torch"]}
    trained_artifact_abi = {
        "component_id": "estimator",
        "component_role": "near_field_range_angle_estimator",
        "entrypoint_id": "near_field_estimator",
        "required_operation_inputs": ["problem"],
        "inputs": {"array_ri": {"dtype": "float32", "shape": ["batch", "antenna", 2]}},
        "outputs": {"range_angle": {"dtype": "float32", "shape": ["batch", 2]}},
        "binding_params": {"mode": "learned_artifact", "artifact_manifest_path": "trained_artifact.yaml", "artifact_entrypoint": "near_field_estimator"},
    }
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "far_field_angle_grid", "status": "implemented", "parameter_bindings": {"mode": "far_field_steering"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "polar_range_angle_codebook", "status": "implemented", "parameter_bindings": {"mode": "polar_codebook"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "simulation_truth", "status": "implemented", "parameter_bindings": {"mode": "oracle_focus"}},
        {"runner": "benchmark_run", "backend": "onnxruntime", "implementation": "portable_trained_artifact_runtime", "status": "implemented", "parameter_bindings": {"mode": "learned_artifact"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "reference_estimators", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "supervised_range_angle_estimation", "status": "implemented"},
    ]
    params_schema = _artifact_params(
        default_mode="polar_codebook",
        modes=["far_field_steering", "polar_codebook", "oracle_focus", "learned_artifact"],
        entrypoint="near_field_estimator",
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        arrays, metadata = _load_npz(ctx.require_input("problem").path)
        observation = np.asarray(arrays["array_observation"], dtype=np.complex64)
        mode = str(ctx.params.get("mode") or "polar_codebook")
        checkpoint_sha = ""
        if mode == "far_field_steering":
            estimate = _far_field_search(observation, metadata)
        elif mode == "polar_codebook":
            estimate = _polar_search(observation, metadata)
        elif mode == "oracle_focus":
            truth_arrays, _ = _load_npz(ctx.require_input("truth").path)
            estimate = np.asarray(truth_arrays["range_angle"], dtype=np.float32)
        elif mode == "learned_artifact":
            array_ri = np.stack([observation.real, observation.imag], axis=-1).astype(np.float32)
            outputs, checkpoint_sha = _run_artifact(
                ctx,
                default_entrypoint="near_field_estimator",
                inputs={"array_ri": array_ri},
                label="Near-field estimator artifact",
            )
            if "range_angle" not in outputs:
                raise OperationError("Near-field estimator artifact did not return `range_angle`")
            estimate = np.asarray(outputs["range_angle"], dtype=np.float32)
        else:
            raise OperationError("Unsupported near-field estimator mode `%s`" % mode)
        if estimate.shape != (observation.shape[0], 2) or not np.all(np.isfinite(estimate)):
            raise OperationError("Near-field estimate must have shape [batch, 2] and be finite")
        out_metadata = {**metadata, "array": "range_angle", "estimator_mode": mode}
        if checkpoint_sha:
            out_metadata["artifact_manifest_sha256"] = checkpoint_sha
        path = ctx.output_path("estimate", ".npz")
        np.savez_compressed(path, range_angle=estimate.astype(np.float32), metadata_json=json.dumps(out_metadata, sort_keys=True))
        return OperationResult(outputs={"estimate": artifact(self.output_kinds["estimate"], path, out_metadata)}, metadata=out_metadata)


class NearFieldMetricsOperation(Operation):
    id = "metrics.near_field_focusing"
    name = "Near-field range-angle and focusing metrics"
    input_kinds = {
        "problem": ["near_field.array_observation.numpy"],
        "truth": ["near_field.range_angle_truth.numpy"],
        "estimate": ["near_field.range_angle_estimate.numpy"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        problem, metadata = _load_npz(ctx.require_input("problem").path)
        truth_arrays, _ = _load_npz(ctx.require_input("truth").path)
        estimate_arrays, estimate_metadata = _load_npz(ctx.require_input("estimate").path)
        truth = np.asarray(truth_arrays["range_angle"], dtype=np.float64)
        estimate = np.asarray(estimate_arrays["range_angle"], dtype=np.float64)
        range_error = estimate[:, 0] - truth[:, 0]
        angle_error = estimate[:, 1] - truth[:, 1]
        true_steering = _near_field_steering(
            truth[:, 0], truth[:, 1], antennas=int(metadata["antennas"]), carrier_frequency_ghz=float(metadata["carrier_frequency_ghz"])
        )
        if str(estimate_metadata.get("estimator_mode")) == "far_field_steering":
            estimate_ranges = np.full(estimate.shape[0], 1e6, dtype=np.float32)
        else:
            estimate_ranges = estimate[:, 0]
        estimated_steering = _near_field_steering(
            estimate_ranges, estimate[:, 1], antennas=int(metadata["antennas"]), carrier_frequency_ghz=float(metadata["carrier_frequency_ghz"])
        )
        gain = np.abs(np.sum(np.conj(true_steering) * estimated_steering, axis=1)) ** 2
        gain /= np.maximum(
            np.sum(np.abs(true_steering) ** 2, axis=1)
            * np.sum(np.abs(estimated_steering) ** 2, axis=1),
            1e-12,
        )
        range_rmse = float(np.sqrt(np.mean(range_error**2)))
        angle_rmse = float(np.sqrt(np.mean(angle_error**2)))
        focusing = float(np.mean(gain))
        metrics = {
            "near_field.range_rmse_m": range_rmse,
            "near_field.angle_rmse_deg": angle_rmse,
            "near_field.normalized_focusing_gain": focusing,
            "channel.snr_db": float(metadata.get("snr_db") or 0.0),
            "task.score": float(focusing / (1.0 + range_rmse / 5.0 + angle_rmse / 10.0)),
        }
        report = {"schema_version": 1, "task": "near_field_range_angle_focusing", "method": str(estimate_metadata.get("estimator_mode") or ""), "metrics": metrics}
        path = ctx.output_path("report", ".json")
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return OperationResult(outputs={"report": artifact("metrics.report", path)}, metrics=metrics, metadata=report)


def _beam_centers(count: int) -> np.ndarray:
    return np.linspace(-60.0, 60.0, int(count), dtype=np.float32)


class LeoNtnTrackingScenarioOperation(Operation):
    id = "source.leo_ntn_tracking_scenario"
    name = "Synthetic LEO-NTN Doppler and beam-handover scenario"
    output_kinds = {
        "problem": "ntn.tracking_history.numpy",
        "truth": "ntn.future_state_truth.numpy",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch"]}
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "bounded_kinematic_leo_pass", "status": "implemented"},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "bounded_kinematic_leo_pass", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "captured_future_state_supervision", "status": "implemented"},
    ]
    params_schema = object_schema(
        {
            "example_count": {"type": "integer", "default": 64, "minimum": 1},
            "history_length": {"type": "integer", "default": 6, "minimum": 3},
            "history_step_s": {"type": "number", "default": 0.5, "exclusiveMinimum": 0.0},
            "prediction_horizon_s": {"type": "number", "default": 1.0, "exclusiveMinimum": 0.0},
            "beam_count": {"type": "integer", "default": 9, "minimum": 3},
            "max_doppler_hz": {"type": "number", "default": 48000.0, "exclusiveMinimum": 0.0},
            "snr_db": {"type": "number", "default": 15.0},
            "seed": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        count = int(ctx.params.get("example_count") or 64)
        history_length = int(ctx.params.get("history_length") or 6)
        step_s = float(ctx.params.get("history_step_s") or 0.5)
        horizon = float(ctx.params.get("prediction_horizon_s") or 1.0)
        beams = int(ctx.params.get("beam_count") or 9)
        max_doppler = float(ctx.params.get("max_doppler_hz") or 48000.0)
        snr_db = float(ctx.params.get("snr_db") or 0.0)
        rng = np.random.RandomState(ctx.seed("leo_ntn_tracking"))
        current_angle = rng.uniform(-48.0, 48.0, size=count)
        angular_rate = rng.uniform(-3.0, 3.0, size=count)
        angular_accel = rng.uniform(-0.12, 0.12, size=count)
        times = -step_s * np.arange(history_length - 1, -1, -1, dtype=np.float64)
        history_angle = current_angle[:, None] + angular_rate[:, None] * times + 0.5 * angular_accel[:, None] * times**2
        history_doppler = max_doppler * np.sin(np.deg2rad(history_angle))
        doppler_noise = max_doppler * 10.0 ** (-snr_db / 20.0)
        angle_noise = 8.0 * 10.0 ** (-snr_db / 20.0)
        measured_doppler = history_doppler + rng.normal(scale=doppler_noise, size=history_doppler.shape)
        measured_angle = history_angle + rng.normal(scale=angle_noise, size=history_angle.shape)
        normalized_time = np.broadcast_to(times[None, :] / max(step_s * (history_length - 1), 1e-6), history_angle.shape)
        features = np.stack(
            [normalized_time, measured_doppler / max_doppler, measured_angle / 60.0],
            axis=-1,
        ).astype(np.float32)
        future_angle = current_angle + angular_rate * horizon + 0.5 * angular_accel * horizon**2
        future_angle = np.clip(future_angle, -60.0, 60.0)
        future_doppler = max_doppler * np.sin(np.deg2rad(future_angle))
        centers = _beam_centers(beams)
        best_beam = np.argmin(np.abs(future_angle[:, None] - centers[None, :]), axis=1)
        truth = np.stack([future_doppler, future_angle, best_beam], axis=1).astype(np.float32)
        metadata = {
            "scenario": "synthetic_bounded_kinematic_leo_pass_v1",
            "array": "track_features",
            "history_length": history_length,
            "history_step_s": step_s,
            "prediction_horizon_s": horizon,
            "beam_count": beams,
            "max_doppler_hz": max_doppler,
            "snr_db": snr_db,
            "example_count": count,
            "seed": int(ctx.seed("leo_ntn_tracking")),
        }
        problem_path = ctx.output_path("problem", ".npz")
        np.savez_compressed(
            problem_path,
            track_features=features,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        truth_path = ctx.output_path("truth", ".npz")
        np.savez_compressed(
            truth_path,
            future_state=truth,
            metadata_json=json.dumps({**metadata, "array": "future_state"}, sort_keys=True),
        )
        return OperationResult(
            outputs={
                "problem": artifact(self.output_kinds["problem"], problem_path, metadata),
                "truth": artifact(self.output_kinds["truth"], truth_path, {**metadata, "array": "future_state"}),
            },
            metrics={"channel.snr_db": snr_db, "ntn.prediction_horizon_s": horizon},
            metadata=metadata,
        )


def _ntn_reference_decision(
    features: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    linear: bool,
) -> np.ndarray:
    max_doppler = float(metadata["max_doppler_hz"])
    beams = int(metadata["beam_count"])
    horizon = float(metadata["prediction_horizon_s"])
    step_s = float(metadata["history_step_s"])
    doppler = features[..., 1] * max_doppler
    angle = features[..., 2] * 60.0
    if linear:
        doppler_prediction = doppler[:, -1] + horizon * (doppler[:, -1] - doppler[:, -2]) / step_s
        angle_prediction = angle[:, -1] + horizon * (angle[:, -1] - angle[:, -2]) / step_s
    else:
        doppler_prediction = doppler[:, -1]
        angle_prediction = angle[:, -1]
    centers = _beam_centers(beams)
    beam = np.argmin(np.abs(angle_prediction[:, None] - centers[None, :]), axis=1)
    return np.stack([doppler_prediction, angle_prediction, beam], axis=1).astype(np.float32)


class LeoNtnTrackingAdapterOperation(Operation):
    id = "model.leo_ntn_tracking_adapter"
    name = "LEO-NTN Doppler and beam-handover adapter"
    input_kinds = {"problem": ["ntn.tracking_history.numpy"]}
    optional_input_kinds = {"truth": ["ntn.future_state_truth.numpy"]}
    output_kinds = {"decision": "ntn.future_state_decision.numpy"}
    differentiability = {"framework": "torch", "gradient": "full", "trainable_params": True, "exportable": True, "reason": "A bounded history tensor feeds a portable joint Doppler/beam predictor."}
    backends = {"benchmark_run": ["numpy", "onnxruntime"], "dataset_capture": ["numpy", "onnxruntime"], "differentiable_export": ["torch"]}
    trained_artifact_abi = {
        "component_id": "tracker",
        "component_role": "leo_ntn_doppler_beam_tracker",
        "entrypoint_id": "ntn_tracker",
        "required_operation_inputs": ["problem"],
        "inputs": {"track_features": {"dtype": "float32", "shape": ["batch", "history", 3]}},
        "outputs": {"decision": {"dtype": "float32", "shape": ["batch", "decision"]}},
        "binding_params": {"mode": "learned_artifact", "artifact_manifest_path": "trained_artifact.yaml", "artifact_entrypoint": "ntn_tracker"},
    }
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "hold_last_observation", "status": "implemented", "parameter_bindings": {"mode": "hold_last"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "linear_extrapolation", "status": "implemented", "parameter_bindings": {"mode": "linear_extrapolation"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "simulation_future_state", "status": "implemented", "parameter_bindings": {"mode": "oracle_future"}},
        {"runner": "benchmark_run", "backend": "onnxruntime", "implementation": "portable_trained_artifact_runtime", "status": "implemented", "parameter_bindings": {"mode": "learned_artifact"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "reference_trackers", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "supervised_future_state_prediction", "status": "implemented"},
    ]
    params_schema = _artifact_params(
        default_mode="linear_extrapolation",
        modes=["hold_last", "linear_extrapolation", "oracle_future", "learned_artifact"],
        entrypoint="ntn_tracker",
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        arrays, metadata = _load_npz(ctx.require_input("problem").path)
        features = np.asarray(arrays["track_features"], dtype=np.float32)
        mode = str(ctx.params.get("mode") or "linear_extrapolation")
        checkpoint_sha = ""
        if mode == "hold_last":
            decision = _ntn_reference_decision(features, metadata, linear=False)
        elif mode == "linear_extrapolation":
            decision = _ntn_reference_decision(features, metadata, linear=True)
        elif mode == "oracle_future":
            truth_arrays, _ = _load_npz(ctx.require_input("truth").path)
            decision = np.asarray(truth_arrays["future_state"], dtype=np.float32).copy()
        elif mode == "learned_artifact":
            outputs, checkpoint_sha = _run_artifact(
                ctx,
                default_entrypoint="ntn_tracker",
                inputs={"track_features": features},
                label="LEO-NTN tracker artifact",
            )
            if "decision" not in outputs:
                raise OperationError("LEO-NTN tracker artifact did not return `decision`")
            raw = np.asarray(outputs["decision"], dtype=np.float32)
            beams = int(metadata["beam_count"])
            if raw.shape != (features.shape[0], beams + 1):
                raise OperationError("LEO-NTN learned decision must have shape [batch, beam_count + 1]")
            selected = np.argmax(raw[:, 1:], axis=1)
            centers = _beam_centers(beams)
            decision = np.stack([raw[:, 0], centers[selected], selected], axis=1).astype(np.float32)
        else:
            raise OperationError("Unsupported LEO-NTN tracker mode `%s`" % mode)
        if decision.shape != (features.shape[0], 3) or not np.all(np.isfinite(decision)):
            raise OperationError("LEO-NTN decision must have shape [batch, 3] and be finite")
        out_metadata = {**metadata, "array": "future_state", "tracker_mode": mode}
        if checkpoint_sha:
            out_metadata["artifact_manifest_sha256"] = checkpoint_sha
        path = ctx.output_path("decision", ".npz")
        np.savez_compressed(path, future_state=decision.astype(np.float32), metadata_json=json.dumps(out_metadata, sort_keys=True))
        return OperationResult(outputs={"decision": artifact(self.output_kinds["decision"], path, out_metadata)}, metadata=out_metadata)


class LeoNtnTrackingMetricsOperation(Operation):
    id = "metrics.leo_ntn_tracking"
    name = "LEO-NTN Doppler and handover metrics"
    input_kinds = {
        "truth": ["ntn.future_state_truth.numpy"],
        "decision": ["ntn.future_state_decision.numpy"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        truth_arrays, metadata = _load_npz(ctx.require_input("truth").path)
        decision_arrays, decision_metadata = _load_npz(ctx.require_input("decision").path)
        truth = np.asarray(truth_arrays["future_state"], dtype=np.float64)
        decision = np.asarray(decision_arrays["future_state"], dtype=np.float64)
        doppler_mae = float(np.mean(np.abs(decision[:, 0] - truth[:, 0])))
        beam_accuracy = float(np.mean(decision[:, 2].astype(np.int64) == truth[:, 2].astype(np.int64)))
        angle_mae = float(np.mean(np.abs(decision[:, 1] - truth[:, 1])))
        normalized_doppler = doppler_mae / max(float(metadata["max_doppler_hz"]), 1.0)
        metrics = {
            "ntn.doppler_mae_hz": doppler_mae,
            "ntn.beam_handover_accuracy": beam_accuracy,
            "ntn.beam_outage_rate": 1.0 - beam_accuracy,
            "ntn.pointing_mae_deg": angle_mae,
            "ntn.prediction_horizon_s": float(metadata["prediction_horizon_s"]),
            "channel.snr_db": float(metadata.get("snr_db") or 0.0),
            "task.score": float(beam_accuracy * math.exp(-normalized_doppler)),
        }
        report = {"schema_version": 1, "task": "leo_ntn_doppler_beam_tracking", "method": str(decision_metadata.get("tracker_mode") or ""), "metrics": metrics}
        path = ctx.output_path("report", ".json")
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return OperationResult(outputs={"report": artifact("metrics.report", path)}, metrics=metrics, metadata=report)
