from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from noema_lab.core.executor import LocalExecutor
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationRegistry,
    OperationResult,
    object_schema,
)
from noema_lab.core.planner import (
    EXECUTION_PLAN_KIND,
    EXECUTION_PLAN_SCHEMA_VERSION,
    RecipePlanningError,
    plan_recipe,
    validate_recipe_against_registry,
)
from noema_lab.core.recipes import compile_recipe, load_recipe, recipe_from_dict
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.storage import LocalStore
from noema_lab.core.verification import verify_run_bundle
from noema_lab.ops import build_registry
from noema_lab.ops.channel.digital import DigitalDemodulateOperation
from noema_lab.ops.models.external import DeepJsccExternalEncodeOperation


ROOT = Path(__file__).resolve().parents[1]


class _MultiBackendOperation(Operation):
    id = "test.multi_backend"
    name = "Synthetic multi-backend operation"
    backends = {
        "benchmark_run": ["numpy", "sionna"],
        "dataset_capture": ["numpy"],
        "differentiable_export": [],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "numpy_reference",
            "status": "implemented",
        },
        {
            "runner": "benchmark_run",
            "backend": "sionna",
            "implementation": "sionna_reference",
            "status": "implemented",
            "notes": "Synthetic alternate backend.",
        },
        {
            "runner": "dataset_capture",
            "backend": "numpy",
            "implementation": "numpy_capture",
            "status": "implemented",
        },
    ]
    params_schema = object_schema(
        {
            "wireless_backend": {
                "type": "string",
                "default": "auto",
                "enum": ["auto", "numpy", "sionna"],
            },
            "count": {"type": "integer", "default": 1, "minimum": 1},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult(
            metadata={
                "executed": True,
                "wireless_backend": ctx.params.get("wireless_backend"),
            }
        )


class _NumpyOnlyOperation(_MultiBackendOperation):
    id = "test.numpy_only"
    name = "Synthetic NumPy-only operation"
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": [],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "numpy_only",
            "status": "implemented",
        },
        {
            "runner": "dataset_capture",
            "backend": "numpy",
            "implementation": "numpy_capture",
            "status": "implemented",
        },
    ]


class _SymbolSourceOperation(Operation):
    id = "test.symbol_source"
    name = "Synthetic symbol source"
    output_kinds = {"symbols": "channel.symbols.complex_numpy"}
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": ["torch"],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "numpy_source",
            "status": "implemented",
        },
        {
            "runner": "dataset_capture",
            "backend": "numpy",
            "implementation": "numpy_source",
            "status": "implemented",
        },
        {
            "runner": "differentiable_export",
            "backend": "torch",
            "implementation": "torch_source",
            "status": "implemented",
        },
    ]

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


class _NoMaterializationOperation(Operation):
    id = "test.no_materialization"
    name = "Synthetic operation without a runner binding"
    materializations = []

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


class _RuntimeBoundOperation(Operation):
    id = "test.runtime_bound"
    name = "Synthetic parameter-bound runtime operation"
    backends = {
        "benchmark_run": ["external", "onnxruntime"],
        "dataset_capture": ["external", "onnxruntime"],
        "differentiable_export": [],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "external",
            "implementation": "external_callable_runtime",
            "parameter_bindings": {"runtime": "external_callable"},
        },
        {
            "runner": "benchmark_run",
            "backend": "onnxruntime",
            "implementation": "portable_artifact_runtime",
            "parameter_bindings": {"runtime": "learned_artifact"},
        },
    ]
    params_schema = object_schema(
        {
            "runtime": {
                "type": "string",
                "default": "external_callable",
                "enum": ["external_callable", "learned_artifact"],
            }
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        runtime = str(ctx.params.get("runtime"))
        return OperationResult(
            metadata={
                "runtime": runtime,
                "runtime_backend": (
                    "onnxruntime" if runtime == "learned_artifact" else "external"
                ),
            }
        )


class _InvalidParameterBindingOperation(Operation):
    id = "test.invalid_parameter_binding"
    name = "Synthetic invalid parameter binding operation"
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": [],
        "differentiable_export": [],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "invalid_count_runtime",
            "parameter_bindings": {"count": "not-an-integer"},
        }
    ]
    params_schema = object_schema(
        {"count": {"type": "integer", "default": 1, "minimum": 1}}
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


class _DuplicateMaterializationIdentityOperation(Operation):
    id = "test.duplicate_materialization_identity"
    name = "Synthetic duplicate materialization identity operation"
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": [],
        "differentiable_export": [],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "shared_runtime",
            "parameter_bindings": {"mode": "first"},
        },
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "shared_runtime",
            "parameter_bindings": {"mode": "second"},
        },
    ]
    params_schema = object_schema(
        {
            "mode": {
                "type": "string",
                "default": "first",
                "enum": ["first", "second"],
            }
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


class _LimitedGetRegistry(OperationRegistry):
    def __init__(self, maximum_gets: int) -> None:
        super().__init__()
        self.maximum_gets = maximum_gets
        self.get_count = 0

    def get(self, operation_id: str) -> Operation:
        self.get_count += 1
        if self.get_count > self.maximum_gets:
            raise AssertionError("executor re-resolved an operation after planning")
        return super().get(operation_id)


def _registry(operation: Operation) -> OperationRegistry:
    registry = OperationRegistry()
    registry.register(operation)
    return registry


def _recipe(operation_id: str, params=None):
    step = {"id": "work", "op": operation_id}
    if params is not None:
        step["params"] = dict(params)
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "execution_plan_test",
            "steps": [step],
        }
    )


class ExecutionPlannerTests(unittest.TestCase):
    def test_duplicate_materialization_identity_is_rejected(self):
        operation = _DuplicateMaterializationIdentityOperation()

        with self.assertRaisesRegex(
            OperationError,
            r"duplicates materialization identity benchmark_run/numpy/shared_runtime",
        ):
            _registry(operation)

    def test_plan_is_versioned_immutable_and_digest_stable(self):
        operation = _MultiBackendOperation()
        registry = _registry(operation)
        recipe = _recipe(operation.id, {"wireless_backend": "auto"})

        first = plan_recipe(recipe, registry)
        second = plan_recipe(recipe, registry)

        self.assertEqual(first.schema_version, EXECUTION_PLAN_SCHEMA_VERSION)
        self.assertEqual(first.kind, EXECUTION_PLAN_KIND)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertIsInstance(first.steps, tuple)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.runner = "dataset_capture"
        with self.assertRaises(TypeError):
            first.steps[0].implementation_metadata["status"] = "changed"

        serialized = first.to_dict()
        supplied_sha = serialized.pop("sha256")
        self.assertEqual(supplied_sha, canonical_json_sha256(serialized))
        contract = operation.describe()
        self.assertEqual(
            first.steps[0].operation_contract_sha256,
            canonical_json_sha256(contract),
        )
        self.assertEqual(
            first.operation_contracts_sha256,
            canonical_json_sha256({operation.id: contract}),
        )

    def test_explicit_step_backend_and_planner_override_are_respected(self):
        operation = _MultiBackendOperation()
        registry = _registry(operation)
        recipe = _recipe(operation.id, {"wireless_backend": "sionna"})

        selected = plan_recipe(recipe, registry).steps[0]
        self.assertEqual(selected.backend, "sionna")
        self.assertEqual(selected.implementation, "sionna_reference")
        self.assertEqual(
            selected.implementation_metadata["selection"]["source"],
            "step.params.wireless_backend",
        )
        self.assertEqual(
            selected.materialization_id,
            "test.multi_backend@benchmark_run/sionna/sionna_reference",
        )

        overridden = plan_recipe(recipe, registry, backend="numpy").steps[0]
        self.assertEqual(overridden.backend, "numpy")
        self.assertEqual(
            overridden.implementation_metadata["selection"]["source"],
            "planner.backend",
        )

    def test_auto_uses_declared_contract_order_deterministically(self):
        operation = _MultiBackendOperation()
        registry = _registry(operation)
        recipe = _recipe(operation.id, {"wireless_backend": "auto"})

        selected = plan_recipe(recipe, registry).steps[0]

        self.assertEqual(selected.backend, "numpy")
        self.assertEqual(selected.implementation, "numpy_reference")
        selection = selected.implementation_metadata["selection"]
        self.assertEqual(selection["source"], "step.params.wireless_backend")
        self.assertEqual(selection["requested_backend"], "auto")
        self.assertEqual(selection["candidate_index"], 0)
        self.assertEqual(
            selected.params_for_execution(recipe.steps[0].params)[
                "wireless_backend"
            ],
            "numpy",
        )

    def test_real_wireless_backend_parameter_selects_sionna_contract(self):
        registry = build_registry()
        authored = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml")
        effective = compile_recipe(
            authored,
            mode="compat",
            registry=registry,
        ).require_recipe(effective=True)
        for step in effective.steps:
            if step.op in {
                "model.deepjscc_external_encode",
                "model.deepjscc_external_decode",
            }:
                step.params["runtime"] = "external_callable"

        # This test exercises materialization selection, not optional-package
        # discovery.  Production preflight must continue to reject an explicit
        # Sionna request when the wireless extra is absent.
        with mock.patch(
            "noema_lab.ops.channel.digital._sionna_available",
            return_value=True,
        ):
            plan = plan_recipe(effective, registry)
        wireless = next(step for step in plan.steps if step.step_id == "wireless_channel")

        self.assertEqual(wireless.operation_id, "wireless.channel")
        self.assertEqual(wireless.backend, "sionna")
        self.assertEqual(
            wireless.implementation, "sionna_awgn_matched_artifact"
        )

    def test_export_only_training_interfaces_report_the_runner_mismatch(self):
        registry = build_registry()
        for recipe_name in (
            "deepjscc_kodak_awgn_train.yaml",
            "csi_feedback_sionna_train.yaml",
        ):
            with self.subTest(recipe=recipe_name):
                recipe = load_recipe(ROOT / "recipes" / recipe_name)
                with self.assertRaisesRegex(
                    RecipePlanningError,
                    (
                        r"params select .*training_interface.*implemented only for "
                        r"runner=differentiable_export; runner=benchmark_run cannot execute"
                    ),
                ):
                    plan_recipe(recipe, registry)

    def test_wireless_materialization_rejects_unsupported_channel_backend_pair(self):
        registry = build_registry()
        registry.register(_SymbolSourceOperation())
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "unsupported_sionna_ofdm",
                "steps": [
                    {"id": "source", "op": "test.symbol_source"},
                    {
                        "id": "work",
                        "op": "wireless.channel",
                        "inputs": {"symbols": "source.symbols"},
                        "params": {
                            "channel": "ofdm_cdl",
                            "wireless_backend": "sionna",
                            "receiver_processing": "matched",
                            "channel_state_mode": "none",
                        },
                    },
                ],
            }
        )

        with self.assertRaisesRegex(
            RecipePlanningError, "do not select an implemented materialization"
        ):
            plan_recipe(recipe, registry)

    def test_wireless_torch_export_is_parameter_bound(self):
        registry = build_registry()
        registry.register(_SymbolSourceOperation())
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "bound_torch_rayleigh",
                "steps": [
                    {"id": "source", "op": "test.symbol_source"},
                    {
                        "id": "work",
                        "op": "wireless.channel",
                        "inputs": {"symbols": "source.symbols"},
                        "params": {
                            "channel": "flat_rayleigh",
                            "wireless_backend": "auto",
                            "receiver_processing": "none",
                            "channel_state_mode": "none",
                        },
                    },
                ],
            }
        )

        planned = next(
            step
            for step in plan_recipe(
                recipe,
                registry,
                runner="differentiable_export",
            ).steps
            if step.step_id == "work"
        )

        self.assertEqual(planned.backend, "torch")
        self.assertEqual(
            planned.implementation, "torch_flat_rayleigh_none_module"
        )
        self.assertEqual(
            planned.implementation_metadata["selection"]["parameter_bindings"],
            {
                "channel": "flat_rayleigh",
                "receiver_processing": "none",
                "channel_state_mode": "none",
                "wireless_backend": "auto",
            },
        )

    def test_parameter_binding_selects_and_executes_the_same_runtime(self):
        operation = _RuntimeBoundOperation()
        registry = _registry(operation)
        recipe = _recipe(operation.id, {"runtime": "learned_artifact"})

        planned = plan_recipe(recipe, registry).steps[0]

        self.assertEqual(planned.backend, "onnxruntime")
        self.assertEqual(planned.implementation, "portable_artifact_runtime")
        self.assertEqual(
            planned.implementation_metadata["parameter_overrides"],
            {"runtime": "learned_artifact"},
        )
        self.assertEqual(
            planned.implementation_metadata["selection"]["source"],
            "step.params.runtime",
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = LocalExecutor(registry, LocalStore(Path(tmp))).run(recipe)
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
        executed = summary["steps"][0]
        self.assertEqual(executed["execution_binding"]["backend"], "onnxruntime")
        self.assertEqual(executed["metadata"]["runtime"], "learned_artifact")
        self.assertEqual(executed["metadata"]["runtime_backend"], "onnxruntime")

    def test_deepjscc_learned_artifact_binds_onnxruntime_runtime(self):
        registry = OperationRegistry()

        class ImageSourceOperation(Operation):
            id = "test.image_source"
            name = "Synthetic image source"
            output_kinds = {"images": "image.batch.numpy"}

            def run(self, ctx: OperationContext) -> OperationResult:
                return OperationResult()

        registry.register(ImageSourceOperation())
        registry.register(DeepJsccExternalEncodeOperation())
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "deepjscc_runtime_binding",
                "steps": [
                    {"id": "data", "op": "test.image_source"},
                    {
                        "id": "sender",
                        "op": "model.deepjscc_external_encode",
                        "inputs": {"images": "data.images"},
                        "params": {
                            "runtime": "learned_artifact",
                            "artifact_manifest_path": "trained_artifact.yaml",
                        },
                    },
                ],
            }
        )

        sender = plan_recipe(recipe, registry).steps[1]

        self.assertEqual(sender.backend, "onnxruntime")
        self.assertEqual(sender.implementation, "portable_trained_artifact_runtime")
        self.assertEqual(
            sender.params_for_execution(recipe.steps[1].params)["runtime"],
            "learned_artifact",
        )
        self.assertEqual(
            sender.implementation_metadata["selection"]["parameter_bindings"],
            {"runtime": "learned_artifact"},
        )

    def test_explicit_backend_rebinds_runtime_before_execution(self):
        operation = _RuntimeBoundOperation()
        recipe = _recipe(operation.id, {"runtime": "learned_artifact"})

        planned = plan_recipe(
            recipe,
            _registry(operation),
            backend="external",
        ).steps[0]

        self.assertEqual(planned.backend, "external")
        self.assertEqual(planned.implementation, "external_callable_runtime")
        self.assertEqual(
            planned.params_for_execution(recipe.steps[0].params)["runtime"],
            "external_callable",
        )

    def test_parameter_overrides_are_schema_validated_after_binding(self):
        operation = _InvalidParameterBindingOperation()

        with self.assertRaisesRegex(
            RecipePlanningError,
            "params are invalid after binding.*must be integer",
        ):
            plan_recipe(
                _recipe(operation.id, {"count": 1}),
                _registry(operation),
                backend="numpy",
            )

    def test_schema_declared_data_plane_selector_translates_runtime_values(self):
        registry = build_registry()
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "data_plane_binding",
                "steps": [
                    {
                        "id": "source",
                        "op": "source.random_bits",
                        "params": {
                            "bit_count": 8,
                            "data_plane_backend": "auto",
                        },
                    },
                    {
                        "id": "boundary",
                        "op": "channel.bit_boundary",
                        "inputs": {"bits": "source.bits"},
                        "params": {"data_plane_backend": "cpp_native"},
                    },
                ],
            }
        )

        plan = plan_recipe(recipe, registry)
        source, boundary = plan.steps

        self.assertEqual(source.backend, "numpy")
        self.assertEqual(
            source.params_for_execution(recipe.steps[0].params)["data_plane_backend"],
            "python_numpy",
        )
        self.assertEqual(
            source.implementation_metadata["subordinate_runtime_selectors"][
                "data_plane_backend"
            ],
            {
                "scope": "subordinate",
                "requested_value": "auto",
                "automatic": True,
                "target_backend": "numpy",
                "effective_value": "python_numpy",
            },
        )
        self.assertEqual(boundary.backend, "cpp")
        self.assertEqual(
            boundary.params_for_execution(recipe.steps[1].params)["data_plane_backend"],
            "cpp_native",
        )
        self.assertEqual(
            boundary.implementation_metadata["selection"]["source"],
            "step.params.data_plane_backend",
        )

    def test_subordinate_auto_data_plane_selector_is_recorded_separately(self):
        registry = build_registry()
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "data_plane_auto_binding",
                "steps": [
                    {
                        "id": "source",
                        "op": "source.random_bits",
                        "params": {
                            "bit_count": 8,
                            "data_plane_backend": "auto",
                        },
                    }
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = LocalExecutor(registry, LocalStore(Path(tmp))).run(recipe)
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
        executed = summary["steps"][0]
        self.assertEqual(executed["execution_binding"]["backend"], "numpy")
        binding_metadata = executed["execution_binding"]["implementation_metadata"]
        self.assertEqual(
            binding_metadata["parameter_overrides"]["data_plane_backend"],
            "python_numpy",
        )
        self.assertEqual(
            binding_metadata["subordinate_runtime_selectors"]["data_plane_backend"],
            {
                "scope": "subordinate",
                "requested_value": "auto",
                "automatic": True,
                "target_backend": "numpy",
                "effective_value": "python_numpy",
            },
        )
        self.assertEqual(
            executed["metadata"]["data_plane_backend"],
            "python_numpy",
        )

    def test_planned_auto_modem_and_wireless_execution_is_fully_bound(self):
        registry = build_registry()
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "planned_auto_modem_wireless",
                "steps": [
                    {
                        "id": "source",
                        "op": "source.random_bits",
                        "params": {
                            "bit_count": 32,
                            "data_plane_backend": "auto",
                        },
                    },
                    {
                        "id": "modulator",
                        "op": "modulation.digital_modulate",
                        "inputs": {"bits": "source.bits"},
                        "params": {
                            "modulation": "qpsk",
                            "data_plane_backend": "auto",
                        },
                    },
                    {
                        "id": "channel",
                        "op": "wireless.channel",
                        "inputs": {"symbols": "modulator.symbols"},
                        "params": {
                            "channel": "awgn",
                            "snr_db": 20.0,
                            "wireless_backend": "auto",
                            "data_plane_backend": "auto",
                            "seed": 17,
                        },
                    },
                    {
                        "id": "demodulator",
                        "op": "demodulation.digital_demodulate",
                        "inputs": {"rx_symbols": "channel.rx_symbols"},
                        "params": {
                            "modulation": "qpsk",
                            "data_plane_backend": "auto",
                        },
                    },
                ],
            }
        )

        plan = plan_recipe(recipe, registry)
        planned = {step.step_id: step for step in plan.steps}
        self.assertEqual(planned["modulator"].backend, "numpy")
        self.assertEqual(
            planned["modulator"].implementation,
            "numpy_qpsk_modulator",
        )
        self.assertEqual(planned["channel"].backend, "numpy")
        self.assertEqual(
            planned["channel"].implementation,
            "numpy_awgn_matched",
        )
        self.assertEqual(planned["demodulator"].backend, "numpy")
        self.assertEqual(
            planned["demodulator"].implementation,
            "numpy_qpsk_demodulator",
        )
        for step_id in ("source", "modulator", "channel", "demodulator"):
            authored = next(step for step in recipe.steps if step.id == step_id)
            self.assertEqual(
                planned[step_id].params_for_execution(authored.params)[
                    "data_plane_backend"
                ],
                "python_numpy",
            )
        channel_params = planned["channel"].params_for_execution(
            next(step for step in recipe.steps if step.id == "channel").params
        )
        self.assertEqual(channel_params["wireless_backend"], "numpy")
        self.assertEqual(
            planned["channel"].implementation_metadata["selection"][
                "requested_backend"
            ],
            "auto",
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = LocalExecutor(registry, LocalStore(Path(tmp))).run(recipe)
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
        executed = {step["id"]: step for step in summary["steps"]}
        for step_id in ("source", "modulator", "channel", "demodulator"):
            self.assertEqual(
                executed[step_id]["metadata"]["data_plane_backend"],
                "python_numpy",
            )
            self.assertEqual(
                executed[step_id]["execution_binding"][
                    "implementation_metadata"
                ]["parameter_overrides"]["data_plane_backend"],
                "python_numpy",
            )
        self.assertEqual(
            executed["channel"]["metadata"]["wireless_backend"],
            "numpy",
        )
        self.assertNotIn(
            "backend_fallback",
            executed["channel"]["metadata"],
        )

    def test_every_planned_wireless_auto_operation_binds_numpy(self):
        registry = build_registry()
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "all_wireless_auto_operations",
                "steps": [
                    {
                        "id": "bits",
                        "op": "source.random_bits",
                        "params": {"bit_count": 16},
                    },
                    {
                        "id": "digital_link",
                        "op": "wireless.digital_link",
                        "inputs": {"bits": "bits.bits"},
                        "params": {
                            "channel": "awgn",
                            "modulation": "qpsk",
                            "wireless_backend": "auto",
                            "data_plane_backend": "auto",
                            "seed": 3,
                        },
                    },
                    {
                        "id": "channel_truth",
                        "op": "source.ai_phy_channel_realization",
                        "params": {
                            "scenario": "flat_siso",
                            "example_count": 4,
                            "subcarriers": 4,
                            "seed": 5,
                        },
                    },
                    {
                        "id": "pilots",
                        "op": "source.ai_phy_pilot_pattern",
                        "params": {
                            "scenario": "unit",
                            "tx_antennas": 1,
                            "subcarriers": 4,
                            "seed": 7,
                        },
                    },
                    {
                        "id": "pilot_observation",
                        "op": "wireless.pilot_observation",
                        "inputs": {
                            "channel": "channel_truth.channel",
                            "pilots": "pilots.pilots",
                        },
                        "params": {
                            "wireless_backend": "auto",
                            "snr_db": 15.0,
                            "seed": 11,
                        },
                    },
                    {
                        "id": "legacy_pilot_source",
                        "op": "source.ai_phy_pilot_channel",
                        "params": {
                            "wireless_backend": "auto",
                            "example_count": 4,
                            "subcarriers": 4,
                            "seed": 13,
                        },
                    },
                ],
            }
        )

        plan = plan_recipe(recipe, registry)
        planned = {step.step_id: step for step in plan.steps}
        for step_id in (
            "digital_link",
            "pilot_observation",
            "legacy_pilot_source",
        ):
            authored = next(step for step in recipe.steps if step.id == step_id)
            self.assertEqual(planned[step_id].backend, "numpy")
            self.assertEqual(
                planned[step_id].params_for_execution(authored.params)[
                    "wireless_backend"
                ],
                "numpy",
            )
            self.assertEqual(
                planned[step_id].implementation_metadata["selection"][
                    "requested_backend"
                ],
                "auto",
            )

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp))
            run_dir = LocalExecutor(registry, store).run(recipe)
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            verification = verify_run_bundle(store, run_dir.name)
        executed = {step["id"]: step for step in summary["steps"]}
        for step_id in (
            "digital_link",
            "pilot_observation",
            "legacy_pilot_source",
        ):
            self.assertEqual(
                executed[step_id]["execution_binding"]["backend"],
                "numpy",
            )
            self.assertEqual(
                executed[step_id]["metadata"]["wireless_backend"],
                "numpy",
            )
            self.assertEqual(
                executed[step_id]["execution_binding"][
                    "implementation_metadata"
                ]["parameter_overrides"]["wireless_backend"],
                "numpy",
            )
        self.assertEqual(
            executed["digital_link"]["metadata"]["data_plane_backend"],
            "python_numpy",
        )
        self.assertEqual(verification["status"], "valid", verification)

    def test_subordinate_cpp_data_plane_selector_remains_valid_and_evidenced(self):
        registry = build_registry()
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "unsupported_data_plane_binding",
                "steps": [
                    {
                        "id": "source",
                        "op": "source.random_bits",
                        "params": {
                            "bit_count": 8,
                            "data_plane_backend": "cpp_native",
                        },
                    }
                ],
            }
        )

        source = plan_recipe(recipe, registry).steps[0]

        self.assertEqual(source.backend, "numpy")
        self.assertEqual(
            source.params_for_execution(recipe.steps[0].params)["data_plane_backend"],
            "cpp_native",
        )
        self.assertEqual(
            source.implementation_metadata["subordinate_runtime_selectors"][
                "data_plane_backend"
            ],
            {
                "scope": "subordinate",
                "requested_value": "cpp_native",
                "automatic": False,
                "target_backend": "cpp",
            },
        )

    def test_cpp_modulator_materialization_rejects_unsupported_qam16(self):
        registry = build_registry()
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "unsupported_cpp_qam16_modulator",
                "steps": [
                    {
                        "id": "source",
                        "op": "source.random_bits",
                        "params": {"bit_count": 16},
                    },
                    {
                        "id": "modulator",
                        "op": "modulation.digital_modulate",
                        "inputs": {"bits": "source.bits"},
                        "params": {
                            "modulation": "qam16",
                            "data_plane_backend": "cpp_native",
                        },
                    },
                ],
            }
        )

        with self.assertRaisesRegex(
            RecipePlanningError,
            "params do not select an implemented materialization",
        ):
            plan_recipe(recipe, registry)

        recipe.steps[1].params["modulation"] = "qpsk"
        modulator = plan_recipe(recipe, registry).steps[1]
        self.assertEqual(modulator.backend, "cpp")
        self.assertEqual(modulator.implementation, "cpp_qpsk_modulator")

    def test_cpp_demodulator_materialization_rejects_unsupported_qam16(self):
        class RxSymbolSourceOperation(Operation):
            id = "test.rx_symbol_source"
            name = "Synthetic received-symbol source"
            output_kinds = {"symbols": "channel.rx_symbols.complex_numpy"}

            def run(self, ctx: OperationContext) -> OperationResult:
                return OperationResult()

        registry = OperationRegistry()
        registry.register(RxSymbolSourceOperation())
        registry.register(DigitalDemodulateOperation())
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "unsupported_cpp_qam16_demodulator",
                "steps": [
                    {"id": "source", "op": "test.rx_symbol_source"},
                    {
                        "id": "demodulator",
                        "op": "demodulation.digital_demodulate",
                        "inputs": {"rx_symbols": "source.symbols"},
                        "params": {
                            "modulation": "qam16",
                            "data_plane_backend": "cpp_native",
                        },
                    },
                ],
            }
        )

        with self.assertRaisesRegex(
            RecipePlanningError,
            "params do not select an implemented materialization",
        ):
            plan_recipe(recipe, registry)

        recipe.steps[1].params["modulation"] = "bpsk"
        demodulator = plan_recipe(recipe, registry).steps[1]
        self.assertEqual(demodulator.backend, "cpp")
        self.assertEqual(demodulator.implementation, "cpp_bpsk_demodulator")

    def test_local_executor_rejects_non_operation_run_export_runner(self):
        operation = _MultiBackendOperation()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with self.assertRaisesRegex(
                RecipePlanningError,
                "cannot execute runner differentiable_export through Operation.run",
            ):
                LocalExecutor(_registry(operation), LocalStore(workspace)).run(
                    _recipe(operation.id),
                    runner="differentiable_export",
                )
            self.assertFalse((workspace / "runs").exists())

    def test_binding_failure_precedes_all_run_directory_side_effects(self):
        operation = _NumpyOnlyOperation()
        registry = _registry(operation)
        recipe = _recipe(operation.id, {"wireless_backend": "sionna"})

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with self.assertRaisesRegex(
                RecipePlanningError,
                "no implemented materialization.*backend=sionna",
            ):
                LocalExecutor(registry, LocalStore(workspace)).run(recipe)
            self.assertFalse((workspace / "runs").exists())
            self.assertEqual(list(workspace.iterdir()), [])

    def test_unresolved_matrix_is_rejected_before_run_side_effects(self):
        operation = _MultiBackendOperation()
        registry = _registry(operation)
        payload = _recipe(operation.id).to_dict()
        payload["metadata"] = {
            "matrix": {
                "dimensions": {"backend": ["numpy", "sionna"]},
                "step_params": {
                    "work": {"wireless_backend": {"matrix": "backend"}}
                },
            }
        }
        recipe = recipe_from_dict(payload)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with self.assertRaisesRegex(
                ValueError,
                "single-run execution requires a concrete recipe",
            ):
                LocalExecutor(registry, LocalStore(workspace)).run(recipe)
            self.assertFalse((workspace / "runs").exists())
            self.assertEqual(list(workspace.iterdir()), [])

    def test_validation_only_api_does_not_silently_choose_a_runner(self):
        operation = _NoMaterializationOperation()
        registry = _registry(operation)
        recipe = _recipe(operation.id)

        self.assertIsNone(validate_recipe_against_registry(recipe, registry))
        with self.assertRaisesRegex(
            RecipePlanningError,
            "no implemented materialization",
        ):
            plan_recipe(recipe, registry)

    def test_executor_consumes_bound_operation_and_persists_plan_evidence(self):
        operation = _MultiBackendOperation()
        registry = _LimitedGetRegistry(maximum_gets=2)
        registry.register(operation)
        authored = _recipe(operation.id)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = LocalExecutor(registry, LocalStore(Path(tmp))).run(authored)
            execution_plan = json.loads(
                (run_dir / "execution-plan.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            effective = json.loads(
                (run_dir / "recipe.json").read_text(encoding="utf-8")
            )

        self.assertEqual(registry.get_count, 2)
        self.assertEqual(execution_plan["sha256"], manifest["execution_plan"]["sha256"])
        self.assertEqual(execution_plan["sha256"], summary["execution_plan"]["sha256"])
        self.assertEqual(execution_plan["recipe"]["sha256"], manifest["recipe"]["effective_sha256"])
        self.assertEqual(
            canonical_json_sha256(effective),
            manifest["recipe"]["effective_sha256"],
        )
        self.assertNotEqual(
            manifest["recipe"]["authored_sha256"],
            manifest["recipe"]["effective_sha256"],
        )
        self.assertEqual(
            summary["steps"][0]["execution_binding"],
            manifest["steps"][0]["execution_binding"],
        )
        self.assertEqual(
            summary["steps"][0]["execution_binding"]["binding_sha256"],
            execution_plan["steps"][0]["binding_sha256"],
        )
        self.assertEqual(summary["steps"][0]["metadata"]["wireless_backend"], "numpy")
        self.assertEqual(
            execution_plan["steps"][0]["implementation_metadata"]["parameter_overrides"],
            {"wireless_backend": "numpy"},
        )
        self.assertEqual(
            manifest["operation_contracts"]["sha256"],
            execution_plan["operation_contracts"]["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
