import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

_REAL_FIND_SPEC = importlib.util.find_spec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.cli.main import main
from noema_lab.core.capture import run_dataset_capture_recipe
from noema_lab.core.matrix import matrix_variant_id
from noema_lab.core.operations import Operation, OperationContext, OperationRegistry, OperationResult
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import load_recipe, recipe_from_dict
from noema_lab.core.reproducibility import (
    canonical_json_sha256,
    derive_seed,
    recipe_fingerprint,
)
from noema_lab.core.storage import LocalStore
from noema_lab.core.training_plans import (
    TrainingPlan,
    apply_training_plan,
    scenario_recipe_fingerprint,
)
from noema_lab.core.trained_artifacts import inspect_trained_artifact
from noema_lab.ops import build_registry
from noema_lab.training.differentiability import (
    TrainingDependencyError,
    require_sionna_available,
    sionna_available,
    torch_available,
)
from noema_lab.training.export_graph import build_export_graph_from_recipe
from noema_lab.training.exporter import (
    DifferentiableExportError,
    _parse_numeric_values,
    available_exporters,
    export_differentiable_scenario,
    exporter_ids,
)
from noema_lab.training import generic_capture_export
from noema_lab.training.scenario import deepjscc_awgn_scenario, neural_receiver_capture_scenario
from demo_trainings.prepare_example import prepare_example
from noema_lab.training.sionna_blocks import (
    AwgnChannelBlock,
    FlatRayleighChannelBlock,
    PowerNormalizationBlock,
    QamPamMapperBlock,
    SionnaAwgnChannelBlock,
    SionnaFlatFadingChannelBlock,
    SionnaMapperBlock,
    SionnaSoftDemapperBlock,
    SoftDemapperBlock,
)


def _find_spec_without_sionna(name, *args, **kwargs):
    if name == "sionna":
        return None
    return _REAL_FIND_SPEC(name)



def _deepjscc_image_recipe():
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "deepjscc_kodak_awgn_train",
            "metadata": {
                "research_stage": "differentiable_export_mvp",
                "training_performed": False,
                "channel_enabled": True,
                "codec_output_form": "symbols",
                "sweeps": {"channel.snr_db": "8:2:12"},
            },
            "steps": [
                {
                    "id": "data",
                    "op": "source.image_dataset",
                    "params": {
                        "dataset": "kodak",
                        "dataset_dir": ".noema/datasets/kodak",
                        "image_ids": "kodim01,kodim02,kodim03,kodim04,kodim05",
                        "crop_size": 8,
                        "repeat_count": 1,
                    },
                },
                {"id": "sender", "op": "model.deepjscc_external_encode", "inputs": {"images": "data.images"}},
                {"id": "tx_power_normalize", "op": "channel.symbol_power_normalize", "inputs": {"symbols": "sender.symbols"}, "params": {"target_power": 1.0}},
                {"id": "tx_symbol_boundary", "op": "channel.symbol_boundary", "inputs": {"symbols": "tx_power_normalize.symbols"}},
                {
                    "id": "wireless_channel",
                    "op": "wireless.channel",
                    "inputs": {"symbols": "tx_symbol_boundary.symbols"},
                    "params": {"channel": "awgn", "snr_db": 10, "wireless_backend": "numpy"},
                },
                {"id": "rx_symbol_boundary", "op": "channel.symbol_boundary", "inputs": {"symbols": "wireless_channel.rx_symbols"}},
                {"id": "receiver", "op": "model.deepjscc_external_decode", "inputs": {"symbols": "rx_symbol_boundary.symbols"}},
                {"id": "evaluation", "op": "metrics.image_reconstruction", "inputs": {"reference": "data.images", "reconstruction": "receiver.images"}},
            ],
        }
    )


class NumericSweepParsingTests(unittest.TestCase):
    def test_numeric_sweep_accepts_ranges_and_explicit_lists(self):
        self.assertEqual(_parse_numeric_values("1:2:8"), [1.0, 3.0, 5.0, 7.0])
        self.assertEqual(_parse_numeric_values("1,2,8"), [1.0, 2.0, 8.0])


class _SyntheticExportOperation(Operation):
    params_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    def __init__(
        self,
        operation_id,
        name,
        input_kinds=None,
        output_kinds=None,
        differentiability=None,
        replaceable=False,
        required_operation_inputs=None,
    ):
        self.id = operation_id
        self.name = name
        self.input_kinds = dict(input_kinds or {})
        self.output_kinds = dict(output_kinds or {})
        self.differentiability = dict(differentiability or {})
        if self.differentiability.get("exportable") and self.differentiability.get("gradient") in {"full", "surrogate"}:
            self.backends = {
                "benchmark_run": ["numpy"],
                "dataset_capture": ["numpy"],
                "differentiable_export": ["torch"],
            }
        if replaceable:
            component_id = operation_id.rsplit(".", 1)[-1]
            self.params_schema = {
                "type": "object",
                "properties": {
                    "artifact_manifest_path": {"type": "string", "default": ""},
                    "artifact_entrypoint": {"type": "string", "default": component_id},
                },
                "required": [],
                "additionalProperties": False,
            }
            self.trained_artifact_abi = {
                "component_id": component_id,
                "component_role": "synthetic_export_component",
                "entrypoint_id": component_id,
                "required_operation_inputs": list(
                    required_operation_inputs
                    if required_operation_inputs is not None
                    else self.input_kinds
                ),
                "inputs": {
                    name: {"dtype": "operation_defined", "shape": ["..."]}
                    for name in self.input_kinds
                },
                "outputs": {
                    name: {"dtype": "operation_defined", "shape": ["..."]}
                    for name in self.output_kinds
                },
                "binding_params": {
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": component_id,
                },
            }

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


def _neural_receiver_registry():
    registry = OperationRegistry()
    registry.register(
        _SyntheticExportOperation(
            "source.synthetic_receiver_pairs",
            "Synthetic receiver capture rows",
            output_kinds={
                "rx_symbols": "channel.rx_symbols.complex_numpy",
                "target_bits": "channel.coded_bits.numpy",
            },
        )
    )
    registry.register(
        _SyntheticExportOperation(
            "model.synthetic_neural_receiver",
            "Synthetic trainable neural receiver",
            input_kinds={
                "features": ["channel.rx_symbols.complex_numpy", "channel.llr.numpy"],
                "target_bits": ["channel.coded_bits.numpy", "channel.payload_bits.numpy"],
            },
            output_kinds={"logits": "channel.llr.numpy"},
            differentiability={
                "framework": "torch",
                "gradient": "full",
                "trainable_params": True,
                "exportable": True,
            },
            replaceable=True,
            required_operation_inputs=["features"],
        )
    )
    return registry


def _neural_receiver_recipe():
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "neural_receiver_capture_train",
            "steps": [
                {"id": "capture", "op": "source.synthetic_receiver_pairs"},
                {
                    "id": "receiver",
                    "op": "model.synthetic_neural_receiver",
                    "inputs": {"features": "capture.rx_symbols", "target_bits": "capture.target_bits"},
                },
            ],
        }
    )


def _branched_replacement_fixture():
    registry = OperationRegistry()
    registry.register(
        _SyntheticExportOperation(
            "test.source",
            "Source",
            output_kinds={"out": "test.tensor"},
        )
    )
    registry.register(
        _SyntheticExportOperation(
            "test.replace_b",
            "Replacement B",
            input_kinds={"x": ["test.tensor"]},
            output_kinds={"out": "test.tensor"},
            replaceable=True,
        )
    )
    for operation_id in ("test.branch_c1", "test.branch_c2"):
        registry.register(
            _SyntheticExportOperation(
                operation_id,
                operation_id,
                input_kinds={"x": ["test.tensor"]},
                output_kinds={"out": "test.tensor"},
                differentiability={
                    "framework": "torch",
                    "gradient": "full",
                    "trainable_params": False,
                    "exportable": True,
                },
            )
        )
    registry.register(
        _SyntheticExportOperation(
            "test.merge_d",
            "Merge D",
            input_kinds={"left": ["test.tensor"], "right": ["test.tensor"]},
            output_kinds={"out": "test.tensor"},
            differentiability={
                "framework": "torch",
                "gradient": "full",
                "trainable_params": False,
                "exportable": True,
            },
        )
    )
    registry.register(
        _SyntheticExportOperation(
            "test.metric",
            "Metric",
            input_kinds={"prediction": ["test.tensor"], "reference": ["test.tensor"]},
            output_kinds={"report": "metrics.report.json"},
        )
    )
    recipe = recipe_from_dict(
        {
            "schema_version": 1,
            "name": "branched_replacement",
            "steps": [
                {"id": "a", "op": "test.source"},
                {"id": "b", "op": "test.replace_b", "inputs": {"x": "a.out"}},
                {"id": "c1", "op": "test.branch_c1", "inputs": {"x": "b.out"}},
                {"id": "c2", "op": "test.branch_c2", "inputs": {"x": "b.out"}},
                {
                    "id": "d",
                    "op": "test.merge_d",
                    "inputs": {"left": "c1.out", "right": "c2.out"},
                },
                {
                    "id": "evaluation",
                    "op": "test.metric",
                    "inputs": {"prediction": "d.out", "reference": "a.out"},
                },
            ],
        }
    )
    return recipe, registry


def _torch_or_skip(testcase):
    if not torch_available():
        testcase.skipTest("PyTorch is not installed")
    import torch
    return torch


class DifferentiableExportTests(unittest.TestCase):
    def test_sionna_missing_error_message_is_clear(self):
        with mock.patch("importlib.util.find_spec", side_effect=_find_spec_without_sionna):
            self.assertFalse(sionna_available())
            with self.assertRaisesRegex(TrainingDependencyError, "uv sync --extra wireless"):
                require_sionna_available()

    def test_power_normalization_awgn_and_rayleigh_keep_gradients(self):
        torch = _torch_or_skip(self)
        symbols = torch.randn(64, dtype=torch.complex64, requires_grad=True)
        normalized = PowerNormalizationBlock()(symbols)
        self.assertAlmostEqual(float(torch.mean(torch.abs(normalized) ** 2).detach().cpu()), 1.0, places=5)
        awgn_out = AwgnChannelBlock(snr_db=30.0, backend="torch", seed=7)(normalized)
        rayleigh_out = FlatRayleighChannelBlock(snr_db=30.0, seed=9)(normalized)
        loss = torch.mean(torch.abs(awgn_out) ** 2) + 0.1 * torch.mean(torch.abs(rayleigh_out) ** 2)
        loss.backward()
        self.assertIsNotNone(symbols.grad)
        self.assertTrue(bool(torch.isfinite(symbols.grad.real).all()))
        self.assertTrue(bool(torch.isfinite(symbols.grad.imag).all()))

    def test_qam_mapper_and_soft_demapper_shapes(self):
        torch = _torch_or_skip(self)
        bits = torch.tensor([0, 0, 0, 1, 1, 0, 1, 1], dtype=torch.int64)
        symbols = QamPamMapperBlock("qpsk")(bits)
        self.assertEqual(tuple(symbols.shape), (4,))
        self.assertTrue(symbols.dtype.is_complex)
        llr = SoftDemapperBlock("qpsk", noise_variance=0.1)(symbols)
        self.assertEqual(tuple(llr.shape), (8,))
        self.assertTrue(bool(torch.isfinite(llr).all()))

    def test_sionna_phy_blocks_declare_honest_differentiability(self):
        self.assertEqual(SionnaMapperBlock.differentiability["framework"], "sionna")
        self.assertEqual(SionnaMapperBlock.differentiability["gradient"], "stop")
        self.assertEqual(SionnaSoftDemapperBlock.differentiability["gradient"], "full")
        self.assertEqual(SionnaAwgnChannelBlock.differentiability["framework"], "sionna")
        self.assertEqual(SionnaFlatFadingChannelBlock.differentiability["framework"], "sionna")
        for block in (
            SionnaMapperBlock,
            SionnaSoftDemapperBlock,
            SionnaAwgnChannelBlock,
            SionnaFlatFadingChannelBlock,
        ):
            self.assertTrue(block.differentiability["exportable"])
        if not sionna_available():
            for constructor in (
                SionnaMapperBlock,
                SionnaSoftDemapperBlock,
                SionnaAwgnChannelBlock,
                SionnaFlatFadingChannelBlock,
            ):
                with self.assertRaises(TrainingDependencyError):
                    constructor()

    @unittest.skipUnless(sionna_available(), "Sionna 2.x wireless extra is not installed")
    def test_sionna2_phy_chain_preserves_pytorch_gradients(self):
        torch = _torch_or_skip(self)
        bits = torch.tensor(
            [0, 0, 0, 1, 1, 0, 1, 1],
            dtype=torch.int64,
        )
        mapped = SionnaMapperBlock("qpsk")(bits)
        symbols = mapped.detach().requires_grad_(True)
        noisy = SionnaAwgnChannelBlock(snr_db=30.0, seed=7)(symbols)
        llr = SionnaSoftDemapperBlock(
            "qpsk",
            noise_variance=1e-3,
        )(noisy)
        llr.square().mean().backward()
        self.assertIsNotNone(symbols.grad)
        self.assertTrue(bool(torch.isfinite(symbols.grad).all()))

        faded_input = torch.ones(
            8,
            dtype=torch.complex64,
            requires_grad=True,
        )
        faded = SionnaFlatFadingChannelBlock(
            snr_db=30.0,
            seed=9,
        )(faded_input)
        faded.abs().mean().backward()
        self.assertIsNotNone(faded_input.grad)
        self.assertTrue(bool(torch.isfinite(faded_input.grad).all()))

    @unittest.skipUnless(sionna_available(), "Sionna 2.x wireless extra is not installed")
    def test_sionna2_mapper_and_demapper_match_noema_modem_contract(self):
        torch = _torch_or_skip(self)
        cases = {
            "bpsk": [0, 1, 0, 1, 1],
            "qpsk": [0, 0, 0, 1, 1],
            "qam16": [0, 0, 0, 0, 0, 1, 1, 0, 1],
        }
        for modulation, values in cases.items():
            with self.subTest(modulation=modulation):
                bits = torch.tensor(values, dtype=torch.int64)
                expected = QamPamMapperBlock(
                    modulation,
                    normalize_power=False,
                )(bits)
                actual = SionnaMapperBlock(
                    modulation,
                    normalize_power=False,
                )(bits)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

                expected_llr = SoftDemapperBlock(
                    modulation,
                    noise_variance=0.2,
                )(expected)
                actual_llr = SionnaSoftDemapperBlock(
                    modulation,
                    noise_variance=0.2,
                )(actual)
                torch.testing.assert_close(
                    actual_llr,
                    expected_llr,
                    rtol=1e-5,
                    atol=1e-5,
                )
                padded = bits
                width = {"bpsk": 1, "qpsk": 2, "qam16": 4}[modulation]
                pad = (-int(bits.numel())) % width
                if pad:
                    padded = torch.cat(
                        [bits, torch.zeros(pad, dtype=bits.dtype)]
                    )
                torch.testing.assert_close(
                    (actual_llr < 0).to(torch.int64),
                    padded,
                    rtol=0,
                    atol=0,
                )

    @unittest.skipUnless(sionna_available(), "Sionna 2.x wireless extra is not installed")
    def test_export_graph_materializes_sionna_pytorch_channel(self):
        recipe = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml")
        graph = build_export_graph_from_recipe(
            recipe, build_registry(), backend="sionna"
        )
        self.assertEqual(graph.metadata["backend"], "sionna")
        self.assertTrue(
            any(isinstance(block, SionnaAwgnChannelBlock) for block in graph.blocks)
        )

    def test_export_graph_builds_minimal_awgn_phy_path(self):
        _torch_or_skip(self)
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "text_bart_awgn_export",
                "metadata": {
                    "research_stage": "phase3_differentiable_export",
                    "task_id": "text_semantic_similarity",
                    "dataset_id": "semantic_text_smoke",
                    "training_performed": False,
                    "channel_enabled": True,
                },
                "steps": [
                    {"id": "data", "op": "source.text_dataset", "params": {"dataset": "semantic_text_smoke"}},
                    {"id": "sender", "op": "model.text_bart_jscc_encode", "inputs": {"texts": "data.texts"}},
                    {"id": "tx_power_normalize", "op": "channel.symbol_power_normalize", "inputs": {"symbols": "sender.symbols"}, "params": {"target_power": 1.0}},
                    {"id": "tx_symbol_boundary", "op": "channel.symbol_boundary", "inputs": {"symbols": "tx_power_normalize.symbols"}},
                    {"id": "wireless_channel", "op": "wireless.channel", "inputs": {"symbols": "tx_symbol_boundary.symbols"}, "params": {"channel": "awgn", "snr_db": 20, "wireless_backend": "numpy"}},
                    {"id": "rx_symbol_boundary", "op": "channel.symbol_boundary", "inputs": {"symbols": "wireless_channel.rx_symbols"}},
                    {"id": "receiver", "op": "model.text_bart_jscc_decode", "inputs": {"symbols": "rx_symbol_boundary.symbols"}},
                    {"id": "evaluation", "op": "metrics.text_semantic_similarity", "inputs": {"reference": "data.texts", "candidate": "receiver.texts"}},
                ],
            }
        )
        graph = build_export_graph_from_recipe(recipe, build_registry(), backend="torch")
        payload = graph.to_dict()
        self.assertEqual(payload["recipe"], "text_bart_awgn_export")
        self.assertEqual(payload["metadata"]["scope"], "minimal_differentiable_phy_mvp")
        block_names = [item["block"] for item in payload["blocks"]]
        self.assertIn("PowerNormalizationBlock", block_names)
        self.assertIn("AwgnChannelBlock", block_names)
        self.assertEqual(payload["feasibility"]["recommended_mode"], "dataset_capture")

    def test_export_graph_derives_channel_seed_from_recipe_master_seed(self):
        _torch_or_skip(self)
        recipe = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml")
        recipe.metadata["seed"] = 73
        wireless_step = next(
            step for step in recipe.steps if step.id == "wireless_channel"
        )
        wireless_step.params.pop("seed", None)
        graph = build_export_graph_from_recipe(
            recipe, build_registry(), backend="torch"
        )
        channel = next(
            item
            for item in graph.to_dict()["blocks"]
            if item["block"] == "AwgnChannelBlock"
        )
        self.assertEqual(
            channel["materialization_contract"]["seed"],
            derive_seed(
                73,
                recipe.name,
                "wireless_channel",
                "wireless_channel",
            ),
        )

    def test_export_graph_excludes_selected_replacement_implementations(self):
        _torch_or_skip(self)
        recipe = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml")
        graph = build_export_graph_from_recipe(
            recipe,
            build_registry(),
            backend="torch",
            replacement_steps=("sender", "receiver"),
        )
        payload = graph.to_dict()
        self.assertEqual(payload["metadata"]["scope"], "replacement_downstream_support")
        self.assertEqual(
            payload["metadata"]["selected_replacement_steps"],
            ["receiver", "sender"],
        )
        self.assertEqual(
            payload["metadata"]["support_step_ids"],
            ["rx_symbol_boundary", "tx_power", "tx_symbol_boundary", "wireless_channel"],
        )

    def test_export_graph_excludes_auto_selected_replacement_by_default(self):
        _torch_or_skip(self)
        recipe = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml")
        graph = build_export_graph_from_recipe(recipe, build_registry(), backend="torch")
        payload = graph.to_dict()
        self.assertEqual(payload["metadata"]["scope"], "replacement_downstream_support")
        self.assertEqual(
            payload["metadata"]["selected_replacement_steps"],
            ["receiver", "sender"],
        )
        self.assertNotIn("sender", payload["metadata"]["support_step_ids"])
        self.assertNotIn("receiver", payload["metadata"]["support_step_ids"])

    def test_legacy_sequential_graph_rejects_branched_support_dag(self):
        _torch_or_skip(self)
        recipe, registry = _branched_replacement_fixture()
        graph = build_export_graph_from_recipe(
            recipe,
            registry,
            backend="torch",
            replacement_steps=("b",),
            loss_steps=("evaluation",),
        )
        self.assertEqual(graph.metadata["support_topology"], "branched")
        self.assertEqual(graph.metadata["support_step_ids"], ["c1", "c2", "d"])
        self.assertFalse(graph.metadata["executable_sequential"])
        self.assertIn("branched DAG", graph.metadata["execution_issue"])
        with self.assertRaisesRegex(TrainingDependencyError, "branched DAG"):
            graph.torch_module()

    def test_legacy_sequential_graph_rejects_side_input_and_wrong_output_port(self):
        _torch_or_skip(self)
        registry = build_registry()
        registry.register(
            _SyntheticExportOperation(
                "test.symbol_source",
                "Symbol source",
                output_kinds={"symbols": "channel.symbols.complex_numpy"},
            )
        )
        registry.register(
            _SyntheticExportOperation(
                "test.channel_state_source",
                "Channel-state source",
                output_kinds={"state": "channel.ofdm_channel_state.numpy"},
            )
        )
        registry.register(
            _SyntheticExportOperation(
                "test.symbol_replacement",
                "Symbol replacement",
                input_kinds={"symbols": ["channel.symbols.complex_numpy"]},
                output_kinds={"symbols": "channel.symbols.complex_numpy"},
                replaceable=True,
            )
        )
        registry.register(
            _SyntheticExportOperation(
                "metrics.synthetic_allocation",
                "Allocation metric",
                input_kinds={"allocation": ["channel.power_allocation.numpy"]},
                output_kinds={"report": "metrics.report.json"},
            )
        )
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "legacy_sequential_port_mismatch",
                "steps": [
                    {"id": "source", "op": "test.symbol_source"},
                    {"id": "state", "op": "test.channel_state_source"},
                    {
                        "id": "replacement",
                        "op": "test.symbol_replacement",
                        "inputs": {"symbols": "source.symbols"},
                    },
                    {
                        "id": "allocator",
                        "op": "model.symbol_power_allocator",
                        "inputs": {
                            "symbols": "replacement.symbols",
                            "channel_state": "state.state",
                        },
                        "params": {
                            "policy": "snr_sigmoid",
                            "granularity": "per_subcarrier",
                        },
                    },
                    {
                        "id": "evaluation",
                        "op": "metrics.synthetic_allocation",
                        "inputs": {"allocation": "allocator.allocation"},
                    },
                ],
            }
        )

        graph = build_export_graph_from_recipe(
            recipe,
            registry,
            backend="torch",
            replacement_steps=("replacement",),
            loss_steps=("evaluation",),
        )

        self.assertEqual(graph.metadata["support_topology"], "linear")
        self.assertEqual(graph.metadata["support_step_ids"], ["allocator"])
        self.assertEqual(graph.metadata["materialized_support_step_ids"], ["allocator"])
        self.assertFalse(graph.metadata["executable_sequential"])
        self.assertIn("active recipe port wiring", graph.metadata["execution_issue"])
        self.assertIn("channel_state, symbols", graph.metadata["execution_issue"])
        self.assertIn("output port(s) allocation", graph.metadata["execution_issue"])
        with self.assertRaisesRegex(TrainingDependencyError, "active recipe port wiring"):
            graph.torch_module()

    @unittest.skipUnless(sionna_available(), "Sionna 2.x wireless extra is not installed")
    def test_export_graph_advertises_sionna2_pytorch_channel_as_differentiable(self):
        _torch_or_skip(self)
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "text_bart_awgn_sionna_export",
                "metadata": {
                    "research_stage": "phase3_differentiable_export",
                    "task_id": "text_semantic_similarity",
                    "dataset_id": "semantic_text_smoke",
                    "training_performed": False,
                    "channel_enabled": True,
                },
                "steps": [
                    {"id": "data", "op": "source.text_dataset", "params": {"dataset": "semantic_text_smoke"}},
                    {"id": "sender", "op": "model.text_bart_jscc_encode", "inputs": {"texts": "data.texts"}},
                    {"id": "tx_power_normalize", "op": "channel.symbol_power_normalize", "inputs": {"symbols": "sender.symbols"}, "params": {"target_power": 1.0}},
                    {"id": "tx_symbol_boundary", "op": "channel.symbol_boundary", "inputs": {"symbols": "tx_power_normalize.symbols"}},
                    {"id": "wireless_channel", "op": "wireless.channel", "inputs": {"symbols": "tx_symbol_boundary.symbols"}, "params": {"channel": "awgn", "snr_db": 20, "wireless_backend": "sionna"}},
                    {"id": "rx_symbol_boundary", "op": "channel.symbol_boundary", "inputs": {"symbols": "wireless_channel.rx_symbols"}},
                    {"id": "receiver", "op": "model.text_bart_jscc_decode", "inputs": {"symbols": "rx_symbol_boundary.symbols"}},
                    {"id": "evaluation", "op": "metrics.text_semantic_similarity", "inputs": {"reference": "data.texts", "candidate": "receiver.texts"}},
                ],
            }
        )
        graph = build_export_graph_from_recipe(
            recipe, build_registry(), backend="sionna"
        )
        self.assertEqual(graph.metadata["backend"], "sionna")
        channel = next(
            block
            for block in graph.blocks
            if isinstance(block, SionnaAwgnChannelBlock)
        )
        self.assertEqual(channel.differentiability["gradient"], "full")
        self.assertTrue(channel.differentiability["exportable"])

    def test_training_scenarios_are_serializable(self):
        self.assertEqual(deepjscc_awgn_scenario(18).to_dict()["mode"], "differentiable_export")
        self.assertEqual(neural_receiver_capture_scenario().to_dict()["mode"], "dataset_capture")

    def test_exporter_registry_exposes_expected_exporters(self):
        ids = exporter_ids()
        self.assertIn("deepjscc-image", ids)
        self.assertIn("neural-receiver", ids)
        self.assertIn("resource-allocation", ids)
        self.assertIn("dataset-capture-only", ids)
        exporters = {item.id: item for item in available_exporters()}
        self.assertTrue(exporters["deepjscc-image"].supports(_deepjscc_image_recipe(), build_registry()))
        self.assertTrue(exporters["neural-receiver"].supports(_neural_receiver_recipe(), _neural_receiver_registry()))
        resource_recipe = load_recipe(ROOT / "recipes" / "resource_water_filling_baseline.yaml")
        self.assertTrue(exporters["resource-allocation"].supports(resource_recipe, build_registry()))

    def test_resource_allocation_export_is_label_free_and_preserves_canonical_topology(self):
        recipe = load_recipe(ROOT / "recipes" / "resource_water_filling_baseline.yaml")
        with tempfile.TemporaryDirectory() as root:
            out_dir = Path(root) / "resource_training"
            payload = export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["tx_power"],
                loss="resource.negative_shannon_spectral_efficiency",
                framework="torch",
                out_dir=out_dir,
                exporter="resource-allocation",
                include_starter=True,
            )
            starter_dir = out_dir / "reference_training"
            self.assertEqual(payload["exporter"], "training-contract")
            self.assertEqual(payload["starter_exporter"], "resource-allocation")
            exported_plan = yaml.safe_load(
                (out_dir / "training_plan.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                exported_plan["objective"],
                "resource.negative_shannon_spectral_efficiency",
            )
            self.assertEqual(exported_plan["starter"], "resource-allocation")
            for relative in [
                "model.py",
                "losses.py",
                "datamodule.py",
                "train.py",
                "evaluate.py",
                "requirements.txt",
                "training_template.yaml",
                "train_config.yaml",
                "capture_train_recipe.yaml",
                "capture_validation_recipe.yaml",
                "capture_test_recipe.yaml",
                "power_allocation_contract.yaml",
                "project_manifest.yaml",
            ]:
                self.assertTrue((starter_dir / relative).is_file(), relative)
            config = yaml.safe_load((starter_dir / "train_config.yaml").read_text(encoding="utf-8"))
            self.assertFalse(config["objective"]["uses_oracle_labels"])
            self.assertIsNone(config["data"]["oracle_allocation_tap"])
            self.assertEqual(config["constraints"]["nonnegative_power"], "exact_euclidean_simplex_projection")
            source = recipe.to_dict()
            self.assertFalse((starter_dir / "trained_artifact.yaml").exists())
            self.assertFalse((starter_dir / "benchmark_recipe.yaml").exists())
            self.assertFalse((starter_dir / "benchmark_pack.yaml").exists())
            source_seed = next(step for step in source["steps"] if step["id"] == "channel_state")["params"]["seed"]
            capture_seeds = {}
            for split in ("train", "validation", "test"):
                capture = yaml.safe_load((starter_dir / f"capture_{split}_recipe.yaml").read_text(encoding="utf-8"))
                self.assertEqual(capture["dataset_capture"]["split"], split)
                self.assertEqual(capture["dataset_capture"]["taps"], [{"id": "channel_gains", "from": "channel_state.state"}])
                capture_seeds[split] = next(
                    step for step in capture["steps"] if step["id"] == "channel_state"
                )["params"]["seed"]
            self.assertTrue(all(seed != source_seed for seed in capture_seeds.values()))
            self.assertEqual(len(set(capture_seeds.values())), 3)
            manifest = yaml.safe_load((out_dir / "project_manifest.yaml").read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "noema.training_interface_bundle@1")
            starter_metadata = manifest["external_training"]["optional_demo_scaffold"]
            self.assertFalse(starter_metadata["normative"])
            self.assertEqual(starter_metadata["exporter"], "resource-allocation")
            self.assertEqual(manifest["training"]["owner"], "external_researcher")
            self.assertEqual(manifest["evaluation"]["owner"], "noema_ordinary_recipe_or_benchmark")
            self.assertEqual(
                [(job["split"], job["requested_samples"]) for job in manifest["capture_jobs"]],
                [("train", 667), ("validation", 167), ("test", 166)],
            )
            self.assertTrue((out_dir / "data").is_dir())
            self.assertEqual(
                [Path(job["output_dir"]) for job in manifest["capture_jobs"]],
                [
                    out_dir / "data" / "train",
                    out_dir / "data" / "validation",
                    out_dir / "data" / "test",
                ],
            )
            self.assertTrue(
                all(job["sample_unit"] == "recipe records" for job in manifest["capture_jobs"])
            )
            self.assertEqual(payload["capture_jobs"], manifest["capture_jobs"])
            self.assertEqual(payload["trained_artifacts"], manifest["trained_artifacts"])
            self.assertEqual(
                manifest["trained_artifacts"][0]["operations"],
                ["model.symbol_power_allocator"],
            )
            self.assertEqual(config["training"]["artifact_manifest_path"], "../trained_artifact.yaml")
            self.assertEqual(
                config["training"]["artifact_component_path"],
                "../artifacts/power_policy.onnx",
            )
            self.assertEqual(config["data"]["train_capture_dirs"], ["../data/train"])
            self.assertEqual(
                config["data"]["validation_capture_dirs"],
                ["../data/validation"],
            )
            self.assertEqual(config["data"]["test_capture_dirs"], ["../data/test"])
            starter_manifest = yaml.safe_load((starter_dir / "project_manifest.yaml").read_text(encoding="utf-8"))
            self.assertEqual(starter_manifest["kind"], "noema.standalone_training_project")
            self.assertEqual(
                starter_manifest["training"]["requires_python"],
                ">=3.11,<3.14",
            )
            self.assertEqual(
                starter_manifest["training"]["dependency_file"],
                str(starter_dir / "requirements.txt"),
            )
            self.assertEqual(
                starter_manifest["training"]["artifact_manifest_path"],
                manifest["trained_artifacts"][0]["manifest_path"],
            )
            requirements = (starter_dir / "requirements.txt").read_text(encoding="utf-8")
            self.assertIn("torch>=2.9.1,<3", requirements)
            self.assertIn("numpy>=1.24,<3", requirements)
            self.assertIn("PyYAML>=6.0,<7", requirements)

            researcher_file = out_dir / "custom_trainer.py"
            researcher_file.write_text("# researcher-owned\n", encoding="utf-8")
            captured_file = out_dir / "data" / "train" / "research_capture.bin"
            captured_file.parent.mkdir(parents=True)
            captured_file.write_bytes(b"preserve captured data")
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["tx_power"],
                loss="resource.negative_shannon_spectral_efficiency",
                framework="torch",
                out_dir=out_dir,
                exporter="resource-allocation",
                force=True,
                include_starter=True,
            )
            self.assertEqual(
                researcher_file.read_text(encoding="utf-8"),
                "# researcher-owned\n",
            )
            self.assertEqual(captured_file.read_bytes(), b"preserve captured data")

    def test_forced_bundle_export_refuses_an_unrecognized_nonempty_directory(self):
        recipe = load_recipe(ROOT / "recipes" / "resource_water_filling_baseline.yaml")
        with tempfile.TemporaryDirectory() as root:
            out_dir = Path(root) / "not_a_bundle"
            out_dir.mkdir()
            researcher_file = out_dir / "do_not_overwrite.txt"
            researcher_file.write_text("owned by researcher\n", encoding="utf-8")

            with self.assertRaisesRegex(
                DifferentiableExportError,
                "not a recognized Noema training bundle",
            ):
                export_differentiable_scenario(
                    recipe,
                    build_registry(),
                    optimizable_steps=["tx_power"],
                    loss="researcher.defined",
                    framework="torch",
                    out_dir=out_dir,
                    force=True,
                )

            self.assertEqual(
                researcher_file.read_text(encoding="utf-8"),
                "owned by researcher\n",
            )
            self.assertFalse((out_dir / "training_contract.yaml").exists())

    def test_forced_neutral_reexport_removes_only_verified_demo_launchers(self):
        recipe = load_recipe(ROOT / "recipes" / "resource_water_filling_baseline.yaml")
        with tempfile.TemporaryDirectory() as root:
            out_dir = Path(root) / "resource_contract"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["tx_power"],
                loss="researcher.defined",
                framework="torch",
                out_dir=out_dir,
                project_root=ROOT,
            )
            prepare_example(out_dir, "resource-allocation", project_root=ROOT)
            for relative in ("RUN_DEMO.md", "train_demo.py", "evaluate_demo.py"):
                self.assertTrue((out_dir / relative).is_file())
            researcher_file = out_dir / "research_notes.md"
            researcher_file.write_text("keep me\n", encoding="utf-8")
            captured_file = out_dir / "data" / "train" / "research_capture.bin"
            captured_file.parent.mkdir(parents=True, exist_ok=True)
            captured_file.write_bytes(b"preserve captured data")

            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["tx_power"],
                loss="researcher.defined",
                framework="torch",
                out_dir=out_dir,
                project_root=ROOT,
                force=True,
            )

            for relative in ("RUN_DEMO.md", "train_demo.py", "evaluate_demo.py"):
                self.assertFalse((out_dir / relative).exists())
            self.assertTrue((out_dir / "reference_training").is_dir())
            self.assertEqual(researcher_file.read_text(encoding="utf-8"), "keep me\n")
            self.assertEqual(captured_file.read_bytes(), b"preserve captured data")
            manifest = yaml.safe_load(
                (out_dir / "project_manifest.yaml").read_text(encoding="utf-8")
            )
            self.assertIsNone(
                manifest["external_training"]["optional_demo_scaffold"]
            )

    def test_resource_data_contract_capture_assets_and_allocator_abi_are_integrity_bound(self):
        recipe = load_recipe(ROOT / "recipes" / "resource_water_filling_baseline.yaml")
        with tempfile.TemporaryDirectory() as root:
            out_dir = Path(root) / "resource_contract"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["tx_power"],
                loss="researcher.defined",
                framework="torch",
                out_dir=out_dir,
            )
            manifest = yaml.safe_load((out_dir / "project_manifest.yaml").read_text(encoding="utf-8"))
            contract = yaml.safe_load((out_dir / "training_contract.yaml").read_text(encoding="utf-8"))
            data_contract = yaml.safe_load((out_dir / "data_contract.yaml").read_text(encoding="utf-8"))
            artifact_template = yaml.safe_load(
                (out_dir / "trained_artifact.template.yaml").read_text(encoding="utf-8")
            )

            data_reference = manifest["data_contract"]
            self.assertEqual(data_reference["path"], "data_contract.yaml")
            self.assertEqual(data_reference["sha256"], canonical_json_sha256(data_contract))
            self.assertEqual(
                data_reference["file_sha256"],
                hashlib.sha256((out_dir / "data_contract.yaml").read_bytes()).hexdigest(),
            )
            self.assertEqual(data_contract["mode"], "captured_generic_tensors")
            self.assertEqual(
                [(item["reference"], item["required"]) for item in data_contract["signals"]],
                [("channel_state.state", True)],
            )
            self.assertEqual(data_contract["ownership"]["capture_execution"], "noema")
            self.assertEqual(data_contract["ownership"]["model_training"], "external_researcher")
            for job in manifest["capture_jobs"]:
                self.assertEqual(job["owner"], "noema")
                self.assertEqual(job["consumer"], "external_researcher")
                capture_path = out_dir / job["bundle_recipe_path"]
                capture = yaml.safe_load(capture_path.read_text(encoding="utf-8"))
                self.assertEqual(job["recipe_sha256"], canonical_json_sha256(capture))
                self.assertEqual(job["recipe_file_sha256"], hashlib.sha256(capture_path.read_bytes()).hexdigest())
                self.assertEqual(capture["dataset_capture"]["taps"], job["expected_taps"])

            slot = contract["trainable_slots"][0]
            self.assertEqual(slot["inputs"]["channel_state"]["dtype"], "float32")
            self.assertEqual(contract["conditioning"][0]["source"], "channel_state.params.noise_variance")
            self.assertEqual(contract["artifact_return"]["bindings"][0]["required_inputs"], ["channel_state"])
            self.assertEqual(
                artifact_template["compatible_operations"][0]["required_inputs"],
                ["channel_state"],
            )

            validation = subprocess.run(
                [sys.executable, "validate_contract.py"],
                cwd=out_dir,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validation.returncode, 0, validation.stderr)

            data_contract["signals"][0]["role"] = "tampered_role"
            (out_dir / "data_contract.yaml").write_text(
                yaml.safe_dump(data_contract, sort_keys=False),
                encoding="utf-8",
            )
            invalid_data = subprocess.run(
                [sys.executable, "validate_contract.py"],
                cwd=out_dir,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(invalid_data.returncode, 0)
            self.assertIn("training data contract", invalid_data.stderr)

            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["tx_power"],
                loss="researcher.defined",
                framework="torch",
                out_dir=out_dir,
                force=True,
            )
            capture_path = out_dir / "capture_train_recipe.yaml"
            capture = yaml.safe_load(capture_path.read_text(encoding="utf-8"))
            capture["dataset_capture"]["taps"].append(
                {"id": "power_allocation", "from": "tx_power.allocation"}
            )
            capture_path.write_text(yaml.safe_dump(capture, sort_keys=False), encoding="utf-8")
            invalid_capture = subprocess.run(
                [sys.executable, "validate_contract.py"],
                cwd=out_dir,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(invalid_capture.returncode, 0)
            self.assertIn("capture recipe", invalid_capture.stderr)

            candidate = out_dir / "candidate.onnx"
            candidate.write_bytes(b"placeholder")
            package = subprocess.run(
                [
                    sys.executable,
                    "package_artifact.py",
                    str(candidate),
                    "--format",
                    "onnx",
                    "--runtime",
                    "onnx",
                ],
                cwd=out_dir,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(package.returncode, 0)
            self.assertFalse((out_dir / "artifact_package_request.yaml").exists())

    @unittest.skipUnless(sionna_available(), "Sionna is not installed")
    def test_resource_train_capture_projects_pruned_allocator_matrix_and_runs(self):
        payload = load_recipe(
            ROOT / "recipes" / "resource_equal_power_baseline.yaml"
        ).to_dict()
        data_step = next(step for step in payload["steps"] if step["id"] == "data")
        data_step["params"].update({"bit_count": 256, "batch_size": 1})
        recipe = apply_training_plan(
            recipe_from_dict(payload),
            TrainingPlan(
                selected_steps=("tx_power",),
                dataset_capture={
                    "taps": [
                        {
                            "id": "channel_gains",
                            "from": "channel_state.state",
                        }
                    ],
                    "split_plan": {
                        "total_samples": 3,
                        "counts": {"train": 1, "validation": 1, "test": 1},
                    },
                },
            ),
        )
        self.assertIn("tx_power", recipe.metadata["matrix"]["step_params"])

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            out_dir = root_path / "resource_contract"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["tx_power"],
                loss="researcher.defined",
                framework="torch",
                out_dir=out_dir,
                project_root=root_path,
            )

            # The authored scenario retains its Run All matrix. The capture
            # graph still contains channel_state, so its physical power-budget
            # sweep must remain while the pruned tx_power binding disappears.
            self.assertIn("tx_power", recipe.metadata["matrix"]["step_params"])
            capture_recipe = load_recipe(out_dir / "capture_train_recipe.yaml")
            self.assertEqual(
                capture_recipe.metadata["matrix"],
                {
                    "dimensions": {
                        "resource.average_transmit_power_budget": [0.5, 1.0, 2.0]
                    },
                    "step_params": {
                        "channel_state": {
                            "average_power_budget": {
                                "matrix": "resource.average_transmit_power_budget"
                            }
                        }
                    },
                },
            )
            self.assertNotIn(
                "tx_power",
                {step.id for step in capture_recipe.steps},
            )

            result = run_dataset_capture_recipe(
                capture_recipe,
                build_registry(),
                LocalStore(root_path / ".noema"),
                root_path / "captured_train",
            )
            self.assertEqual(result["captured_samples"], 1)

    def test_capture_matrix_projection_keeps_only_retained_bindings_and_dimensions(self):
        payload = load_recipe(
            ROOT / "recipes" / "resource_equal_power_baseline.yaml"
        ).to_dict()
        matrix = payload["metadata"]["matrix"]
        matrix["dimensions"]["capture.noise_variance"] = [0.1, 0.2]
        matrix["step_params"]["channel_state"] = {
            "noise_variance": {"matrix": "capture.noise_variance"}
        }
        recipe = recipe_from_dict(payload)
        projected_metadata = dict(recipe.to_dict()["metadata"])

        generic_capture_export._project_capture_matrix(
            recipe,
            projected_metadata,
            retained_step_ids={"channel_state"},
        )

        self.assertEqual(
            projected_metadata["matrix"],
            {
                "dimensions": {"capture.noise_variance": [0.1, 0.2]},
                "step_params": {
                    "channel_state": {
                        "noise_variance": {"matrix": "capture.noise_variance"}
                    }
                },
            },
        )
        self.assertIn(
            "resource.average_transmit_power_budget",
            recipe.metadata["matrix"]["dimensions"],
        )
        self.assertIn("tx_power", recipe.metadata["matrix"]["step_params"])

    @unittest.skipUnless(torch_available(), "PyTorch is not installed")
    def test_resource_allocation_export_trains_without_oracle_labels_and_returns_safe_checkpoint(self):
        recipe = load_recipe(ROOT / "recipes" / "resource_water_filling_baseline.yaml")
        rng = np.random.default_rng(123)
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            out_dir = root_path / "resource_training"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["tx_power"],
                loss="resource.negative_shannon_spectral_efficiency",
                framework="torch",
                out_dir=out_dir,
                exporter="resource-allocation",
                include_starter=True,
            )
            starter_dir = out_dir / "reference_training"
            train_capture = root_path / "capture_train"
            validation_capture = root_path / "capture_validation"
            test_capture = root_path / "capture_test"
            _write_gain_capture(train_capture, "train", rng.lognormal(size=(24, 4)).astype(np.float32))
            _write_gain_capture(validation_capture, "validation", rng.lognormal(size=(12, 4)).astype(np.float32))
            _write_gain_capture(test_capture, "test", rng.lognormal(size=(12, 4)).astype(np.float32))
            config_path = starter_dir / "train_config.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["data"].update(
                {
                    "feature_tap": "channel_gains",
                    "train_capture_dirs": [str(train_capture)],
                    "validation_capture_dirs": [str(validation_capture)],
                    "test_capture_dirs": [str(test_capture)],
                    "subcarrier_count": 4,
                }
            )
            config["model"]["hidden_dim"] = 8
            config["training"].update(
                {
                    "epochs": 1,
                    "batch_size": 8,
                    "initialization_seeds": [7],
                    "early_stopping_patience": 0,
                    "device": "cpu",
                    "average_power_budget_range": [1.0, 1.0],
                    "validation_average_power_budgets": [1.0],
                }
            )
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "train.py"],
                cwd=starter_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            policy_path = out_dir / "artifacts" / "power_policy.onnx"
            self.assertTrue(policy_path.is_file())
            manifest_path = out_dir / "trained_artifact.yaml"
            artifact_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            expected_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
            train_capture_sha = hashlib.sha256(
                (train_capture / "schema.json").read_bytes()
            ).hexdigest()
            validation_capture_sha = hashlib.sha256(
                (validation_capture / "schema.json").read_bytes()
            ).hexdigest()
            test_capture_sha = hashlib.sha256(
                (test_capture / "schema.json").read_bytes()
            ).hexdigest()
            self.assertEqual(artifact_manifest["schema_version"], 2)
            self.assertEqual(artifact_manifest["components"][0]["sha256"], expected_sha)
            self.assertEqual(artifact_manifest["components"][0]["format"], "onnx")
            self.assertEqual(
                artifact_manifest["training"]["train_capture_schema_sha256"],
                [train_capture_sha],
            )
            self.assertEqual(
                artifact_manifest["training"]["validation_capture_schema_sha256"],
                [validation_capture_sha],
            )
            binding = artifact_manifest["compatible_operations"][0]
            self.assertEqual(binding["operation"], "model.symbol_power_allocator")
            self.assertEqual(binding["required_inputs"], ["channel_state"])
            self.assertEqual(
                binding["params"],
                {
                    "policy": "learned_artifact",
                    "granularity": "per_subcarrier",
                    "budget_mode": "fixed_average",
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": "power_policy",
                },
            )
            self.assertFalse(artifact_manifest["training"]["uses_oracle_labels"])
            inspected = inspect_trained_artifact(
                manifest_path,
                project_root=root_path,
                registry=build_registry(),
            )
            self.assertTrue(inspected["ready"], inspected["issues"])
            self.assertEqual(inspected["components"][0]["actual_sha256"], expected_sha)
            evaluated = subprocess.run(
                [sys.executable, "evaluate.py"],
                cwd=starter_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)
            report = json.loads((starter_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(report["oracle_used_during_training"])
            self.assertEqual(
                report["trained_artifact"]["manifest_sha256"],
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                report["trained_artifact"]["components"][0]["sha256"], expected_sha
            )
            self.assertEqual(
                report["trained_artifact"]["components"][0]["actual_sha256"],
                expected_sha,
            )
            self.assertEqual(
                report["training_provenance"]["train_capture_schema_sha256"],
                [train_capture_sha],
            )
            self.assertEqual(
                report["training_provenance"]["validation_capture_schema_sha256"],
                [validation_capture_sha],
            )
            self.assertEqual(report["test_capture_schema_sha256"], [test_capture_sha])
            self.assertIn("relative_rate_regret", report["operating_points"][0])
            self.assertIn("mean_kkt_normalized_residual", report["operating_points"][0])


    def test_differentiable_export_writes_neutral_contract_bundle_by_default(self):
        recipe = _deepjscc_image_recipe()
        registry = build_registry()
        with tempfile.TemporaryDirectory() as root:
            out_dir = Path(root) / "differentiable_exports" / "deepjscc"
            payload = export_differentiable_scenario(
                recipe,
                registry,
                optimizable_steps=["sender", "receiver"],
                loss="image.mse",
                framework="torch",
                out_dir=out_dir,
                source_path=Path("recipes/deepjscc_kodak_awgn_train.yaml"),
                project_root=ROOT,
            )
            self.assertEqual(payload["status"], "exported")
            self.assertEqual(payload["kind"], "noema.training_interface_bundle@1")
            self.assertEqual(payload["exporter"], "training-contract")
            self.assertFalse(payload["include_starter"])
            for relative in [
                "training_contract.yaml",
                "scenario_graph.json",
                "project_manifest.yaml",
                "noema_recipe.yaml",
                "interfaces.py",
                "validate_contract.py",
                "package_artifact.py",
                "test_vectors/contract_vectors.json",
                "README.md",
                "training_plan.yaml",
            ]:
                self.assertTrue((out_dir / relative).is_file(), relative)
            for architecture_or_training_file in ("model.py", "losses.py", "train.py", "train_config.yaml"):
                self.assertFalse((out_dir / architecture_or_training_file).exists())
            self.assertFalse((out_dir / "reference_training").exists())
            self.assertFalse((out_dir / "trained_artifact.yaml").exists())
            contract = yaml.safe_load((out_dir / "training_contract.yaml").read_text(encoding="utf-8"))
            self.assertEqual(contract["kind"], "noema.trainable_slot_contract@1")
            self.assertEqual(contract["source_recipe"]["name"], recipe.name)
            self.assertEqual(contract["source_recipe"]["sha256"], scenario_recipe_fingerprint(recipe))
            self.assertEqual(contract["training_policy"]["architecture"], "external")
            self.assertEqual(contract["training_policy"]["loss"], "external")
            self.assertEqual(contract["training_policy"]["trainer"], "external")
            signals = {item["id"]: item for item in contract["signals"]}
            self.assertEqual(signals["source_image"]["source"], "data.images")
            self.assertEqual(signals["reconstruction"]["source"], "receiver.images")
            self.assertEqual(
                [(slot["step_id"], slot["role"]) for slot in contract["trainable_slots"]],
                [("sender", "encoder"), ("receiver", "decoder")],
            )
            copied = yaml.safe_load((out_dir / "noema_recipe.yaml").read_text(encoding="utf-8"))
            self.assertEqual(copied["name"], recipe.name)
            manifest = yaml.safe_load((out_dir / "project_manifest.yaml").read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "noema.training_interface_bundle@1")
            self.assertEqual(manifest["external_training"]["architecture"], "not_supplied")
            self.assertEqual(manifest["external_training"]["loss"], "not_supplied")
            self.assertEqual(manifest["external_training"]["trainer"], "not_supplied")
            self.assertIsNone(manifest["external_training"]["optional_demo_scaffold"])
            self.assertNotIn("data_contract", manifest)
            self.assertEqual(manifest["capture_jobs"], [])
            self.assertEqual(
                manifest["artifact_return"]["groups"][0]["artifact_application"],
                "all_group_bindings",
            )
            self.assertFalse((out_dir / "data_contract.yaml").exists())

    def test_deepjscc_demo_project_materializes_its_own_image_split(self):
        from PIL import Image

        recipe = _deepjscc_image_recipe()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "research_images"
            image_dir.mkdir()
            image_ids = ["frame_%02d.png" % index for index in range(5)]
            for index, image_id in enumerate(image_ids):
                values = np.full((8, 8, 3), index * 20, dtype=np.uint8)
                Image.fromarray(values, mode="RGB").save(image_dir / image_id)
            data_step = next(step for step in recipe.steps if step.id == "data")
            data_step.params = {
                "dataset": "kodak",
                "dataset_dir": str(image_dir),
                "image_ids": ",".join(image_ids),
                "crop_size": 8,
                "repeat_count": 1,
            }
            out_dir = root / "deepjscc_contract"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["sender", "receiver"],
                loss="image.mse",
                framework="torch",
                out_dir=out_dir,
                project_root=root,
            )
            prepare_example(out_dir, "deepjscc-image", project_root=root)
            contract_path = out_dir / "data_contract.yaml"
            contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
            self.assertEqual(
                contract["ownership"],
                {
                    "source_selection": "noema_recipe",
                    "partition_materialization": "noema_export",
                    "file_integrity_validation": "noema_contract_and_external_trainer",
                    "dataset_consumption": "external_researcher",
                    "model_training": "external_researcher",
                    "test_evaluation": "noema_ordinary_recipe_or_benchmark",
                },
            )
            splits = {row["id"]: row for row in contract["splits"]}
            self.assertEqual(
                [item["image_id"] for item in splits["train"]["files"]],
                image_ids[:4],
            )
            self.assertEqual(
                [item["image_id"] for item in splits["validation"]["files"]],
                image_ids[4:],
            )
            for row in contract["splits"]:
                for item in row["files"]:
                    self.assertEqual(
                        item["sha256"],
                        hashlib.sha256(Path(item["resolved_path"]).read_bytes()).hexdigest(),
                    )
            config = yaml.safe_load(
                (out_dir / "reference_training" / "train_config.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(config["data"]["train_image_ids"], image_ids[:4])
            self.assertEqual(config["data"]["validation_image_ids"], image_ids[4:])
            self.assertFalse(config["data"]["test_images_exposed_to_trainer"])

            loader_check = """
import yaml
from pathlib import Path
from datamodule import build_train_loader, build_validation_loader
config = yaml.safe_load(Path('train_config.yaml').read_text(encoding='utf-8'))
assert len(build_train_loader(config).dataset) == 4
assert len(build_validation_loader(config).dataset) == 1
"""
            starter_dir = out_dir / "reference_training"
            valid = subprocess.run(
                [sys.executable, "-c", loader_check],
                cwd=starter_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
            contract_validation = subprocess.run(
                [sys.executable, "validate_contract.py"],
                cwd=out_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                contract_validation.returncode,
                0,
                contract_validation.stdout + contract_validation.stderr,
            )
            (image_dir / image_ids[0]).write_bytes(
                (image_dir / image_ids[0]).read_bytes() + b"changed"
            )
            tampered = subprocess.run(
                [sys.executable, "-c", loader_check],
                cwd=starter_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("failed SHA-256 verification", tampered.stderr)
            tampered_contract = subprocess.run(
                [sys.executable, "validate_contract.py"],
                cwd=out_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertNotEqual(tampered_contract.returncode, 0)
            self.assertIn("source image hash mismatch", tampered_contract.stderr)

    def test_deepjscc_demo_honors_manifest_split_and_explicit_validation_count(self):
        from PIL import Image

        recipe = _deepjscc_image_recipe()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "research_images"
            image_dir.mkdir()
            sample_rows = []
            selected_ids = []
            identity = {"name": "identity"}
            for index in range(5):
                filename = "frame_%02d.jpg" % index
                path = image_dir / filename
                Image.fromarray(
                    np.full((8, 8, 3), index * 20, dtype=np.uint8),
                    mode="RGB",
                ).save(path)
                sample_id = "research:%02d" % index
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                selected_ids.append(sample_id)
                sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "path": filename,
                        "sha256": digest,
                        "source_id": sample_id,
                        "group_id": sample_id,
                        "source_sha256": digest,
                        "ancestry_ids": [sample_id],
                        "transform": identity,
                        "transform_fingerprint_sha256": canonical_json_sha256(
                            identity
                        ),
                    }
                )
            manifest_path = root / "research_manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "kind": "noema.image_dataset_manifest",
                        "id": "research-images",
                        "version": "v1",
                        "root": image_dir.name,
                        "publication_ready": False,
                        "publication_blocker": "test fixture",
                        "splits": {"training_all": selected_ids},
                        "samples": sample_rows,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            data_step = next(step for step in recipe.steps if step.id == "data")
            data_step.params = {
                "dataset": "research-images",
                "dataset_dir": str(image_dir),
                "manifest_path": str(manifest_path),
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "split": "training_all",
                "image_ids": "",
                "training_validation_count": 2,
                "crop_size": 8,
                "repeat_count": 1,
            }
            out_dir = root / "deepjscc_contract"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["sender", "receiver"],
                loss="image.mse",
                framework="torch",
                out_dir=out_dir,
                project_root=root,
            )
            prepare_example(out_dir, "deepjscc-image", project_root=root)
            contract = yaml.safe_load(
                (out_dir / "data_contract.yaml").read_text(encoding="utf-8")
            )
            splits = {row["id"]: row for row in contract["splits"]}
            self.assertEqual(
                [item["image_id"] for item in splits["train"]["files"]],
                selected_ids[:3],
            )
            self.assertEqual(
                [item["image_id"] for item in splits["validation"]["files"]],
                selected_ids[3:],
            )
            self.assertEqual(
                contract["partition_policy"]["id"],
                "ordered_recipe_selection_explicit_validation_count_v1",
            )
            self.assertEqual(contract["partition_policy"]["validation_count"], 2)
            self.assertEqual(
                contract["dataset"]["manifest"]["split"], "training_all"
            )

    def test_deepjscc_export_rejects_cross_framework_sionna(self):
        with tempfile.TemporaryDirectory() as root:
            recipe = _deepjscc_image_recipe()
            channel = next(
                step for step in recipe.steps if step.id == "wireless_channel"
            )
            channel.params["wireless_backend"] = "sionna"
            neutral_dir = Path(root) / "neutral"
            payload = export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["sender", "receiver"],
                loss="researcher.defined",
                framework="torch-sionna",
                out_dir=neutral_dir,
                exporter="deepjscc-image",
            )
            self.assertEqual(payload["exporter"], "training-contract")
            scenario_graph = json.loads(
                (neutral_dir / "scenario_graph.json").read_text(encoding="utf-8")
            )
            self.assertEqual(scenario_graph["framework"], "torch-sionna")
            frozen_channel = next(
                node
                for node in scenario_graph["nodes"]
                if node["id"] == "wireless_channel"
            )
            self.assertEqual(
                frozen_channel["materialization"]["backend"],
                "sionna",
            )
            self.assertEqual(
                frozen_channel["materialization"]["implementation"],
                "sionna_awgn_matched_pytorch_module",
            )
            self.assertNotIn(
                "selected_for_framework",
                frozen_channel["materialization"],
            )
            frozen_power = next(
                node
                for node in scenario_graph["nodes"]
                if node["id"] == "tx_power_normalize"
            )
            self.assertEqual(
                frozen_power["materialization"]["backend"],
                "torch",
            )
            self.assertNotIn(
                "selected_for_framework",
                frozen_power["materialization"],
            )
            with self.assertRaisesRegex(
                DifferentiableExportError,
                "requires framework=torch.*generic differentiable export graph",
            ):
                export_differentiable_scenario(
                    recipe,
                    build_registry(),
                    optimizable_steps=["sender", "receiver"],
                    loss="image.mse",
                    framework="torch-sionna",
                    out_dir=Path(root) / "bad",
                    exporter="deepjscc-image",
                    include_starter=True,
                )

    @unittest.skipUnless(torch_available(), "PyTorch is not installed")
    def test_exported_deepjscc_scenario_backpropagates_through_frozen_awgn(self):
        with tempfile.TemporaryDirectory() as root:
            out_dir = Path(root) / "deepjscc_gradient"
            export_differentiable_scenario(
                _deepjscc_image_recipe(),
                build_registry(),
                optimizable_steps=["sender", "receiver"],
                loss="image.mse",
                framework="torch",
                out_dir=out_dir,
                exporter="deepjscc-image",
                include_starter=True,
            )
            starter_dir = out_dir / "reference_training"
            code = """
import torch
import yaml
from pathlib import Path
from losses import image_mse
from scenario import build_scenario

config = yaml.safe_load(Path('train_config.yaml').read_text(encoding='utf-8'))
config['model']['symbol_channels'] = 2
scenario = build_scenario(config)
batch = {'image': torch.rand(2, 3, 8, 8)}
output = scenario(batch, snr_db=10.0)
assert 'loss' not in output
image_mse(output['reconstruction'], batch['image']).backward()
encoder_grads = [p.grad for p in scenario.model.encoder.parameters()]
decoder_grads = [p.grad for p in scenario.model.decoder.parameters()]
assert encoder_grads and all(g is not None and torch.isfinite(g).all() for g in encoder_grads)
assert decoder_grads and all(g is not None and torch.isfinite(g).all() for g in decoder_grads)
assert list(scenario.channel.parameters()) == []
assert list(scenario.power_normalize.parameters()) == []
"""
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=starter_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_checkpoint_return_smoke_trains_validates_benchmarks_and_records_sha(self):
        _torch_or_skip(self)
        from PIL import Image

        recipe = _deepjscc_image_recipe()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            image_ids = []
            for index in range(5):
                image_id = "fixture_%02d.png" % index
                pixels = np.full((8, 8, 3), index * 32, dtype=np.uint8)
                Image.fromarray(pixels, mode="RGB").save(image_dir / image_id)
                image_ids.append(image_id)
            data_step = next(step for step in recipe.steps if step.id == "data")
            data_step.params.update(
                {
                    "dataset_dir": str(image_dir),
                    "image_ids": ",".join(image_ids),
                }
            )
            out_dir = root / "differentiable_exports" / "deepjscc_return"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["sender", "receiver"],
                loss="image.mse",
                framework="torch",
                out_dir=out_dir,
                source_path=Path("recipes/deepjscc_kodak_awgn_train.yaml"),
                project_root=root,
            )
            prepare_example(out_dir, "deepjscc-image", project_root=root)
            starter_dir = out_dir / "reference_training"

            config_path = starter_dir / "train_config.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config.setdefault("data", {}).update(
                {
                    "crop_size": 8,
                }
            )
            config.setdefault("model", {})["symbol_channels"] = 2
            config.setdefault("training", {}).update(
                {
                    "device": "cpu",
                    "epochs": 1,
                    "batch_size": 2,
                    "num_workers": 0,
                    "early_stopping_patience": 0,
                    "initialization_seeds": [23],
                }
            )
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            completed = subprocess.run(
                [sys.executable, "train.py"],
                cwd=starter_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            encoder_path = out_dir / "artifacts" / "encoder.onnx"
            decoder_path = out_dir / "artifacts" / "decoder.onnx"
            self.assertTrue(encoder_path.is_file())
            self.assertTrue(decoder_path.is_file())
            encoder_sha = hashlib.sha256(encoder_path.read_bytes()).hexdigest()
            decoder_sha = hashlib.sha256(decoder_path.read_bytes()).hexdigest()

            manifest_path = out_dir / "trained_artifact.yaml"
            artifact_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact_manifest["schema_version"], 2)
            self.assertEqual(artifact_manifest["runtime"]["backend"], "onnxruntime")
            self.assertEqual(
                {(item["id"], item["format"], item["sha256"]) for item in artifact_manifest["components"]},
                {("encoder", "onnx", encoder_sha), ("decoder", "onnx", decoder_sha)},
            )
            artifact = inspect_trained_artifact(
                manifest_path,
                project_root=root,
                registry=build_registry(),
            )
            self.assertTrue(artifact["ready"], artifact["issues"])
            self.assertEqual(
                {item["id"]: item["actual_sha256"] for item in artifact["components"]},
                {"encoder": encoder_sha, "decoder": decoder_sha},
            )
            self.assertEqual(artifact["application"], {"mode": "all_group_bindings"})
            bindings = {binding["role"]: binding for binding in artifact["compatible_operations"]}
            self.assertEqual(set(bindings), {"encoder", "decoder"})
            self.assertEqual(
                {binding["binding_group"] for binding in bindings.values()},
                {"deepjscc_sender_receiver"},
            )
            self.assertEqual(bindings["encoder"]["operation"], "model.deepjscc_external_encode")
            self.assertEqual(bindings["decoder"]["operation"], "model.deepjscc_external_decode")
            self.assertEqual(bindings["encoder"]["params"]["runtime"], "learned_artifact")

            images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
            images[:, 2:6, 2:6, :] = [32, 128, 224]
            input_path = root / "images.npz"
            np.savez_compressed(input_path, images=images)
            recipe_path = root / "returned_adapter_recipe.yaml"
            returned_recipe = recipe.to_dict()
            returned_recipe["name"] = "checkpoint_return_canonical_smoke"
            returned_recipe.setdefault("metadata", {})["training_performed"] = True
            for step in returned_recipe["steps"]:
                if step["id"] == "data":
                    step["op"] = "source.local_npz_images"
                    step["params"] = {"path": str(input_path), "array": "images"}
                elif step["id"] == "sender":
                    step["params"] = dict(bindings["encoder"]["params"])
                    step["params"]["artifact_manifest_path"] = str(manifest_path.resolve())
                elif step["id"] == "receiver":
                    step["params"] = dict(bindings["decoder"]["params"])
                    step["params"]["artifact_manifest_path"] = str(manifest_path.resolve())
                elif step["id"] == "wireless_channel":
                    step["params"] = {
                        "channel": "awgn",
                        "snr_db": 30.0,
                        "noise_mode": "snr_at_unit_power",
                        "wireless_backend": "numpy",
                    }
            recipe_path.write_text(
                yaml.safe_dump(returned_recipe, sort_keys=False),
                encoding="utf-8",
            )
            validate_recipe_against_registry(load_recipe(recipe_path), build_registry())

            benchmark_path = root / "checkpoint_return_benchmark.yaml"
            benchmark_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "id": "checkpoint_return_smoke",
                        "version": "1",
                        "name": "Checkpoint Return Smoke",
                        "dataset": {"id": "images", "modality": "image"},
                        "task": {"id": "image_reconstruction", "kind": "reconstruction"},
                        "metrics": [{"id": "quality.psnr_db"}],
                        "metadata": {"tier": "smoke"},
                        "recipes": [
                            {
                                "id": "returned_deepjscc_adapter",
                                "label": "Returned DeepJSCC artifact",
                                "role": "candidate",
                                "path": str(recipe_path),
                                "params": {
                                    "matrix_selection": {"channel.snr_db": 10}
                                },
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            benchmark_stdout = io.StringIO()
            with contextlib.redirect_stdout(benchmark_stdout):
                code = main(
                    [
                        "--workspace",
                        str(root / "workspace"),
                        "benchmark",
                        "run",
                        str(benchmark_path),
                        "--retain-backing-runs",
                    ]
                )
            self.assertEqual(code, 0, benchmark_stdout.getvalue())
            result_line = [line for line in benchmark_stdout.getvalue().splitlines() if line.startswith("result: ")][0]
            benchmark_result = json.loads(Path(result_line.split("result: ", 1)[1]).read_text(encoding="utf-8"))
            self.assertEqual(benchmark_result["status"], "completed")
            self.assertEqual(benchmark_result["recipes"][0]["status"], "completed")

            run_manifest_path = Path(benchmark_result["recipes"][0]["manifest"])
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            executed_recipe = json.loads(
                (run_manifest_path.parent / "recipe.json").read_text(encoding="utf-8")
            )
            selection = {"channel.snr_db": 10}
            self.assertEqual(executed_recipe["metadata"]["matrix_selection"], selection)
            self.assertEqual(
                executed_recipe["metadata"]["matrix_variant_id"],
                matrix_variant_id(selection),
            )
            self.assertNotIn("sweeps", executed_recipe["metadata"])
            steps = {step["id"]: step for step in run_manifest["steps"]}
            self.assertIn("tx_power_normalize", steps)
            self.assertIn("wireless_channel", steps)
            sender_metadata = steps["sender"]["outputs"]["symbols"]["metadata"]
            self.assertEqual(sender_metadata["artifact_manifest_path"], str(manifest_path.resolve()))
            self.assertEqual(sender_metadata["artifact_entrypoint"], "encoder")
            self.assertGreater(sender_metadata["symbol_count"], 0)

    def test_neural_receiver_export_writes_receiver_training_harness(self):
        recipe = _neural_receiver_recipe()
        with tempfile.TemporaryDirectory() as root:
            out_dir = Path(root) / "differentiable_exports" / "receiver"
            payload = export_differentiable_scenario(
                recipe,
                _neural_receiver_registry(),
                optimizable_steps=["receiver"],
                loss="bit.bce",
                framework="torch",
                out_dir=out_dir,
                exporter="neural-receiver",
                include_starter=True,
            )
            starter_dir = out_dir / "reference_training"
            self.assertEqual(payload["status"], "exported")
            self.assertEqual(payload["exporter"], "training-contract")
            self.assertEqual(payload["starter_exporter"], "neural-receiver")
            root_manifest = yaml.safe_load((out_dir / "project_manifest.yaml").read_text(encoding="utf-8"))
            self.assertEqual(root_manifest["training"]["owner"], "external_researcher")
            self.assertEqual(
                root_manifest["evaluation"]["owner"],
                "noema_ordinary_recipe_or_benchmark",
            )
            self.assertEqual(
                root_manifest["external_training"]["optional_demo_scaffold"]["exporter"],
                "neural-receiver",
            )
            for relative in [
                "model.py",
                "train.py",
                "datamodule.py",
                "structured_input.py",
                "losses.py",
                "evaluate.py",
                "train_config.yaml",
                "noema_recipe.yaml",
                "training_template.yaml",
                "requirements.txt",
            ]:
                self.assertTrue((starter_dir / relative).is_file(), relative)
            config = yaml.safe_load((starter_dir / "train_config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["training_template"], "neural_receiver.supervised_qpsk")
            self.assertEqual(config["data"]["feature_tap"], "rx_symbols")
            self.assertEqual(config["data"]["target_tap"], "target_bits")
            self.assertEqual(config["recipe"]["receiver_step"], "receiver")
            starter_manifest = yaml.safe_load(
                (starter_dir / "project_manifest.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                starter_manifest["trained_artifacts"][0]["operation"],
                "model.synthetic_neural_receiver",
            )
            self.assertEqual(starter_manifest["source_recipe_sha256"], recipe_fingerprint(recipe))

    def test_neural_receiver_starter_rejects_duplicate_config_and_schema_keys(self):
        recipe = _neural_receiver_recipe()
        with tempfile.TemporaryDirectory() as root:
            out_dir = Path(root) / "receiver"
            export_differentiable_scenario(
                recipe,
                _neural_receiver_registry(),
                optimizable_steps=["receiver"],
                loss="bit.bce",
                framework="torch",
                out_dir=out_dir,
                exporter="neural-receiver",
                include_starter=True,
            )
            starter_dir = out_dir / "reference_training"
            config_path = starter_dir / "train_config.yaml"
            original_config = config_path.read_text(encoding="utf-8")
            config_path.write_text(
                original_config + "\ntraining:\n  epochs: 999\n",
                encoding="utf-8",
            )
            training = subprocess.run(
                [sys.executable, "train.py"],
                cwd=starter_dir,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(training.returncode, 0)
            self.assertIn(
                "Duplicate YAML mapping key `training`",
                training.stderr,
            )

            config_path.write_text(original_config, encoding="utf-8")
            capture_dir = starter_dir / "data" / "train"
            capture_dir.mkdir(parents=True)
            (capture_dir / "schema.json").write_text(
                '{"kind":"noema.capture_dataset","kind":"shadow",'
                '"split":"train"}\n',
                encoding="utf-8",
            )
            schema_load = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from datamodule import load_capture_dataset; "
                        "load_capture_dataset(['data/train'], "
                        "feature_tap='rx_symbols', target_tap='target_bits', "
                        "expected_split='train')"
                    ),
                ],
                cwd=starter_dir,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(schema_load.returncode, 0)
            self.assertIn(
                "Duplicate JSON object key `kind`",
                schema_load.stderr,
            )

    def test_differentiable_export_cli_json(self):
        recipe = _deepjscc_image_recipe()
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            recipe_path = root_path / "deepjscc.yaml"
            recipe_path.write_text(yaml.safe_dump(recipe.to_dict(), sort_keys=False), encoding="utf-8")
            out_dir = root_path / "exported"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([
                    "differentiable",
                    "export",
                    str(recipe_path),
                    "--optimizable",
                    "sender,receiver",
                    "--framework",
                    "torch",
                    "--out",
                    str(out_dir),
                    "--json",
                ])
            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "exported")
            self.assertEqual(payload["optimizable_steps"], ["sender", "receiver"])
            self.assertEqual(payload["exporter"], "training-contract")
            self.assertFalse(payload["include_starter"])
            self.assertTrue((out_dir / "training_contract.yaml").is_file())
            self.assertTrue((out_dir / "scenario_graph.json").is_file())
            self.assertFalse((out_dir / "train.py").exists())
            self.assertFalse((out_dir / "model.py").exists())
            self.assertFalse((out_dir / "losses.py").exists())

    def test_differentiable_export_rejects_unsupported_recipe(self):
        recipe = load_recipe(ROOT / "recipes" / "text_bart_jscc_clean.yaml")
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(
                DifferentiableExportError,
                "trained-artifact ABI",
            ):
                export_differentiable_scenario(
                    recipe,
                    build_registry(),
                    optimizable_steps=["sender", "receiver"],
                    loss="researcher.defined",
                    framework="torch",
                    out_dir=Path(root) / "neutral",
                    exporter="deepjscc-image",
                )
            with self.assertRaisesRegex(DifferentiableExportError, "image-to-symbol sender"):
                export_differentiable_scenario(
                    recipe,
                    build_registry(),
                    optimizable_steps=["sender", "receiver"],
                    loss="image.mse",
                    framework="torch",
                    out_dir=Path(root) / "bad",
                    exporter="deepjscc-image",
                    include_starter=True,
                )

    def test_differentiable_export_requires_wireless_channel(self):
        recipe = _deepjscc_image_recipe()
        for step in recipe.steps:
            if step.id == "wireless_channel":
                step.op = "channel.identity_symbol_link"
                step.params = {"label": "disabled"}
            if step.id == "rx_symbol_boundary":
                step.inputs = {"symbols": "wireless_channel.symbols"}
        with tempfile.TemporaryDirectory() as root:
            neutral = export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["sender", "receiver"],
                loss="researcher.defined",
                framework="torch",
                out_dir=Path(root) / "neutral",
                exporter="deepjscc-image",
            )
            self.assertEqual(neutral["exporter"], "training-contract")
            with self.assertRaisesRegex(DifferentiableExportError, "requires a wireless.channel step"):
                export_differentiable_scenario(
                    recipe,
                    build_registry(),
                    optimizable_steps=["sender", "receiver"],
                    loss="image.mse",
                    framework="torch",
                    out_dir=Path(root) / "bad",
                    exporter="deepjscc-image",
                    include_starter=True,
                )

    def test_sionna_awgn_requires_the_explicit_sionna_block_type(self):
        with self.assertRaisesRegex(
            TrainingDependencyError, "SionnaAwgnChannelBlock"
        ):
            AwgnChannelBlock(snr_db=25.0, backend="sionna")


def _write_gain_capture(path: Path, split: str, gains: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    shards = path / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    shard_path = shards / "shard_0000.npz"
    np.savez_compressed(shard_path, channel_gains=np.asarray(gains, dtype=np.float32))
    schema = {
        "schema_version": 1,
        "kind": "noema.capture_dataset",
        "split": split,
        "captured_samples": int(gains.shape[0]),
        "shards": [{"path": "shards/shard_0000.npz", "captured_samples": int(gains.shape[0])}],
        "tap_schemas": {
            "channel_gains": {
                "id": "channel_gains",
                "dtype": "float32",
                "record_shape": [int(gains.shape[1])],
            }
        },
    }
    (path / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
