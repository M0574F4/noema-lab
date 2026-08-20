import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.cli.main import main
from noema_lab.core.artifacts import Artifact, artifact, file_sha256
from noema_lab.core.boundaries import (
    boundary_contract,
    validate_artifact_file,
    validate_bits_per_pixel,
    validate_channel_symbols,
    validate_image_tensor,
    validate_metric_scalar,
    validate_npz_artifact_schema,
    validate_payload_bits,
    validate_rate_accounting_point,
    validate_semantic_embedding,
    validate_text_utf8_bytes,
)
from noema_lab.core.executor import ExecutionCancelled, LocalExecutor
from noema_lab.core.external_adapters import validate_adapter_manifest
from noema_lab.core.graph import recipe_graph
from noema_lab.core.lint import lint_recipe_invariants
from noema_lab.core.materialization import MaterializationRegistry, build_materialization_registry
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationRegistry, OperationResult
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import load_recipe, recipe_from_dict
from noema_lab.core.research import research_specs_from_recipe
from noema_lab.core.reproducibility import canonical_json_sha256, derive_seed
from noema_lab.core.runner_contracts import operation_runner_supports
from noema_lab.core.storage import LocalStore
from noema_lab.core.suites_catalog import load_suites_catalog
from noema_lab.core.training import inspect_training_feasibility
from noema_lab.core.verification import verify_benchmark_result, verify_run_bundle
from noema_lab.training.differentiability import torch_available
from noema_lab.ops.channel.tensor import BitsToLatentsOperation
from noema_lab.ops.channel.digital import (
    BitBoundaryCheckpointOperation,
    BitCountMatchOperation,
    IdentityDemodulateOperation,
    IdentityModulateOperation,
    IdentitySymbolLinkOperation,
    SymbolBoundaryCheckpointOperation,
    SymbolCountMatchOperation,
    SymbolPowerNormalizeOperation,
    WirelessChannelOperation,
)
from noema_lab.ops import build_registry
from noema_lab.ops.foundation import _masked_lm_render
from noema_lab.ops.models.catalog import diffusers_load_kwargs, load_model_catalog
from noema_lab.ops.models.eflic import _aoti_load_package_compat as _eflic_aoti_load_package_compat
from noema_lab.ops.models.learned_codecs import _aoti_load_package_compat as _compressai_aoti_load_package_compat
from noema_lab.ops.models.upstream_lic import (
    _google_drive_direct_download_url,
    _validate_checkpoint_file,
)
from noema_lab.ui.server import _learned_checkpoint_readiness, start_ui_server_in_thread


class _SyntheticTrainingOperation(Operation):
    params_schema = {
        "type": "object",
        "properties": {
            "artifact_manifest_path": {"type": "string", "default": ""},
            "artifact_entrypoint": {"type": "string", "default": ""},
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, operation_id, name, input_kinds=None, output_kinds=None, differentiability=None, backends=None, materializations=None):
        self.id = operation_id
        self.name = name
        self.input_kinds = dict(input_kinds or {})
        self.output_kinds = dict(output_kinds or {})
        self.differentiability = dict(differentiability or {})
        self.backends = dict(backends or {})
        if not self.backends and self.differentiability.get("exportable") and self.differentiability.get("gradient") in {"full", "surrogate"}:
            backend = "sionna" if self.differentiability.get("framework") == "sionna" else "torch"
            self.backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": [backend]}
        self.materializations = list(materializations) if materializations is not None else None
        if self.differentiability.get("trainable_params"):
            component_id = operation_id.rsplit(".", 1)[-1]
            self.trained_artifact_abi = {
                "component_id": component_id,
                "component_role": "synthetic_training_component",
                "entrypoint_id": component_id,
                "required_operation_inputs": list(self.input_kinds),
                "inputs": {name: {"dtype": "operation_defined", "shape": ["..."]} for name in self.input_kinds},
                "outputs": {name: {"dtype": "operation_defined", "shape": ["..."]} for name in self.output_kinds},
                "binding_params": {
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": component_id,
                },
            }

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


def _synthetic_training_registry():
    registry = OperationRegistry()
    registry.register(
        _SyntheticTrainingOperation(
            "source.synthetic_images",
            "Synthetic image source",
            output_kinds={"images": "image.batch.numpy"},
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "model.synthetic_encoder",
            "Synthetic trainable encoder",
            input_kinds={"images": ["image.batch.numpy"]},
            output_kinds={"symbols": "channel.symbols.complex_numpy"},
            differentiability={
                "framework": "torch",
                "gradient": "full",
                "trainable_params": True,
                "exportable": True,
            },
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "model.synthetic_bit_encoder",
            "Synthetic trainable bit encoder",
            input_kinds={"images": ["image.batch.numpy"]},
            output_kinds={"bits": "channel.payload_bits.numpy"},
            differentiability={
                "framework": "torch",
                "gradient": "surrogate",
                "trainable_params": True,
                "exportable": True,
            },
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "wireless.sionna_awgn",
            "Synthetic Sionna AWGN channel",
            input_kinds={"symbols": ["channel.symbols.complex_numpy"]},
            output_kinds={"symbols": "channel.rx_symbols.complex_numpy"},
            differentiability={
                "framework": "sionna",
                "gradient": "full",
                "trainable_params": False,
                "exportable": True,
            },
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "channel.hard_demod",
            "Synthetic hard demodulator",
            input_kinds={"symbols": ["channel.rx_symbols.complex_numpy", "channel.symbols.complex_numpy"]},
            output_kinds={"symbols": "channel.rx_symbols.complex_numpy"},
            differentiability={
                "framework": "numpy",
                "gradient": "stop",
                "trainable_params": False,
                "exportable": False,
                "reason": "Hard decisions stop gradients.",
            },
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "channel.crc32_check",
            "Synthetic CRC check",
            input_kinds={"bits": ["channel.payload_bits.numpy"]},
            output_kinds={"bits": "channel.payload_bits.numpy"},
            differentiability={
                "framework": "numpy",
                "gradient": "stop",
                "trainable_params": False,
                "exportable": False,
                "reason": "CRC is a hard non-differentiable packet decision.",
            },
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "source.synthetic_bits",
            "Synthetic bit source",
            output_kinds={"bits": "channel.bits.numpy"},
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "channel.synthetic_encoder",
            "Synthetic channel encoder",
            input_kinds={"bits": ["channel.bits.numpy", "channel.payload_bits.numpy"]},
            output_kinds={"coded_bits": "channel.coded_bits.numpy"},
            differentiability={
                "framework": "numpy",
                "gradient": "stop",
                "trainable_params": False,
                "exportable": False,
                "reason": "Channel coding is a hard digital transform in this synthetic receiver recipe.",
            },
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "modulation.synthetic_qpsk",
            "Synthetic QPSK mapper",
            input_kinds={"bits": ["channel.coded_bits.numpy"]},
            output_kinds={"symbols": "channel.symbols.complex_numpy"},
            differentiability={
                "framework": "numpy",
                "gradient": "stop",
                "trainable_params": False,
                "exportable": False,
                "reason": "Hard bit-to-constellation mapping stops gradients to the bits.",
            },
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "model.synthetic_neural_receiver",
            "Synthetic trainable neural receiver",
            input_kinds={
                "features": ["channel.rx_symbols.complex_numpy", "channel.llr.numpy"],
                "target_bits": ["channel.coded_bits.numpy", "channel.payload_bits.numpy", "channel.bits.numpy"],
            },
            output_kinds={"bits": "channel.demod_bits.numpy"},
            differentiability={
                "framework": "torch",
                "gradient": "full",
                "trainable_params": True,
                "exportable": True,
            },
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "model.synthetic_decoder",
            "Synthetic trainable decoder",
            input_kinds={"symbols": ["channel.rx_symbols.complex_numpy", "channel.symbols.complex_numpy"]},
            output_kinds={"images": "image.batch.numpy"},
            differentiability={
                "framework": "torch",
                "gradient": "full",
                "trainable_params": True,
                "exportable": True,
            },
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "model.synthetic_bit_decoder",
            "Synthetic trainable bit decoder",
            input_kinds={"bits": ["channel.payload_bits.numpy"]},
            output_kinds={"images": "image.batch.numpy"},
            differentiability={
                "framework": "torch",
                "gradient": "full",
                "trainable_params": True,
                "exportable": True,
            },
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "codec.synthetic_jpeg",
            "Synthetic classical codec",
            input_kinds={"images": ["image.batch.numpy"]},
            output_kinds={"images": "image.batch.numpy"},
            differentiability={
                "framework": "numpy",
                "gradient": "stop",
                "trainable_params": False,
                "exportable": False,
                "reason": "Classical codec is not differentiable.",
            },
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "metrics.synthetic_loss",
            "Synthetic image loss",
            input_kinds={"reference": ["image.batch.numpy"], "reconstruction": ["image.batch.numpy"]},
            output_kinds={"report": "metrics.report"},
        )
    )
    registry.register(
        _SyntheticTrainingOperation(
            "metrics.synthetic_bit_loss",
            "Synthetic bit loss",
            input_kinds={
                "reference": ["channel.coded_bits.numpy", "channel.payload_bits.numpy", "channel.bits.numpy"],
                "candidate": ["channel.demod_bits.numpy", "channel.payload_bits.numpy", "channel.bits.numpy"],
            },
            output_kinds={"report": "metrics.report"},
        )
    )
    return registry


class CliCoreTests(unittest.TestCase):
    def test_project_metadata_declares_neural_extras(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        lockfile = (ROOT / "uv.lock").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.11,<3.14"', pyproject)
        self.assertIn('license = "MIT"', pyproject)
        self.assertIn('license-files = ["LICENSE"]', pyproject)
        package_data = next(
            line
            for line in pyproject.split("[tool.setuptools.package-data]", 1)[1].splitlines()
            if line.startswith('"noema_lab" = ')
        )
        for resource in (
            "research_catalog.yaml",
            "recipe_templates.yaml",
            "suites_catalog.yaml",
        ):
            self.assertIn(f'"{resource}"', package_data)
        self.assertIn("[project.optional-dependencies]", pyproject)
        self.assertIn('"matplotlib>=3.7"', pyproject)
        self.assertIn('"Pillow>=9.0"', pyproject)
        self.assertIn("compressai = [", pyproject)
        self.assertIn('"compressai>=1.2.0"', pyproject)
        self.assertIn("foundation = [", pyproject)
        self.assertIn('"transformers>=4.40"', pyproject)
        self.assertIn('name = "compressai"', lockfile)
        self.assertIn('name = "transformers"', lockfile)
        self.assertIn('name = "matplotlib"', lockfile)
        self.assertIn('name = "pillow"', lockfile)
        self.assertTrue(
            "extra == 'compressai'" in lockfile or 'extra == "compressai"' in lockfile,
            "compressai extra marker missing from uv.lock",
        )
        self.assertTrue((ROOT / "LICENSE").is_file())
        self.assertTrue((ROOT / "CITATION.cff").is_file())
        self.assertTrue((ROOT / ".github" / "workflows" / "ci.yml").is_file())

    def test_diffusers_vqmodel_catalog_supplies_subfolder(self):
        catalog = load_model_catalog()
        kwargs = diffusers_load_kwargs(catalog, "vqmodel", "CompVis/ldm-celebahq-256", {})
        self.assertEqual(kwargs.get("subfolder"), "vqvae")
        override = diffusers_load_kwargs(
            catalog,
            "vqmodel",
            "CompVis/ldm-celebahq-256",
            {"subfolder": "custom_vqvae"},
        )
        self.assertEqual(override.get("subfolder"), "custom_vqvae")

    def test_bits_to_latents_sanitizes_readonly_float_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bits_path = root / "bits.npz"
            nan_payload = np.array([np.nan], dtype=np.float32).tobytes()
            bits = np.unpackbits(np.frombuffer(nan_payload, dtype=np.uint8)).astype(np.uint8)
            metadata = {
                "byte_count": len(nan_payload),
                "tensor_dtype": "float32",
                "tensor_shape": [1],
            }
            np.savez_compressed(bits_path, bits=bits, metadata_json=json.dumps(metadata))
            step_dir = root / "step"
            ctx = OperationContext(
                recipe_name="test",
                step_id="payload_unpacker",
                params={"sanitize": True},
                inputs={
                    "bits": Artifact(
                        kind="channel.payload_bits.numpy",
                        path=bits_path,
                        metadata=metadata,
                    )
                },
                run_dir=root,
                step_dir=step_dir,
            )
            result = BitsToLatentsOperation().run(ctx)
            with np.load(result.outputs["latents"].path, allow_pickle=False) as payload:
                latents = payload["latents"]
            self.assertTrue(np.isfinite(latents).all())
            self.assertEqual(float(latents[0]), 0.0)

    def test_bit_boundary_requires_canonical_uint8_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bits_path = root / "bits.npz"
            metadata = {"bit_count": 4, "payload_bit_count": 4}
            np.savez_compressed(
                bits_path,
                bits=np.array([0, 1, 1, 0], dtype=np.uint8),
                metadata_json=json.dumps(metadata),
            )
            ctx = OperationContext(
                recipe_name="test",
                step_id="tx_bit_boundary",
                params={"label": "modulator_input", "role": "modulator_input"},
                inputs={"bits": Artifact("channel.coded_bits.numpy", bits_path, metadata)},
                run_dir=root,
                step_dir=root / "tx_boundary",
            )
            result = BitBoundaryCheckpointOperation().run(ctx)
            self.assertEqual(result.outputs["bits"].kind, "channel.bits.numpy")
            self.assertEqual(result.metrics["channel.fixed.modulator_input.bit_count"], 4)
            self.assertEqual(result.outputs["bits"].metadata["boundary_contract"], "channel.bits")
            with np.load(result.outputs["bits"].path, allow_pickle=False) as payload:
                self.assertEqual(payload["bits"].dtype, np.dtype("uint8"))

            bad_path = root / "bad_bits.npz"
            np.savez_compressed(
                bad_path,
                bits=np.array([0.0, 1.0], dtype=np.float32),
                metadata_json=json.dumps({"bit_count": 2}),
            )
            bad_ctx = OperationContext(
                recipe_name="test",
                step_id="tx_bit_boundary",
                params={"label": "modulator_input"},
                inputs={"bits": Artifact("channel.coded_bits.numpy", bad_path, {"bit_count": 2})},
                run_dir=root,
                step_dir=root / "bad_boundary",
            )
            with self.assertRaises(OperationError):
                BitBoundaryCheckpointOperation().run(bad_ctx)

    def test_canonical_boundary_validators_cover_shared_contracts(self):
        self.assertEqual(boundary_contract("payload.bits")["dtype"], "uint8")
        bits, bit_metadata = validate_payload_bits(np.array([0, 1, 1, 0], dtype=np.uint8), "payload_test")
        self.assertEqual(bits.dtype, np.dtype("uint8"))
        self.assertEqual(bit_metadata["boundary_contract"], "payload.bits")
        self.assertEqual(bit_metadata["bit_count"], 4)
        with self.assertRaisesRegex(OperationError, "uint8"):
            validate_payload_bits(np.array([0.0, 1.0], dtype=np.float32), "bad_payload")
        with self.assertRaisesRegex(OperationError, "0 or 1"):
            validate_payload_bits(np.array([0, 2], dtype=np.uint8), "bad_payload")

        complex_symbols, complex_metadata = validate_channel_symbols(
            np.array([1 + 1j, -0.5 + 0.25j], dtype=np.complex64),
            "complex_symbols",
        )
        self.assertEqual(complex_symbols.dtype, np.dtype("complex64"))
        self.assertEqual(complex_metadata["representation"], "complex_baseband")
        real_symbols, real_metadata = validate_channel_symbols(
            np.ones((2, 3), dtype=np.float32),
            "real_symbols",
            representation="real",
        )
        self.assertEqual(real_symbols.dtype, np.dtype("float32"))
        self.assertEqual(real_metadata["representation"], "real_valued")
        with self.assertRaisesRegex(OperationError, "non-finite"):
            validate_channel_symbols(np.array([np.nan + 0j], dtype=np.complex64), "bad_symbols")

        images, image_metadata = validate_image_tensor(np.zeros((1, 4, 5, 3), dtype=np.uint8))
        self.assertEqual(images.shape, (1, 4, 5, 3))
        self.assertEqual(image_metadata["boundary_contract"], "image.tensor")
        with self.assertRaisesRegex(OperationError, r"\[N, H, W, C\]"):
            validate_image_tensor(np.zeros((4, 5, 3), dtype=np.uint8))

        raw_text, text_metadata = validate_text_utf8_bytes("hello")
        self.assertEqual(raw_text, b"hello")
        self.assertEqual(text_metadata["encoding"], "utf-8")
        with self.assertRaisesRegex(OperationError, "UTF-8"):
            validate_text_utf8_bytes(b"\xff")

        embedding, embedding_metadata = validate_semantic_embedding(np.ones((2, 4), dtype=np.float64))
        self.assertEqual(embedding.dtype, np.dtype("float32"))
        self.assertEqual(embedding_metadata["embedding_dim"], 4)
        with self.assertRaisesRegex(OperationError, "finite"):
            validate_semantic_embedding(np.array([[np.inf]], dtype=np.float32))

        self.assertEqual(validate_metric_scalar(1.5, "score"), 1.5)
        with self.assertRaisesRegex(OperationError, "finite"):
            validate_metric_scalar(float("nan"), "score")
        self.assertEqual(validate_rate_accounting_point("tx_bits", 8, "bits"), 8)
        with self.assertRaisesRegex(OperationError, "integer"):
            validate_rate_accounting_point("tx_bits", 8.5, "bits")
        self.assertAlmostEqual(validate_bits_per_pixel(8, 4, 2.0), 2.0)
        with self.assertRaisesRegex(OperationError, "does not match"):
            validate_bits_per_pixel(8, 4, 3.0)

    def test_artifact_file_and_npz_schema_validators_are_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "bits.npz"
            np.savez_compressed(path, bits=np.array([0, 1, 0, 1], dtype=np.uint8), metadata_json="{}")
            bit_artifact = artifact("channel.bits.numpy", path, {"bit_count": 4})
            report = validate_artifact_file(bit_artifact, expected_kind="channel.bits.numpy")
            self.assertEqual(report["kind"], "channel.bits.numpy")
            schema_report = validate_npz_artifact_schema(bit_artifact, ["bits"], expected_kind="channel.bits.numpy")
            self.assertEqual(schema_report["arrays"]["bits"]["dtype"], "uint8")

            tampered = Artifact("channel.bits.numpy", path, {"bit_count": 4}, sha256="0" * 64)
            with self.assertRaisesRegex(OperationError, "SHA-256"):
                validate_artifact_file(tampered)
            with self.assertRaisesRegex(OperationError, "missing required"):
                validate_npz_artifact_schema(bit_artifact, ["symbols"])
            with self.assertRaisesRegex(OperationError, "expected artifact kind"):
                validate_artifact_file(bit_artifact, expected_kind="image.batch.numpy")

    def test_bit_count_match_rejects_channel_length_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_path = root / "ref.npz"
            cand_path = root / "cand.npz"
            np.savez_compressed(ref_path, bits=np.array([0, 1, 0], dtype=np.uint8), metadata_json="{}")
            np.savez_compressed(cand_path, bits=np.array([0, 1], dtype=np.uint8), metadata_json="{}")
            ctx = OperationContext(
                recipe_name="test",
                step_id="channel_bit_count_match",
                params={"label": "channel_io"},
                inputs={
                    "reference": Artifact("channel.bits.numpy", ref_path, {}),
                    "candidate": Artifact("channel.bits.numpy", cand_path, {}),
                },
                run_dir=root,
                step_dir=root / "count_match",
            )
            with self.assertRaises(OperationError):
                BitCountMatchOperation().run(ctx)

    def test_symbol_boundary_requires_canonical_complex_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            metadata = {"symbol_count": 2, "channel_use_count": 2}
            np.savez_compressed(
                symbols_path,
                symbols=np.array([1 + 0.25j, -0.5 + 0.75j], dtype=np.complex64),
                metadata_json=json.dumps(metadata),
            )
            ctx = OperationContext(
                recipe_name="test",
                step_id="tx_symbol_boundary",
                params={"label": "modulator_input", "role": "symbol_channel_input"},
                inputs={"symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata)},
                run_dir=root,
                step_dir=root / "tx_symbol_boundary",
            )
            result = SymbolBoundaryCheckpointOperation().run(ctx)
            self.assertEqual(result.outputs["symbols"].kind, "channel.symbols.complex_numpy")
            self.assertEqual(result.metrics["channel.fixed.modulator_input.symbol_count"], 2)
            self.assertEqual(result.outputs["symbols"].metadata["boundary_contract"], "channel.symbols")
            with np.load(result.outputs["symbols"].path, allow_pickle=False) as payload:
                self.assertEqual(payload["symbols"].dtype, np.dtype("complex64"))

            normalized = SymbolPowerNormalizeOperation().run(
                OperationContext(
                    recipe_name="test",
                    step_id="tx_power_normalize",
                    params={"label": "tx_power_normalize", "target_power": 1.0},
                    inputs={"symbols": result.outputs["symbols"]},
                    run_dir=root,
                    step_dir=root / "tx_power_normalize",
                )
            )
            with np.load(normalized.outputs["symbols"].path, allow_pickle=False) as payload:
                normalized_symbols = payload["symbols"]
            self.assertEqual(normalized.outputs["symbols"].kind, "channel.symbols.complex_numpy")
            self.assertAlmostEqual(float(np.mean(np.abs(normalized_symbols) ** 2)), 1.0, places=6)
            self.assertIn("channel.tx_power.after", normalized.metrics)

            linked = IdentitySymbolLinkOperation().run(
                OperationContext(
                    recipe_name="test",
                    step_id="wireless_channel",
                    params={"label": "disabled_symbol_channel"},
                    inputs={"symbols": normalized.outputs["symbols"]},
                    run_dir=root,
                    step_dir=root / "identity_symbol_link",
                )
            )
            self.assertEqual(linked.outputs["symbols"].kind, "channel.symbols.complex_numpy")
            self.assertEqual(linked.outputs["symbols"].metadata["boundary_contract"], "channel.symbols")
            self.assertIn("channel.rx_antenna_power.average", linked.metrics)
            self.assertIn("channel.rx_output_power.average", linked.metrics)
            self.assertAlmostEqual(
                linked.metrics["channel.rx_antenna_power.average"],
                linked.metrics["channel.rx_output_power.average"],
                places=7,
            )

            SymbolCountMatchOperation().run(
                OperationContext(
                    recipe_name="test",
                    step_id="channel_symbol_count_match",
                    params={"label": "channel_io"},
                    inputs={"reference": result.outputs["symbols"], "candidate": linked.outputs["symbols"]},
                    run_dir=root,
                    step_dir=root / "symbol_count_match",
                )
            )

            bad_path = root / "bad_symbols.npz"
            np.savez_compressed(
                bad_path,
                symbols=np.array([0.0, 1.0], dtype=np.float32),
                metadata_json=json.dumps({"symbol_count": 2}),
            )
            bad_ctx = OperationContext(
                recipe_name="test",
                step_id="tx_symbol_boundary",
                params={"label": "modulator_input"},
                inputs={"symbols": Artifact("channel.symbols.complex_numpy", bad_path, {"symbol_count": 2})},
                run_dir=root,
                step_dir=root / "bad_symbol_boundary",
            )
            with self.assertRaises(OperationError):
                SymbolBoundaryCheckpointOperation().run(bad_ctx)

    def test_identity_phy_null_modem_roundtrips_canonical_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bits_path = root / "bits.npz"
            metadata = {"bit_count": 6, "payload_bit_count": 6}
            original_bits = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)
            np.savez_compressed(bits_path, bits=original_bits, metadata_json=json.dumps(metadata))
            boundary = BitBoundaryCheckpointOperation().run(
                OperationContext(
                    recipe_name="test",
                    step_id="tx_bit_boundary",
                    params={"label": "modulator_input", "role": "modulator_input"},
                    inputs={"bits": Artifact("channel.coded_bits.numpy", bits_path, metadata)},
                    run_dir=root,
                    step_dir=root / "tx_bit_boundary",
                )
            )
            modulated = IdentityModulateOperation().run(
                OperationContext(
                    recipe_name="test",
                    step_id="modulator",
                    params={"label": "identity_phy_modulator"},
                    inputs={"bits": boundary.outputs["bits"]},
                    run_dir=root,
                    step_dir=root / "modulator",
                )
            )
            self.assertEqual(modulated.outputs["symbols"].kind, "channel.symbols.complex_numpy")
            self.assertEqual(modulated.outputs["symbols"].metadata["identity_phy_stage"], "modulator")
            with np.load(modulated.outputs["symbols"].path, allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["symbols"].real.astype(np.uint8), original_bits)
                self.assertEqual(payload["symbols"].dtype, np.dtype("complex64"))

            linked = IdentitySymbolLinkOperation().run(
                OperationContext(
                    recipe_name="test",
                    step_id="wireless_channel",
                    params={"label": "identity_phy_channel"},
                    inputs={"symbols": modulated.outputs["symbols"]},
                    run_dir=root,
                    step_dir=root / "wireless_channel",
                )
            )
            self.assertIn("channel.rx_antenna_power.average", linked.metrics)
            self.assertEqual(linked.outputs["symbols"].metadata["rx_antenna_power_average"], linked.metrics["channel.rx_antenna_power.average"])
            demodulated = IdentityDemodulateOperation().run(
                OperationContext(
                    recipe_name="test",
                    step_id="demodulator",
                    params={"label": "identity_phy_demodulator"},
                    inputs={"rx_symbols": linked.outputs["symbols"]},
                    run_dir=root,
                    step_dir=root / "demodulator",
                )
            )
            self.assertEqual(demodulated.outputs["bits"].kind, "channel.demod_bits.numpy")
            self.assertEqual(demodulated.outputs["bits"].metadata["identity_phy_stage"], "demodulator")
            with np.load(demodulated.outputs["bits"].path, allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["bits"], original_bits)
            with np.load(demodulated.outputs["llr"].path, allow_pickle=False) as payload:
                self.assertEqual(payload["llr"].dtype, np.dtype("float32"))

    def test_wireless_channel_schema_exposes_realistic_presets(self):
        op = build_registry().get("wireless.channel")
        description = op.describe()
        properties = description["params_schema"]["properties"]
        channel_enum = properties["channel"]["enum"]
        self.assertIn("interference_awgn", channel_enum)
        self.assertIn("mimo_flat", channel_enum)
        self.assertIn("ofdm_tdl", channel_enum)
        self.assertIn("wireless_backend", properties)
        self.assertEqual(properties["wireless_backend"]["enum"], ["auto", "numpy", "sionna"])
        self.assertIn(
            "auto resolves deterministically",
            properties["wireless_backend"]["description"],
        )
        self.assertEqual(properties["noise_mode"]["enum"], ["snr_at_unit_power", "fixed_variance"])
        self.assertIn("noise_variance", properties)
        self.assertIn("ofdm_cdl", description["wireless_presets"])
        self.assertIn("availability", description)

    def test_operation_describe_includes_default_differentiability_metadata(self):
        description = build_registry().get("source.image_dataset").describe()
        self.assertEqual(
            description["differentiability"],
            {
                "framework": "numpy",
                "gradient": "none",
                "trainable_params": False,
                "exportable": False,
            },
        )
        self.assertEqual(description["backends"], {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": []})
        self.assertEqual(description["equivalence"]["type"], "behavioral")
        self.assertEqual(description["formats"]["artifact"], "operation-defined")
        self.assertEqual(description["formats"]["tensor"], "none")
        self.assertEqual(
            description["training_capabilities"],
            {"built_in_fine_tuning": False, "portable_replacement": False},
        )
        self.assertIn(
            {"runner": "benchmark_run", "backend": "numpy", "implementation": "default", "status": "implemented"},
            description["materializations"],
        )

    def test_operation_describe_exposes_explicit_differentiability_metadata(self):
        description = build_registry().get("wireless.channel").describe()
        differentiability = description["differentiability"]
        self.assertEqual(differentiability["framework"], "torch")
        self.assertEqual(differentiability["gradient"], "full")
        self.assertFalse(differentiability["trainable_params"])
        self.assertTrue(differentiability["exportable"])
        self.assertIn("Sionna 2.x/PyTorch", differentiability["reason"])
        self.assertEqual(description["backends"]["benchmark_run"], ["numpy", "sionna"])
        self.assertEqual(
            description["backends"]["differentiable_export"],
            ["torch", "sionna"],
        )
        self.assertEqual(description["equivalence"]["type"], "statistical")
        self.assertEqual(description["formats"]["artifact"], "npz")
        self.assertEqual(description["formats"]["tensor"], "torch.Tensor")
        self.assertEqual(
            description["training_capabilities"],
            {"built_in_fine_tuning": False, "portable_replacement": False},
        )
        self.assertTrue(
            any(
                item["runner"] == "differentiable_export"
                and item["backend"] == "sionna"
                for item in description["materializations"]
            )
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["ops", "show", "wireless.channel", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["differentiability"]["framework"], "torch")
        self.assertEqual(payload["differentiability"]["gradient"], "full")
        self.assertEqual(
            payload["backends"]["differentiable_export"],
            ["torch", "sionna"],
        )
        self.assertEqual(payload["equivalence"]["type"], "statistical")
        self.assertEqual(payload["formats"]["tensor"], "torch.Tensor")

        human_output = io.StringIO()
        with contextlib.redirect_stdout(human_output):
            code = main(["ops", "show", "wireless.channel"])
        self.assertEqual(code, 0)
        self.assertIn("differentiability", human_output.getvalue())
        self.assertIn("backends", human_output.getvalue())
        self.assertIn("equivalence", human_output.getvalue())
        self.assertIn("formats", human_output.getvalue())

    def test_operation_training_capabilities_are_independent(self):
        registry = build_registry()
        replacement = registry.get("model.deepjscc_external_encode").describe()
        fine_tunable = registry.get("model.text_bart_jscc_encode").describe()
        support = registry.get("wireless.channel").describe()

        self.assertEqual(
            replacement["training_capabilities"],
            {"built_in_fine_tuning": False, "portable_replacement": True},
        )
        self.assertEqual(
            fine_tunable["training_capabilities"],
            {"built_in_fine_tuning": False, "portable_replacement": False},
        )
        self.assertEqual(
            support["training_capabilities"],
            {"built_in_fine_tuning": False, "portable_replacement": False},
        )

        class InvalidFineTuningCapability(Operation):
            id = "test.invalid_fine_tuning_capability"
            name = "Invalid fine-tuning capability"
            fine_tuning_supported = "yes"

            def run(self, ctx: OperationContext) -> OperationResult:
                return OperationResult()

        with self.assertRaisesRegex(OperationError, "fine_tuning_supported must be a boolean"):
            InvalidFineTuningCapability().describe()

        class MissingFineTuningProvider(Operation):
            id = "test.missing_fine_tuning_provider"
            name = "Missing fine-tuning provider"
            fine_tuning_supported = True

            def run(self, ctx: OperationContext) -> OperationResult:
                return OperationResult()

        with self.assertRaisesRegex(OperationError, "requires a callable fine_tuning_provider"):
            MissingFineTuningProvider().describe()

        class FineTuningProvider(Operation):
            id = "test.fine_tuning_provider"
            name = "Fine-tuning provider"
            fine_tuning_supported = True

            @staticmethod
            def fine_tuning_provider(*_args, **_kwargs):
                return {"status": "ready"}

            def run(self, ctx: OperationContext) -> OperationResult:
                return OperationResult()

        self.assertTrue(
            FineTuningProvider().describe()["training_capabilities"]["built_in_fine_tuning"]
        )

        class InvalidPortableReplacementBinding(Operation):
            id = "test.invalid_portable_replacement_binding"
            name = "Invalid portable replacement binding"
            input_kinds = {"x": ["test.tensor"]}
            output_kinds = {"y": "test.tensor"}
            params_schema = {
                "type": "object",
                "properties": {
                    "artifact_manifest_path": {"type": "string"},
                    "artifact_entrypoint": {"type": "string"},
                },
                "required": [],
                "additionalProperties": False,
            }
            trained_artifact_abi = {
                "component_id": "candidate",
                "component_role": "candidate",
                "entrypoint_id": "forward",
                "required_operation_inputs": ["x"],
                "inputs": {"x": {"dtype": "float32", "shape": ["batch"]}},
                "outputs": {"y": {"dtype": "float32", "shape": ["batch"]}},
                "binding_params": {
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": "different_entrypoint",
                },
            }

            def run(self, ctx: OperationContext) -> OperationResult:
                return OperationResult()

        with self.assertRaisesRegex(OperationError, "must match entrypoint_id"):
            InvalidPortableReplacementBinding().describe()

        class MissingPortableReplacementRole(InvalidPortableReplacementBinding):
            id = "test.missing_portable_replacement_role"
            trained_artifact_abi = {
                key: value
                for key, value in InvalidPortableReplacementBinding.trained_artifact_abi.items()
                if key != "component_role"
            }

        with self.assertRaisesRegex(
            OperationError,
            r"trained_artifact_abi\.component_role must be non-empty",
        ):
            MissingPortableReplacementRole().describe()

    def test_operation_backend_materialization_metadata_is_normalized_and_validated(self):
        class AliasBackendOperation(Operation):
            id = "test.alias_backend"
            name = "Alias backend"
            backends = {"benchmark_run": ["onnx", "pytorch"], "differentiable_export": "sionna"}
            equivalence = {"type": "numerical", "tolerance": {"atol": 1e-6}}
            formats = {"artifact": "npz", "tensor": "torch.Tensor"}
            materializations = [
                {
                    "runner": "benchmark_run",
                    "backend": "onnx",
                    "implementation": "test implementation",
                    "status": "experimental",
                }
            ]

            def run(self, ctx: OperationContext) -> OperationResult:
                return OperationResult()

        payload = AliasBackendOperation().describe()
        self.assertEqual(payload["backends"]["benchmark_run"], ["onnxruntime", "torch"])
        self.assertEqual(payload["backends"]["differentiable_export"], ["sionna"])
        self.assertEqual(payload["equivalence"]["type"], "numerical")
        self.assertEqual(payload["equivalence"]["tolerance"], {"atol": 1e-6})
        self.assertEqual(payload["materializations"][0]["backend"], "onnxruntime")

        class InvalidBackendOperation(Operation):
            id = "test.invalid_backend"
            name = "Invalid backend"
            backends = {"benchmark_run": ["jax"]}

            def run(self, ctx: OperationContext) -> OperationResult:
                return OperationResult()

        with self.assertRaisesRegex(OperationError, "backends.benchmark"):
            InvalidBackendOperation().describe()

    def test_materialization_registry_resolves_runner_backend_implementations(self):
        registry = build_registry()
        materializations = build_materialization_registry(registry)

        resolved = materializations.resolve("wireless.channel", runner="benchmark_run", backend="numpy")
        self.assertEqual(resolved.operation_id, "wireless.channel")
        self.assertEqual(resolved.runner, "benchmark_run")
        self.assertEqual(resolved.backend, "numpy")
        self.assertEqual(resolved.implementation, "numpy_awgn_matched")
        self.assertIs(resolved.operation, registry.get("wireless.channel"))

        sionna = materializations.resolve(
            "wireless.channel",
            runner="differentiable_export",
            backend="sionna",
        )
        self.assertEqual(sionna.backend, "sionna")
        self.assertEqual(
            sionna.implementation,
            "sionna_awgn_matched_pytorch_module",
        )

        torch_alias = materializations.resolve("wireless.channel", runner="differentiable_export", backend="pytorch")
        self.assertEqual(torch_alias.backend, "torch")
        self.assertEqual(torch_alias.implementation, "torch_awgn_matched_module")

        with self.assertRaisesRegex(OperationError, "No compatible materialization"):
            materializations.resolve("channel.identity_link", runner="differentiable_export", backend="torch")

    def test_operation_runner_support_uses_materialization_contracts(self):
        registry = build_registry()
        wireless = operation_runner_supports(registry.get("wireless.channel").describe())
        self.assertTrue(wireless["benchmark_run"]["supported"])
        self.assertTrue(wireless["dataset_capture"]["supported"])
        self.assertTrue(wireless["differentiable_export"]["supported"])

        crc = operation_runner_supports(registry.get("channel.crc32_check").describe())
        self.assertTrue(crc["benchmark_run"]["supported"])
        self.assertTrue(crc["dataset_capture"]["supported"])
        self.assertFalse(crc["differentiable_export"]["supported"])
        self.assertIn("gradient break", crc["differentiable_export"]["reason"])

    def test_materialization_registry_can_be_built_from_synthetic_operations(self):
        class SyntheticMaterializedOperation(Operation):
            id = "test.synthetic_materialized"
            name = "Synthetic materialized op"
            backends = {"benchmark_run": ["numpy"], "differentiable_export": ["torch", "external"]}
            materializations = [
                {"runner": "benchmark_run", "backend": "numpy", "implementation": "numpy_artifact"},
                {"runner": "differentiable_export", "backend": "torch", "implementation": "torch_module"},
                {"runner": "differentiable_export", "backend": "external", "implementation": "external_module", "status": "experimental"},
            ]

            def run(self, ctx: OperationContext) -> OperationResult:
                return OperationResult()

        registry = OperationRegistry()
        op = SyntheticMaterializedOperation()
        registry.register(op)
        materializations = MaterializationRegistry.from_operation_registry(registry)
        self.assertEqual(
            [item.implementation for item in materializations.list("test.synthetic_materialized", runner="differentiable_export")],
            ["torch_module", "external_module"],
        )
        self.assertEqual(
            materializations.resolve("test.synthetic_materialized", "differentiable_export", "torch").operation,
            op,
        )
        with self.assertRaisesRegex(OperationError, "No compatible materialization"):
            materializations.resolve("test.synthetic_materialized", "differentiable_export", "external")
        external = materializations.resolve(
            "test.synthetic_materialized",
            "differentiable_export",
            "external",
            require_implemented=False,
        )
        self.assertEqual(external.status, "experimental")

        class InvalidEquivalenceOperation(Operation):
            id = "test.invalid_equivalence"
            name = "Invalid equivalence"
            equivalence = {"type": "identical"}

            def run(self, ctx: OperationContext) -> OperationResult:
                return OperationResult()

        with self.assertRaisesRegex(OperationError, "equivalence.type"):
            InvalidEquivalenceOperation().describe()

    def test_operation_differentiability_metadata_is_normalized_and_validated(self):
        class AliasDifferentiabilityOperation(Operation):
            id = "test.alias_differentiability"
            name = "Alias differentiability"
            differentiability = {
                "framework": "pytorch",
                "gradient": "full",
                "trainable_params": "true",
                "exportable": "false",
            }

            def run(self, ctx: OperationContext) -> OperationResult:
                return OperationResult()

        payload = AliasDifferentiabilityOperation().describe()
        self.assertEqual(payload["differentiability"]["framework"], "torch")
        self.assertTrue(payload["differentiability"]["trainable_params"])
        self.assertFalse(payload["differentiability"]["exportable"])

        class InvalidDifferentiabilityOperation(Operation):
            id = "test.invalid_differentiability"
            name = "Invalid differentiability"
            differentiability = {
                "framework": "jax",
                "gradient": "full",
                "trainable_params": False,
                "exportable": False,
            }

            def run(self, ctx: OperationContext) -> OperationResult:
                return OperationResult()

        with self.assertRaisesRegex(OperationError, "differentiability.framework"):
            InvalidDifferentiabilityOperation().describe()

    def test_external_adapter_manifest_can_override_differentiability_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_path = root / "adapter.py"
            adapter_path.write_text(
                "def encode_bits(images, params):\n"
                "    import numpy as np\n"
                "    return {'array': np.zeros(8, dtype=np.uint8), 'metadata': {'bit_count': 8}}\n",
                encoding="utf-8",
            )
            manifest_path = root / "noema_adapter.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "name: differentiable_adapter",
                        "operations:",
                        "  - id: model.diff_adapter_encode_bits",
                        "    name: Differentiable adapter encode bits",
                        "    wraps: model.external_encode_bits",
                        "    adapter:",
                        "      path: adapter.py",
                        "      callable: encode_bits",
                        "      call_style: array_params",
                        "    differentiability:",
                        "      framework: torch",
                        "      gradient: surrogate",
                        "      trainable_params: true",
                        "      exportable: true",
                        "      reason: Adapter declares a straight-through estimator.",
                    ]
                ),
                encoding="utf-8",
            )
            registry = build_registry([manifest_path])
            description = registry.get("model.diff_adapter_encode_bits").describe()
            self.assertEqual(description["differentiability"]["framework"], "torch")
            self.assertEqual(description["differentiability"]["gradient"], "surrogate")
            self.assertTrue(description["differentiability"]["trainable_params"])
            self.assertTrue(description["differentiability"]["exportable"])

    def test_differentiable_inspect_reports_full_gradient_possible_for_jscc_path(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "synthetic_deepjscc",
                "steps": [
                    {"id": "data", "op": "source.synthetic_images"},
                    {"id": "sender", "op": "model.synthetic_encoder", "inputs": {"images": "data.images"}},
                    {"id": "sionna_awgn", "op": "wireless.sionna_awgn", "inputs": {"symbols": "sender.symbols"}},
                    {"id": "receiver", "op": "model.synthetic_decoder", "inputs": {"symbols": "sionna_awgn.symbols"}},
                    {"id": "evaluation", "op": "metrics.synthetic_loss", "inputs": {"reference": "data.images", "reconstruction": "receiver.images"}},
                ],
            }
        )
        report = inspect_training_feasibility(recipe, _synthetic_training_registry())
        self.assertEqual(report["status"], "full_gradient_possible")
        self.assertEqual(report["recommended_mode"], "differentiable_export")
        self.assertEqual(report["gradient_breaks"], [])
        self.assertEqual(report["gradient_path"], ["sender", "sionna_awgn", "receiver", "loss"])
        self.assertEqual(report["portable_artifact_return_blocks"], ["sender", "receiver"])
        self.assertEqual(report["interface_only_trainable_blocks"], [])

    def test_differentiable_inspect_reports_selected_paths_for_deepjscc(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "synthetic_deepjscc_path_aware",
                "steps": [
                    {"id": "data", "op": "source.synthetic_images"},
                    {"id": "sender", "op": "model.synthetic_encoder", "inputs": {"images": "data.images"}},
                    {"id": "sionna_awgn", "op": "wireless.sionna_awgn", "inputs": {"symbols": "sender.symbols"}},
                    {"id": "receiver", "op": "model.synthetic_decoder", "inputs": {"symbols": "sionna_awgn.symbols"}},
                    {"id": "evaluation", "op": "metrics.synthetic_loss", "inputs": {"reference": "data.images", "reconstruction": "receiver.images"}},
                ],
            }
        )
        report = inspect_training_feasibility(
            recipe,
            _synthetic_training_registry(),
            optimizable_steps="sender,receiver",
            loss="evaluation",
        )
        self.assertEqual(report["status"], "full_gradient_possible")
        self.assertEqual(report["selected_optimizable_steps"], ["sender", "receiver"])
        self.assertEqual(report["selected_loss_steps"], ["evaluation"])
        paths = {item["optimizable_step_id"]: item for item in report["paths"]}
        self.assertEqual(paths["sender"]["source_to_optimizable_path"], ["data", "sender"])
        self.assertEqual(paths["sender"]["optimizable_to_loss_path"], ["sender", "sionna_awgn", "receiver", "evaluation"])
        self.assertEqual(paths["sender"]["optimizable_channel_loss_path"], ["sender", "sionna_awgn", "receiver", "evaluation"])
        self.assertEqual(paths["receiver"]["source_to_optimizable_path"], ["data", "sender", "sionna_awgn", "receiver"])
        self.assertFalse(report["path_issues"])

    def test_differentiable_inspect_reports_receiver_only_when_hard_block_breaks_sender_gradient(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "synthetic_receiver_training",
                "steps": [
                    {"id": "data", "op": "source.synthetic_images"},
                    {"id": "sender", "op": "model.synthetic_encoder", "inputs": {"images": "data.images"}},
                    {"id": "sionna_awgn", "op": "wireless.sionna_awgn", "inputs": {"symbols": "sender.symbols"}},
                    {"id": "hard_demod", "op": "channel.hard_demod", "inputs": {"symbols": "sionna_awgn.symbols"}},
                    {"id": "receiver", "op": "model.synthetic_decoder", "inputs": {"symbols": "hard_demod.symbols"}},
                    {"id": "evaluation", "op": "metrics.synthetic_loss", "inputs": {"reference": "data.images", "reconstruction": "receiver.images"}},
                ],
            }
        )
        report = inspect_training_feasibility(recipe, _synthetic_training_registry())
        self.assertEqual(report["status"], "partial_gradient_possible")
        self.assertEqual(report["recommended_mode"], "receiver_only")
        self.assertEqual(report["gradient_path"], ["receiver", "loss"])
        self.assertEqual([item["step_id"] for item in report["gradient_breaks"]], ["hard_demod"])

    def test_replacement_target_ignores_original_implementation_gradient(self):
        registry = _synthetic_training_registry()
        target = registry.get("model.synthetic_decoder")
        target.differentiability = {
            "framework": "numpy",
            "gradient": "none",
            "trainable_params": False,
            "exportable": False,
            "reason": "The built-in implementation is replaced and is never on the training route.",
        }
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "replacement_target_gradient_is_irrelevant",
                "steps": [
                    {"id": "data", "op": "source.synthetic_images"},
                    {"id": "sender", "op": "model.synthetic_encoder", "inputs": {"images": "data.images"}},
                    {"id": "channel", "op": "wireless.sionna_awgn", "inputs": {"symbols": "sender.symbols"}},
                    {"id": "target", "op": "model.synthetic_decoder", "inputs": {"symbols": "channel.symbols"}},
                    {"id": "evaluation", "op": "metrics.synthetic_loss", "inputs": {"reference": "data.images", "reconstruction": "target.images"}},
                ],
            }
        )
        report = inspect_training_feasibility(
            recipe,
            registry,
            optimizable_steps="target",
            loss="evaluation",
        )
        self.assertIn("target", report["replacement_candidate_blocks"])
        self.assertEqual(report["status"], "full_gradient_possible")
        self.assertFalse(report["paths"][0]["gradient_breaks"])

    def test_differentiable_inspect_reports_exact_breaks_on_selected_paths(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "synthetic_hard_demod_path_aware",
                "steps": [
                    {"id": "data", "op": "source.synthetic_images"},
                    {"id": "sender", "op": "model.synthetic_encoder", "inputs": {"images": "data.images"}},
                    {"id": "sionna_awgn", "op": "wireless.sionna_awgn", "inputs": {"symbols": "sender.symbols"}},
                    {"id": "hard_demod", "op": "channel.hard_demod", "inputs": {"symbols": "sionna_awgn.symbols"}},
                    {"id": "receiver", "op": "model.synthetic_decoder", "inputs": {"symbols": "hard_demod.symbols"}},
                    {"id": "evaluation", "op": "metrics.synthetic_loss", "inputs": {"reference": "data.images", "reconstruction": "receiver.images"}},
                ],
            }
        )
        report = inspect_training_feasibility(
            recipe,
            _synthetic_training_registry(),
            optimizable_steps=["sender", "receiver"],
            loss="evaluation",
        )
        self.assertEqual(report["recommended_mode"], "receiver_only")
        paths = {item["optimizable_step_id"]: item for item in report["paths"]}
        self.assertEqual([item["step_id"] for item in paths["sender"]["gradient_breaks"]], ["hard_demod"])
        self.assertFalse(paths["receiver"]["source_gradient_breaks"])
        self.assertFalse(paths["receiver"]["upstream_gradient_required"])
        self.assertEqual(
            [item["step_id"] for item in paths["receiver"]["upstream_non_differentiable_steps"]],
            ["hard_demod"],
        )
        self.assertEqual([item["step_id"] for item in report["gradient_breaks"]], ["hard_demod"])

    def test_differentiable_inspect_replacement_target_ignores_upstream_gradient_breaks(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "synthetic_neural_receiver_capture",
                "steps": [
                    {"id": "source_bits", "op": "source.synthetic_bits"},
                    {"id": "channel_encoder", "op": "channel.synthetic_encoder", "inputs": {"bits": "source_bits.bits"}},
                    {"id": "modulator", "op": "modulation.synthetic_qpsk", "inputs": {"bits": "channel_encoder.coded_bits"}},
                    {"id": "wireless_channel", "op": "wireless.sionna_awgn", "inputs": {"symbols": "modulator.symbols"}},
                    {
                        "id": "receiver",
                        "op": "model.synthetic_neural_receiver",
                        "inputs": {"features": "wireless_channel.symbols", "target_bits": "channel_encoder.coded_bits"},
                    },
                    {"id": "evaluation", "op": "metrics.synthetic_bit_loss", "inputs": {"reference": "channel_encoder.coded_bits", "candidate": "receiver.bits"}},
                ],
            }
        )
        report = inspect_training_feasibility(
            recipe,
            _synthetic_training_registry(),
            optimizable_steps="receiver",
            loss="evaluation",
        )
        self.assertEqual(report["status"], "full_gradient_possible")
        self.assertEqual(report["recommended_mode"], "differentiable_export")
        self.assertEqual(report["gradient_path"], ["receiver", "loss"])
        self.assertFalse(report["gradient_breaks"])
        tap_by_id = {item["id"]: item for item in report["suggested_capture_taps"]}
        self.assertEqual(tap_by_id["rx_symbols"]["from"], "wireless_channel.symbols")
        self.assertEqual(tap_by_id["target_bits"]["from"], "channel_encoder.coded_bits")

    def test_differentiable_inspect_reports_crc_as_gradient_break(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "synthetic_crc_receiver_training",
                "steps": [
                    {"id": "data", "op": "source.synthetic_images"},
                    {"id": "sender", "op": "model.synthetic_bit_encoder", "inputs": {"images": "data.images"}},
                    {"id": "crc_check", "op": "channel.crc32_check", "inputs": {"bits": "sender.bits"}},
                    {"id": "receiver", "op": "model.synthetic_bit_decoder", "inputs": {"bits": "crc_check.bits"}},
                    {"id": "evaluation", "op": "metrics.synthetic_loss", "inputs": {"reference": "data.images", "reconstruction": "receiver.images"}},
                ],
            }
        )
        report = inspect_training_feasibility(recipe, _synthetic_training_registry())
        self.assertEqual(report["status"], "partial_gradient_possible")
        self.assertEqual(report["recommended_mode"], "receiver_only")
        self.assertEqual(report["gradient_path"], ["receiver", "loss"])
        self.assertEqual([item["step_id"] for item in report["gradient_breaks"]], ["crc_check"])
        self.assertIn("CRC", report["gradient_breaks"][0]["reason"])
        self.assertTrue(report["runner_support"]["summary"]["benchmark_run"]["supported"])
        self.assertIn("can benchmark", report["mode_explanation"])
        self.assertIn("gradient break", report["mode_explanation"])

    def test_differentiable_inspect_reports_capture_only_for_non_differentiable_benchmark(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "synthetic_classical_codec",
                "steps": [
                    {"id": "data", "op": "source.synthetic_images"},
                    {"id": "codec", "op": "codec.synthetic_jpeg", "inputs": {"images": "data.images"}},
                    {"id": "evaluation", "op": "metrics.synthetic_loss", "inputs": {"reference": "data.images", "reconstruction": "codec.images"}},
                ],
            }
        )
        report = inspect_training_feasibility(recipe, _synthetic_training_registry())
        self.assertEqual(report["status"], "dataset_capture_only")
        self.assertEqual(report["recommended_mode"], "dataset_capture")
        self.assertTrue(report["dataset_capture"]["possible"])

    def test_differentiable_inspect_classical_codec_is_not_end_to_end_differentiable_export(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "synthetic_jpeg_not_e2e_trainable",
                "steps": [
                    {"id": "data", "op": "source.synthetic_images"},
                    {"id": "codec", "op": "codec.synthetic_jpeg", "inputs": {"images": "data.images"}},
                    {"id": "evaluation", "op": "metrics.synthetic_loss", "inputs": {"reference": "data.images", "reconstruction": "codec.images"}},
                ],
            }
        )
        report = inspect_training_feasibility(recipe, _synthetic_training_registry(), loss="evaluation")
        self.assertIn(report["recommended_mode"], {"dataset_capture", "benchmark_run"})
        self.assertNotEqual(report["recommended_mode"], "differentiable_export")

    def test_differentiable_inspect_reports_benchmark_only_when_no_training_or_capture_path_exists(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "synthetic_data_only",
                "steps": [{"id": "data", "op": "source.synthetic_images"}],
            }
        )
        report = inspect_training_feasibility(recipe, _synthetic_training_registry())
        self.assertEqual(report["status"], "not_trainable")
        self.assertEqual(report["recommended_mode"], "benchmark_run")

    def test_differentiable_inspect_cli_outputs_json_and_human_reports(self):
        recipe_path = ROOT / "recipes" / "text_bart_jscc_clean.yaml"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["differentiable", "inspect", str(recipe_path), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["recipe"], "text_continuous_symbol_jscc")
        self.assertEqual(payload["recommended_mode"], "dataset_capture")
        self.assertNotIn("sender", payload["optimizable_candidate_blocks"])
        self.assertNotIn("sender", payload["fine_tunable_blocks"])
        self.assertNotIn("receiver", payload["fine_tunable_blocks"])
        self.assertIn("runner_support", payload)
        self.assertIn("mode_explanation", payload)
        self.assertIn("differentiability", build_registry().get("channel.symbol_boundary").describe())

        selected_output = io.StringIO()
        with contextlib.redirect_stdout(selected_output):
            code = main([
                "differentiable",
                "inspect",
                str(recipe_path),
                "--replacement",
                "sender,receiver",
                "--loss",
                "evaluation",
                "--json",
            ])
        self.assertEqual(code, 0)
        selected_payload = json.loads(selected_output.getvalue())
        self.assertEqual(selected_payload["selected_optimizable_steps"], ["sender", "receiver"])
        self.assertEqual(selected_payload["selected_loss_steps"], ["evaluation"])
        self.assertTrue(selected_payload["paths"])
        self.assertTrue(selected_payload["path_issues"])

        human_output = io.StringIO()
        with contextlib.redirect_stdout(human_output):
            code = main(["differentiable", "inspect", str(recipe_path)])
        self.assertEqual(code, 0)
        self.assertIn("Recommendation: dataset_capture", human_output.getvalue())

    def test_disabled_channel_identities_report_honest_gradient_behavior(self):
        operations = build_registry()
        bit_identity = operations.get("channel.identity_link").describe()["differentiability"]
        symbol_identity = operations.get("channel.identity_symbol_link").describe()["differentiability"]
        self.assertEqual(bit_identity["gradient"], "stop")
        self.assertFalse(bit_identity["exportable"])
        self.assertIn("discrete bitstream", bit_identity["reason"])
        self.assertEqual(symbol_identity["gradient"], "full")
        self.assertTrue(symbol_identity["exportable"])

    def test_wireless_realistic_numpy_presets_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            n = np.arange(128, dtype=np.float32)
            symbols = np.exp(1j * 0.13 * n).astype(np.complex64)
            metadata = {"modulation": "qpsk", "bit_count": int(symbols.size * 2)}
            np.savez_compressed(symbols_path, symbols=symbols, metadata_json=json.dumps(metadata))
            ctx = OperationContext(
                recipe_name="test",
                step_id="wireless_channel",
                params={
                    "channel": "ofdm_tdl",
                    "snr_db": 30.0,
                    "wireless_backend": "numpy",
                    "ofdm_fft_size": 32,
                    "num_ofdm_symbols": 4,
                    "seed": 11,
                },
                inputs={"symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata)},
                run_dir=root,
                step_dir=root / "wireless_channel",
            )
            result = WirelessChannelOperation().run(ctx)
            self.assertEqual(result.outputs["rx_symbols"].kind, "channel.rx_symbols.complex_numpy")
            self.assertEqual(result.outputs["rx_symbols"].metadata["wireless_preset"], "ofdm_tdl")
            self.assertEqual(
                result.outputs["rx_symbols"].metadata["wireless_backend"],
                "numpy",
            )
            self.assertEqual(
                result.outputs["rx_symbols"].metadata["data_plane_backend"],
                "python_numpy",
            )
            self.assertEqual(result.metrics["channel.ofdm_fft_size"], 32)
            self.assertEqual(result.metrics["channel.num_ofdm_symbols"], 4)
            self.assertIn("channel.rx_power.average", result.metrics)
            self.assertGreater(result.metrics["channel.rx_power.average"], 0)
            self.assertIn("rx_power_average", result.outputs["rx_symbols"].metadata)
            with np.load(result.outputs["rx_symbols"].path, allow_pickle=False) as payload:
                rx_symbols = payload["symbols"]
            self.assertEqual(rx_symbols.shape, symbols.shape)
            self.assertEqual(rx_symbols.dtype, np.dtype("complex64"))

    def test_flat_rayleigh_reports_antenna_and_equalized_power(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            n = np.arange(64, dtype=np.float32)
            symbols = np.exp(1j * 0.07 * n).astype(np.complex64)
            metadata = {"modulation": "qpsk", "bit_count": int(symbols.size * 2), "power_unit": "normalized"}
            np.savez_compressed(symbols_path, symbols=symbols, metadata_json=json.dumps(metadata))
            ctx = OperationContext(
                recipe_name="test",
                step_id="wireless_channel",
                params={
                    "channel": "flat_rayleigh",
                    "snr_db": 12.0,
                    "wireless_backend": "numpy",
                    "seed": 11,
                },
                inputs={"symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata)},
                run_dir=root,
                step_dir=root / "wireless_channel",
            )
            result = WirelessChannelOperation().run(ctx)
            self.assertIn("channel.rx_antenna_power.average", result.metrics)
            self.assertIn("channel.rx_equalized_power.average", result.metrics)
            self.assertIn("channel.rx_output_power.average", result.metrics)
            self.assertIn("channel.gain.average", result.metrics)
            self.assertGreater(result.metrics["channel.rx_antenna_power.average"], 0)
            self.assertGreater(result.metrics["channel.rx_equalized_power.average"], 0)
            self.assertAlmostEqual(
                result.metrics["channel.rx_output_power.average"],
                result.metrics["channel.rx_equalized_power.average"],
                places=6,
            )
            metadata_out = result.outputs["rx_symbols"].metadata
            self.assertTrue(metadata_out["channel_equalized"])
            self.assertEqual(metadata_out["equalizer"], "perfect_csi_one_tap")
            self.assertIn("rx_antenna_power_average", metadata_out)
            self.assertIn("rx_equalized_power_average", metadata_out)

    def test_fixed_noise_variance_reports_reference_and_effective_snr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            symbols = np.full(256, np.sqrt(2.0) + 0j, dtype=np.complex64)
            metadata = {"power_unit": "normalized"}
            np.savez_compressed(symbols_path, symbols=symbols, metadata_json=json.dumps(metadata))
            result = WirelessChannelOperation().run(
                OperationContext(
                    recipe_name="fixed_noise",
                    step_id="wireless_channel",
                    params={
                        "channel": "awgn",
                        "noise_mode": "fixed_variance",
                        "noise_variance": 0.5,
                        "wireless_backend": "numpy",
                    },
                    inputs={"symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata)},
                    run_dir=root,
                    step_dir=root / "wireless_channel",
                )
            )
            self.assertAlmostEqual(result.metrics["channel.noise_variance"], 0.5, places=7)
            self.assertAlmostEqual(result.metrics["channel.reference_snr_db"], 10.0 * np.log10(2.0), places=6)
            self.assertAlmostEqual(result.metrics["channel.tx_effective_snr_db"], 10.0 * np.log10(4.0), places=6)
            self.assertEqual(result.outputs["rx_symbols"].metadata["noise_mode"], "fixed_variance")

    def test_forced_sionna_backend_requires_optional_dependency(self):
        if importlib.util.find_spec("sionna") is not None:
            self.skipTest("Sionna is installed in this environment")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            metadata = {"modulation": "bpsk", "bit_count": 8}
            np.savez_compressed(symbols_path, symbols=np.ones(8, dtype=np.complex64), metadata_json=json.dumps(metadata))
            ctx = OperationContext(
                recipe_name="test",
                step_id="wireless_channel",
                params={"channel": "awgn", "snr_db": 12.0, "wireless_backend": "sionna"},
                inputs={"symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata)},
                run_dir=root,
                step_dir=root / "wireless_channel",
            )
            with self.assertRaisesRegex(OperationError, "uv sync --extra wireless"):
                WirelessChannelOperation().run(ctx)

    def test_suites_catalog_loads_active_and_planned_suites(self):
        catalog = load_suites_catalog()
        self.assertEqual(catalog.schema_version, 1)
        self.assertIn("semantic_comm", catalog.suites)
        self.assertEqual([suite.id for suite in catalog.active_suites()], ["semantic_comm"])
        semantic = catalog.suite("semantic_comm")
        self.assertIsNotNone(semantic)
        self.assertEqual(semantic.status, "active")
        self.assertIn("image_reconstruction", semantic.supported_tasks)
        self.assertTrue(any(pack.path == "benchmarks/benchmark_v1/kodak_image_reconstruction_v1.yaml" for pack in semantic.benchmark_packs))
        neural_receiver = catalog.suite("neural_receiver")
        self.assertIsNotNone(neural_receiver)
        self.assertEqual(neural_receiver.status, "experimental")
        self.assertIn("neural_receiver_demapping", neural_receiver.supported_tasks)
        self.assertTrue(
            any(
                pack.path == "benchmarks/neural_receiver_ai_phy/qpsk_awgn_receiver_v1.yaml"
                for pack in neural_receiver.benchmark_packs
            )
        )
        expected_experimental = {
            "mimo_ofdm": "mimo_ofdm.channel_estimation_v1",
            "channel_estimation": "channel_estimation.pilot_awgn_v1",
            "beamforming_precoding": "beamforming_precoding.beam_selection_v1",
            "localization_sensing": "localization_sensing.range_localization_v1",
            "resource_allocation": "resource_allocation.power_allocation_v1",
        }
        for suite_id, pack_id in expected_experimental.items():
            suite = catalog.suite(suite_id)
            self.assertIsNotNone(suite)
            self.assertEqual(suite.status, "experimental")
            self.assertTrue(suite.supported_tasks)
            self.assertIn(pack_id, {pack.id for pack in suite.benchmark_packs})

    def test_suites_catalog_references_existing_benchmark_packs(self):
        catalog = load_suites_catalog()
        catalog_ids = set(catalog.suites)
        for suite in catalog.suites.values():
            for pack in suite.benchmark_packs:
                self.assertTrue((ROOT / pack.path).exists(), pack.path)

        for recipe_path in sorted((ROOT / "recipes").glob("*.yaml")):
            recipe = load_recipe(recipe_path)
            self.assertIn(recipe.suite.get("id"), catalog_ids, recipe_path.name)

        from noema_lab.core.benchmarks import load_benchmark_pack

        for pack_path in sorted((ROOT / "benchmarks").rglob("*.yaml")):
            pack = load_benchmark_pack(pack_path)
            self.assertIn(pack.suite.get("id"), catalog_ids, str(pack_path.relative_to(ROOT)))

    def test_suite_cli_lists_shows_and_lists_benchmarks(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["suite", "list"])
        self.assertEqual(code, 0)
        self.assertIn("active\tsemantic_comm\tSemantic Communication", output.getvalue())
        self.assertIn("experimental\tchannel_estimation\tChannel Estimation", output.getvalue())
        self.assertIn("experimental\tbeamforming_precoding\tBeamforming / Precoding", output.getvalue())

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["suite", "list", "--json"])
        self.assertEqual(code, 0)
        listed = json.loads(output.getvalue())
        self.assertIn("semantic_comm", {suite["id"] for suite in listed["suites"]})

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["suite", "show", "semantic_comm", "--json"])
        self.assertEqual(code, 0)
        suite = json.loads(output.getvalue())
        self.assertEqual(suite["id"], "semantic_comm")
        self.assertEqual(suite["status"], "active")
        self.assertTrue(suite["benchmark_packs"])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["suite", "benchmarks", "semantic_comm", "--json"])
        self.assertEqual(code, 0)
        benchmarks = json.loads(output.getvalue())
        self.assertEqual(benchmarks["suite"]["id"], "semantic_comm")
        self.assertIn("benchmark_v1.image_reconstruction.kodak", {item["id"] for item in benchmarks["benchmarks"]})

        error_output = io.StringIO()
        with contextlib.redirect_stderr(error_output):
            code = main(["suite", "show", "missing_suite"])
        self.assertEqual(code, 1)
        self.assertIn("unknown suite: missing_suite", error_output.getvalue())

    def test_recipe_suite_metadata_round_trips(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "suite_metadata_smoke",
                "suite": {
                    "id": "semantic_comm",
                    "name": "Semantic Communication",
                    "status": "active",
                    "version": "v1",
                },
                "steps": [{"id": "data", "op": "source.synthetic_images"}],
            }
        )
        self.assertEqual(recipe.suite["id"], "semantic_comm")
        self.assertEqual(recipe.to_dict()["suite"]["status"], "active")

    def test_recipe_validates_and_graphs(self):
        registry = build_registry()
        recipe = load_recipe(ROOT / "recipes" / "compressai_kodak_default.yaml")
        validate_recipe_against_registry(recipe, registry)
        graph = recipe_graph(recipe, registry)
        self.assertEqual(graph["recipe"], "image_source_coded_link")
        self.assertEqual(len(graph["nodes"]), 18)
        self.assertEqual(len(graph["edges"]), 19)
        data_node = next(node for node in graph["nodes"] if node["id"] == "data")
        self.assertIn("differentiability", data_node)
        self.assertIn("materializations", data_node)
        self.assertTrue(data_node["runner_support"]["benchmark_run"]["supported"])
        self.assertTrue(data_node["runner_support"]["dataset_capture"]["supported"])
        self.assertFalse(data_node["runner_support"]["differentiable_export"]["supported"])

    def test_recipe_lint_accepts_canonical_bit_transport_spine(self):
        registry = build_registry()
        recipe = load_recipe(ROOT / "recipes" / "text_semantic_utf8_clean.yaml")
        report = lint_recipe_invariants(recipe, registry)
        self.assertEqual(report["status"], "passed")
        self.assertIn("bit_transport", report["profiles"])
        self.assertEqual(report["error_count"], 0)
        self.assertTrue(report["runner_support"]["summary"]["benchmark_run"]["supported"])
        self.assertTrue(report["runner_support"]["summary"]["dataset_capture"]["supported"])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                [
                    "recipe",
                    "lint",
                    str(ROOT / "recipes" / "text_semantic_utf8_clean.yaml"),
                    "--json",
                ]
            )
        self.assertEqual(code, 0)
        cli_report = json.loads(output.getvalue())
        self.assertEqual(cli_report["status"], "passed")
        self.assertIn("bit_transport", cli_report["profiles"])
        self.assertTrue(cli_report["runner_support"]["summary"]["benchmark_run"]["supported"])

    def test_recipe_lint_accepts_canonical_symbol_transport_spine(self):
        registry = build_registry()
        recipe = load_recipe(ROOT / "recipes" / "text_bart_jscc_clean.yaml")
        report = lint_recipe_invariants(recipe, registry)
        self.assertEqual(report["status"], "passed")
        self.assertIn("symbol_transport", report["profiles"])
        self.assertNotIn("bit_transport", report["profiles"])
        self.assertEqual(report["error_count"], 0)
        self.assertTrue(report["runner_support"]["summary"]["benchmark_run"]["supported"])
        self.assertTrue(report["runner_support"]["summary"]["differentiable_export"]["supported"])

    def test_recipe_lint_accepts_identity_phy_disabled_channel_spine(self):
        registry = build_registry()
        data = load_recipe(ROOT / "recipes" / "text_semantic_utf8_clean.yaml").to_dict()
        data["name"] = "text_semantic_utf8_identity_phy"
        data.setdefault("metadata", {})["channel_mode"] = "identity_phy"
        data["metadata"]["channel_disabled_mode"] = "identity_phy"
        steps = list(data["steps"])
        wireless_index = next(index for index, step in enumerate(steps) if step["id"] == "wireless_channel")
        steps[wireless_index:wireless_index + 1] = [
            {
                "id": "modulator",
                "op": "modulation.identity_modulate",
                "inputs": {"bits": "tx_bit_boundary.bits"},
                "params": {"label": "identity_phy_modulator"},
            },
            {
                "id": "tx_power",
                "op": "channel.symbol_power_identity",
                "inputs": {"symbols": "modulator.symbols"},
                "params": {"label": "tx_power_off"},
            },
            {
                "id": "wireless_channel",
                "op": "channel.identity_symbol_link",
                "inputs": {"symbols": "tx_power.symbols"},
                "params": {"label": "identity_phy_channel"},
            },
            {
                "id": "demodulator",
                "op": "demodulation.identity_demodulate",
                "inputs": {"rx_symbols": "wireless_channel.symbols"},
                "params": {"label": "identity_phy_demodulator"},
            },
        ]
        for step in steps:
            if step["id"] == "rx_bit_boundary":
                step["inputs"] = {"bits": "demodulator.bits"}
                step["params"] = {"label": "identity_phy_output", "role": "identity_phy_output"}
        data["steps"] = steps
        recipe = recipe_from_dict(data)
        validate_recipe_against_registry(recipe, registry)
        graph = recipe_graph(recipe, registry)
        self.assertIn("modulator", {node["id"] for node in graph["nodes"]})
        self.assertIn("tx_power", {node["id"] for node in graph["nodes"]})
        self.assertIn("demodulator", {node["id"] for node in graph["nodes"]})
        report = lint_recipe_invariants(recipe, registry)
        self.assertEqual(report["status"], "passed")
        self.assertIn("bit_transport", report["profiles"])
        self.assertEqual(report["error_count"], 0)

    def test_recipe_lint_rejects_missing_bit_count_anchor(self):
        registry = build_registry()
        recipe = load_recipe(ROOT / "recipes" / "text_semantic_utf8_clean.yaml")
        data = recipe.to_dict()
        data["name"] = "broken_text_missing_bit_match"
        data["steps"] = [
            step for step in data["steps"] if step["id"] != "channel_bit_count_match"
        ]
        broken = recipe_from_dict(data)
        report = lint_recipe_invariants(broken, registry)
        self.assertEqual(report["status"], "failed")
        self.assertIn("required_step_missing", {issue["code"] for issue in report["issues"]})

        with tempfile.TemporaryDirectory() as tmp:
            recipe_path = Path(tmp) / "broken_recipe.json"
            recipe_path.write_text(json.dumps(data), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(["recipe", "lint", str(recipe_path), "--json"])
            self.assertEqual(code, 1)
            cli_report = json.loads(stdout.getvalue())
            self.assertEqual(cli_report["status"], "failed")

    def test_recipe_lint_rejects_benchmark_unsupported_materialization(self):
        class TrainOnlyOperation(Operation):
            id = "test.train_only_block"
            name = "Train-only block"
            backends = {"benchmark_run": [], "dataset_capture": [], "differentiable_export": ["torch"]}
            differentiability = {
                "framework": "torch",
                "gradient": "full",
                "trainable_params": True,
                "exportable": True,
            }
            output_kinds = {"value": "metric.scalar"}

            def run(self, ctx: OperationContext) -> OperationResult:
                return OperationResult()

        registry = OperationRegistry()
        registry.register(TrainOnlyOperation())
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "train_only_not_benchmarkable",
                "steps": [{"id": "train_block", "op": "test.train_only_block"}],
            }
        )
        report = lint_recipe_invariants(recipe, registry)
        self.assertEqual(report["status"], "failed")
        self.assertIn("benchmark_runner_unsupported", {issue["code"] for issue in report["issues"]})
        self.assertFalse(report["runner_support"]["summary"]["benchmark_run"]["supported"])

    def test_recipe_specs_are_inferred_and_exposed_by_cli(self):
        registry = build_registry()
        recipe = load_recipe(ROOT / "recipes" / "compressai_kodak_default.yaml")
        validate_recipe_against_registry(recipe, registry)
        specs = research_specs_from_recipe(recipe)
        self.assertEqual(specs["source"], "inferred_from_recipe")
        self.assertEqual(specs["dataset"]["id"], "kodak")
        self.assertEqual(specs["dataset"]["modality"], "image")
        self.assertEqual(specs["task"]["id"], "image_reconstruction")
        self.assertEqual(specs["benchmark"]["dataset"], "kodak")
        self.assertIn("quality.psnr_db", specs["task"]["metrics"])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["recipe", "specs", str(ROOT / "recipes" / "compressai_kodak_default.yaml")])
        self.assertEqual(code, 0)
        cli_specs = json.loads(output.getvalue())
        self.assertEqual(cli_specs["task"]["kind"], "reconstruction")
        self.assertEqual(cli_specs["catalog_validation"]["status"], "valid")

        catalog_output = io.StringIO()
        with contextlib.redirect_stdout(catalog_output):
            code = main(["research", "show", "task", "image_reconstruction"])
        self.assertEqual(code, 0)
        catalog_task = json.loads(catalog_output.getvalue())
        self.assertEqual(catalog_task["status"], "supported")
        self.assertIn("quality.psnr_db", catalog_task["metrics"])

        validation_output = io.StringIO()
        with contextlib.redirect_stdout(validation_output):
            code = main(
                [
                    "research",
                    "validate-recipe",
                    str(ROOT / "recipes" / "compressai_kodak_default.yaml"),
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(validation_output.getvalue())["catalog_validation"]["status"], "valid")

    def test_recipe_research_metadata_validation_rejects_bad_values(self):
        registry = build_registry()
        bad_seed = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "bad_seed",
                "metadata": {"seed": -1},
                "steps": [{"id": "data", "op": "source.local_npz_images", "params": {"path": "x"}}],
            }
        )
        with self.assertRaises(Exception):
            validate_recipe_against_registry(bad_seed, registry)

        bad_metric = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "bad_metric",
                "metadata": {
                    "research": {
                        "dataset": {"id": "toy", "modality": "image"},
                        "task": {"id": "task", "kind": "reconstruction"},
                        "metrics": [{"id": "metric", "direction": "best"}],
                    }
                },
                "steps": [{"id": "data", "op": "source.local_npz_images", "params": {"path": "x"}}],
            }
        )
        with self.assertRaises(Exception):
            validate_recipe_against_registry(bad_metric, registry)

    def test_recipe_run_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = np.zeros((1, 24, 24, 3), dtype=np.uint8)
            input_path = root / "images.npz"
            np.savez_compressed(input_path, images=images)
            recipe_path = root / "jpeg_recipe.yaml"
            recipe_path.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "name: jpeg_cli_smoke",
                        "steps:",
                        "  - id: data",
                        "    op: source.local_npz_images",
                        "    params:",
                        f"      path: {input_path}",
                        "      array: images",
                        "  - id: sender",
                        "    op: model.jpeg_encode",
                        "    inputs:",
                        "      images: data.images",
                        "    params:",
                        "      quality: 75",
                        "  - id: receiver",
                        "    op: model.jpeg_decode",
                        "    inputs:",
                        "      bits: sender.bits",
                        "  - id: evaluation",
                        "    op: metrics.image_reconstruction",
                        "    inputs:",
                        "      reference: data.images",
                        "      reconstruction: receiver.images",
                    ]
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(
                    [
                        "--workspace",
                        tmp,
                        "recipe",
                        "run",
                        str(recipe_path),
                    ]
                )
            self.assertEqual(code, 0)
            runs = LocalStore(Path(tmp)).list_runs()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["status"], "completed")
            summary = LocalStore(Path(tmp)).get_run(runs[0]["run_id"])
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["manifest"], "manifest.json")
            self.assertIn("recipe_sha256", summary)
            self.assertIn("seed_policy", summary)
            self.assertEqual(summary["steps"][-1]["op"], "metrics.image_reconstruction")
            self.assertIn("quality.psnr_db", summary["steps"][-1]["metrics"])
            image_metadata = summary["steps"][0]["outputs"]["images"]["metadata"]
            self.assertEqual(image_metadata["arrays"]["images"]["dtype"], "uint8")
            self.assertIn("timing.step.wall_time_s", summary["steps"][0]["metrics"])
            manifest = LocalStore(Path(tmp)).read_json(
                Path(tmp) / "runs" / runs[0]["run_id"] / "manifest.json"
            )
            self.assertEqual(manifest["kind"], "noema.run_manifest")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["recipe"]["sha256"], summary["recipe_sha256"])
            self.assertEqual(manifest["recipe"]["research"]["task"]["id"], "image_reconstruction")
            self.assertIn("python", manifest["environment"])
            self.assertIn("pyproject.toml", manifest["environment"]["project_files"])
            self.assertIn("sha256", manifest["operation_contracts"])
            self.assertIn("source.local_npz_images", manifest["operation_contracts"]["operations"])
            self.assertGreaterEqual(len(manifest["artifacts"]), 4)
            first_artifact = manifest["artifacts"][0]
            self.assertEqual(first_artifact["kind"], "image.batch.numpy")
            self.assertEqual(first_artifact["dtype"], "uint8")
            self.assertEqual(first_artifact["shape"], [1, 24, 24, 3])
            manifest_output = io.StringIO()
            with contextlib.redirect_stdout(manifest_output):
                manifest_code = main(
                    [
                        "--workspace",
                        tmp,
                        "runs",
                        "manifest",
                        runs[0]["run_id"],
                    ]
                )
            self.assertEqual(manifest_code, 0)
            self.assertEqual(json.loads(manifest_output.getvalue())["kind"], "noema.run_manifest")

    def test_operation_context_derives_independent_step_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx_a = OperationContext(
                recipe_name="seeded_recipe",
                step_id="sender_noise",
                params={},
                inputs={},
                run_dir=root,
                step_dir=root / "a",
                master_seed=123,
            )
            ctx_b = OperationContext(
                recipe_name="seeded_recipe",
                step_id="receiver_noise",
                params={},
                inputs={},
                run_dir=root,
                step_dir=root / "b",
                master_seed=123,
            )
            self.assertEqual(
                ctx_a.seed("semantic_indices"),
                derive_seed(123, "seeded_recipe", "sender_noise", "semantic_indices"),
            )
            self.assertNotEqual(ctx_a.seed("semantic_indices"), ctx_b.seed("semantic_indices"))
            explicit = OperationContext(
                recipe_name="seeded_recipe",
                step_id="sender_noise",
                params={"seed": 7},
                inputs={},
                run_dir=root,
                step_dir=root / "c",
                master_seed=123,
            )
            self.assertEqual(explicit.seed("semantic_indices"), 7)

    def test_image_dataset_repeat_count_expands_batch_for_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = build_registry()
            recipe = recipe_from_dict(
                {
                    "schema_version": 1,
                    "name": "repeat_source",
                    "steps": [
                        {
                            "id": "data",
                            "op": "source.image_dataset",
                            "inputs": {},
                            "params": {
                                "dataset": "kodak",
                                "dataset_dir": ".noema/datasets/kodak",
                                "image_ids": "kodim01,kodim02",
                                "crop_size": 64,
                                "repeat_count": 3,
                            },
                        }
                    ],
                }
            )
            store = LocalStore(Path(tmp))
            run_dir = LocalExecutor(registry, store).run(recipe)
            summary = store.get_run(run_dir.name)
            metadata = summary["steps"][0]["outputs"]["images"]["metadata"]
            self.assertEqual(metadata["base_image_count"], 2)
            self.assertEqual(metadata["repeat_count"], 3)
            self.assertEqual(metadata["shape"][0], 6)
            self.assertEqual(summary["steps"][0]["metadata"]["image_count"], 6)

    def test_image_dataset_allows_mixed_uncropped_kodak_shapes(self):
        from noema_lab.ops.source import image_dataset as image_dataset_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "kodak"
            dataset_dir.mkdir()
            _write_rgb_png(dataset_dir / "kodim01.png", np.zeros((5, 7, 3), dtype=np.uint8))
            _write_rgb_png(dataset_dir / "kodim04.png", np.full((6, 4, 3), 128, dtype=np.uint8))
            original_download = image_dataset_module.download_kodak_dataset
            image_dataset_module.download_kodak_dataset = lambda _path: None
            try:
                registry = build_registry()
                recipe = recipe_from_dict(
                    {
                        "schema_version": 1,
                        "name": "mixed_uncropped_source",
                        "steps": [
                            {
                                "id": "data",
                                "op": "source.image_dataset",
                                "inputs": {},
                                "params": {
                                    "dataset": "kodak",
                                    "dataset_dir": str(dataset_dir),
                                    "image_ids": "kodim01,kodim04",
                                    "crop_size": 0,
                                },
                            }
                        ],
                    }
                )
                store = LocalStore(root / "workspace")
                run_dir = LocalExecutor(registry, store).run(recipe)
            finally:
                image_dataset_module.download_kodak_dataset = original_download

            summary = store.get_run(run_dir.name)
            metadata = summary["steps"][0]["outputs"]["images"]["metadata"]
            self.assertEqual(metadata["shape"], [2, 6, 7, 3])
            self.assertEqual(metadata["original_shapes"], [[1, 5, 7, 3], [1, 6, 4, 3]])
            self.assertTrue(metadata["padded_to_common_shape"])

    def test_codec_timing_metrics_are_recorded_for_model_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = np.zeros((2, 24, 24, 3), dtype=np.uint8)
            images[0, 4:20, 4:20, :] = [48, 160, 220]
            input_path = root / "images.npz"
            np.savez_compressed(input_path, images=images)
            registry = build_registry()
            recipe = recipe_from_dict(
                {
                    "schema_version": 1,
                    "name": "timed_jpeg_codec",
                    "metadata": {
                        "codec_timing": {
                            "enabled": True,
                            "warmup_runs": 0,
                            "timed_runs": 1,
                            "runner": "operation",
                        }
                    },
                    "steps": [
                        {
                            "id": "data",
                            "op": "source.local_npz_images",
                            "inputs": {},
                            "params": {"path": str(input_path), "array": "images"},
                        },
                        {
                            "id": "sender",
                            "op": "model.jpeg_encode",
                            "inputs": {"images": "data.images"},
                            "params": {"quality": 75},
                        },
                    ],
                }
            )
            store = LocalStore(root)
            events = []
            run_dir = LocalExecutor(registry, store).run(recipe, event_sink=events.append)
            summary = store.get_run(run_dir.name)
            self.assertIn("memory.run.peak_rss_bytes", summary["metrics"])
            self.assertGreaterEqual(summary["metrics"]["memory.run.peak_rss_bytes"], 0)
            sender = summary["steps"][1]
            self.assertNotIn("codec_timing.timed_mean_s", sender["metrics"])
            timing = sender["metadata"]["codec_timing"]
            self.assertEqual(timing["runner"], "local_python")
            self.assertEqual(timing["scope"], "per_example_stage")
            self.assertEqual(timing["measurement_protocol"], "single_execution_per_example")
            measurements = timing["measurements"]
            self.assertGreaterEqual(len(measurements), 4)
            self.assertEqual(
                {record["stage"] for record in measurements},
                {"encoder.payload_encode", "encoder.total"},
            )
            self.assertEqual(
                {record["example_index"] for record in measurements if "example_index" in record},
                {0, 1},
            )
            progress_events = [
                event for event in events
                if event["kind"] == "step_progress" and event["step_id"] == "sender"
            ]
            encoding_events = [event for event in progress_events if event["phase"] == "encoding"]
            self.assertGreaterEqual(len(encoding_events), 3)
            self.assertEqual({event["unit"] for event in encoding_events}, {"examples"})
            self.assertEqual(encoding_events[-1]["completed"], 2)
            self.assertEqual(encoding_events[-1]["total"], 2)

    def test_jpeg_codec_round_trip_records_real_payload_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = np.zeros((2, 24, 24, 3), dtype=np.uint8)
            images[0, :, :, 0] = np.arange(24, dtype=np.uint8)[None, :] * 8
            images[0, :, :, 1] = np.arange(24, dtype=np.uint8)[:, None] * 8
            images[1, 4:20, 4:20, :] = [32, 180, 220]
            input_path = root / "images.npz"
            np.savez_compressed(input_path, images=images)
            registry = build_registry()
            recipe = recipe_from_dict(
                {
                    "schema_version": 1,
                    "name": "jpeg_round_trip",
                    "steps": [
                        {
                            "id": "data",
                            "op": "source.local_npz_images",
                            "inputs": {},
                            "params": {"path": str(input_path), "array": "images"},
                        },
                        {
                            "id": "sender",
                            "op": "model.jpeg_encode",
                            "inputs": {"images": "data.images"},
                            "params": {"quality": 70, "subsampling": "420"},
                        },
                        {
                            "id": "receiver",
                            "op": "model.jpeg_decode",
                            "inputs": {"bits": "sender.bits"},
                            "params": {},
                        },
                        {
                            "id": "evaluation",
                            "op": "metrics.image_reconstruction",
                            "inputs": {
                                "reference": "data.images",
                                "reconstruction": "receiver.images",
                            },
                            "params": {},
                        },
                    ],
                }
            )
            store = LocalStore(root)
            run_dir = LocalExecutor(registry, store).run(recipe)
            summary = store.get_run(run_dir.name)
            steps = {step["id"]: step for step in summary["steps"]}
            bit_metadata = steps["sender"]["outputs"]["bits"]["metadata"]
            self.assertEqual(bit_metadata["codec"], "jpeg")
            self.assertGreater(bit_metadata["bit_count"], 0)
            self.assertEqual(bit_metadata["bit_count"], bit_metadata["byte_count"] * 8)
            self.assertEqual(bit_metadata["payload_bit_count"], bit_metadata["bit_count"])
            self.assertEqual(bit_metadata["bit_storage"], "unpacked_uint8")
            self.assertEqual(bit_metadata["source_bit_storage"], "packed_bytes")
            self.assertEqual(
                steps["sender"]["outputs"]["bits"]["kind"],
                "channel.payload_bits.numpy",
            )
            self.assertEqual(
                steps["receiver"]["outputs"]["images"]["metadata"]["shape"],
                [2, 24, 24, 3],
            )
            self.assertIn("quality.psnr_db", steps["evaluation"]["metrics"])

    def test_external_packed_byte_codec_counts_channel_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_path = root / "adapter.py"
            adapter_path.write_text(
                "\n".join(
                    [
                        "import numpy as np",
                        "",
                        "def encode(images, params):",
                        "    return {'array': b'\\xa0', 'metadata': {'bit_count': 3}}",
                        "",
                        "def decode(payload, params):",
                        "    if payload.shape != (1,) or int(payload[0]) != 160:",
                        "        raise RuntimeError('expected packed byte 0xa0')",
                        "    return np.zeros((1, 8, 8, 3), dtype=np.uint8)",
                    ]
                ),
                encoding="utf-8",
            )
            images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
            input_path = root / "images.npz"
            np.savez_compressed(input_path, images=images)
            registry = build_registry()
            recipe = recipe_from_dict(
                {
                    "schema_version": 1,
                    "name": "external_packed_bytes",
                    "steps": [
                        {
                            "id": "data",
                            "op": "source.local_npz_images",
                            "inputs": {},
                            "params": {"path": str(input_path), "array": "images"},
                        },
                        {
                            "id": "sender",
                            "op": "model.external_encode_bits",
                            "inputs": {"images": "data.images"},
                            "params": {
                                "path": str(adapter_path),
                                "callable": "encode",
                                "bit_storage": "packed_bytes",
                                "bit_order": "big",
                            },
                        },
                        {
                            "id": "receiver",
                            "op": "model.external_decode_bits",
                            "inputs": {"bits": "sender.bits"},
                            "params": {
                                "path": str(adapter_path),
                                "callable": "decode",
                                "bit_storage": "packed_bytes",
                                "bit_order": "big",
                            },
                        },
                        {
                            "id": "evaluation",
                            "op": "metrics.image_reconstruction",
                            "inputs": {
                                "reference": "data.images",
                                "reconstruction": "receiver.images",
                            },
                            "params": {},
                        },
                    ],
                }
            )
            store = LocalStore(root)
            run_dir = LocalExecutor(registry, store).run(recipe)
            summary = store.get_run(run_dir.name)
            steps = {step["id"]: step for step in summary["steps"]}
            bit_metadata = steps["sender"]["outputs"]["bits"]["metadata"]
            self.assertEqual(bit_metadata["byte_count"], 1)
            self.assertEqual(bit_metadata["bit_count"], 3)
            self.assertEqual(bit_metadata["payload_bit_count"], 3)
            self.assertEqual(bit_metadata["bit_storage"], "unpacked_uint8")
            self.assertEqual(bit_metadata["adapter_bit_storage"], "packed_bytes")
            self.assertEqual(steps["sender"]["metrics"]["channel.payload_bit_count"], 3)

    def test_external_adapter_sdk_manifest_registers_operation_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_path = root / "adapter.py"
            adapter_path.write_text(
                "\n".join(
                    [
                        "import numpy as np",
                        "",
                        "def encode_bits(images, params):",
                        "    return {'array': np.array([1, 0, 1], dtype=np.uint8), 'metadata': {'bit_count': 3}}",
                        "",
                        "def decode_bits(bits, params):",
                        "    shape = tuple(params.get('image_shape') or [1, 8, 8, 3])",
                        "    return np.zeros(shape, dtype=np.uint8)",
                    ]
                ),
                encoding="utf-8",
            )
            manifest_path = root / "noema_adapter.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "name: demo_sdk_codec",
                        "version: 0.1.0",
                        "operations:",
                        "  - id: model.demo_sdk_encode_bits",
                        "    name: Demo SDK encode bits",
                        "    wraps: model.external_encode_bits",
                        "    adapter:",
                        "      path: adapter.py",
                        "      callable: encode_bits",
                        "      call_style: array_params",
                        "    fixed_params:",
                        "      bit_storage: unpacked_bits",
                        "  - id: model.demo_sdk_decode_bits",
                        "    name: Demo SDK decode bits",
                        "    wraps: model.external_decode_bits",
                        "    adapter:",
                        "      path: adapter.py",
                        "      callable: decode_bits",
                        "      call_style: array_params",
                        "    fixed_params:",
                        "      bit_storage: unpacked_bits",
                        "    params_schema:",
                        "      type: object",
                        "      properties:",
                        "        image_shape:",
                        "          type: array",
                        "          default: [1, 8, 8, 3]",
                        "      additionalProperties: true",
                    ]
                ),
                encoding="utf-8",
            )
            registry = build_registry([manifest_path])
            operation = registry.get("model.demo_sdk_encode_bits").describe()
            self.assertEqual(operation["external_adapter"]["name"], "demo_sdk_codec")
            self.assertEqual(operation["output_kinds"]["bits"], "channel.payload_bits.numpy")

            images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
            input_path = root / "images.npz"
            np.savez_compressed(input_path, images=images)
            recipe = recipe_from_dict(
                {
                    "schema_version": 1,
                    "name": "sdk_manifest_run",
                    "steps": [
                        {
                            "id": "data",
                            "op": "source.local_npz_images",
                            "inputs": {},
                            "params": {"path": str(input_path), "array": "images"},
                        },
                        {
                            "id": "sender",
                            "op": "model.demo_sdk_encode_bits",
                            "inputs": {"images": "data.images"},
                            "params": {},
                        },
                        {
                            "id": "receiver",
                            "op": "model.demo_sdk_decode_bits",
                            "inputs": {"bits": "sender.bits"},
                            "params": {"image_shape": [1, 8, 8, 3]},
                        },
                        {
                            "id": "evaluation",
                            "op": "metrics.image_reconstruction",
                            "inputs": {"reference": "data.images", "reconstruction": "receiver.images"},
                            "params": {},
                        },
                    ],
                }
            )
            validate_recipe_against_registry(recipe, registry)
            run_dir = LocalExecutor(registry, LocalStore(root)).run(recipe)
            summary = LocalStore(root).get_run(run_dir.name)
            steps = {step["id"]: step for step in summary["steps"]}
            self.assertEqual(steps["sender"]["outputs"]["bits"]["metadata"]["bit_count"], 3)
            self.assertEqual(
                steps["sender"]["metadata"]["external_adapter_sdk"]["operation_id"],
                "model.demo_sdk_encode_bits",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["adapter", "validate", str(manifest_path), "--json"])
            self.assertEqual(code, 0)
            validation = json.loads(output.getvalue())
            self.assertEqual(validation["operation_count"], 2)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["--adapter", str(manifest_path), "ops", "show", "model.demo_sdk_encode_bits", "--json"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["external_adapter"]["name"], "demo_sdk_codec")

    def test_checkpoint_backed_external_adapter_records_training_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoints" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"noema checkpoint bytes")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            adapter_path = root / "adapter.py"
            adapter_path.write_text(
                "\n".join(
                    [
                        "import pathlib",
                        "import numpy as np",
                        "",
                        "def encode_bits(images, params):",
                        "    if not pathlib.Path(params['checkpoint_path']).is_file():",
                        "        raise RuntimeError('missing checkpoint')",
                        "    return {'array': np.array([1, 0, 1, 1], dtype=np.uint8), 'metadata': {'bit_count': 4}}",
                        "",
                        "def decode_bits(bits, params):",
                        "    return np.zeros((1, 8, 8, 3), dtype=np.uint8)",
                    ]
                ),
                encoding="utf-8",
            )
            manifest_path = root / "noema_adapter.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "name: checkpoint_sdk_codec",
                        "version: 0.2.0",
                        "training:",
                        "  source_recipe: recipes/deepjscc_kodak_awgn.yaml",
                        "  source_recipe_sha256: abc123",
                        "  differentiable_export_id: export-001",
                        "  capture_id: capture-001",
                        "  framework: torch",
                        "  checkpoint_path: checkpoints/model.pt",
                        "  input_schema:",
                        "    dtype: float32",
                        "    shape: [N, H, W, C]",
                        "  output_schema:",
                        "    dtype: uint8",
                        "    shape: [N]",
                        "  model_card: MODEL_CARD.md",
                        "operations:",
                        "  - id: model.checkpoint_sdk_encode_bits",
                        "    name: Checkpoint SDK encode bits",
                        "    wraps: model.external_encode_bits",
                        "    adapter:",
                        "      path: adapter.py",
                        "      callable: encode_bits",
                        "      call_style: array_params",
                        "    fixed_params:",
                        "      bit_storage: unpacked_bits",
                        "  - id: model.checkpoint_sdk_decode_bits",
                        "    name: Checkpoint SDK decode bits",
                        "    wraps: model.external_decode_bits",
                        "    adapter:",
                        "      path: adapter.py",
                        "      callable: decode_bits",
                        "      call_style: array_params",
                        "    fixed_params:",
                        "      bit_storage: unpacked_bits",
                    ]
                ),
                encoding="utf-8",
            )
            registry = build_registry([manifest_path])
            validation = validate_adapter_manifest(manifest_path, build_registry())
            self.assertEqual(validation["training"]["checkpoint_sha256"], checkpoint_sha)
            self.assertEqual(validation["training"]["checkpoint_path"], str(checkpoint.resolve()))
            operation = registry.get("model.checkpoint_sdk_encode_bits").describe()
            self.assertEqual(operation["external_adapter"]["training"]["checkpoint_sha256"], checkpoint_sha)

            images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
            input_path = root / "images.npz"
            np.savez_compressed(input_path, images=images)
            recipe = recipe_from_dict(
                {
                    "schema_version": 1,
                    "name": "checkpoint_sdk_manifest_run",
                    "steps": [
                        {"id": "data", "op": "source.local_npz_images", "params": {"path": str(input_path), "array": "images"}},
                        {"id": "sender", "op": "model.checkpoint_sdk_encode_bits", "inputs": {"images": "data.images"}},
                        {"id": "receiver", "op": "model.checkpoint_sdk_decode_bits", "inputs": {"bits": "sender.bits"}},
                        {"id": "evaluation", "op": "metrics.image_reconstruction", "inputs": {"reference": "data.images", "reconstruction": "receiver.images"}},
                    ],
                }
            )
            store = LocalStore(root)
            run_dir = LocalExecutor(registry, store).run(recipe)
            summary = store.get_run(run_dir.name)
            sender = {step["id"]: step for step in summary["steps"]}["sender"]
            training = sender["metadata"]["external_adapter_sdk"]["training"]
            self.assertEqual(training["checkpoint_sha256"], checkpoint_sha)
            self.assertEqual(training["checkpoint_path"], str(checkpoint.resolve()))
            self.assertEqual(sender["outputs"]["bits"]["metadata"]["external_adapter_sdk"]["training"]["checkpoint_sha256"], checkpoint_sha)
            manifest = store.get_manifest(run_dir.name)
            self.assertEqual(
                manifest["operation_contracts"]["operations"]["model.checkpoint_sdk_encode_bits"]["external_adapter"]["training"]["checkpoint_sha256"],
                checkpoint_sha,
            )
            manifest_sender = {step["id"]: step for step in manifest["steps"]}["sender"]
            self.assertEqual(manifest_sender["external_adapter_sdk"]["training"]["checkpoint_sha256"], checkpoint_sha)

    def test_checkpoint_backed_external_adapter_requires_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_path = root / "adapter.py"
            adapter_path.write_text("def encode_bits(images, params):\n    return images\n", encoding="utf-8")
            manifest_path = root / "noema_adapter.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "name: missing_checkpoint_adapter",
                        "training:",
                        "  framework: torch",
                        "  checkpoint_path: checkpoints/missing.pt",
                        "operations:",
                        "  - id: model.missing_checkpoint_encode_bits",
                        "    wraps: model.external_encode_bits",
                        "    adapter:",
                        "      path: adapter.py",
                        "      callable: encode_bits",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(OperationError, "training.checkpoint_path does not exist"):
                validate_adapter_manifest(manifest_path, build_registry(), import_callables=False)

    def test_explicit_comms_recipe_run_writes_channel_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_path = root / "adapter.py"
            adapter_path.write_text(
                "\n".join(
                    [
                        "import numpy as np",
                        "",
                        "def encode(images, params):",
                        "    return {'array': np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8), 'metadata': {'bit_count': 8}}",
                    ]
                ),
                encoding="utf-8",
            )
            images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
            input_path = root / "images.npz"
            np.savez_compressed(input_path, images=images)
            registry = build_registry()
            recipe = recipe_from_dict(
                {
                    "schema_version": 1,
                    "name": "explicit_bit_comms",
                    "steps": [
                        {
                            "id": "data",
                            "op": "source.local_npz_images",
                            "inputs": {},
                            "params": {"path": str(input_path), "array": "images"},
                        },
                        {
                            "id": "sender",
                            "op": "model.external_encode_bits",
                            "inputs": {"images": "data.images"},
                            "params": {"path": str(adapter_path), "callable": "encode"},
                        },
                        {
                            "id": "channel_encoder",
                            "op": "channel.repetition_encoder",
                            "inputs": {"bits": "sender.bits"},
                            "params": {"factor": 3},
                        },
                        {
                            "id": "modulator",
                            "op": "modulation.digital_modulate",
                            "inputs": {"bits": "channel_encoder.coded_bits"},
                            "params": {"modulation": "bpsk"},
                        },
                        {
                            "id": "wireless_channel",
                            "op": "wireless.channel",
                            "inputs": {"symbols": "modulator.symbols"},
                            "params": {"channel": "awgn", "snr_db": 80.0, "seed": 23},
                        },
                        {
                            "id": "demodulator",
                            "op": "demodulation.digital_demodulate",
                            "inputs": {"rx_symbols": "wireless_channel.rx_symbols"},
                            "params": {"modulation": "auto"},
                        },
                        {
                            "id": "coded_ber",
                            "op": "metrics.bit_error_rate",
                            "inputs": {
                                "reference": "channel_encoder.coded_bits",
                                "candidate": "demodulator.bits",
                            },
                            "params": {"label": "coded"},
                        },
                        {
                            "id": "channel_decoder",
                            "op": "channel.repetition_decoder",
                            "inputs": {"coded_bits": "demodulator.bits"},
                            "params": {"factor": 3},
                        },
                        {
                            "id": "payload_ber",
                            "op": "metrics.bit_error_rate",
                            "inputs": {
                                "reference": "sender.bits",
                                "candidate": "channel_decoder.bits",
                            },
                            "params": {"label": "payload"},
                        },
                    ],
                }
            )
            run_dir = LocalExecutor(registry, LocalStore(root)).run(recipe)
            summary = LocalStore(root).get_run(run_dir.name)
            steps = {step["id"]: step for step in summary["steps"]}
            self.assertIn("channel.coded.ber", steps["coded_ber"]["metrics"])
            self.assertIn("channel.payload.ber", steps["payload_ber"]["metrics"])
            self.assertEqual(
                steps["demodulator"]["outputs"]["llr"]["kind"],
                "channel.llr.numpy",
            )

    def test_capacity_oracle_protected_digital_baseline_reports_outage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = np.full((1, 16, 16, 3), 80, dtype=np.uint8)
            images[:, 4:12, 4:12, :] = [20, 180, 220]
            input_path = root / "images.npz"
            np.savez_compressed(input_path, images=images)

            def run_at(snr_db):
                recipe = recipe_from_dict(
                    {
                        "schema_version": 1,
                        "name": "jpeg_capacity_oracle_smoke",
                        "steps": [
                            {"id": "data", "op": "source.local_npz_images", "params": {"path": str(input_path), "array": "images"}},
                            {"id": "sender", "op": "model.jpeg_encode", "inputs": {"images": "data.images"}, "params": {"quality": 75}},
                            {"id": "payload_bit_boundary", "op": "channel.bit_boundary", "inputs": {"bits": "sender.bits"}, "params": {"label": "payload", "role": "payload"}},
                            {"id": "tx_bit_boundary", "op": "channel.bit_boundary", "inputs": {"bits": "payload_bit_boundary.bits"}, "params": {"label": "tx", "role": "protected_link_input"}},
                            {
                                "id": "wireless_channel",
                                "op": "channel.capacity_oracle_digital_link",
                                "inputs": {"bits": "tx_bit_boundary.bits"},
                                "params": {"snr_db": snr_db, "ldpc_rate": 0.5, "on_decode_failure": "gray_image"},
                            },
                            {"id": "rx_bit_boundary", "op": "channel.bit_boundary", "inputs": {"bits": "wireless_channel.bits"}, "params": {"label": "rx", "role": "protected_link_output"}},
                            {"id": "receiver", "op": "model.jpeg_decode", "inputs": {"bits": "rx_bit_boundary.bits"}, "params": {"on_error": "gray_image"}},
                            {"id": "evaluation", "op": "metrics.image_reconstruction", "inputs": {"reference": "data.images", "reconstruction": "receiver.images"}},
                        ],
                    }
                )
                store = LocalStore(root / ("workspace_%s" % str(snr_db).replace("-", "m")))
                run_dir = LocalExecutor(build_registry(), store).run(recipe)
                return {step["id"]: step for step in store.get_run(run_dir.name)["steps"]}

            success_steps = run_at(12.0)
            success_metrics = success_steps["wireless_channel"]["metrics"]
            self.assertEqual(success_metrics["channel.packet_success_rate"], 1.0)
            self.assertEqual(success_metrics["channel.outage_rate"], 0.0)
            self.assertGreater(success_metrics["channel.channel_use_count"], 0)
            self.assertIn("rate.payload_bpp", success_metrics)
            self.assertIn("rate.coded_bpp", success_metrics)
            self.assertIn("rate.padded_bpp", success_metrics)
            self.assertNotIn("rate.bpp", success_metrics)

            outage_steps = run_at(-12.0)
            outage_metrics = outage_steps["wireless_channel"]["metrics"]
            self.assertEqual(outage_metrics["channel.packet_success_rate"], 0.0)
            self.assertEqual(outage_metrics["channel.outage_rate"], 1.0)
            self.assertIn("quality.psnr_db", outage_steps["evaluation"]["metrics"])
            self.assertEqual(outage_steps["wireless_channel"]["outputs"]["bits"]["metadata"]["on_decode_failure"], "gray_image")

    def test_phase7_capacity_oracle_baseline_recipes_validate(self):
        registry = build_registry()
        for name in [
            "jpeg_q75_kodak_capacity_oracle_awgn.yaml",
            "compressai_q3_kodak_capacity_oracle_awgn.yaml",
        ]:
            recipe = load_recipe(ROOT / "recipes" / name)
            validate_recipe_against_registry(recipe, registry)
            wireless = next(step for step in recipe.steps if step.id == "wireless_channel")
            self.assertEqual(wireless.op, "channel.capacity_oracle_digital_link")
            self.assertEqual(float(wireless.params["ldpc_rate"]), 0.5)
            self.assertEqual(wireless.params["modulation"], "qpsk")
            matrix = recipe.metadata.get("matrix", {})
            self.assertIn("channel.snr_db", matrix.get("dimensions", {}))
            self.assertEqual(
                matrix.get("step_params", {})
                .get("wireless_channel", {})
                .get("snr_db"),
                {"matrix": "channel.snr_db"},
            )

    def test_matrix_expands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "matrix_recipe.yaml"
            recipe_path.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "name: channel_snr_matrix",
                        "metadata:",
                        "  matrix:",
                        "    dimensions:",
                        "      bit_count: [4, 8, 12]",
                        "      seed: [1, 2]",
                        "    step_params:",
                        "      data:",
                        "        bit_count:",
                        "          matrix: bit_count",
                        "        seed:",
                        "          matrix: seed",
                        "steps:",
                        "  - id: data",
                        "    op: source.random_bits",
                        "    params:",
                        "      bit_count: 1",
                        "      seed: 0",
                    ]
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(
                    [
                        "--workspace",
                        tmp,
                        "recipe",
                        "expand-matrix",
                        str(recipe_path),
                    ]
                )
            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["expanded_count"], 6)
            self.assertEqual(
                payload["recipes"][0]["steps"][0]["params"]["bit_count"],
                4,
            )

    def test_template_cli_instantiates_stable_id_with_typed_overrides(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                [
                    "template",
                    "instantiate",
                    "ai_phy.pilot_channel_estimation.adapter",
                    "--name",
                    "pilot_estimation_from_template",
                    "--set",
                    'estimator.mode="least_squares"',
                    "--json",
                ]
            )

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["recipe"]["name"], "pilot_estimation_from_template")
        estimator = next(
            step for step in payload["recipe"]["steps"] if step["id"] == "estimator"
        )
        self.assertEqual(estimator["params"]["mode"], "least_squares")
        provenance = payload["provenance"]
        self.assertEqual(
            provenance["template_id"],
            "ai_phy.pilot_channel_estimation.adapter",
        )
        self.assertEqual(
            payload["recipe"]["metadata"]["template_provenance"],
            provenance,
        )
        self.assertFalse(
            any(key.startswith("ui_") for key in payload["recipe"]["metadata"])
        )

    def test_recipe_run_rejects_unresolved_matrix_and_run_matrix_executes_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            recipe_path = root / "random_bits_matrix.yaml"
            recipe_path.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "name: random_bits_matrix",
                        "metadata:",
                        "  matrix:",
                        "    dimensions:",
                        "      seed: [3, 5]",
                        "    step_params:",
                        "      data:",
                        "        seed:",
                        "          matrix: seed",
                        "steps:",
                        "  - id: data",
                        "    op: source.random_bits",
                        "    params:",
                        "      bit_count: 8",
                        "      seed: 0",
                    ]
                ),
                encoding="utf-8",
            )

            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                code = main(
                    [
                        "--workspace",
                        str(workspace),
                        "recipe",
                        "run",
                        str(recipe_path),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("concrete recipe", error.getvalue())
            self.assertFalse((workspace / "runs").exists())

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(
                    [
                        "--workspace",
                        str(workspace),
                        "recipe",
                        "run-matrix",
                        str(recipe_path),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(output.getvalue().count("completed variant"), 2)
            run_dirs = sorted((workspace / "runs").iterdir())
            self.assertEqual(len(run_dirs), 2)
            recipes = [
                json.loads((run_dir / "recipe.json").read_text(encoding="utf-8"))
                for run_dir in run_dirs
            ]
            self.assertEqual(
                sorted(recipe["metadata"]["matrix_selection"]["seed"] for recipe in recipes),
                [3, 5],
            )
            self.assertEqual(
                len({recipe["metadata"]["matrix_variant_id"] for recipe in recipes}),
                2,
            )
            self.assertTrue(
                all((run_dir / "execution-plan.json").is_file() for run_dir in run_dirs)
            )

    def test_benchmark_pack_validates_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = np.zeros((1, 16, 16, 3), dtype=np.uint8)
            images[:, 2:14, 2:14, :] = [40, 160, 220]
            input_path = root / "images.npz"
            np.savez_compressed(input_path, images=images)
            recipe_path = root / "jpeg_recipe.yaml"
            recipe_path.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "name: jpeg_benchmark_recipe",
                        "steps:",
                        "  - id: data",
                        "    op: source.local_npz_images",
                        "    params:",
                        f"      path: {input_path}",
                        "      array: images",
                        "  - id: sender",
                        "    op: model.jpeg_encode",
                        "    inputs:",
                        "      images: data.images",
                        "    params:",
                        "      quality: 75",
                        "  - id: receiver",
                        "    op: model.jpeg_decode",
                        "    inputs:",
                        "      bits: sender.bits",
                        "  - id: evaluation",
                        "    op: metrics.image_reconstruction",
                        "    inputs:",
                        "      reference: data.images",
                        "      reconstruction: receiver.images",
                    ]
                ),
                encoding="utf-8",
            )
            benchmark_path = root / "benchmark.yaml"
            benchmark_path.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "id: local_image_smoke",
                        "version: '1'",
                        "name: Local Image Smoke",
                        "dataset:",
                        "  id: images",
                        "  modality: image",
                        "task:",
                        "  id: image_reconstruction",
                        "  kind: reconstruction",
                        "metrics:",
                        "  - id: quality.psnr_db",
                        "recipes:",
                        "  - id: jpeg",
                        "    label: JPEG local",
                        "    role: baseline",
                        f"    path: {recipe_path}",
                    ]
                ),
                encoding="utf-8",
            )

            validate_output = io.StringIO()
            with contextlib.redirect_stdout(validate_output):
                code = main(["--workspace", str(root / "workspace"), "benchmark", "validate", str(benchmark_path)])
            self.assertEqual(code, 0)
            validation = json.loads(validate_output.getvalue())
            self.assertEqual(validation["recipe_count"], 1)
            self.assertEqual(validation["recipes"][0]["research"]["task"]["id"], "image_reconstruction")

            summary_output = io.StringIO()
            with contextlib.redirect_stdout(summary_output):
                code = main(
                    [
                        "--workspace",
                        str(root / "workspace"),
                        "benchmark",
                        "validate",
                        str(benchmark_path),
                        "--summary",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("validated benchmark: Local Image Smoke", summary_output.getvalue())
            self.assertIn("recipes: 1 [baseline=1]", summary_output.getvalue())
            self.assertIn("catalog: valid", summary_output.getvalue())

            run_output = io.StringIO()
            with contextlib.redirect_stdout(run_output):
                code = main(["--workspace", str(root / "workspace"), "benchmark", "run", str(benchmark_path)])
            self.assertEqual(code, 0)
            result_line = [line for line in run_output.getvalue().splitlines() if line.startswith("result: ")][0]
            result_path = Path(result_line.split("result: ", 1)[1])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["kind"], "noema.benchmark_result")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["benchmark"]["id"], "local_image_smoke")
            self.assertEqual(result["benchmark"]["task"]["id"], "image_reconstruction")
            self.assertEqual(result["benchmark"]["catalog_validation"]["status"], "valid")
            self.assertEqual(result["recipes"][0]["status"], "completed")
            self.assertIn("quality.psnr_db", result["recipes"][0]["metrics"])
            self.assertIn("reports", result)
            metrics_path = Path(result["reports"]["metrics_csv"]["path"])
            recipes_path = Path(result["reports"]["recipes_csv"]["path"])
            summary_path = Path(result["reports"]["summary_markdown"]["path"])
            self.assertTrue(metrics_path.exists())
            self.assertTrue(recipes_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertIn("quality.psnr_db", metrics_path.read_text(encoding="utf-8"))
            self.assertIn("JPEG local", recipes_path.read_text(encoding="utf-8"))
            self.assertIn("# Local Image Smoke", summary_path.read_text(encoding="utf-8"))

            export_output = io.StringIO()
            with contextlib.redirect_stdout(export_output):
                code = main(
                    [
                        "--workspace",
                        str(root / "workspace"),
                        "benchmark",
                        "export",
                        result_path.parent.name,
                    ]
                )
            self.assertEqual(code, 0)
            export_payload = json.loads(export_output.getvalue())
            self.assertEqual(export_payload["result_id"], result_path.parent.name)
            self.assertIn("summary_markdown", export_payload["reports"])

            results_output = io.StringIO()
            with contextlib.redirect_stdout(results_output):
                code = main(["--workspace", str(root / "workspace"), "benchmark", "results"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(results_output.getvalue())["results"][0]["benchmark_id"], "local_image_smoke")

    def test_benchmark_plot_exports_hero_figures_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            store = LocalStore(workspace)
            result_id = "hero_plot_fixture"
            result_dir = store.get_benchmark_result_dir(result_id)
            result_dir.mkdir(parents=True)
            result = {
                "schema_version": 1,
                "kind": "noema.benchmark_result",
                "benchmark": {
                    "id": "hero_plot_benchmark",
                    "version": "1",
                    "name": "Hero Plot Benchmark",
                    "dataset": {"id": "kodak"},
                    "task": {"id": "image_reconstruction"},
                    "metrics": [{"id": "quality.psnr_db"}, {"id": "channel.packet_success_rate"}],
                },
                "status": "completed",
                "created_at_utc": "2026-07-10T00:00:00Z",
                "completed_at_utc": "2026-07-10T00:00:01Z",
                "recipes": [
                    {
                        "id": "jpeg_snr0",
                        "label": "JPEG + protected digital link",
                        "role": "baseline",
                        "run_id": "run_jpeg_0",
                        "status": "completed",
                        "metrics": {"channel.snr_db": 0, "quality.psnr_db": 12.0, "channel.packet_success_rate": 0.0, "channel.outage_rate": 1.0},
                    },
                    {
                        "id": "jpeg_snr10",
                        "label": "JPEG + protected digital link",
                        "role": "baseline",
                        "run_id": "run_jpeg_10",
                        "pairing_id": "seed-1",
                        "aggregation_cell_id": "kodak-snr10",
                        "statistical_unit": "paired-seed",
                        "status": "completed",
                        "metrics": {"channel.snr_db": 10, "quality.psnr_db": 31.0, "channel.packet_success_rate": 1.0, "channel.outage_rate": 0.0},
                    },
                    {
                        "id": "jpeg_snr10_seed2",
                        "label": "JPEG + protected digital link",
                        "role": "baseline",
                        "run_id": "run_jpeg_10_seed2",
                        "pairing_id": "seed-2",
                        "aggregation_cell_id": "kodak-snr10",
                        "statistical_unit": "paired-seed",
                        "status": "completed",
                        "metrics": {"channel.snr_db": 10, "quality.psnr_db": 29.0, "channel.packet_success_rate": 1.0, "channel.outage_rate": 0.0},
                    },
                    {
                        "id": "deepjscc_snr0",
                        "label": "DeepJSCC baseline",
                        "role": "baseline",
                        "run_id": "run_jscc_0",
                        "status": "completed",
                        "metrics": {"channel.snr_db": 0, "quality.psnr_db": 19.0, "channel.packet_success_rate": 1.0, "channel.outage_rate": 0.0},
                    },
                    {
                        "id": "deepjscc_snr10",
                        "label": "DeepJSCC baseline",
                        "role": "baseline",
                        "run_id": "run_jscc_10",
                        "status": "completed",
                        "metrics": {"channel.snr_db": 10, "quality.psnr_db": 28.0, "channel.packet_success_rate": 1.0, "channel.outage_rate": 0.0},
                    },
                ],
            }
            store.write_json(result_dir / "result.json", result)

            plot_output = io.StringIO()
            with contextlib.redirect_stdout(plot_output):
                code = main(
                    [
                        "--workspace",
                        str(workspace),
                        "benchmark",
                        "plot",
                        result_id,
                        "--plot",
                        "graceful-degradation",
                        "--x",
                        "channel.snr_db",
                        "--y",
                        "quality.psnr_db",
                        "--group",
                        "method",
                        "--method-order",
                        "DeepJSCC baseline,JPEG + protected digital link",
                        "--packet-success-panel",
                        "--style",
                        "paper",
                        "--out",
                        "figures/graceful_degradation.png",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0, plot_output.getvalue())
            payload = json.loads(plot_output.getvalue())
            image_path = Path(payload["plot"]["path"])
            data_path = Path(payload["plot"]["data_csv_path"])
            self.assertTrue(image_path.is_file())
            self.assertEqual(image_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(payload["plot"]["x_metric"], "channel.snr_db")
            self.assertEqual(payload["plot"]["y_metric"], "quality.psnr_db")
            self.assertEqual(payload["plot"]["group_by"], "method")
            self.assertTrue(payload["plot"]["packet_success_panel"])
            self.assertEqual(
                payload["plot"]["sha256"], hashlib.sha256(image_path.read_bytes()).hexdigest()
            )
            self.assertEqual(payload["plot"]["size_bytes"], image_path.stat().st_size)
            self.assertEqual(
                payload["plot"]["data_csv_sha256"],
                hashlib.sha256(data_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(payload["plot"]["data_csv_size_bytes"], data_path.stat().st_size)
            self.assertTrue(data_path.is_file())
            data_text = data_path.read_text(encoding="utf-8")
            self.assertIn("channel.snr_db", data_text)
            self.assertIn("quality.psnr_db", data_text)
            self.assertIn("JPEG + protected digital link", data_text)
            rows = list(csv.DictReader(data_text.splitlines()))
            self.assertEqual(len(rows), payload["plot"]["point_count"])
            self.assertEqual(rows[0]["series"], "DeepJSCC baseline")
            jpeg_snr10 = next(row for row in rows if row["series"] == "JPEG + protected digital link" and row["x_value"] == "10.0")
            self.assertEqual(jpeg_snr10["sample_count"], "2")
            self.assertAlmostEqual(float(jpeg_snr10["y_value"]), 30.0)
            self.assertGreater(float(jpeg_snr10["y_ci95"]), 0.0)
            jpeg_outage = next(row for row in rows if row["series"] == "JPEG + protected digital link" and row["x_value"] == "0.0")
            self.assertEqual(jpeg_outage["outage_marker"], "1")

            style_path = result_dir / "paper_style.json"
            style_path.write_text(json.dumps({"figure_width": 5.8, "figure_height": 3.8, "dpi": 160}), encoding="utf-8")
            svg_output = io.StringIO()
            with contextlib.redirect_stdout(svg_output):
                code = main(
                    [
                        "--workspace",
                        str(workspace),
                        "benchmark",
                        "plot",
                        result_id,
                        "--plot",
                        "graceful-degradation",
                        "--x",
                        "channel.snr_db",
                        "--y",
                        "quality.psnr_db",
                        "--style-config",
                        str(style_path),
                        "--out",
                        "figures/graceful_degradation_vector.svg",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0, svg_output.getvalue())
            svg_path = Path(json.loads(svg_output.getvalue())["plot"]["path"])
            self.assertTrue(svg_path.is_file())
            self.assertIn("<svg", svg_path.read_text(encoding="utf-8")[:500])

            packet_output = io.StringIO()
            with contextlib.redirect_stdout(packet_output):
                code = main(
                    [
                        "--workspace",
                        str(workspace),
                        "benchmark",
                        "plot",
                        result_id,
                        "--plot",
                        "packet-success",
                        "--x",
                        "channel.snr_db",
                        "--out",
                        "figures/packet_success.pdf",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0, packet_output.getvalue())
            packet_payload = json.loads(packet_output.getvalue())
            packet_pdf = Path(packet_payload["plot"]["path"])
            self.assertTrue(packet_pdf.is_file())
            self.assertEqual(packet_pdf.read_bytes()[:4], b"%PDF")
            updated = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
            self.assertNotIn("plots", updated)
            plot_sidecars = sorted(result_dir.rglob("*.plot.json"))
            self.assertEqual(len(plot_sidecars), 3)
            bound_paths = {
                json.loads(path.read_text(encoding="utf-8"))["plot"][
                    "relative_path"
                ]
                for path in plot_sidecars
            }
            self.assertEqual(
                bound_paths,
                {
                    "figures/graceful_degradation.png",
                    "figures/graceful_degradation_vector.svg",
                    "figures/packet_success.pdf",
                },
            )

    def test_development_benchmark_v1_validates_and_is_listed(self):
        benchmark_path = ROOT / "benchmarks" / "benchmark_v1" / "kodak_image_reconstruction_v1.yaml"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["benchmark", "validate", str(benchmark_path)])
        self.assertEqual(code, 0)
        validation = json.loads(output.getvalue())
        self.assertEqual(validation["benchmark_tier"], "experimental")
        self.assertEqual(validation["metadata"]["protocol_version"], "1.2.0")
        self.assertEqual(validation["recipe_count"], 2)
        self.assertEqual(validation["recipes"][0]["lint"]["status"], "passed")

        list_output = io.StringIO()
        with contextlib.redirect_stdout(list_output):
            code = main(["benchmark", "list", "--directory", str(ROOT / "benchmarks"), "--json"])
        self.assertEqual(code, 0)
        listed = json.loads(list_output.getvalue())
        ids = {item["id"] for item in listed["benchmarks"]}
        self.assertIn("benchmark_v1.image_reconstruction.kodak", ids)
        self.assertIn("semantic_comm", {group["suite"]["id"] for group in listed["suites"]})

        human_list_output = io.StringIO()
        with contextlib.redirect_stdout(human_list_output):
            code = main(["benchmark", "list", "--directory", str(ROOT / "benchmarks")])
        self.assertEqual(code, 0)
        self.assertIn("Semantic Communication [semantic_comm, active]", human_list_output.getvalue())
        self.assertIn("benchmark_v1.image_reconstruction.kodak", human_list_output.getvalue())

    def test_external_non_codec_adapter_can_run_benchmark_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            adapter_path = ROOT / "examples" / "adapters" / "classification_task"
            benchmark_path = ROOT / "examples" / "benchmarks" / "external_classification_adapter_smoke.yaml"

            validate_output = io.StringIO()
            with contextlib.redirect_stdout(validate_output):
                code = main(
                    [
                        "--adapter",
                        str(adapter_path),
                        "adapter",
                        "validate",
                        str(adapter_path / "noema_adapter.yaml"),
                        "--json",
                    ]
                )
            self.assertEqual(code, 0)
            adapter_validation = json.loads(validate_output.getvalue())
            self.assertEqual(adapter_validation["operation_count"], 2)

            benchmark_output = io.StringIO()
            with contextlib.redirect_stdout(benchmark_output):
                code = main(
                    [
                        "--workspace",
                        str(workspace),
                        "--adapter",
                        str(adapter_path),
                        "benchmark",
                        "run",
                        str(benchmark_path),
                    ]
                )
            self.assertEqual(code, 0)
            result_line = [line for line in benchmark_output.getvalue().splitlines() if line.startswith("result: ")][0]
            result = json.loads(Path(result_line.split("result: ", 1)[1]).read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed")
            metrics = result["recipes"][0]["metrics"]
            self.assertEqual(metrics["task.accuracy"], 0.75)
            self.assertEqual(metrics["external.classification.accuracy"], 0.75)

    def test_submission_bundle_validation_rejects_fabricated_result_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "result_bundle"
            result_dir.mkdir()
            (result_dir / "result.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "noema.benchmark_result",
                        "benchmark": {
                            "id": "benchmark_v1.image_reconstruction.kodak",
                            "version": "1.0.0",
                            "sha256": "abc",
                        },
                        "status": "completed",
                        "recipes": [],
                        "reports": {"metrics_csv": {"path": "metrics.csv"}},
                    }
                ),
                encoding="utf-8",
            )
            submission_path = root / "submission.json"
            submission_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "comparison_division": "closed",
                        "result_bundle": "result_bundle",
                        "method": {"name": "Example method", "authors": ["A. Researcher"]},
                        "benchmark": {
                            "id": "benchmark_v1.image_reconstruction.kodak",
                            "version": "1.0.0",
                        },
                        "runtime": {"hardware": "CPU", "software": "noema-lab 0.1.0", "dependencies": []},
                        "reproducibility": {
                            "recipe_sha256": "abc",
                            "seed_policy": "noema deterministic",
                            "seeds": [23],
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["submission", "validate", str(submission_path), "--json"])
            self.assertEqual(code, 1)
            validation = json.loads(output.getvalue())
            self.assertEqual(validation["status"], "invalid")
            self.assertEqual(validation["verdict"], "rejected")
            self.assertIsNotNone(validation["verification"])
            self.assertEqual(validation["verification"]["status"], "invalid")

    def test_text_semantic_benchmark_validates_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            benchmark_path = ROOT / "benchmarks" / "text_semantic_smoke_v1.yaml"

            validate_output = io.StringIO()
            with contextlib.redirect_stdout(validate_output):
                code = main(["--workspace", str(workspace), "benchmark", "validate", str(benchmark_path)])
            self.assertEqual(code, 0)
            validation = json.loads(validate_output.getvalue())
            self.assertEqual(validation["catalog_validation"]["status"], "valid")
            self.assertEqual(validation["recipes"][0]["research"]["task"]["id"], "text_semantic_similarity")

            run_output = io.StringIO()
            with contextlib.redirect_stdout(run_output):
                code = main(["--workspace", str(workspace), "benchmark", "run", str(benchmark_path)])
            self.assertEqual(code, 0)
            result_line = [line for line in run_output.getvalue().splitlines() if line.startswith("result: ")][0]
            result_path = Path(result_line.split("result: ", 1)[1])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["benchmark"]["task"]["id"], "text_semantic_similarity")
            self.assertEqual(result["benchmark"]["dataset"]["id"], "semantic_text_smoke")
            self.assertEqual(len(result["recipes"]), 2)
            clean_metrics = result["recipes"][0]["metrics"]
            noisy_metrics = result["recipes"][1]["metrics"]
            self.assertEqual(clean_metrics["text.exact_match"], 1.0)
            self.assertEqual(clean_metrics["semantic.lexical_similarity"], 1.0)
            self.assertLess(noisy_metrics["semantic.lexical_similarity"], 1.0)
            self.assertIn(
                "text.unigram_bleu_proxy",
                Path(result["reports"]["metrics_csv"]["path"]).read_text(encoding="utf-8"),
            )

    def test_text_semantic_presets_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            benchmark_path = ROOT / "benchmarks" / "text_semantic_presets_v1.yaml"

            validate_output = io.StringIO()
            with contextlib.redirect_stdout(validate_output):
                code = main(["--workspace", str(workspace), "benchmark", "validate", str(benchmark_path)])
            self.assertEqual(code, 0)
            validation = json.loads(validate_output.getvalue())
            self.assertEqual(validation["catalog_validation"]["status"], "valid")
            recipe_ids = [item["id"] for item in validation["recipes"]]
            self.assertEqual(recipe_ids, ["utf8_clean", "utf8_mask_repair", "bart_jscc_clean"])

            recipe_output = io.StringIO()
            with contextlib.redirect_stdout(recipe_output):
                code = main(
                    [
                        "--workspace",
                        str(workspace),
                        "recipe",
                        "validate",
                        str(ROOT / "recipes" / "text_semantic_utf8_mask_repair.yaml"),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("valid recipe: text_semantic_utf8_mask_repair", recipe_output.getvalue())

    def test_phase6_task_oriented_benchmark_core_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            benchmark_path = ROOT / "benchmarks" / "task_oriented_smoke_v1.yaml"

            validate_output = io.StringIO()
            with contextlib.redirect_stdout(validate_output):
                code = main(["--workspace", str(workspace), "benchmark", "validate", str(benchmark_path)])
            self.assertEqual(code, 0)
            validation = json.loads(validate_output.getvalue())
            self.assertEqual(validation["catalog_validation"]["status"], "valid")
            self.assertEqual(validation["catalog_validation"]["task"]["id"], "classification")
            self.assertEqual(validation["recipes"][0]["research"]["task"]["id"], "classification")

            run_output = io.StringIO()
            with contextlib.redirect_stdout(run_output):
                code = main(["--workspace", str(workspace), "benchmark", "run", str(benchmark_path)])
            self.assertEqual(code, 0)
            result_line = [line for line in run_output.getvalue().splitlines() if line.startswith("result: ")][0]
            result_path = Path(result_line.split("result: ", 1)[1])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["benchmark"]["task"]["id"], "classification")
            metrics = result["recipes"][0]["metrics"]
            self.assertEqual(metrics["task.accuracy"], 0.75)
            self.assertEqual(metrics["task.exact_match"], 0.75)
            self.assertIn("classification.balanced_accuracy", metrics)
            self.assertIn("task.accuracy", Path(result["reports"]["metrics_csv"]["path"]).read_text(encoding="utf-8"))

    def test_phase7_typed_semantic_artifact_recipe_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            recipe_path = ROOT / "recipes" / "semantic_artifacts_smoke.yaml"

            validate_output = io.StringIO()
            with contextlib.redirect_stdout(validate_output):
                code = main(["--workspace", str(workspace), "recipe", "validate", str(recipe_path)])
            self.assertEqual(code, 0)
            self.assertIn("valid recipe: semantic_artifacts_smoke", validate_output.getvalue())

            run_output = io.StringIO()
            with contextlib.redirect_stdout(run_output):
                code = main(["--workspace", str(workspace), "recipe", "run", str(recipe_path)])
            self.assertEqual(code, 0)
            summary_line = [line for line in run_output.getvalue().splitlines() if line.startswith("summary: ")][0]
            summary = json.loads(Path(summary_line.split("summary: ", 1)[1]).read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            steps = {step["id"]: step for step in summary["steps"]}
            self.assertEqual(steps["data"]["metadata"]["dataset"], "semantic_artifact_smoke")
            self.assertEqual(steps["data"]["metadata"]["artifact_contract_stage"], "typed_semantic_artifacts")
            data_outputs = steps["data"]["outputs"]
            self.assertEqual(data_outputs["clip_embeddings"]["kind"], "vision.embedding.clip.numpy")
            self.assertEqual(data_outputs["multimodal_embeddings"]["kind"], "multimodal.embedding.numpy")
            self.assertEqual(data_outputs["detections_reference"]["kind"], "vision.detections.json")
            self.assertEqual(data_outputs["segmentation_reference"]["kind"], "vision.segmentation_mask.numpy")
            self.assertEqual(data_outputs["scene_graph"]["kind"], "vision.scene_graph.json")
            self.assertEqual(data_outputs["semantic_map"]["kind"], "vision.semantic_map.json")
            self.assertEqual(data_outputs["captions_reference"]["kind"], "text.caption.json")
            self.assertEqual(data_outputs["vqa_reference"]["kind"], "vqa.answers.json")
            self.assertEqual(data_outputs["rankings"]["kind"], "retrieval.rankings.json")
            self.assertEqual(data_outputs["importance_map"]["kind"], "semantic.importance_map.numpy")
            self.assertEqual(data_outputs["video_frames"]["kind"], "video.frame_sequence.numpy")
            self.assertIn("detection.f1_at_iou_0p5", steps["detection_evaluation"]["metrics"])
            self.assertIn("segmentation.miou", steps["segmentation_evaluation"]["metrics"])
            self.assertIn(
                "caption.unigram_bleu_proxy", steps["captioning_evaluation"]["metrics"]
            )
            self.assertEqual(
                steps["vqa_evaluation"]["metrics"]["vqa.single_reference_exact_match"],
                1.0,
            )
            self.assertIn("retrieval.recall_at_1", steps["retrieval_evaluation"]["metrics"])

    def test_phase8_goal_oriented_vqa_recipe_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            recipe_path = ROOT / "recipes" / "vqa_goal_oriented_smoke.yaml"

            validate_output = io.StringIO()
            with contextlib.redirect_stdout(validate_output):
                code = main(["--workspace", str(workspace), "recipe", "validate", str(recipe_path)])
            self.assertEqual(code, 0)
            self.assertIn("valid recipe: vqa_goal_oriented_smoke", validate_output.getvalue())

            run_output = io.StringIO()
            with contextlib.redirect_stdout(run_output):
                code = main(["--workspace", str(workspace), "recipe", "run", str(recipe_path)])
            self.assertEqual(code, 0)
            summary_line = [line for line in run_output.getvalue().splitlines() if line.startswith("summary: ")][0]
            summary = json.loads(Path(summary_line.split("summary: ", 1)[1]).read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            steps = {step["id"]: step for step in summary["steps"]}
            self.assertEqual(steps["data"]["outputs"]["questions"]["kind"], "vqa.questions.json")
            self.assertEqual(steps["semantic_selector"]["outputs"]["packet"]["kind"], "vqa.semantic_packet.json")
            self.assertEqual(steps["sender"]["outputs"]["bits"]["kind"], "channel.payload_bits.numpy")
            self.assertEqual(
                steps["tx_bit_boundary"]["metrics"]["channel.fixed.modulator_input.bit_count"],
                steps["sender"]["metrics"]["channel.payload_bit_count"],
            )
            self.assertEqual(steps["channel_bit_count_match"]["metrics"]["channel.fixed.channel_io.bit_count_match"], 1)
            self.assertEqual(
                steps["evaluation"]["metrics"]["vqa.single_reference_exact_match"],
                1.0,
            )
            self.assertEqual(steps["evaluation"]["metrics"]["task.exact_match"], 1.0)

    def test_phase8_goal_oriented_vqa_benchmark_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            benchmark_path = ROOT / "benchmarks" / "vqa_goal_oriented_smoke_v1.yaml"

            validate_output = io.StringIO()
            with contextlib.redirect_stdout(validate_output):
                code = main(["--workspace", str(workspace), "benchmark", "validate", str(benchmark_path)])
            self.assertEqual(code, 0)
            validation = json.loads(validate_output.getvalue())
            self.assertEqual(validation["catalog_validation"]["status"], "valid")
            self.assertEqual(validation["catalog_validation"]["dataset"]["id"], "coco_vqa_smoke")
            self.assertEqual(validation["catalog_validation"]["task"]["id"], "visual_question_answering")
            self.assertEqual(validation["recipes"][0]["research"]["dataset"]["source"], "source.vqa_smoke")

    def test_phase5_semantic_state_kb_recipe_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            recipe_path = ROOT / "recipes" / "text_semantic_state_kb_clean.yaml"

            run_output = io.StringIO()
            with contextlib.redirect_stdout(run_output):
                code = main(["--workspace", str(workspace), "recipe", "run", str(recipe_path)])
            self.assertEqual(code, 0)
            summary_line = [line for line in run_output.getvalue().splitlines() if line.startswith("summary: ")][0]
            summary = json.loads(Path(summary_line.split("summary: ", 1)[1]).read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            steps = {step["id"]: step for step in summary["steps"]}
            self.assertEqual(steps["knowledge_base"]["op"], "foundation.knowledge_base")
            self.assertEqual(steps["semantic_encoder"]["op"], "foundation.text_semantic_state_encode")
            self.assertEqual(steps["sender"]["op"], "foundation.semantic_state_payload_encode")
            self.assertEqual(steps["payload_decoder"]["op"], "foundation.semantic_state_payload_decode")
            faithfulness = steps["faithfulness"]["metrics"]
            self.assertEqual(faithfulness["faithfulness.concept_f1"], 1.0)
            self.assertEqual(faithfulness["faithfulness.fact_f1"], 1.0)
            self.assertEqual(faithfulness["faithfulness.kb_fact_precision"], 1.0)
            self.assertEqual(
                faithfulness["faithfulness.unsupported_assertion_rate"], 0.0
            )
            self.assertEqual(faithfulness["faithfulness.fact_omission_rate"], 0.0)
            self.assertGreater(steps["sender"]["metrics"]["channel.payload_bit_count"], 0)

    def test_phase5_sender_masking_is_scored_against_clean_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            recipe = load_recipe(ROOT / "recipes" / "text_semantic_state_kb_clean.yaml").to_dict()
            recipe["name"] = "text_semantic_state_masked"
            insert_index = next(index for index, step in enumerate(recipe["steps"]) if step["id"] == "sender")
            recipe["steps"].insert(
                insert_index,
                {
                    "id": "sender_source_perturbation",
                    "op": "noise.source_text_perturbation",
                    "inputs": {"texts": "data.texts"},
                    "params": {"mode": "mask_words", "probability": 1.0, "mask_token": "[MASK]", "seed": 1},
                },
            )
            recipe["steps"].insert(
                insert_index + 1,
                {
                    "id": "noisy_semantic_encoder",
                    "op": "foundation.text_semantic_state_encode",
                    "inputs": {"texts": "sender_source_perturbation.texts"},
                    "params": {"extractor": "local_rules", "max_concepts": 16, "include_text": True},
                },
            )
            recipe["steps"].insert(
                insert_index + 2,
                {
                    "id": "noisy_semantic_ground",
                    "op": "foundation.semantic_state_ground",
                    "inputs": {"state": "noisy_semantic_encoder.state", "kb": "knowledge_base.kb"},
                    "params": {"min_token_overlap": 1, "max_matches_per_state": 12},
                },
            )
            recipe["steps"][insert_index + 3]["inputs"]["state"] = "noisy_semantic_ground.state"
            recipe_path = Path(tmp) / "masked.yaml"
            recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")

            run_output = io.StringIO()
            with contextlib.redirect_stdout(run_output):
                code = main(["--workspace", str(workspace), "recipe", "run", str(recipe_path)])
            self.assertEqual(code, 0)
            summary_line = [line for line in run_output.getvalue().splitlines() if line.startswith("summary: ")][0]
            summary = json.loads(Path(summary_line.split("summary: ", 1)[1]).read_text(encoding="utf-8"))
            steps = {step["id"]: step for step in summary["steps"]}
            self.assertLess(steps["faithfulness"]["metrics"]["faithfulness.concept_f1"], 1.0)
            self.assertEqual(steps["evaluation"]["metrics"]["text.exact_match"], 0.0)
            self.assertEqual(steps["receiver"]["op"], "foundation.semantic_state_to_text")

    def test_text_mask_repair_pass_through_mode_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "texts.json"
            input_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "text.batch",
                        "dataset": "unit",
                        "examples": [{"id": "a", "text": "hello [MASK]"}],
                    }
                ),
                encoding="utf-8",
            )
            op = build_registry().get("foundation.text_mask_repair")
            result = op.run(
                OperationContext(
                    recipe_name="text_repair",
                    step_id="receiver_repair",
                    params={"generator": "local_template"},
                    inputs={"texts": Artifact("text.batch.json", input_path, {})},
                    run_dir=root,
                    step_dir=root / "receiver_repair",
                )
            )
            self.assertEqual(result.outputs["texts"].kind, "text.batch.json")
            repaired = json.loads(result.outputs["texts"].path.read_text(encoding="utf-8"))
            self.assertEqual(repaired["examples"][0]["text"], "hello [MASK]")
            self.assertEqual(result.metrics["text.mask_repair_applied"], 0)

    def test_masked_lm_repair_fills_multiple_masks_one_at_a_time(self):
        class Tokenizer:
            mask_token = "[MASK]"
            unk_token = "[UNK]"

        class FillMask:
            tokenizer = Tokenizer()

            def __init__(self):
                self.tokens = iter(["using", "special"])
                self.prompts = []

            def __call__(self, text, top_k=1):
                self.prompts.append(text)
                self.assert_single_mask(text)
                return [{"token_str": next(self.tokens), "score": 0.9}]

            @staticmethod
            def assert_single_mask(text):
                if text.count("[MASK]") != 1:
                    raise AssertionError("fill-mask prompt should contain exactly one [MASK]: %s" % text)

        fill_mask = FillMask()
        rendered, fills = _masked_lm_render(
            fill_mask,
            "A researcher transmits a short weather report [MASK] a [MASK] semantic channel.",
            1,
            "A researcher transmits a short weather report over a noisy semantic channel.",
        )
        self.assertNotIn("[MASK]", rendered)
        self.assertEqual(rendered, "A researcher transmits a short weather report using a special semantic channel.")
        self.assertEqual([fill["token"] for fill in fills], ["using", "special"])
        self.assertEqual(len(fill_mask.prompts), 2)

    def test_masked_lm_repair_normalizes_jscc_mask_token_variants(self):
        class Tokenizer:
            mask_token = "[MASK]"
            unk_token = "[UNK]"

        class FillMask:
            tokenizer = Tokenizer()

            def __call__(self, text, top_k=1):
                if text.count("[MASK]") != 1:
                    raise AssertionError("expected one canonical mask prompt: %s" % text)
                return [{"token_str": "special", "score": 0.8}]

        rendered, fills = _masked_lm_render(
            FillMask(),
            "A researcher transmits a short weather report using a [MASk] semantic channel.",
            1,
            "",
        )
        self.assertEqual(rendered, "A researcher transmits a short weather report using a special semantic channel.")
        self.assertNotIn("[MASk]", rendered)
        self.assertNotIn("[MASK]", rendered)
        self.assertEqual(fills[0]["token"], "special")

    def test_text_pipeline_order_contracts_are_explicit(self):
        utf8_recipe = load_recipe(ROOT / "recipes" / "text_semantic_utf8_source_perturbation.yaml").to_dict()
        utf8_steps = [step["id"] for step in utf8_recipe["steps"]]
        self.assertLess(utf8_steps.index("data"), utf8_steps.index("sender_source_perturbation"))
        self.assertLess(utf8_steps.index("sender_source_perturbation"), utf8_steps.index("sender"))
        self.assertLess(utf8_steps.index("sender"), utf8_steps.index("wireless_channel"))
        self.assertLess(utf8_steps.index("wireless_channel"), utf8_steps.index("receiver"))
        self.assertLess(utf8_steps.index("receiver"), utf8_steps.index("evaluation"))
        self.assertEqual(
            next(step for step in utf8_recipe["steps"] if step["id"] == "sender")["inputs"]["texts"],
            "sender_source_perturbation.texts",
        )
        self.assertEqual(
            next(step for step in utf8_recipe["steps"] if step["id"] == "evaluation")["inputs"]["candidate"],
            "receiver.texts",
        )

        bart_recipe = load_recipe(ROOT / "recipes" / "text_bart_jscc_clean.yaml").to_dict()
        bart_steps = [step["id"] for step in bart_recipe["steps"]]
        for before, after in [
            ("data", "sender"),
            ("sender", "tx_power"),
            ("tx_power", "tx_symbol_boundary"),
            ("tx_symbol_boundary", "wireless_channel"),
            ("wireless_channel", "rx_symbol_boundary"),
            ("rx_symbol_boundary", "receiver"),
            ("receiver", "evaluation"),
        ]:
            self.assertLess(bart_steps.index(before), bart_steps.index(after))
        self.assertEqual(
            next(step for step in bart_recipe["steps"] if step["id"] == "receiver")["inputs"]["symbols"],
            "rx_symbol_boundary.symbols",
        )

    def test_phase5_foundation_operations_are_registered(self):
        operations = {operation.id: operation.describe() for operation in build_registry().list()}
        self.assertEqual(operations["foundation.knowledge_base"]["output_kinds"]["kb"], "foundation.kb.json")
        self.assertEqual(operations["foundation.text_semantic_state_encode"]["output_kinds"]["state"], "semantic.state.json")
        self.assertEqual(operations["foundation.semantic_state_payload_encode"]["output_kinds"]["bits"], "channel.payload_bits.numpy")
        self.assertEqual(operations["foundation.text_mask_repair"]["input_kinds"]["texts"], ["text.batch.json"])
        self.assertEqual(operations["source.task_labels_smoke"]["output_kinds"]["reference"], "task.labels.json")
        self.assertEqual(operations["source.semantic_artifacts_smoke"]["output_kinds"]["clip_embeddings"], "vision.embedding.clip.numpy")
        self.assertEqual(operations["source.semantic_artifacts_smoke"]["output_kinds"]["scene_graph"], "vision.scene_graph.json")
        self.assertEqual(operations["source.semantic_artifacts_smoke"]["output_kinds"]["importance_map"], "semantic.importance_map.numpy")
        self.assertEqual(operations["source.vqa_smoke"]["output_kinds"]["questions"], "vqa.questions.json")
        self.assertEqual(operations["foundation.vqa_semantic_select"]["output_kinds"]["packet"], "vqa.semantic_packet.json")
        self.assertEqual(operations["foundation.vqa_payload_encode"]["output_kinds"]["bits"], "channel.payload_bits.numpy")
        self.assertEqual(operations["foundation.vqa_payload_decode"]["output_kinds"]["packet"], "vqa.semantic_packet.json")
        self.assertEqual(operations["foundation.vqa_answer_from_packet"]["output_kinds"]["answers"], "vqa.answers.json")
        self.assertEqual(operations["foundation.vqa_transformers_answer"]["output_kinds"]["answers"], "vqa.answers.json")
        self.assertEqual(operations["foundation.vqa_transformers_answer"]["input_kinds"]["images"], ["image.batch.numpy"])
        self.assertEqual(operations["metrics.classification"]["input_kinds"]["candidate"], ["task.predictions.json", "task.labels.json"])
        self.assertEqual(operations["metrics.vqa"]["output_kinds"]["report"], "metrics.report")
        self.assertEqual(operations["metrics.detection"]["input_kinds"]["reference"], ["vision.detections.json"])
        self.assertEqual(operations["metrics.segmentation"]["input_kinds"]["candidate"], ["vision.segmentation_mask.numpy"])
        self.assertEqual(operations["metrics.captioning"]["input_kinds"]["candidate"], ["text.batch.json", "text.caption.json"])
        self.assertEqual(operations["metrics.retrieval"]["input_kinds"]["rankings"], ["retrieval.rankings.json"])
        self.assertEqual(operations["model.text_bart_jscc_encode"]["output_kinds"]["symbols"], "channel.symbols.complex_numpy")
        self.assertEqual(operations["model.text_bart_jscc_decode"]["input_kinds"]["symbols"], ["channel.symbols.complex_numpy", "channel.rx_symbols.complex_numpy"])
        self.assertEqual(operations["channel.symbol_boundary"]["output_kinds"]["symbols"], "channel.symbols.complex_numpy")
        self.assertEqual(operations["channel.identity_symbol_link"]["output_kinds"]["symbols"], "channel.symbols.complex_numpy")
        self.assertEqual(
            operations["demodulation.digital_demodulate"]["input_kinds"]["rx_symbols"],
            ["channel.symbols.complex_numpy", "channel.rx_symbols.complex_numpy"],
        )
        self.assertEqual(operations["channel.symbol_count_match"]["input_kinds"]["candidate"], ["channel.symbols.complex_numpy", "channel.rx_symbols.complex_numpy"])
        self.assertEqual(operations["foundation.clip_text_embed"]["output_kinds"]["embeddings"], "foundation.embedding.numpy")
        self.assertEqual(operations["foundation.sam_segment"]["output_kinds"]["state"], "semantic.state.json")
        self.assertIn("faithfulness.concept_f1", (ROOT / "src" / "noema_lab" / "research_catalog.yaml").read_text(encoding="utf-8"))
        self.assertIn("visual_question_answering", (ROOT / "src" / "noema_lab" / "research_catalog.yaml").read_text(encoding="utf-8"))
        self.assertIn("retrieval.recall_at_1", (ROOT / "src" / "noema_lab" / "research_catalog.yaml").read_text(encoding="utf-8"))

    def test_ui_server_serves_graph_and_runs_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark_dir = Path(tmp) / "benchmarks" / "20260707T000000Z_ui_smoke"
            benchmark_dir.mkdir(parents=True)
            (benchmark_dir / "result.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "noema.benchmark_result",
                        "benchmark": {
                            "id": "ui_smoke",
                            "version": "1",
                            "name": "UI Smoke Benchmark",
                            "suite": {
                                "id": "semantic_comm",
                                "name": "Semantic Communication",
                                "status": "active",
                                "version": "v1",
                            },
                        },
                        "status": "completed",
                        "created_at_utc": "2026-07-07T00:00:00.000Z",
                        "completed_at_utc": "2026-07-07T00:00:01.000Z",
                        "recipes": [],
                    }
                ),
                encoding="utf-8",
            )
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                Path(tmp),
                ROOT,
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                index = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8")
                self.assertIn("Noema", index)
                self.assertIn("recipeTabs", index)
                self.assertIn("topbar-tabs", index)
                self.assertIn("trainingTabButton", index)
                self.assertIn("trainingView", index)
                self.assertIn("graphHeaderControls", index)
                self.assertIn("recipeOpenButton", index)
                self.assertNotIn("recipeLibrarySelect", index)
                self.assertIn("<h2>Recipe</h2>", index)
                self.assertNotIn("<h2>Recipe Config</h2>", index)
                self.assertIn("recipeConfigurator", index)
                self.assertIn("settingsOverlay", index)
                self.assertNotIn("Add Block", index)
                self.assertNotIn(">Runs<", index)
                suites = _get_json(base + "/api/suites")
                suite_by_id = {item["id"]: item for item in suites["suites"]}
                self.assertIn("semantic_comm", suite_by_id)
                self.assertIn("channel_estimation", suite_by_id)
                self.assertEqual(suite_by_id["semantic_comm"]["status"], "active")
                self.assertEqual(suite_by_id["channel_estimation"]["status"], "experimental")
                self.assertTrue(suite_by_id["channel_estimation"].get("benchmark_packs"))
                benchmark_results = _get_json(base + "/api/benchmarks/results")
                self.assertEqual(benchmark_results["results"][0]["benchmark_id"], "ui_smoke")
                self.assertEqual(benchmark_results["results"][0]["suite"]["id"], "semantic_comm")
                recipes = _get_json(base + "/api/recipes")
                recipe_names = {item["name"] for item in recipes["recipes"]}
                self.assertIn("resource_water_filling_baseline", recipe_names)
                benchmark_result = _get_json(
                    base + "/api/benchmarks/results/" + urllib.parse.quote("20260707T000000Z_ui_smoke")
                )
                self.assertEqual(benchmark_result["result"]["benchmark"]["name"], "UI Smoke Benchmark")
                research_catalog = _get_json(base + "/api/research/catalog")
                task_ids = [item["id"] for item in research_catalog["tasks"]]
                self.assertIn("image_reconstruction", task_ids)
                self.assertIn("text_semantic_similarity", task_ids)
                self.assertIn("classification", task_ids)
                self.assertIn("visual_question_answering", task_ids)
                template_catalog = _get_json(base + "/api/recipe-templates")
                self.assertEqual(template_catalog["status"], "valid")
                templates_by_task = {
                    item["task_id"]: item for item in template_catalog["templates"]
                    if item.get("default")
                }
                self.assertNotIn("classification", templates_by_task)
                self.assertEqual(
                    templates_by_task["visual_question_answering"]["recipe_path"],
                    "recipes/vqa_pretrained_transformers_smoke.yaml",
                )
                self.assertTrue(all(item["available"] for item in templates_by_task.values()))
                instantiated = _post_json(
                    base + "/api/recipe-templates/instantiate",
                    {
                        "template_id": "ai_phy.pilot_channel_estimation.adapter",
                        "overrides": {"name": "ui_instantiated_pilot_estimation"},
                    },
                )
                self.assertEqual(instantiated["status"], "instantiated")
                self.assertEqual(
                    instantiated["recipe"]["name"],
                    "ui_instantiated_pilot_estimation",
                )
                self.assertEqual(
                    instantiated["provenance"]["template_id"],
                    "ai_phy.pilot_channel_estimation.adapter",
                )
                self.assertEqual(
                    instantiated["recipe"]["metadata"]["template_provenance"],
                    instantiated["provenance"],
                )
                self.assertFalse(
                    any(
                        str(key).startswith("ui_")
                        for key in instantiated["recipe"]["metadata"]
                    )
                )
                matrix_source = _get_json(
                    base + "/api/recipe?path="
                    + urllib.parse.quote("recipes/deepjscc_kodak_awgn_train.yaml")
                )["recipe"]
                matrix_expansion = _post_json(
                    base + "/api/recipe/matrix/expand",
                    {"recipe": matrix_source},
                )
                self.assertEqual(matrix_expansion["expanded_count"], 6)
                self.assertEqual(
                    [
                        item["metadata"]["matrix_selection"]["channel.snr_db"]
                        for item in matrix_expansion["recipes"]
                    ],
                    [-4, 0, 4, 8, 12, 16],
                )
                self.assertTrue(
                    all("matrix" not in item["metadata"] for item in matrix_expansion["recipes"])
                )
                app_js = urllib.request.urlopen(base + "/static/app.js", timeout=5).read().decode("utf-8")
                self.assertIn("metrics.vqa", app_js)
                self.assertIn('const fromSweep = summarySweepNumber(summary, "channel.snr_db");', app_js)
                self.assertIn('formatMetricDisplay("SNR", summarySnrDb(summary, details))', app_js)
                self.assertIn("data-communication-settings", app_js)
                self.assertIn("data-channel-response-settings", app_js)
                self.assertIn("data-channel-response-entry", app_js)
                self.assertIn("data-channel-response-legend-id", app_js)
                self.assertIn("Shannon Spectral Efficiency", app_js)
                self.assertIn("resourceSeriesHasPowerSweep", app_js)
                self.assertIn("data-results-suite-filter", app_js)
                self.assertIn("suite-status-tag", app_js)
                self.assertIn("suiteIsSelectableForResults", app_js)
                self.assertIn("Differentiable Export", app_js)
                self.assertIn("/api/recipe/differentiable-inspect", app_js)
                self.assertIn("/api/recipe/differentiable-export", app_js)
                self.assertIn("/api/recipe/differentiable-export-graph", app_js)
                self.assertIn("/api/dataset-capture-jobs", app_js)
                self.assertIn("Training bundle", app_js)
                self.assertIn("Capture all datasets", app_js)
                self.assertNotIn("Include demo starter", app_js)
                self.assertNotIn("Demo starter", app_js)
                self.assertIn("Export training bundle", app_js)
                self.assertIn("model ABI + data contract", app_js)
                self.assertNotIn("Export Standalone Training Project", app_js)
                self.assertNotIn("include_trainer", app_js)
                self.assertNotIn('"noema.standalone_training_project"', app_js)
                self.assertNotIn("request.exporter", app_js)
                self.assertNotIn("allocator block marks the CSI/power interface and artifact return target", app_js)
                self.assertIn("Dataset capture", app_js)
                self.assertIn("External model", app_js)
                self.assertIn("differentiableExportHasManagedProject", app_js)
                self.assertIn("data-exported-project-capture-all", app_js)
                self.assertIn("data-exported-project-capture", app_js)
                self.assertIn("/api/workbench/exported-project", app_js)
                self.assertIn("/api/trained-artifacts", app_js)
                self.assertIn("data-schema-trained-artifact", app_js)
                self.assertNotIn("Use trained artifact", app_js)
                self.assertIn("exportedProjectCapturePlanMarkup", app_js)
                self.assertNotIn("data-exported-project-benchmark", app_js)
                self.assertNotIn("Run equal vs water filling vs learned", app_js)
                self.assertNotIn('effectiveExporter === "resource-allocation" ? exportedTrainingProjectWorkflowMarkup', app_js)
                self.assertIn("Learned allocation", app_js)
                self.assertIn("Captured signals", app_js)
                self.assertIn("Dataset size and splits", app_js)
                self.assertNotIn("data-dataset-capture-config-field", app_js)
                self.assertNotIn("data-dataset-capture-draft", app_js)
                self.assertNotIn("data-dataset-capture-add-tap", app_js)
                self.assertNotIn("data-dataset-capture-remove-tap", app_js)
                self.assertNotIn("Select on Graph", app_js)
                self.assertNotIn("Trainable slots and return targets", app_js)
                self.assertNotIn("Selected replacement blocks", app_js)
                self.assertIn("data-differentiable-export-slot", app_js)
                self.assertIn("Frozen support", app_js)
                self.assertNotIn("data-differentiable-open-graph-picker", app_js)
                self.assertNotIn("data-differentiable-export-mode-done", app_js)
                self.assertIn("data-differentiable-export-submit", app_js)
                self.assertNotIn("data-dataset-capture-submit", app_js)
                self.assertNotIn("nodeDifferentiableExportProbe", app_js)
                self.assertIn("nodeDifferentiabilityBadge", app_js)
                self.assertIn("GRAPH_RUNNER_MODES", app_js)
                self.assertIn("nodeContractBadges", app_js)
                self.assertIn("nodeBackendInfo", app_js)
                self.assertIn("nodeEquivalenceInfo", app_js)
                self.assertIn("Workflow view", app_js)
                self.assertIn("openRecipeLibraryDialog", app_js)
                self.assertIn("data-recipe-library-dialog", app_js)
                self.assertIn("openRecipeFromLibrary", app_js)
                self.assertIn("Changes affect this open recipe only", app_js)
                self.assertIn("data-graph-add-block", app_js)
                self.assertIn("data-open-recipe-in-new-tab", app_js)
                self.assertIn("data-node-output", app_js)
                self.assertIn("data-node-input", app_js)
                self.assertIn("pendingGraphLinkMarkup", app_js)
                self.assertNotIn("data-graph-edit-links", app_js)
                self.assertIn("bindSchemaDrivenRecipeConfigurator", app_js)
                self.assertIn("compatibleOutputReferences", app_js)
                self.assertNotIn("data-resource-allocation-entry", app_js)
                self.assertIn("data-resource-allocation-snapshot", app_js)
                self.assertIn("resource-power-line", app_js)
                self.assertIn("shared by all runs", app_js)
                self.assertIn("data-resource-allocation-legend-id", app_js)
                self.assertIn("Subcarrier SNR &amp; Power", app_js)
                self.assertIn("No differentiable export", app_js)
                self.assertIn("bit-perfect bypass", app_js)
                self.assertIn("identity PHY", app_js)
                self.assertNotIn("Trainable block markers", app_js)
                self.assertNotIn("Phase 3", app_js)
                self.assertIn("xLog", app_js)
                self.assertIn('colorBy: "recipe"', app_js)
                self.assertNotIn("resultColorBy", app_js)
                graph = _get_json(
                    base
                    + "/api/recipe/graph?path="
                    + urllib.parse.quote("recipes/compressai_kodak_default.yaml")
                )
                self.assertEqual(graph["recipe"], "image_source_coded_link")
                self.assertIn("runner_support", graph["nodes"][0])
                self.assertIn("equivalence", graph["nodes"][0])
                training = _post_json(
                    base + "/api/recipe/differentiable-inspect",
                    {
                        "recipe": {
                            "schema_version": 1,
                            "name": "ui_text_jscc_training",
                            "steps": [
                                {"id": "data", "op": "source.text_dataset", "params": {"dataset": "semantic_text_smoke"}},
                                {"id": "sender", "op": "model.text_bart_jscc_encode", "inputs": {"texts": "data.texts"}},
                                {"id": "tx_symbol_boundary", "op": "channel.symbol_boundary", "inputs": {"symbols": "sender.symbols"}},
                                {"id": "wireless_channel", "op": "wireless.channel", "inputs": {"symbols": "tx_symbol_boundary.symbols"}, "params": {"channel": "awgn", "snr_db": 20, "wireless_backend": "numpy"}},
                                {"id": "rx_symbol_boundary", "op": "channel.symbol_boundary", "inputs": {"symbols": "wireless_channel.rx_symbols"}},
                                {"id": "receiver", "op": "model.text_bart_jscc_decode", "inputs": {"symbols": "rx_symbol_boundary.symbols"}},
                                {"id": "evaluation", "op": "metrics.text_semantic_similarity", "inputs": {"reference": "data.texts", "candidate": "receiver.texts"}},
                            ],
                        }
                    },
                )
                self.assertEqual(training["inspection"]["recommended_mode"], "dataset_capture")
                self.assertNotIn("sender", training["inspection"]["fine_tunable_blocks"])
                self.assertFalse(training["inspection"]["replacement_candidate_blocks"])
                self.assertIn("exportable_differentiable_blocks", training["inspection"])
                if training.get("export_graph"):
                    block_names = [item["block"] for item in training["export_graph"]["blocks"]]
                    self.assertIn("AwgnChannelBlock", block_names)
                else:
                    self.assertIn("differentiable export", training.get("export_graph_error", "").lower())
                if not torch_available():
                    graph_export_error = _post_json_error(
                        base + "/api/recipe/differentiable-export-graph",
                        {
                            "recipe": load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml").to_dict(),
                            "backend": "torch",
                            "out": str(Path(tmp) / "training_graph_export_ui"),
                            "force": True,
                        },
                    )
                    self.assertEqual(graph_export_error["status_code"], 400)
                    self.assertIn("optional dependencies", graph_export_error["body"].get("error", ""))
                else:
                    graph_export_out = Path(tmp) / "training_graph_export_ui"
                    training_graph_export = _post_json(
                        base + "/api/recipe/differentiable-export-graph",
                        {
                            "recipe": load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml").to_dict(),
                            "backend": "torch",
                            "out": str(graph_export_out),
                            "force": True,
                        },
                    )
                    self.assertEqual(training_graph_export["status"], "exported")
                    self.assertEqual(training_graph_export["export"]["kind"], "differentiable_graph")
                    self.assertTrue((graph_export_out / "export_graph.json").is_file())
                    self.assertTrue((graph_export_out / "scenario.py").is_file())
                export_out = Path(tmp) / "differentiable_export_ui"
                differentiable_export = _post_json(
                    base + "/api/recipe/differentiable-export",
                    {
                        "recipe": load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml").to_dict(),
                        "optimizable": "sender,receiver",
                        "loss": "image.mse",
                        "starter": "deepjscc-image",
                        "include_starter": True,
                        "framework": "torch",
                        "out": str(export_out),
                        "force": True,
                    },
                )
                self.assertEqual(differentiable_export["status"], "exported")
                self.assertEqual(
                    differentiable_export["export"]["kind"],
                    "noema.training_interface_bundle@1",
                )
                self.assertFalse(differentiable_export["export"]["include_starter"])
                self.assertNotIn("demo_starter", differentiable_export["export"])
                self.assertFalse((export_out / "train.py").exists())
                self.assertFalse((export_out / "model.py").exists())
                self.assertFalse((export_out / "reference_training").exists())
                self.assertTrue((export_out / "training_contract.yaml").is_file())
                exported_training_plan = yaml.safe_load(
                    (export_out / "training_plan.yaml").read_text(encoding="utf-8")
                )
                exported_contract = yaml.safe_load(
                    (export_out / "training_contract.yaml").read_text(encoding="utf-8")
                )
                self.assertEqual(exported_training_plan["loss_steps"], ["evaluation"])
                self.assertNotIn("objective", exported_training_plan)
                self.assertNotIn("starter", exported_training_plan)
                self.assertEqual(exported_contract["recipe_loss_steps"], ["evaluation"])
                self.assertTrue((export_out / "scenario_graph.json").is_file())
                self.assertTrue((export_out / "trained_artifact.template.yaml").is_file())
                self.assertTrue((export_out / "project_manifest.yaml").is_file())
                self.assertFalse((export_out / "adapter_template").exists())
                capture_out = Path(tmp) / "capture_ui"
                capture = _post_json(
                    base + "/api/recipe/dataset-capture-run",
                    {
                        "recipe": {
                            "schema_version": 1,
                            "name": "ui_capture_smoke",
                            "dataset_capture": {
                                "split": "train",
                                "samples": 2,
                                "taps": [
                                    {"id": "received_embedding", "from": "data.clip_embeddings"},
                                    {"id": "target_mask", "from": "data.segmentation_reference"},
                                ],
                            },
                            "steps": [
                                {
                                    "id": "data",
                                    "op": "source.semantic_artifacts_smoke",
                                    "params": {"embedding_dim": 4},
                                }
                            ],
                        },
                        "out": str(capture_out),
                        "force": True,
                    },
                )
                self.assertEqual(capture["status"], "captured")
                self.assertEqual(capture["dataset_capture"]["tap_count"], 2)
                self.assertTrue((capture_out / "schema.json").is_file())
                self.assertTrue((capture_out / "shards" / "shard_0000.npz").is_file())
                run = _post_json(
                    base + "/api/recipe/run-payload",
                    {
                        "recipe": {
                            "schema_version": 1,
                            "name": "ui_payload_recipe",
                            "steps": [
                                {
                                    "id": "data",
                                    "op": "source.image_dataset",
                                    "params": {
                                        "dataset": "kodak",
                                        "dataset_dir": ".noema/datasets/kodak",
                                        "image_ids": "kodim01",
                                        "crop_size": 128,
                                    },
                                    "inputs": {},
                                }
                            ],
                        }
                    },
                )
                self.assertEqual(run["status"], "completed")
                summary = _get_json(base + "/api/runs/" + urllib.parse.quote(run["run_id"]))
                self.assertEqual(summary["recipe"]["name"], "ui_payload_recipe")
                image_path = summary["steps"][0]["outputs"]["images"]["path"]
                image = urllib.request.urlopen(
                    base + "/api/image?path=" + urllib.parse.quote(image_path),
                    timeout=5,
                ).read()
                self.assertEqual(image[:2], b"BM")
                runs = _get_json(base + "/api/runs")
                run_ids = {item["run_id"] for item in runs["runs"]}
                self.assertEqual(len(runs["runs"]), 2)
                self.assertIn(run["run_id"], run_ids)
                self.assertIn(capture["dataset_capture"]["run_id"], run_ids)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

    def test_ui_run_job_api_logs_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                Path(tmp),
                ROOT,
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                payload = {
                    "recipe": {
                        "schema_version": 1,
                        "name": "ui_job_recipe",
                        "steps": [
                            {
                                "id": "data",
                                "op": "source.image_dataset",
                                "params": {
                                    "dataset": "kodak",
                                    "dataset_dir": ".noema/datasets/kodak",
                                    "image_ids": "kodim01",
                                    "crop_size": 64,
                                },
                                "inputs": {},
                            }
                        ],
                    }
                }
                job = _post_json(base + "/api/run-jobs", payload)
                self.assertIn(job["status"], {"queued", "running", "completed"})
                for _ in range(80):
                    job = _get_json(base + "/api/run-jobs/" + urllib.parse.quote(job["job_id"]))
                    if job["status"] in {"completed", "failed", "canceled"}:
                        break
                    time.sleep(0.1)
                self.assertEqual(job["status"], "completed")
                self.assertTrue(job["run_id"])
                kinds = [event["kind"] for event in job["events"]]
                self.assertIn("step_started", kinds)
                self.assertIn("step_completed", kinds)
                summary = _get_json(base + "/api/runs/" + urllib.parse.quote(job["run_id"]))
                self.assertEqual(summary["recipe"]["name"], "ui_job_recipe")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

    def test_ui_exported_project_capture_and_trained_artifact_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".noema"
            project_dir = root / "differentiable_exports" / "demo"
            project_dir.mkdir(parents=True)
            capture_recipe_path = project_dir / "capture_train_recipe.yaml"
            capture_recipe = {
                "schema_version": 1,
                "name": "ui_exported_project_capture",
                "dataset_capture": {
                    "split": "train",
                    "samples": 2,
                    "shard_size": 2,
                    "max_runs": 1,
                    "seed_mode": "fixed_seed",
                    "taps": [{"id": "features", "from": "data.clip_embeddings"}],
                },
                "steps": [
                    {
                        "id": "data",
                        "op": "source.semantic_artifacts_smoke",
                        "params": {"embedding_dim": 4},
                    }
                ],
            }
            capture_recipe_path.write_text(json.dumps(capture_recipe), encoding="utf-8")
            checkpoint_path = project_dir / "checkpoints" / "model.npz"
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_bytes(b"hash-pinned-model")
            checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            artifact_manifest_path = project_dir / "trained_artifact.yaml"
            artifact_manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "noema.trained_block_artifact",
                        "id": "demo.tx_power.csi_allocator",
                        "name": "Demo CSI allocator",
                        "label": "Learned demo allocator",
                        "artifact": {
                            "path": "checkpoints/model.npz",
                            "sha256": checkpoint_sha,
                            "format": "noema_csi_power_deepset_npz_v1",
                        },
                        "compatible_operations": [
                            {
                                "operation": "model.symbol_power_allocator",
                                "required_inputs": ["channel_state"],
                                "params": {
                                    "policy": "learned_checkpoint",
                                    "granularity": "per_subcarrier",
                                    "budget_mode": "fixed_average",
                                    "checkpoint_path": "checkpoints/model.npz",
                                    "checkpoint_sha256": checkpoint_sha,
                                    "checkpoint_format": "noema_csi_power_deepset_npz_v1",
                                    "checkpoint_strict": True,
                                },
                            }
                        ],
                        "source": {"training_template": "test.template"},
                    }
                ),
                encoding="utf-8",
            )
            capture_out = root / ".noema" / "dataset_captures" / "ui_exported_project_capture_train"
            manifest = {
                "schema_version": 1,
                "kind": "noema.standalone_training_project",
                "exporter": "resource-allocation",
                "training_template": "test.template",
                "capture_jobs": [
                    {
                        "split": "train",
                        "label": "Train",
                        "recipe_path": str(capture_recipe_path.relative_to(root)),
                        "output_dir": str(capture_out.relative_to(root)),
                        "requested_samples": 2,
                        "tap_id": "features",
                    }
                ],
                "training": {
                    "working_directory": str(project_dir.relative_to(root)),
                    "command": "uv run python train.py",
                    "checkpoint_path": str(checkpoint_path.relative_to(root)),
                    "history_path": str((project_dir / "training_history.json").relative_to(root)),
                },
                "evaluation": {
                    "metrics_path": str((project_dir / "evaluation_metrics.json").relative_to(root)),
                },
                "trained_artifacts": [
                    {
                        "role": "trained_model",
                        "operation": "model.symbol_power_allocator",
                        "step_id": "tx_power",
                        "manifest_path": str(artifact_manifest_path.relative_to(root)),
                    }
                ],
            }
            (project_dir / "project_manifest.yaml").write_text(json.dumps(manifest), encoding="utf-8")

            server, thread = start_ui_server_in_thread("127.0.0.1", 0, workspace, root)
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                project = _get_json(
                    base
                    + "/api/workbench/exported-project?path="
                    + urllib.parse.quote(str(project_dir.relative_to(root)))
                )
                self.assertEqual(project["captures"][0]["status"], "pending")
                self.assertTrue(project["artifacts_ready"])
                self.assertEqual(project["trained_artifacts"][0]["id"], "demo.tx_power.csi_allocator")
                capture_job = _post_json(
                    base + "/api/dataset-capture-jobs",
                    {
                        "path": str(capture_recipe_path.relative_to(root)),
                        "out": str(capture_out.relative_to(root)),
                        "force": True,
                    },
                )
                for _ in range(100):
                    if capture_job["status"] not in {"queued", "running"}:
                        break
                    time.sleep(0.05)
                    capture_job = _get_json(
                        base
                        + "/api/dataset-capture-jobs/"
                        + urllib.parse.quote(capture_job["job_id"])
                    )
                self.assertEqual(capture_job["status"], "completed", capture_job.get("error"))
                self.assertEqual(capture_job["progress"]["percent"], 100.0)
                self.assertEqual(capture_job["dataset_capture"]["captured_samples"], 2)
                project = _get_json(
                    base
                    + "/api/workbench/exported-project?path="
                    + urllib.parse.quote(str(project_dir.relative_to(root)))
                )
                self.assertEqual(project["captures"][0]["status"], "complete")
                self.assertTrue(project["ready_for_external_training"])
                discovered = _get_json(
                    base + "/api/trained-artifacts?operation=model.symbol_power_allocator"
                )
                self.assertEqual(len(discovered["artifacts"]), 1)
                artifact = discovered["artifacts"][0]
                self.assertTrue(artifact["ready"])
                self.assertEqual(artifact["artifact"]["actual_sha256"], checkpoint_sha)
                binding = artifact["compatible_operations"][0]
                self.assertEqual(binding["required_inputs"], ["channel_state"])
                self.assertEqual(binding["params"]["policy"], "learned_checkpoint")
                self.assertEqual(
                    binding["params"]["checkpoint_path"],
                    str(checkpoint_path.relative_to(root)),
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

    def test_ui_learned_checkpoint_readiness_requires_matching_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "allocator.npz"
            checkpoint.write_bytes(b"safe-checkpoint")
            expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            recipe_path = root / "benchmark_recipe.yaml"
            payload = {
                "schema_version": 1,
                "name": "learned_checkpoint_readiness",
                "metadata": {"training_performed": True},
                "steps": [
                    {
                        "id": "tx_power",
                        "op": "model.symbol_power_allocator",
                        "params": {
                            "policy": "learned_checkpoint",
                            "checkpoint_path": str(checkpoint),
                            "checkpoint_sha256": expected,
                        },
                    }
                ],
            }
            recipe_path.write_text(json.dumps(payload), encoding="utf-8")
            readiness = _learned_checkpoint_readiness(recipe_path)
            self.assertTrue(readiness["valid"])
            self.assertEqual(readiness["actual_sha256"], expected)

            payload["steps"][0]["params"]["checkpoint_sha256"] = "0" * 64
            recipe_path.write_text(json.dumps(payload), encoding="utf-8")
            readiness = _learned_checkpoint_readiness(recipe_path)
            self.assertFalse(readiness["valid"])
            self.assertIn("does not match", "; ".join(readiness["blockers"]))

    def test_executor_cancellation_marks_run_canceled(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = build_registry()
            recipe = recipe_from_dict(
                {
                    "schema_version": 1,
                    "name": "cancel_me",
                    "steps": [
                        {
                            "id": "data",
                            "op": "source.image_dataset",
                            "inputs": {},
                            "params": {
                                "dataset": "kodak",
                                "dataset_dir": ".noema/datasets/kodak",
                                "image_ids": "kodim01",
                                "crop_size": 64,
                            },
                        }
                    ],
                }
            )
            store = LocalStore(Path(tmp))
            cancel_event = threading.Event()
            cancel_event.set()
            events = []
            with self.assertRaises(ExecutionCancelled):
                LocalExecutor(registry, store).run(recipe, event_sink=events.append, cancel_event=cancel_event)
            runs = store.list_runs()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["status"], "canceled")
            self.assertTrue(any(event["kind"] == "run_canceled" for event in events))

    def test_ui_server_saves_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            (project_root / "recipes").mkdir()
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                Path(tmp) / "workspace",
                project_root,
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                payload = {
                    "recipe": {
                        "schema_version": 1,
                        "name": "ui_saved_recipe",
                        "metadata": {
                            "ui_saved": True,
                            "sweeps": {"data.crop_size": "64,128"},
                        },
                        "steps": [
                            {
                                "id": "data",
                                "op": "source.image_dataset",
                                "params": {
                                    "dataset": "kodak",
                                    "dataset_dir": ".noema/datasets/kodak",
                                    "image_ids": "kodim01",
                                    "crop_size": 128,
                                },
                                "inputs": {},
                            }
                        ],
                    }
                }
                result = _post_json(base + "/api/recipe/save", payload)
                self.assertEqual(result["status"], "saved")
                self.assertEqual(result["validation"]["status"], "valid")
                self.assertNotIn("sweeps", result["recipe"]["metadata"])
                self.assertEqual(
                    result["recipe"]["metadata"]["matrix"],
                    {
                        "dimensions": {"data.crop_size": [64, 128]},
                        "step_params": {
                            "data": {
                                "crop_size": {"matrix": "data.crop_size"}
                            }
                        },
                    },
                )
                self.assertTrue((project_root / result["path"]).is_file())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

    def test_ui_dataset_images_api_lists_dataset_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            dataset_dir = project_root / ".noema" / "datasets" / "kodak"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "kodim01.png").write_bytes(b"placeholder")
            (dataset_dir / "not_kodak.png").write_bytes(b"placeholder")
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                Path(tmp) / "workspace",
                project_root,
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                payload = _get_json(
                    base
                    + "/api/datasets/images?dataset=kodak&dataset_dir="
                    + urllib.parse.quote(".noema/datasets/kodak")
                )
                self.assertEqual(payload["count"], 1)
                self.assertFalse(payload["complete"])
                self.assertEqual(payload["images"][0]["id"], "kodim01")
                self.assertEqual(payload["images"][0]["filename"], "kodim01.png")
                self.assertIn("kodim02.png", payload["missing"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

    def test_ops_show_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "unused.json"
            self.assertFalse(output_path.exists())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["--workspace", tmp, "ops", "show", "wireless.digital_link", "--json"])
            self.assertEqual(code, 0)

    def test_model_adapter_contracts_cover_bits_indices_and_latents(self):
        registry = build_registry()
        operations = {operation.id: operation.describe() for operation in registry.list()}
        self.assertNotIn(
            "download",
            operations["source.image_dataset"]["params_schema"]["properties"],
        )
        compressai_models = operations["model.compressai_encode"]["params_schema"]["properties"]["model"]["enum"]
        self.assertIn("bmshj2018_factorized", compressai_models)
        self.assertIn("bmshj2018_factorized_relu", compressai_models)
        self.assertIn("bmshj2018_hyperprior", compressai_models)
        self.assertIn("bmshj2018_hyperprior_vbr", compressai_models)
        self.assertIn("mbt2018_mean", compressai_models)
        self.assertIn("mbt2018_mean_vbr", compressai_models)
        self.assertIn("mbt2018", compressai_models)
        self.assertIn("mbt2018_vbr", compressai_models)
        self.assertIn("cheng2020_anchor", compressai_models)
        self.assertIn("cheng2020_attn", compressai_models)
        self.assertNotIn("ssf2020", compressai_models)
        self.assertIn("vbr_scale_index", operations["model.compressai_encode"]["params_schema"]["properties"])
        self.assertIn("vbr_stage", operations["model.compressai_encode"]["params_schema"]["properties"])
        self.assertEqual(
            operations["model.compressai_encode"]["output_kinds"]["bits"],
            "channel.payload_bits.numpy",
        )
        self.assertEqual(
            operations["model.compressai_analysis_encode"]["output_kinds"]["latents"],
            "semantic.latents.numpy",
        )
        self.assertEqual(
            operations["model.compressai_synthesis_decode"]["output_kinds"]["images"],
            "image.batch.numpy",
        )
        self.assertEqual(
            operations["model.jpeg_encode"]["output_kinds"]["bits"],
            "channel.payload_bits.numpy",
        )
        self.assertEqual(
            operations["source.text_dataset"]["output_kinds"]["texts"],
            "text.batch.json",
        )
        self.assertEqual(
            operations["model.text_utf8_encode"]["output_kinds"]["bits"],
            "channel.payload_bits.numpy",
        )
        self.assertEqual(
            operations["model.text_utf8_decode"]["output_kinds"]["texts"],
            "text.batch.json",
        )
        self.assertEqual(
            operations["metrics.text_semantic_similarity"]["input_kinds"]["reference"],
            ["text.batch.json"],
        )
        self.assertEqual(
            operations["model.tcm_encode"]["output_kinds"]["bits"],
            "channel.payload_bits.numpy",
        )
        self.assertEqual(
            operations["model.hpcm_encode"]["output_kinds"]["bits"],
            "channel.payload_bits.numpy",
        )
        self.assertEqual(
            operations["model.evc_encode"]["output_kinds"]["bits"],
            "channel.payload_bits.numpy",
        )
        self.assertEqual(
            operations["model.eflic_encode"]["output_kinds"]["bits"],
            "channel.payload_bits.numpy",
        )
        self.assertEqual(
            operations["model.eflic_onnx_encode"]["output_kinds"]["bits"],
            "channel.payload_bits.numpy",
        )
        self.assertEqual(
            operations["model.eflic_aoti_encode"]["output_kinds"]["bits"],
            "channel.payload_bits.numpy",
        )
        self.assertEqual(
            operations["model.eflic_openvino_encode"]["output_kinds"]["bits"],
            "channel.payload_bits.numpy",
        )
        self.assertEqual(
            operations["model.eflic_encode_indices"]["output_kinds"]["indices"],
            "semantic.indices.numpy",
        )
        self.assertEqual(
            operations["model.eflic_indices_to_bits"]["output_kinds"]["bits"],
            "channel.payload_bits.numpy",
        )
        self.assertEqual(
            operations["model.eflic_bits_to_indices"]["output_kinds"]["indices"],
            "semantic.indices.numpy",
        )
        self.assertEqual(
            operations["model.eflic_decode_indices"]["output_kinds"]["images"],
            "image.batch.numpy",
        )
        self.assertEqual(
            operations["model.eflic_onnx_encode_indices"]["output_kinds"]["indices"],
            "semantic.indices.numpy",
        )
        self.assertEqual(
            operations["model.eflic_aoti_encode_indices"]["output_kinds"]["indices"],
            "semantic.indices.numpy",
        )
        self.assertEqual(
            operations["model.eflic_openvino_encode_indices"]["output_kinds"]["indices"],
            "semantic.indices.numpy",
        )
        self.assertEqual(
            operations["model.eflic_export_onnx"]["output_kinds"]["model"],
            "model.onnx.bundle",
        )
        self.assertEqual(
            operations["model.eflic_export_aoti"]["output_kinds"]["model"],
            "model.aot_inductor.bundle",
        )
        self.assertIn("repo_path", operations["model.tcm_encode"]["params_schema"]["properties"])
        self.assertIn("checkpoint", operations["model.hpcm_encode"]["params_schema"]["properties"])
        self.assertIn("rate_idx", operations["model.evc_encode"]["params_schema"]["properties"])
        self.assertIn("force_ind", operations["model.eflic_encode"]["params_schema"]["properties"])
        self.assertIn("force_ind", operations["model.eflic_export_onnx"]["params_schema"]["properties"])
        self.assertIn("force_ind", operations["model.eflic_export_aoti"]["params_schema"]["properties"])
        self.assertIn("availability", operations["model.tcm_encode"])
        self.assertIn("availability", operations["model.hpcm_encode"])
        self.assertIn("availability", operations["model.evc_encode"])
        self.assertIn("availability", operations["model.eflic_encode"])
        self.assertIn("quality", operations["model.jpeg_encode"]["params_schema"]["properties"])
        self.assertIn("availability", operations["model.jpeg_encode"])
        self.assertEqual(
            operations["model.external_encode_indices"]["output_kinds"]["indices"],
            "semantic.indices.numpy",
        )
        self.assertEqual(
            operations["model.diffusers_autoencoderkl_encode"]["output_kinds"]["latents"],
            "semantic.latents.numpy",
        )
        self.assertIn("availability", operations["model.compressai_encode"])
        self.assertIn("available", operations["model.compressai_encode"]["availability"])
        self.assertIn("availability", operations["model.diffusers_autoencoderkl_encode"])
        self.assertIn("available", operations["model.diffusers_autoencoderkl_encode"]["availability"])
        self.assertEqual(
            operations["model.deepjscc_external_encode"]["output_kinds"]["symbols"],
            "channel.symbols.complex_numpy",
        )
        self.assertEqual(
            operations["foundation.diffusion_state_to_image"]["output_kinds"]["images"],
            "image.batch.numpy",
        )
        self.assertEqual(
            operations["metrics.embedding_similarity"]["input_kinds"]["reference"],
            ["foundation.embedding.numpy", "vision.embedding.clip.numpy", "multimodal.embedding.numpy"],
        )

    def test_aoti_loader_compat_handles_torch_signature_change(self):
        class NewStyleInductor:
            def __init__(self):
                self.calls = []

            def aoti_load_package(self, path, run_single_threaded=False):
                self.calls.append((path, run_single_threaded))
                return {"path": path, "run_single_threaded": run_single_threaded}

        class OldStyleInductor:
            def __init__(self):
                self.calls = []

            def aoti_load_package(self, path, device_index=-1):
                self.calls.append((path, device_index))
                return {"path": path, "device_index": device_index}

        class TorchStub:
            def __init__(self, inductor):
                self._inductor = inductor

        new_inductor = NewStyleInductor()
        self.assertEqual(
            _compressai_aoti_load_package_compat(TorchStub(new_inductor), "model.pt2", -1),
            {"path": "model.pt2", "run_single_threaded": False},
        )
        self.assertEqual(new_inductor.calls, [("model.pt2", False)])

        old_inductor = OldStyleInductor()
        self.assertEqual(
            _eflic_aoti_load_package_compat(TorchStub(old_inductor), "eflic.pt2", 0),
            {"path": "eflic.pt2", "device_index": 0},
        )
        self.assertEqual(old_inductor.calls, [("eflic.pt2", 0)])

    def test_phase11_diffusion_generation_recipe_validates(self):
        registry = build_registry()
        recipe = load_recipe(ROOT / "recipes" / "diffusion_flickr8k_generation.yaml")
        validate_recipe_against_registry(recipe, registry)
        specs = research_specs_from_recipe(recipe)
        self.assertEqual(specs["task"]["id"], "image_generation")
        self.assertEqual(specs["dataset"]["id"], "flickr8k")
        metric_ids = {item["id"] for item in specs["metrics"]}
        self.assertIn("generation.text_image_clip.cosine_mean", metric_ids)
        self.assertIn("generation.image_image_clip.cosine_mean", metric_ids)

    def test_upstream_google_drive_download_uses_bounded_direct_endpoint(self):
        direct = _google_drive_direct_download_url(
            "https://drive.google.com/file/d/abc123/view?usp=sharing"
        )
        self.assertTrue(direct.startswith("https://drive.usercontent.google.com/download?"))
        self.assertIn("id=abc123", direct)
        self.assertIn("confirm=t", direct)

    def test_upstream_checkpoint_validator_rejects_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pth"
            path.write_text("<!-- login shell --><!doctype html><html><title>Sign in</title></html>", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "HTML page"):
                _validate_checkpoint_file(path)

    def test_model_adapter_recipes_validate_by_artifact_kind(self):
        registry = build_registry()
        base_source = {
            "id": "data",
            "op": "source.image_dataset",
            "params": {
                "dataset": "kodak",
                "dataset_dir": ".noema/datasets/kodak",
                "image_ids": "kodim01",
                "crop_size": 128,
            },
            "inputs": {},
        }
        compressai_recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "compressai_contract_recipe",
                "steps": [
                    base_source,
                    {
                        "id": "encoder",
                        "op": "model.compressai_encode",
                        "inputs": {"images": "data.images"},
                        "params": {"model": "bmshj2018_hyperprior", "quality": 3, "metric": "mse"},
                    },
                    {
                        "id": "channel_encoder",
                        "op": "channel.identity_encoder",
                        "inputs": {"bits": "encoder.bits"},
                        "params": {},
                    },
                ],
            }
        )
        validate_recipe_against_registry(compressai_recipe, registry)

        jpeg_recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "jpeg_contract_recipe",
                "steps": [
                    base_source,
                    {
                        "id": "encoder",
                        "op": "model.jpeg_encode",
                        "inputs": {"images": "data.images"},
                        "params": {"quality": 75, "subsampling": "420"},
                    },
                    {
                        "id": "decoder",
                        "op": "model.jpeg_decode",
                        "inputs": {"bits": "encoder.bits"},
                        "params": {},
                    },
                ],
            }
        )
        validate_recipe_against_registry(jpeg_recipe, registry)

        eflic_recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "eflic_contract_recipe",
                "steps": [
                    base_source,
                    {
                        "id": "encoder",
                        "op": "model.eflic_encode",
                        "inputs": {"images": "data.images"},
                        "params": {
                            "repo_path": ".noema/upstreams/EF-LIC",
                            "checkpoint": ".noema/checkpoints/eflic/checkpoint.pth.tar",
                            "force_ind": 2,
                            "device": "cpu",
                        },
                    },
                    {
                        "id": "decoder",
                        "op": "model.eflic_decode",
                        "inputs": {"bits": "encoder.bits"},
                        "params": {"on_error": "fail"},
                    },
                ],
            }
        )
        validate_recipe_against_registry(eflic_recipe, registry)

        eflic_split_recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "eflic_split_contract_recipe",
                "steps": [
                    base_source,
                    {
                        "id": "sender",
                        "op": "model.eflic_encode_indices",
                        "inputs": {"images": "data.images"},
                        "params": {
                            "repo_path": ".noema/upstreams/EF-LIC",
                            "checkpoint": ".noema/checkpoints/eflic/checkpoint.pth.tar",
                            "force_ind": 2,
                            "device": "cpu",
                        },
                    },
                    {
                        "id": "payload_encoder",
                        "op": "model.eflic_indices_to_bits",
                        "inputs": {"indices": "sender.indices"},
                        "params": {},
                    },
                    {
                        "id": "payload_decoder",
                        "op": "model.eflic_bits_to_indices",
                        "inputs": {"bits": "payload_encoder.bits"},
                        "params": {},
                    },
                    {
                        "id": "receiver",
                        "op": "model.eflic_decode_indices",
                        "inputs": {"indices": "payload_decoder.indices"},
                        "params": {"on_error": "fail"},
                    },
                ],
            }
        )
        validate_recipe_against_registry(eflic_split_recipe, registry)

        latent_recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "latent_contract_recipe",
                "steps": [
                    base_source,
                    {
                        "id": "encoder",
                        "op": "model.diffusers_autoencoderkl_encode",
                        "inputs": {"images": "data.images"},
                        "params": {"model_id": "stabilityai/sd-vae-ft-mse"},
                    },
                    {
                        "id": "packer",
                        "op": "channel.latents_to_bits",
                        "inputs": {"latents": "encoder.latents"},
                        "params": {},
                    },
                    {
                        "id": "unpacker",
                        "op": "channel.bits_to_latents",
                        "inputs": {"bits": "packer.bits"},
                        "params": {},
                    },
                    {
                        "id": "decoder",
                        "op": "model.diffusers_autoencoderkl_decode",
                        "inputs": {"latents": "unpacker.latents"},
                        "params": {"model_id": "stabilityai/sd-vae-ft-mse"},
                    },
                ],
            }
        )
        validate_recipe_against_registry(latent_recipe, registry)

        external_index_recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "external_indices_contract_recipe",
                "steps": [
                    base_source,
                    {
                        "id": "encoder",
                        "op": "model.external_encode_indices",
                        "inputs": {"images": "data.images"},
                        "params": {"module": "my_codec", "callable": "encode"},
                    },
                    {
                        "id": "packer",
                        "op": "channel.indices_to_bits",
                        "inputs": {"indices": "encoder.indices"},
                        "params": {},
                    },
                ],
            }
        )
        validate_recipe_against_registry(external_index_recipe, registry)



    def test_verify_valid_completed_run_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace, run_id = self._write_verifier_run_fixture(Path(tmp))
            report = verify_run_bundle(LocalStore(workspace), run_id)
            self.assertEqual(report["status"], "valid", report)
            self.assertEqual(report["errors"], [])

    def test_verify_run_missing_manifest_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace, run_id = self._write_verifier_run_fixture(Path(tmp))
            (workspace / "runs" / run_id / "manifest.json").unlink()
            report = verify_run_bundle(LocalStore(workspace), run_id)
            self.assertEqual(report["status"], "invalid")
            self.assertTrue(any("manifest.json is missing" in message for message in report["errors"]))

    def test_verify_run_corrupted_json_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace, run_id = self._write_verifier_run_fixture(Path(tmp))
            (workspace / "runs" / run_id / "summary.json").write_text("{bad json", encoding="utf-8")
            report = verify_run_bundle(LocalStore(workspace), run_id)
            self.assertEqual(report["status"], "invalid")
            self.assertTrue(any("could not be parsed" in message for message in report["errors"]))

    def test_verify_run_missing_artifact_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace, run_id = self._write_verifier_run_fixture(Path(tmp))
            (workspace / "runs" / run_id / "artifacts" / "tx_bit_boundary" / "bits.npz").unlink()
            report = verify_run_bundle(LocalStore(workspace), run_id)
            self.assertEqual(report["status"], "invalid")
            self.assertTrue(any("artifact is missing" in message for message in report["errors"]))

    def test_verify_run_artifact_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace, run_id = self._write_verifier_run_fixture(Path(tmp))
            bit_path = workspace / "runs" / run_id / "artifacts" / "tx_bit_boundary" / "bits.npz"
            bit_path.write_bytes(bit_path.read_bytes() + b"changed")
            report = verify_run_bundle(LocalStore(workspace), run_id)
            self.assertEqual(report["status"], "invalid")
            self.assertTrue(any("hash mismatch" in message for message in report["errors"]))

    def test_verify_run_recipe_sha_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace, run_id = self._write_verifier_run_fixture(Path(tmp))
            recipe_path = workspace / "runs" / run_id / "recipe.json"
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            recipe["name"] = "tampered_recipe"
            recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True), encoding="utf-8")
            report = verify_run_bundle(LocalStore(workspace), run_id)
            self.assertEqual(report["status"], "invalid")
            self.assertTrue(any("recipe.json SHA" in message for message in report["errors"]))

    def test_verify_run_invalid_bpp_accounting_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace, run_id = self._write_verifier_run_fixture(Path(tmp), rate_bpp=3.0)
            report = verify_run_bundle(LocalStore(workspace), run_id)
            self.assertEqual(report["status"], "invalid")
            self.assertTrue(any("bpp" in message for message in report["errors"]))

    def test_verify_run_rejects_probability_metric_outside_unit_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace, run_id = self._write_verifier_run_fixture(Path(tmp))
            summary_path = workspace / "runs" / run_id / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["metrics"]["channel.payload.bler"] = 1.01
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            report = verify_run_bundle(LocalStore(workspace), run_id)
            self.assertEqual(report["status"], "invalid")
            self.assertTrue(any("[0, 1]" in message for message in report["errors"]))

    def test_verify_failed_run_is_non_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace, run_id = self._write_verifier_run_fixture(Path(tmp), status="failed")
            report = verify_run_bundle(LocalStore(workspace), run_id)
            self.assertEqual(report["status"], "invalid")
            self.assertTrue(any("non-comparable" in message for message in report["errors"]))

    def test_verify_benchmark_missing_required_metric_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".noema"
            result_dir = workspace / "benchmarks" / "benchmark_result_fixture"
            result_dir.mkdir(parents=True)
            result = {
                "schema_version": 1,
                "kind": "noema.benchmark_result",
                "benchmark": {
                    "id": "benchmark_v1_fixture",
                    "version": "1",
                    "metrics": [{"id": "quality.psnr_db"}],
                    "dataset": {"id": "fixture"},
                    "task": {"id": "image_reconstruction"},
                },
                "status": "completed",
                "created_at_utc": "2026-07-09T00:00:00Z",
                "completed_at_utc": "2026-07-09T00:00:01Z",
                "recipes": [{"id": "candidate", "status": "completed", "metrics": {}}],
            }
            (result_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            report = verify_benchmark_result(LocalStore(workspace), "benchmark_result_fixture")
            self.assertEqual(report["status"], "invalid")
            self.assertTrue(any("missing required metrics" in message for message in report["errors"]))

    def test_cli_runs_verify_json_and_human_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace, run_id = self._write_verifier_run_fixture(Path(tmp))
            (workspace / "runs" / run_id / "manifest.json").unlink()
            json_stdout = io.StringIO()
            json_stderr = io.StringIO()
            with contextlib.redirect_stdout(json_stdout), contextlib.redirect_stderr(json_stderr):
                code = main(["--workspace", str(workspace), "runs", "verify", run_id, "--json"])
            self.assertEqual(code, 1)
            payload = json.loads(json_stdout.getvalue())
            self.assertEqual(payload["status"], "invalid")

            human_stdout = io.StringIO()
            human_stderr = io.StringIO()
            with contextlib.redirect_stdout(human_stdout), contextlib.redirect_stderr(human_stderr):
                code = main(["--workspace", str(workspace), "runs", "verify", run_id])
            self.assertEqual(code, 1)
            self.assertIn("invalid run", human_stdout.getvalue())
            self.assertIn("error:", human_stdout.getvalue())

    def _write_verifier_run_fixture(self, root: Path, *, status: str = "completed", rate_bpp=None):
        workspace = root / ".noema"
        store = LocalStore(workspace)
        source_path = root / "verifier-images.npz"
        pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(1, 8, 8, 3)
        np.savez_compressed(source_path, images=pixels)
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "verify_fixture",
                "metadata": {
                    "rate_count_fixed_point": "tx_bit_boundary.channel.fixed.tx.bit_count"
                },
                "steps": [
                    {
                        "id": "data",
                        "op": "source.local_npz_images",
                        "inputs": {},
                        "params": {"path": str(source_path)},
                    },
                    {
                        "id": "sender",
                        "op": "model.jpeg_encode",
                        "inputs": {"images": "data.images"},
                        "params": {"quality": 75, "subsampling": "420"},
                    },
                    {
                        "id": "tx_bit_boundary",
                        "op": "channel.bit_boundary",
                        "inputs": {"bits": "sender.bits"},
                        "params": {"label": "tx", "role": "tx"},
                    },
                    {
                        "id": "receiver",
                        "op": "model.jpeg_decode",
                        "inputs": {"bits": "tx_bit_boundary.bits"},
                        "params": {"on_error": "fail"},
                    },
                    {
                        "id": "evaluation",
                        "op": "metrics.image_reconstruction",
                        "inputs": {
                            "reference": "data.images",
                            "reconstruction": "receiver.images",
                        },
                        "params": {},
                    },
                ],
            }
        )
        run_dir = LocalExecutor(build_registry(), store).run(recipe)
        run_id = run_dir.name
        summary_path = run_dir / "summary.json"
        manifest_path = run_dir / "manifest.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        tx_step = next(step for step in summary["steps"] if step["id"] == "tx_bit_boundary")
        bit_count = float(tx_step["metrics"]["channel.fixed.tx.bit_count"])
        expected_bpp = bit_count / float(pixels.shape[0] * pixels.shape[1] * pixels.shape[2])
        evaluation = next(step for step in summary["steps"] if step["id"] == "evaluation")
        evaluation["metrics"]["rate_bpp"] = expected_bpp if rate_bpp is None else rate_bpp
        manifest_evaluation = next(step for step in manifest["steps"] if step["id"] == "evaluation")
        manifest_evaluation["metrics"] = dict(evaluation["metrics"])
        manifest_evaluation["metrics_keys"] = sorted(evaluation["metrics"])
        manifest_evaluation["metrics_sha256"] = canonical_json_sha256(
            manifest_evaluation["metrics"]
        )
        for output in manifest_evaluation["outputs"].values():
            output["producer_metrics_sha256"] = manifest_evaluation[
                "metrics_sha256"
            ]
        for output in manifest["artifacts"]:
            if output.get("step_id") == "evaluation":
                output["producer_metrics_sha256"] = manifest_evaluation[
                    "metrics_sha256"
                ]
        summary["status"] = status
        manifest["status"] = status
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        manifest["summary"] = {
            "kind": "noema.run_summary",
            "relative_path": "summary.json",
            "sha256": file_sha256(summary_path),
            "size_bytes": summary_path.stat().st_size,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return workspace, run_id

    def _manifest_artifact(self, run_dir: Path, step_id: str, output_name: str, output: Artifact):
        return {
            "step_id": step_id,
            "output_name": output_name,
            "kind": output.kind,
            "path": str(output.path),
            "relative_path": str(output.path.relative_to(run_dir)),
            "sha256": output.sha256,
            "dtype": output.metadata.get("dtype"),
            "shape": output.metadata.get("shape"),
            "array": output.metadata.get("array"),
            "metadata": dict(output.metadata),
        }


def _get_json(url):
    return json.loads(urllib.request.urlopen(url, timeout=5).read().decode("utf-8"))


def _post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(request, timeout=10).read().decode("utf-8"))


def _post_json_error(url, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8"))
        return {"status_code": exc.code, "body": body}
    raise AssertionError("expected HTTP error from %s" % url)


def _write_rgb_png(path, image):
    import struct
    import zlib

    height, width, channels = image.shape
    if channels != 3:
        raise AssertionError("test PNG helper expects RGB")

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + np.asarray(image[row], dtype=np.uint8).tobytes() for row in range(height))
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(data)


if __name__ == "__main__":
    unittest.main()
