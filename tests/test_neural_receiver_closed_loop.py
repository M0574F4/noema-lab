from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from noema_lab.cli.main import main
from noema_lab.core.recipes import load_recipe, recipe_from_dict
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.trained_artifacts import inspect_trained_artifact
from noema_lab.core.training_plans import TrainingPlan, apply_training_plan
from noema_lab.ops import build_registry
from noema_lab.training.exporter import export_differentiable_scenario
from noema_lab.ui.server import (
    _learned_checkpoint_readiness,
    _runtime_artifact_recipe_blockers,
)


ROOT = Path(__file__).resolve().parents[1]


class NeuralReceiverClosedLoopTests(unittest.TestCase):
    def test_artifact_preflight_surfaces_operation_inspection_failures(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "broken_registry_preflight",
                "steps": [
                    {
                        "id": "source",
                        "op": "source.random_bits",
                        "inputs": {},
                        "params": {"bit_count": 8},
                    }
                ],
            }
        )

        class BrokenRegistry:
            def get(self, _operation_id):
                raise RuntimeError("registry inspection failed")

        blockers = _runtime_artifact_recipe_blockers(
            recipe,
            Path.cwd(),
            BrokenRegistry(),
        )
        self.assertEqual(len(blockers), 1)
        self.assertIn("registry inspection failed", blockers[0])

    def test_receiver_artifact_mode_requires_a_manifest_during_preflight_and_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = load_recipe(
                ROOT / "recipes" / "neural_receiver_qpsk_awgn_adapter.yaml"
            ).to_dict()
            demodulator = next(
                step for step in payload["steps"] if step["id"] == "demodulator"
            )
            demodulator["params"] = {
                "mode": "learned_artifact",
                "modulation": "qpsk",
                "artifact_entrypoint": "neural_receiver",
            }
            recipe = recipe_from_dict(payload)

            blockers = _runtime_artifact_recipe_blockers(
                recipe,
                root,
                build_registry(),
            )
            self.assertIn(
                "demodulator: trained artifact manifest is not selected",
                blockers,
            )

            recipe_path = root / "receiver.yaml"
            recipe_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            readiness = _learned_checkpoint_readiness(recipe_path, root)
            self.assertFalse(readiness["valid"])
            self.assertIn(
                "demodulator: trained artifact manifest is missing",
                readiness["blockers"],
            )

    def test_receiver_preflight_and_readiness_reject_an_unrelated_ready_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unrelated_manifest = _write_unrelated_ready_artifact(root / "unrelated")
            payload = load_recipe(
                ROOT / "recipes" / "neural_receiver_qpsk_awgn_adapter.yaml"
            ).to_dict()
            demodulator = next(
                step for step in payload["steps"] if step["id"] == "demodulator"
            )
            demodulator["params"] = {
                "mode": "learned_artifact",
                "modulation": "qpsk",
                "artifact_manifest_path": str(unrelated_manifest.relative_to(root)),
                "artifact_entrypoint": "neural_receiver",
            }
            recipe = recipe_from_dict(payload)
            expected = (
                "demodulator: trained artifact has no compatible binding for "
                "demodulation.neural_receiver_adapter entrypoint neural_receiver"
            )

            blockers = _runtime_artifact_recipe_blockers(
                recipe,
                root,
                build_registry(),
            )
            self.assertIn(expected, blockers)

            recipe_path = root / "receiver.yaml"
            recipe_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            readiness = _learned_checkpoint_readiness(recipe_path, root)
            self.assertFalse(readiness["valid"])
            self.assertIn(expected, readiness["blockers"])

    def test_real_receiver_export_discovers_labels_and_writes_three_split_jobs(self):
        recipe = apply_training_plan(
            load_recipe(ROOT / "recipes" / "neural_receiver_qpsk_iq_calibration.yaml"),
            TrainingPlan(
                dataset_capture={
                    "taps": [
                        {"id": "rx_symbols", "from": "receiver_frontend.rx_symbols"},
                        {"id": "target_bits", "from": "tx_bit_boundary.bits"},
                    ],
                    "split_plan": {
                        "total_samples": 48,
                        "percentages": {
                            "train": 66.6666666667,
                            "validation": 16.66666666665,
                            "test": 16.66666666665,
                        },
                    },
                }
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "receiver_export"
            payload = export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["demodulator"],
                loss="bit.bce",
                framework="torch",
                out_dir=out,
                project_root=root,
                exporter="neural-receiver",
                include_starter=True,
            )

            self.assertEqual(payload["starter_exporter"], "neural-receiver")
            self.assertEqual(
                [job["split"] for job in payload["capture_jobs"]],
                ["train", "validation", "test"],
            )
            contract = yaml.safe_load((out / "data_contract.yaml").read_text(encoding="utf-8"))
            self.assertEqual(contract["mode"], "captured_generic_tensors")
            self.assertEqual(
                {item["reference"] for item in contract["signals"]},
                {"receiver_frontend.rx_symbols", "tx_bit_boundary.bits"},
            )
            slot = yaml.safe_load((out / "training_contract.yaml").read_text(encoding="utf-8"))[
                "trainable_slots"
            ][0]
            self.assertEqual(slot["runtime_artifact_abi"]["entrypoint_id"], "neural_receiver")
            self.assertEqual(slot["runtime_artifact_abi"]["required_operation_inputs"], ["rx_symbols"])
            train_capture = yaml.safe_load(
                (out / "capture_train_recipe.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(train_capture["dataset_capture"]["seed_mode"], "increment_run_seed")
            self.assertEqual(
                train_capture["dataset_capture"]["taps"],
                [
                    {"id": "rx_symbols", "from": "receiver_frontend.rx_symbols"},
                    {"id": "target_bits", "from": "tx_bit_boundary.bits"},
                ],
            )
            self.assertFalse(
                any("seed" in dict(step.get("params") or {}) for step in train_capture["steps"])
            )
            for relative in (
                "model.py",
                "datamodule.py",
                "losses.py",
                "train.py",
                "evaluate.py",
                "requirements.txt",
            ):
                self.assertTrue((out / "reference_training" / relative).is_file(), relative)
            starter_config = yaml.safe_load(
                (out / "reference_training" / "train_config.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                starter_config["model"]["candidates"],
                [
                    {
                        "id": "affine_iq_calibrator",
                        "architecture": "affine",
                    },
                ],
            )
            self.assertEqual(
                starter_config["training"]["affine_max_iterations"],
                100,
            )
            self.assertEqual(
                starter_config["training"]["initialization_seeds"],
                [23],
            )
            self.assertEqual(
                starter_config["objective"]["checkpoint_selection"],
                [
                    "minimum_validation_bce",
                    "minimum_validation_ber",
                    "configured_candidate_order",
                    "configured_seed_order",
                    "earliest_epoch",
                ],
            )
            validation = subprocess.run(
                [sys.executable, "validate_contract.py"],
                cwd=str(out),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                validation.returncode,
                0,
                "%s\n%s" % (validation.stdout, validation.stderr),
            )
            self.assertIn("contract valid:", validation.stdout)

    def test_schema_v2_receiver_artifact_binds_and_runs_in_an_ordinary_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = _write_identity_receiver_artifact(root / "artifact")
            inspected = inspect_trained_artifact(
                artifact_path,
                project_root=root,
                registry=build_registry(),
            )
            if not inspected["runtime"]["available"]:
                self.skipTest("ONNX Runtime is unavailable")
            self.assertTrue(inspected["ready"], inspected["issues"])
            binding = inspected["compatible_operations"][0]
            self.assertEqual(binding["params"]["mode"], "learned_artifact")
            self.assertEqual(binding["required_inputs"], ["rx_symbols"])

            recipe = load_recipe(
                ROOT / "recipes" / "neural_receiver_qpsk_awgn_adapter.yaml"
            ).to_dict()
            recipe["name"] = "neural_receiver_returned_artifact_smoke"
            # This test constructs one explicit 30 dB smoke recipe rather than
            # executing the template's authored four-coordinate UI matrix.
            recipe["metadata"].pop("matrix", None)
            for step in recipe["steps"]:
                if step["id"] == "data":
                    step["params"]["bit_count"] = 128
                elif step["id"] == "wireless_channel":
                    step["params"]["snr_db"] = 30
                elif step["id"] == "demodulator":
                    step["params"] = {
                        "mode": "learned_artifact",
                        "modulation": "qpsk",
                        "artifact_manifest_path": str(artifact_path),
                        "artifact_entrypoint": "neural_receiver",
                        "artifact_package_sha256": inspected["package_sha256"],
                    }
            recipe_path = root / "recipe.yaml"
            recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    ["--workspace", str(root / "workspace"), "recipe", "run", str(recipe_path)]
                )
            self.assertEqual(code, 0, stdout.getvalue())
            summary_path = next(
                Path(line.split("summary:", 1)[1].strip())
                for line in stdout.getvalue().splitlines()
                if line.startswith("summary:")
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            steps = {step["id"]: step for step in summary["steps"]}
            metadata = steps["demodulator"]["outputs"]["bits"]["metadata"]
            self.assertEqual(metadata["receiver_adapter"], "learned_artifact")
            self.assertEqual(metadata["data_plane_backend"], "python_numpy")
            self.assertEqual(metadata["receiver_runtime_backend"], "onnxruntime")
            preview = metadata["receiver_decision_preview"]
            self.assertEqual(preview["kind"], "memoryless_qpsk_iq_decision_regions")
            self.assertEqual(preview["receiver_mode"], "learned_artifact")
            self.assertEqual(preview["model_sha256"], _sha256(artifact_path))
            self.assertEqual(preview["grid"]["width"], 64)
            self.assertEqual(preview["grid"]["height"], 64)
            self.assertEqual(len(preview["grid"]["class_rows"]), 64)
            self.assertEqual({point["bits"] for point in preview["constellation"]}, {"00", "01", "10", "11"})
            metrics = {}
            for step in summary["steps"]:
                metrics.update(step.get("metrics") or {})
            self.assertLess(float(metrics["channel.coded.ber"]), 0.1)


def _write_identity_receiver_artifact(package: Path) -> Path:
    try:
        import onnx
        from onnx import TensorProto, helper
    except Exception as exc:  # pragma: no cover - optional dependency gate
        raise unittest.SkipTest("onnx is unavailable: %s" % exc) from exc
    package.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "version": 1,
        "kind": "noema.trainable_slot_contract@1",
        "id": "noema.slot.neural-receiver-test",
    }
    contract_path = package / "training_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    component_path = package / "receiver.onnx"
    shape = ["symbol", 2]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["rx_symbols_ri"], ["bit_llr"])],
        "identity_neural_receiver",
        [helper.make_tensor_value_info("rx_symbols_ri", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("bit_llr", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.save_model(model, str(component_path))
    manifest = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": "test.identity-neural-receiver",
        "name": "Identity neural receiver",
        "contract": {
            "id": contract["id"],
            "version": 1,
            "path": contract_path.name,
            "sha256": canonical_json_sha256(contract),
            "file_sha256": _sha256(contract_path),
        },
        "components": [
            {
                "id": "receiver",
                "role": "neural_receiver",
                "path": component_path.name,
                "sha256": _sha256(component_path),
                "format": "onnx",
            }
        ],
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 1,
            "entrypoints": [
                {
                    "id": "neural_receiver",
                    "component": "receiver",
                    "inputs": [
                        {
                            "name": "rx_symbols_ri",
                            "dtype": "float32",
                            "shape": shape,
                            "semantic": "received_qpsk_symbols_real_imag",
                        }
                    ],
                    "outputs": [
                        {
                            "name": "bit_llr",
                            "dtype": "float32",
                            "shape": shape,
                            "semantic": "bit_llr_positive_bit_zero",
                        }
                    ],
                }
            ],
        },
        "application": {"mode": "single_binding"},
        "compatible_operations": [
            {
                "operation": "demodulation.neural_receiver_adapter",
                "runtime_entrypoint": "neural_receiver",
                "required_inputs": ["rx_symbols"],
                "params": {
                    "mode": "learned_artifact",
                    "modulation": "qpsk",
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": "neural_receiver",
                },
            }
        ],
    }
    manifest_path = package / "trained_artifact.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def _write_unrelated_ready_artifact(package: Path) -> Path:
    package.mkdir(parents=True, exist_ok=True)
    component_path = package / "artifact.bin"
    component_path.write_bytes(b"unrelated-but-hash-verified-artifact")
    manifest = {
        "schema_version": 1,
        "kind": "noema.trained_block_artifact",
        "id": "test.unrelated-artifact",
        "name": "Unrelated artifact",
        "artifact": {
            "path": component_path.name,
            "sha256": _sha256(component_path),
            "format": "test-binary-v1",
        },
        "compatible_operations": [
            {
                "operation": "source.random_bits",
                "required_inputs": [],
                "params": {},
            }
        ],
    }
    manifest_path = package / "trained_artifact.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
