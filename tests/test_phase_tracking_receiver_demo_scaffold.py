from __future__ import annotations

import contextlib
import hashlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from demo_trainings.prepare_example import prepare_example
from noema_lab.core.benchmarks import load_benchmark_pack, validate_benchmark_pack
from noema_lab.core.capture import run_dataset_capture_recipe
from noema_lab.core.recipes import load_recipe
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.core.training_plans import TrainingPlan, apply_training_plan
from noema_lab.ops import build_registry
from noema_lab.ops.phase_tracking import _decision_directed_pll_phase as runtime_pll
from noema_lab.ops.phase_tracking import _receiver_features_v3 as runtime_receiver_features
from noema_lab.core.reproducibility import derive_seed
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.training.exporter import export_differentiable_scenario


ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD = ROOT / "demo_trainings" / "neural_receiver_phase_tracking_qpsk"


class PhaseTrackingReceiverDemoScaffoldTests(unittest.TestCase):
    def test_demo_attachment_adds_phase_truth_to_generated_capture_assets(self):
        recipe = apply_training_plan(
            load_recipe(ROOT / "recipes" / "neural_receiver_qpsk_phase_tracking.yaml"),
            TrainingPlan(
                dataset_capture={
                    "taps": [
                        {
                            "id": "rx_symbols",
                            "from": "carrier_impairment.rx_symbols",
                        },
                        {
                            "id": "pilot_context",
                            "from": "modulator.pilot_context",
                        },
                        {"id": "target_bits", "from": "tx_bit_boundary.bits"},
                    ],
                    # Reproduce a bundle created by the older demo helper. The
                    # source recipe already has the equivalent metadata.matrix,
                    # so attachment must remove this duplicate definition.
                    "sweep": {
                        "wireless_channel.snr_db": [-2, 2, 6, 10],
                    },
                    "split_plan": {
                        "total_samples": 12,
                        "percentages": {
                            "train": 50,
                            "validation": 25,
                            "test": 25,
                        },
                    },
                }
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "phase_tracking"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["demodulator"],
                loss="",
                framework="torch",
                out_dir=bundle,
                project_root=root,
            )
            preserved_data = bundle / "data" / "train" / "researcher.bin"
            preserved_data.parent.mkdir(parents=True, exist_ok=True)
            preserved_data.write_bytes(b"preserve captured data")
            preserved_artifact = bundle / "artifacts" / "researcher.onnx"
            preserved_artifact.parent.mkdir(parents=True, exist_ok=True)
            preserved_artifact.write_bytes(b"preserve returned artifact")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                prepare_example(
                    bundle,
                    "phase-tracking-receiver",
                    project_root=root,
                )
            self.assertIn(
                "updated its generated capture plan with its training-only "
                "carrier-phase target and the recipe matrix as its single SNR "
                "variant definition",
                output.getvalue(),
            )

            config = yaml.safe_load(
                (bundle / "reference_training" / "train_config.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                config["data"]["phase_truth_tap"],
                "carrier_impairment_phase_truth",
            )
            self.assertTrue(config["data"]["phase_truth_used_as_training_target"])
            self.assertFalse(config["data"]["phase_truth_used_at_runtime"])
            plan = yaml.safe_load(
                (bundle / "training_plan.yaml").read_text(encoding="utf-8")
            )
            self.assertIn(
                {
                    "id": "carrier_impairment_phase_truth",
                    "from": "carrier_impairment.phase_truth",
                },
                plan["dataset_capture"]["taps"],
            )
            self.assertNotIn("sweep", plan["dataset_capture"])
            source = yaml.safe_load(
                (bundle / "noema_recipe.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                source["metadata"]["matrix"]["dimensions"]["channel.snr_db"],
                [-2, 2, 6, 10],
            )
            self.assertEqual(
                source["metadata"]["matrix"]["step_params"]["wireless_channel"][
                    "snr_db"
                ],
                {"matrix": "channel.snr_db"},
            )
            data_contract_path = bundle / "data_contract.yaml"
            data_contract = yaml.safe_load(
                data_contract_path.read_text(encoding="utf-8")
            )
            phase_signal = next(
                item
                for item in data_contract["signals"]
                if item["reference"] == "carrier_impairment.phase_truth"
            )
            self.assertEqual(phase_signal["role"], "additional_signal")
            self.assertFalse(phase_signal["required"])
            manifest = yaml.safe_load(
                (bundle / "project_manifest.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["data_contract"]["sha256"],
                canonical_json_sha256(data_contract),
            )
            self.assertEqual(
                manifest["data_contract"]["file_sha256"],
                hashlib.sha256(data_contract_path.read_bytes()).hexdigest(),
            )
            for split in ("train", "validation", "test"):
                capture = yaml.safe_load(
                    (bundle / ("capture_%s_recipe.yaml" % split)).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertIn(
                    "carrier_impairment.phase_truth",
                    [
                        str(item.get("from") or "")
                        for item in capture["dataset_capture"]["taps"]
                    ],
                )
                self.assertNotIn("sweep", capture["dataset_capture"])
                self.assertEqual(
                    capture["metadata"]["matrix"]["dimensions"][
                        "channel.snr_db"
                    ],
                    [-2, 2, 6, 10],
                )
                job = next(
                    item for item in manifest["capture_jobs"] if item["split"] == split
                )
                self.assertEqual(
                    job["expected_taps"],
                    capture["dataset_capture"]["taps"],
                )
                capture_path = bundle / ("capture_%s_recipe.yaml" % split)
                self.assertEqual(
                    job["recipe_file_sha256"],
                    hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                )
            capture_result = run_dataset_capture_recipe(
                load_recipe(bundle / "capture_train_recipe.yaml"),
                build_registry(),
                LocalStore(root / ".noema"),
                root / "runtime_capture",
            )
            self.assertEqual(capture_result["captured_samples"], 6)
            self.assertEqual(preserved_data.read_bytes(), b"preserve captured data")
            self.assertEqual(
                preserved_artifact.read_bytes(),
                b"preserve returned artifact",
            )

    def test_specialized_exporter_allows_phase_truth_to_be_omitted(self):
        recipe = apply_training_plan(
            load_recipe(ROOT / "recipes" / "neural_receiver_qpsk_phase_tracking.yaml"),
            TrainingPlan(
                dataset_capture={
                    "taps": [
                        {
                            "id": "rx_symbols",
                            "from": "carrier_impairment.rx_symbols",
                        },
                        {
                            "id": "pilot_context",
                            "from": "modulator.pilot_context",
                        },
                        {"id": "target_bits", "from": "tx_bit_boundary.bits"},
                    ],
                    "sweep": {
                        "wireless_channel.snr_db": [-2, 2, 6, 10],
                    },
                    "split_plan": {
                        "total_samples": 12,
                        "percentages": {
                            "train": 50,
                            "validation": 25,
                            "test": 25,
                        },
                    },
                }
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "phase_tracking"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["demodulator"],
                loss="bit.bce",
                framework="torch",
                out_dir=bundle,
                project_root=root,
                exporter="phase-tracking-receiver",
                include_starter=True,
            )

            config = yaml.safe_load(
                (bundle / "reference_training" / "train_config.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(config["data"]["phase_truth_tap"], "")
            self.assertFalse(config["data"]["phase_truth_used_as_training_target"])
            self.assertFalse(config["data"]["phase_truth_used_at_runtime"])

    def test_demo_attachment_preserves_and_rejects_conflicting_snr_sweep(self):
        recipe = apply_training_plan(
            load_recipe(ROOT / "recipes" / "neural_receiver_qpsk_phase_tracking.yaml"),
            TrainingPlan(
                dataset_capture={
                    "taps": [
                        {
                            "id": "rx_symbols",
                            "from": "carrier_impairment.rx_symbols",
                        },
                        {
                            "id": "pilot_context",
                            "from": "modulator.pilot_context",
                        },
                        {"id": "target_bits", "from": "tx_bit_boundary.bits"},
                    ],
                    "sweep": {"wireless_channel.snr_db": [0, 4, 8]},
                    "split_plan": {
                        "total_samples": 12,
                        "percentages": {
                            "train": 50,
                            "validation": 25,
                            "test": 25,
                        },
                    },
                }
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "phase_tracking"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["demodulator"],
                loss="",
                framework="torch",
                out_dir=bundle,
                project_root=root,
            )
            original_plan = (bundle / "training_plan.yaml").read_bytes()

            with self.assertRaisesRegex(
                ValueError,
                r"cannot combine its recipe matrix with a different "
                r"dataset_capture\.sweep",
            ):
                prepare_example(
                    bundle,
                    "phase-tracking-receiver",
                    project_root=root,
                )

            self.assertEqual(
                (bundle / "training_plan.yaml").read_bytes(),
                original_plan,
            )
            preserved = yaml.safe_load(original_plan)
            self.assertEqual(
                preserved["dataset_capture"]["sweep"],
                {"wireless_channel.snr_db": [0, 4, 8]},
            )
            self.assertNotIn(
                "carrier_impairment.phase_truth",
                [
                    str(item.get("from") or "")
                    for item in preserved["dataset_capture"]["taps"]
                ],
            )

    def test_exporter_captures_public_context_but_excludes_phase_truth_from_abi(self):
        recipe = apply_training_plan(
            load_recipe(ROOT / "recipes" / "neural_receiver_qpsk_phase_tracking.yaml"),
            TrainingPlan(
                dataset_capture={
                    "taps": [
                        {
                            "id": "rx_symbols",
                            "from": "carrier_impairment.rx_symbols",
                        },
                        {
                            "id": "pilot_context",
                            "from": "modulator.pilot_context",
                        },
                        {"id": "target_bits", "from": "tx_bit_boundary.bits"},
                        {
                            "id": "phase_truth",
                            "from": "carrier_impairment.phase_truth",
                        },
                    ],
                    "sweep": {
                        "wireless_channel.snr_db": [-2, 2, 6, 10],
                    },
                    "split_plan": {
                        "total_samples": 12,
                        "percentages": {
                            "train": 50,
                            "validation": 25,
                            "test": 25,
                        },
                    },
                }
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "phase_tracking"
            result = export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["demodulator"],
                loss="bit.bce",
                framework="torch",
                out_dir=bundle,
                project_root=root,
                exporter="phase-tracking-receiver",
                include_starter=True,
            )
            self.assertEqual(result["starter_exporter"], "phase-tracking-receiver")
            slot = yaml.safe_load(
                (bundle / "training_contract.yaml").read_text(encoding="utf-8")
            )["trainable_slots"][0]
            abi = slot["runtime_artifact_abi"]
            self.assertEqual(
                abi["required_operation_inputs"],
                ["rx_symbols", "pilot_context"],
            )
            self.assertEqual(set(abi["inputs"]), {"receiver_features_v3"})
            self.assertNotIn("phase", json.dumps(abi["inputs"]).lower())

            config = yaml.safe_load(
                (bundle / "reference_training" / "train_config.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(config["data"]["phase_truth_tap"], "phase_truth")
            self.assertTrue(config["data"]["phase_truth_used_as_training_target"])
            self.assertFalse(config["data"]["phase_truth_used_at_runtime"])
            self.assertEqual(
                config["objective"]["checkpoint_selection"][:2],
                [
                    "pilot_smoothing_noninferiority_guard",
                    "minimum_validation_ber_improvement",
                ],
            )
            self.assertEqual(config["evaluation"]["pll_alpha"], 0.12)
            self.assertEqual(config["evaluation"]["pll_beta"], 0.005)
            for filename in (
                "model.py",
                "datamodule.py",
                "losses.py",
                "train.py",
                "evaluate.py",
                "build_benchmark.py",
            ):
                self.assertTrue(
                    (bundle / "reference_training" / filename).is_file(),
                    filename,
                )

            starter = bundle / "reference_training"
            history = starter / "training_history.json"
            metrics = starter / "evaluation_metrics.json"
            notes = starter / "researcher_notes.md"
            history.write_bytes(b'{"selected_epoch": 7}\n')
            metrics.write_bytes(b'{"bit_error_rate": 0.05}\n')
            notes.write_text("researcher-owned\n", encoding="utf-8")
            returned_manifest = bundle / "trained_artifact.yaml"
            returned_component = bundle / "artifacts" / "phase_tracking_receiver.onnx"
            returned_component.parent.mkdir(parents=True)
            returned_manifest.write_text("kind: researcher-returned\n", encoding="utf-8")
            returned_component.write_bytes(b"researcher model bytes")
            (starter / "model.py").write_text("# stale checked-in scaffold\n", encoding="utf-8")

            prepare_example(
                bundle,
                "phase-tracking-receiver",
                project_root=root,
            )

            self.assertEqual(history.read_bytes(), b'{"selected_epoch": 7}\n')
            self.assertEqual(metrics.read_bytes(), b'{"bit_error_rate": 0.05}\n')
            self.assertEqual(notes.read_text(encoding="utf-8"), "researcher-owned\n")
            self.assertEqual(
                returned_manifest.read_text(encoding="utf-8"),
                "kind: researcher-returned\n",
            )
            self.assertEqual(returned_component.read_bytes(), b"researcher model bytes")
            self.assertEqual(
                (starter / "model.py").read_bytes(),
                (SCAFFOLD / "model.py").read_bytes(),
            )

    def test_packet_loader_builds_exact_eleven_channel_runtime_features(self):
        module = runpy.run_path(str(SCAFFOLD / "datamodule.py"))
        load_capture_dataset = module["load_capture_dataset"]
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary)
            (capture / "shards").mkdir()
            rx = np.asarray(
                [
                    [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j, 1 + 1j],
                    [-1 + 1j, 1 + 1j, 1 - 1j, -1 - 1j, 1 + 1j],
                ],
                dtype=np.complex64,
            )
            pilot_mask = np.asarray([1, 0, 0, 1, 0], dtype=np.float32)
            known = np.asarray(
                [1 + 1j, 0, 0, 1 - 1j, 0],
                dtype=np.complex64,
            ) / np.sqrt(2.0)
            pilot_context = np.stack(
                (
                    np.broadcast_to(pilot_mask, rx.shape),
                    np.broadcast_to(known.real, rx.shape),
                    np.broadcast_to(known.imag, rx.shape),
                ),
                axis=-1,
            ).astype(np.float32)
            bits = np.asarray(
                [[0, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 1]],
                dtype=np.uint8,
            )
            phase = np.zeros(rx.shape, dtype=np.float32)
            np.savez(
                capture / "shards" / "shard_0000.npz",
                rx=rx,
                pilots=pilot_context,
                bits=bits,
                phase=phase,
            )
            schema = {
                "kind": "noema.capture_dataset",
                "split": "train",
                "captured_samples": 2,
                "tap_schemas": {
                    "rx": {"from": "carrier_impairment.rx_symbols"},
                    "pilots": {"from": "modulator.pilot_context"},
                    "bits": {"from": "tx_bit_boundary.bits"},
                    "phase": {"from": "carrier_impairment.phase_truth"},
                },
                "runs": [
                    {
                        "captured_samples": 2,
                        "sweep": {"wireless_channel.snr_db": 2.0},
                    }
                ],
                "shards": [
                    {
                        "path": "shards/shard_0000.npz",
                        "sample_start": 0,
                    }
                ],
            }
            (capture / "schema.json").write_text(
                json.dumps(schema),
                encoding="utf-8",
            )
            dataset = load_capture_dataset(
                [str(capture)],
                feature_tap="rx_symbols",
                pilot_context_tap="pilot_context",
                target_tap="target_bits",
                phase_truth_tap="",
                feature_reference="carrier_impairment.rx_symbols",
                pilot_context_reference="modulator.pilot_context",
                target_reference="tx_bit_boundary.bits",
                phase_truth_reference="carrier_impairment.phase_truth",
                expected_split="train",
            )
            self.assertEqual(dataset.receiver_features.shape, (2, 5, 11))
            np.testing.assert_allclose(
                dataset.receiver_features,
                runtime_receiver_features(
                    rx,
                    np.broadcast_to(pilot_mask > 0.5, rx.shape),
                    np.broadcast_to(known, rx.shape),
                ),
                rtol=1e-6,
                atol=1e-6,
            )
            self.assertEqual(dataset.target_bits.shape, (2, 5, 2))
            np.testing.assert_array_equal(
                dataset.data_mask[0],
                np.asarray([False, True, True, False, True]),
            )
            np.testing.assert_array_equal(
                dataset.target_bits[0, dataset.data_mask[0]].reshape(-1),
                bits[0],
            )
            self.assertIsNotNone(dataset.phase_truth)

    def test_capture_feature_builder_matches_runtime_with_many_pilots(self):
        module = runpy.run_path(str(SCAFFOLD / "datamodule.py"))
        rng = np.random.RandomState(19)
        rx = (
            rng.normal(size=(2, 32)) + 1j * rng.normal(size=(2, 32))
        ).astype(np.complex64)
        mask = np.zeros((2, 32), dtype=bool)
        mask[:, :8] = True
        mask[:, (15, 24, 31)] = True
        known = np.zeros(rx.shape, dtype=np.complex64)
        known[mask] = (
            np.where(rng.rand(int(mask.sum())) < 0.5, -1.0, 1.0)
            + 1j * np.where(rng.rand(int(mask.sum())) < 0.5, -1.0, 1.0)
        ) / np.sqrt(2.0)
        pilots = np.stack(
            (mask.astype(np.float32), known.real, known.imag),
            axis=-1,
        ).astype(np.float32)
        data_symbols = int((~mask[0]).sum())
        bits = rng.randint(0, 2, size=(2, data_symbols * 2)).astype(np.uint8)
        features = module["_packet_records"](
            rx,
            pilots,
            bits,
            np.zeros(rx.shape, dtype=np.float32),
            Path("synthetic.npz"),
            pilot_smoothing_neighbors=5,
        )[0]
        expected = runtime_receiver_features(
            rx,
            mask,
            known,
            nearest_pilots=5,
        )
        np.testing.assert_allclose(features, expected, rtol=1e-6, atol=1e-6)

    def test_real_capture_roundtrip_preserves_packet_record_axes(self):
        payload = load_recipe(
            ROOT / "recipes" / "neural_receiver_qpsk_phase_tracking.yaml"
        ).to_dict()
        data = next(step for step in payload["steps"] if step["id"] == "data")
        data["params"].update({"bit_count": 32, "batch_size": 1})
        payload["dataset_capture"] = {
            "split": "train",
            "samples": 2,
            "max_runs": 2,
            "shard_size": 2,
            "seed_mode": "increment_run_seed",
            "taps": [
                {"id": "rx", "from": "carrier_impairment.rx_symbols"},
                {"id": "pilots", "from": "modulator.pilot_context"},
                {"id": "bits", "from": "tx_bit_boundary.bits"},
                {"id": "phase", "from": "carrier_impairment.phase_truth"},
            ],
        }
        recipe = recipe_from_dict(payload)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "capture"
            result = run_dataset_capture_recipe(
                recipe,
                build_registry(),
                LocalStore(root / ".noema"),
                output,
            )
            self.assertEqual(result["captured_samples"], 2)
            with np.load(
                output / "shards" / "shard_0000.npz",
                allow_pickle=False,
            ) as shard:
                self.assertEqual(shard["rx"].shape, (2, 33))
                self.assertEqual(shard["pilots"].shape, (2, 33, 3))
                self.assertEqual(shard["bits"].shape, (2, 32))
                self.assertEqual(shard["phase"].shape, (2, 33))
            schema = json.loads(
                (output / "schema.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                schema["tap_schemas"]["pilots"]["record_shape"],
                [33, 3],
            )

    def test_model_starts_from_pilot_smoothing_and_preserves_packet_shape(self):
        module = runpy.run_path(str(SCAFFOLD / "model.py"))
        torch = __import__("torch")
        candidate = module["receiver_candidates"](
            {
                "candidates": [
                    {
                        "id": "small",
                        "architecture": "pilot_smoother_residual_tcn",
                        "hidden_dim": 8,
                        "dilations": [1, 2],
                        "kernel_size": 5,
                    }
                ]
            }
        )[0]
        model = module["build_receiver"](candidate).eval()
        features = torch.randn((3, 41, 11), dtype=torch.float32)
        with torch.no_grad():
            output = model(features)
        self.assertEqual(tuple(output.shape), (3, 41))
        torch.testing.assert_close(output, torch.zeros_like(output))
        self.assertEqual(
            [block.network[0].groups for block in model.blocks],
            [8, 8],
        )

    def test_evaluator_rejects_malformed_or_nonfinite_artifact_outputs(self):
        saved = sys.modules.pop("datamodule", None)
        sys.path.insert(0, str(SCAFFOLD))
        try:
            module = runpy.run_path(str(SCAFFOLD / "evaluate.py"))
        finally:
            sys.path.pop(0)
            sys.modules.pop("datamodule", None)
            if saved is not None:
                sys.modules["datamodule"] = saved
        validate = module["_validated_learned_outputs"]
        phase_shape = (2, 5)
        valid = [np.zeros(phase_shape, dtype=np.float32)]
        phase = validate(valid, expected_phase_shape=phase_shape)
        self.assertEqual(phase.shape, phase_shape)

        with self.assertRaisesRegex(ValueError, "expected residual_phase_rad"):
            validate(
                [valid[0], valid[0]],
                expected_phase_shape=phase_shape,
            )
        with self.assertRaisesRegex(ValueError, "residual_phase_rad"):
            validate(
                [np.zeros((2, 4), dtype=np.float32)],
                expected_phase_shape=phase_shape,
            )
        malformed = [valid[0].copy()]
        malformed[0].flat[0] = np.nan
        with self.assertRaisesRegex(ValueError, "residual_phase_rad"):
            validate(malformed, expected_phase_shape=phase_shape)

    def test_builder_generates_six_paired_methods_with_packet_batches(self):
        module = runpy.run_path(str(SCAFFOLD / "build_benchmark.py"))
        evidence = {"path": "evidence.json", "sha256": "a" * 64}
        pack = module["_build_pack"](
            recipe_reference=str(
                (ROOT / "recipes" / "neural_receiver_qpsk_phase_tracking.yaml").resolve()
            ),
            artifact_reference=".noema/trained_artifact.yaml",
            artifact_package_sha256="b" * 64,
            artifact_evidence=evidence,
            training_history=evidence,
            evaluation_metrics=evidence,
            snr_grid=[0.0, 6.0],
            seeds=[91001, 92001],
            bit_count=262_144,
            packet_bits=1024,
            block_size=1024,
        )
        self.assertEqual(len(pack["recipes"]), 2 * 2 * 6)
        expected_order = [
            "uncompensated_qpsk",
            "pilot_interpolation",
            "pilot_smoothing",
            "decision_directed_pll",
            "learned_receiver",
            "oracle_phase",
        ]
        self.assertEqual(
            pack["metadata"]["demo"]["plots"][0]["method_order"],
            expected_order,
        )
        demo = pack["metadata"]["demo"]
        self.assertEqual(demo["slug"], "learned-qpsk-phase-tracking-receiver")
        self.assertEqual(
            demo["tutorial"],
            "../../../tutorials/learned_qpsk_phase_tracking_demo.html",
        )
        self.assertNotIn(
            "artifact_package_sha256",
            demo["training_evidence"][0],
        )
        self.assertEqual(
            [(plot["x"], plot["y"], plot["style"]["y_scale"]) for plot in demo["plots"]],
            [
                ("channel.snr_db", "channel.coded.ber", "log"),
                ("channel.snr_db", "channel.coded.bler", "log"),
            ],
        )
        self.assertEqual(
            [row["role"] for row in demo["series"]],
            [
                "baseline",
                "baseline",
                "baseline",
                "baseline",
                "candidate",
                "upper_bound",
            ],
        )
        self.assertEqual(demo["training_evidence"][0]["series"], "learned_receiver")
        first_six = pack["recipes"][:6]
        self.assertEqual(
            [row["params"]["metadata"]["benchmark_method"] for row in first_six],
            expected_order,
        )
        for row in pack["recipes"]:
            data = row["params"]["step_params"]["data"]
            self.assertEqual(data["bit_count"], 1024)
            self.assertEqual(data["batch_size"], 256)
        learned = first_six[4]["params"]["step_params"]["demodulator"]
        self.assertEqual(learned["mode"], "learned_artifact")
        self.assertEqual(learned["artifact_package_sha256"], "b" * 64)
        self.assertNotIn("phase_truth", learned)
        self.assertEqual(
            pack["metadata"]["evaluation_packet_count_per_run"],
            256,
        )
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            concrete_source = module["_write_benchmark_recipe_source"](
                yaml.safe_load(
                    (
                        ROOT
                        / "recipes"
                        / "neural_receiver_qpsk_phase_tracking.yaml"
                    ).read_text(encoding="utf-8")
                ),
                temporary_path / "benchmark_recipe.yaml",
            )
            for row in pack["recipes"]:
                row["path"] = str(concrete_source)
            path = temporary_path / "benchmark_pack.yaml"
            path.write_text(
                yaml.safe_dump(pack, sort_keys=False),
                encoding="utf-8",
            )
            validation = validate_benchmark_pack(
                load_benchmark_pack(path),
                build_registry(),
                Path(temporary),
            )
            self.assertEqual(validation["recipe_count"], 24)

    def test_publication_grid_builds_126_concrete_recipes(self):
        module = runpy.run_path(str(SCAFFOLD / "build_benchmark.py"))
        evidence = {"path": "evidence.json", "sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authored = yaml.safe_load(
                (
                    ROOT / "recipes" / "neural_receiver_qpsk_phase_tracking.yaml"
                ).read_text(encoding="utf-8")
            )
            concrete_source = module["_write_benchmark_recipe_source"](
                authored,
                root / "benchmark_recipe.yaml",
            )
            concrete = yaml.safe_load(concrete_source.read_text(encoding="utf-8"))
            self.assertNotIn("matrix", concrete["metadata"])
            self.assertNotIn("matrix_selection", concrete["metadata"])

            pack = module["_build_pack"](
                recipe_reference=str(concrete_source),
                artifact_reference=".noema/trained_artifact.yaml",
                artifact_package_sha256="b" * 64,
                artifact_evidence=evidence,
                training_history=evidence,
                evaluation_metrics=evidence,
                snr_grid=[-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                seeds=[91001, 92001, 93001],
                bit_count=262_144,
                packet_bits=1024,
                block_size=1024,
            )
            self.assertEqual(len(pack["recipes"]), 126)
            for row in pack["recipes"]:
                selection = row["params"]["matrix_selection"]
                self.assertIn("channel.snr_db", selection)
                self.assertNotIn("wireless_channel.snr_db", selection)

            pack_path = root / "benchmark_pack.yaml"
            pack_path.write_text(
                yaml.safe_dump(pack, sort_keys=False),
                encoding="utf-8",
            )
            validation = validate_benchmark_pack(
                load_benchmark_pack(pack_path),
                build_registry(),
                root,
            )
            self.assertEqual(validation["recipe_count"], 126)

    def test_checkpoint_rank_guards_snr_bins_then_uses_ber(self):
        saved_modules = {
            name: sys.modules.pop(name, None)
            for name in ("datamodule", "losses", "model")
        }
        sys.path.insert(0, str(SCAFFOLD))
        try:
            module = runpy.run_path(str(SCAFFOLD / "train.py"))
        finally:
            sys.path.pop(0)
            for name in ("datamodule", "losses", "model"):
                sys.modules.pop(name, None)
                if saved_modules[name] is not None:
                    sys.modules[name] = saved_modules[name]
        self.assertEqual(
            module["CHECKPOINT_SELECTION_ORDER"][:2],
            (
                "material_validation_ber_improvement_required",
                "minimum_per_snr_pilot_smoothing_regressions",
            ),
        )
        non_regressing = module["_checkpoint_rank"](
            validation_ber=0.11,
            validation_phase_loss=0.04,
            validation_phase_rmse=0.2,
            regressed_snr_bins=0,
            worst_per_snr_regression=0.0,
            candidate_index=1,
            seed_index=1,
            epoch_index=3,
        )
        regressing_low_ber = module["_checkpoint_rank"](
            validation_ber=0.09,
            validation_phase_loss=0.03,
            validation_phase_rmse=0.18,
            regressed_snr_bins=1,
            worst_per_snr_regression=0.01,
            candidate_index=0,
            seed_index=0,
            epoch_index=1,
        )
        self.assertLess(non_regressing, regressing_low_ber)

    def test_trainer_refuses_epoch_zero_or_unaccepted_learned_export(self):
        saved_modules = {
            name: sys.modules.pop(name, None)
            for name in ("datamodule", "losses", "model")
        }
        sys.path.insert(0, str(SCAFFOLD))
        try:
            module = runpy.run_path(str(SCAFFOLD / "train.py"))
        finally:
            sys.path.pop(0)
            for name in ("datamodule", "losses", "model"):
                sys.modules.pop(name, None)
                if saved_modules[name] is not None:
                    sys.modules[name] = saved_modules[name]
        require_selection = module["_require_learned_export_selection"]
        candidate = type("Candidate", (), {"id": "candidate-a"})()
        accepted_row = {
            "candidate": {"id": "candidate-a"},
            "seed": 23,
            "epoch": 1,
            "accepted_as_material_improvement": True,
        }
        with self.assertRaisesRegex(
            RuntimeError,
            "refusing to export the untrained epoch-zero",
        ):
            require_selection(
                best_state={"weight": object()},
                selected_candidate=candidate,
                selected_seed=23,
                selected_epoch=0,
                selection_history=[
                    {
                        **accepted_row,
                        "epoch": 0,
                    }
                ],
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "not backed by an accepted",
        ):
            require_selection(
                best_state={"weight": object()},
                selected_candidate=candidate,
                selected_seed=23,
                selected_epoch=1,
                selection_history=[
                    {
                        **accepted_row,
                        "accepted_as_material_improvement": False,
                    }
                ],
            )
        require_selection(
            best_state={"weight": object()},
            selected_candidate=candidate,
            selected_seed=23,
            selected_epoch=1,
            selection_history=[accepted_row],
        )

    def test_builder_rejects_self_labeled_epoch_zero_phase_artifact(self):
        module = runpy.run_path(str(SCAFFOLD / "build_benchmark.py"))
        validate = module["_validate_learned_training_provenance"]
        manifest = {
            "training": {
                "selected_epoch": 0,
                "training_performed": False,
                "learned_checkpoint": False,
                "accepted_as_material_improvement": False,
                "selected_epoch_is_pilot_smoothing_fallback": True,
            }
        }
        with self.assertRaisesRegex(
            ValueError,
            "after epoch zero",
        ):
            validate(manifest)
        manifest["training"].update(
            {
                "selected_epoch": 4,
                "training_performed": True,
                "learned_checkpoint": True,
                "accepted_as_material_improvement": True,
            }
        )
        with self.assertRaisesRegex(ValueError, "fallback"):
            validate(manifest)
        manifest["training"][
            "selected_epoch_is_pilot_smoothing_fallback"
        ] = False
        validate(manifest)

    def test_standalone_pll_matches_runtime_algorithm_and_recipe_gains(self):
        saved = sys.modules.pop("datamodule", None)
        sys.path.insert(0, str(SCAFFOLD))
        try:
            module = runpy.run_path(str(SCAFFOLD / "evaluate.py"))
        finally:
            sys.path.pop(0)
            sys.modules.pop("datamodule", None)
            if saved is not None:
                sys.modules["datamodule"] = saved
        rng = np.random.RandomState(7)
        raw = (
            rng.normal(size=(2, 24)) + 1j * rng.normal(size=(2, 24))
        ).astype(np.complex64)
        mask = np.zeros((2, 24), dtype=bool)
        mask[:, :6] = True
        mask[:, 14] = True
        known = np.zeros(raw.shape, dtype=np.complex64)
        known[mask] = (
            np.where(rng.rand(int(mask.sum())) < 0.5, -1.0, 1.0)
            + 1j * np.where(rng.rand(int(mask.sum())) < 0.5, -1.0, 1.0)
        ) / np.sqrt(2.0)
        observation = np.where(mask, raw * np.conj(known), 0.0 + 0.0j)
        features = runtime_receiver_features(raw, mask, known)
        expected = runtime_pll(
            raw,
            mask,
            known,
            alpha=0.12,
            beta=0.005,
        )
        actual = module["_decision_directed_pll_phase"](
            features,
            alpha=0.12,
            beta=0.005,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)

    def test_benchmark_builder_rejects_capture_seed_reuse(self):
        module = runpy.run_path(str(SCAFFOLD / "build_benchmark.py"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe_path = root / "noema_recipe.yaml"
            recipe_path.write_text("name: source\n", encoding="utf-8")
            for index, split in enumerate(("train", "validation", "test")):
                payload = {
                    "name": "capture_%s" % split,
                    "metadata": {"seed": 23 + index * 100},
                    "dataset_capture": {
                        "seed_mode": "increment_run_seed",
                        "max_runs": 1,
                    },
                    "steps": [
                        {"id": "data", "params": {}},
                        {"id": "wireless_channel", "params": {}},
                        {"id": "carrier_impairment", "params": {}},
                    ],
                }
                (root / ("capture_%s_recipe.yaml" % split)).write_text(
                    yaml.safe_dump(payload, sort_keys=False),
                    encoding="utf-8",
                )
            colliding_seed = derive_seed(
                23,
                "capture_train|dataset_capture_split=train",
                "data",
                "random_bits",
            )
            with self.assertRaisesRegex(ValueError, "not held out"):
                module["_assert_held_out_operation_seeds"](
                    recipe_path,
                    [colliding_seed],
                    {
                        "data": 0,
                        "wireless_channel": 100_000,
                        "carrier_impairment": 200_000,
                    },
                    {
                        "data": "random_bits",
                        "wireless_channel": "wireless_channel",
                        "carrier_impairment": "carrier_phase_impairment",
                    },
                    master_offset=300_000,
                )


if __name__ == "__main__":
    unittest.main()
