from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from noema_lab.core.recipes import load_recipe
from noema_lab.core.training_plans import TrainingPlan, apply_training_plan
from noema_lab.ops import build_registry
from noema_lab.training.csi_feedback_export import (
    CSI_FEEDBACK_EXAMPLE_LOSS,
    CSI_FEEDBACK_LOSS,
    CSI_FEEDBACK_TEMPLATE,
    build_csi_feedback_export_plan,
    suggested_csi_feedback_capture_total,
)
from noema_lab.training.exporter import (
    available_exporters,
    export_differentiable_scenario,
    inspect_training_capture,
)


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "csi_feedback_sionna_train.yaml"
DEMO = ROOT / "demo_trainings" / "csi_feedback_autoencoder"


def _load_demo_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, DEMO / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load demo module %s" % filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_demo_script(name: str, filename: str):
    """Load a starter entry point whose sibling modules use script imports."""

    sibling_names = ("datamodule", "losses", "model")
    previous_modules = {
        module_name: sys.modules.pop(module_name)
        for module_name in sibling_names
        if module_name in sys.modules
    }
    sys.path.insert(0, str(DEMO))
    try:
        return _load_demo_module(name, filename)
    finally:
        sys.path.pop(0)
        for module_name in sibling_names:
            sys.modules.pop(module_name, None)
        sys.modules.update(previous_modules)


class CsiFeedbackTrainingExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.base_recipe = load_recipe(RECIPE)
        cls.recipe = apply_training_plan(
            cls.base_recipe,
            TrainingPlan(
                selected_steps=("feedback_encoder", "feedback_decoder"),
                dataset_capture={
                    "split_plan": {
                        "total_samples": 768,
                        "counts": {
                            "train": 512,
                            "validation": 128,
                            "test": 128,
                        },
                    },
                },
            ),
        )

    def test_capture_split_belongs_to_training_plan_not_canonical_recipe(self):
        self.assertFalse(self.base_recipe.dataset_capture)
        self.assertEqual(
            self.recipe.dataset_capture["split_plan"]["counts"],
            {"train": 512, "validation": 128, "test": 128},
        )

    def test_capture_inspection_requires_abi_input_and_keeps_other_signals_optional(
        self,
    ):
        exporters = {item.id: item for item in available_exporters()}
        self.assertIn("csi-feedback", exporters)
        self.assertTrue(exporters["csi-feedback"].supports(self.recipe, self.registry))
        inspection = inspect_training_capture(
            self.recipe,
            self.registry,
            optimizable_steps=["feedback_encoder", "feedback_decoder"],
            project_root=ROOT,
        )
        self.assertTrue(inspection["ready"], inspection["issue"])
        self.assertEqual(inspection["sample_unit"], "recipe records")
        self.assertEqual(
            inspection["split_plan"]["counts"],
            {"train": 512, "validation": 128, "test": 128},
        )
        self.assertEqual(
            inspection["required_taps"],
            [
                {
                    "id": "channel_state_csi",
                    "from": "channel_state.csi",
                    "role": "replacement_input:feedback_encoder.csi",
                }
            ],
        )
        candidates = {item["from"]: item for item in inspection["candidates"]}
        self.assertTrue(candidates["channel_state.csi"]["selectable"])
        self.assertTrue(
            all(candidate["selectable"] for candidate in candidates.values())
        )
        self.assertEqual(
            candidates["feedback_decoder.reconstruction"]["relationship"],
            "current_replacement_output",
        )

    def test_plan_preserves_quantized_feedback_budget(self):
        plan = build_csi_feedback_export_plan(
            self.recipe,
            self.registry,
            optimizable_steps=["feedback_encoder", "feedback_decoder"],
            loss=CSI_FEEDBACK_LOSS,
            framework="torch",
        )
        self.assertEqual(plan.feedback_mode, "uniform_quantized")
        self.assertEqual(plan.feedback_dimension, 32)
        self.assertEqual(plan.bits_per_latent, 4)
        self.assertEqual(plan.feedback_bits_per_sample, 128)
        self.assertEqual(plan.clip_value, 1.0)

    def test_starter_accepts_neutral_generic_csi_capture_contract(self):
        data = _load_demo_module(
            "noema_test_csi_generic_contract",
            "datamodule.py",
        )
        contract = {
            "mode": "captured_generic_tensors",
            "signals": [
                {
                    "tap_id": "true_csi",
                    "reference": "channel_state.csi",
                    "kind": "channel.miso_ofdm_csi.numpy",
                    "required": True,
                }
            ],
        }
        data._validate_csi_data_contract(contract, feature_tap="true_csi")
        invalid = dict(contract)
        invalid["signals"] = [
            {
                **contract["signals"][0],
                "kind": "channel.unrelated_tensor.numpy",
            }
        ]
        with self.assertRaisesRegex(ValueError, "incompatible kind"):
            data._validate_csi_data_contract(invalid, feature_tap="true_csi")

    def test_capture_recommendation_grows_without_overriding_explicit_split(self):
        self.assertEqual(suggested_csi_feedback_capture_total(self.recipe), 12288)
        plan = build_csi_feedback_export_plan(
            self.recipe,
            self.registry,
            optimizable_steps=["feedback_encoder", "feedback_decoder"],
            loss=CSI_FEEDBACK_LOSS,
            framework="torch",
        )
        self.assertEqual(
            {item.id: item.samples for item in plan.capture_plan.splits},
            {"train": 512, "validation": 128, "test": 128},
        )

    def test_plan_preserves_arbitrary_external_loss_identifier(self):
        identifier = "researcher.hybrid-rate-objective/v7"
        plan = build_csi_feedback_export_plan(
            self.recipe,
            self.registry,
            optimizable_steps=["feedback_encoder", "feedback_decoder"],
            loss=identifier,
            framework="torch",
        )
        self.assertEqual(plan.loss, identifier)

    def test_neutral_contract_is_atomic_without_prescribing_joint_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "contract"
            payload = export_differentiable_scenario(
                self.recipe,
                self.registry,
                optimizable_steps=["feedback_encoder", "feedback_decoder"],
                loss="researcher.defined",
                framework="torch",
                out_dir=out,
                project_root=ROOT,
            )
            contract = yaml.safe_load(
                (out / "training_contract.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(contract["slot_groups"][0]["id"], "csi_feedback_codec")
            self.assertFalse(contract["slot_groups"][0]["joint_training"])
            self.assertEqual(
                contract["slot_groups"][0]["artifact_application"],
                "all_group_bindings",
            )
            data = payload["data_contract"]
            self.assertEqual(data["mode"], "captured_generic_tensors")
            self.assertEqual(
                [(item["reference"], item["required"]) for item in data["signals"]],
                [("channel_state.csi", True)],
            )
            for split, samples in (("train", 512), ("validation", 128), ("test", 128)):
                capture = yaml.safe_load(
                    (out / ("capture_%s_recipe.yaml" % split)).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(len(capture["steps"]), 1)
                self.assertEqual(capture["steps"][0]["op"], "wireless.miso_ofdm_csi")
                self.assertEqual(capture["dataset_capture"]["samples"], samples)
                self.assertEqual(
                    capture["dataset_capture"]["taps"],
                    [{"id": "channel_state_csi", "from": "channel_state.csi"}],
                )
                self.assertNotIn("seed", capture["steps"][0]["params"])
            validation = subprocess.run(
                [sys.executable, "validate_contract.py"],
                cwd=out,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validation.returncode, 0, validation.stderr)

    def test_optional_starter_is_non_normative_and_bound_to_neutral_data_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "contract"
            payload = export_differentiable_scenario(
                self.recipe,
                self.registry,
                optimizable_steps=["feedback_encoder", "feedback_decoder"],
                loss="csi.normalized_mse",
                framework="torch",
                out_dir=out,
                project_root=ROOT,
                exporter="csi-feedback",
                include_starter=True,
            )
            self.assertEqual(payload["starter_exporter"], "csi-feedback")
            self.assertTrue(
                (out / "reference_training" / "build_benchmark.py").is_file()
            )
            config = yaml.safe_load(
                (out / "reference_training" / "train_config.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(config["feedback_link"]["mode"], "uniform_quantized")
            self.assertEqual(config["feedback_link"]["feedback_bits_per_sample"], 128)
            self.assertTrue(config["data"]["contract_sha256"])
            self.assertTrue(config["data"]["contract_file_sha256"])
            self.assertEqual(config["training_template"], CSI_FEEDBACK_TEMPLATE)
            self.assertEqual(
                config["external_training_request"]["loss_identifier"],
                "csi.normalized_mse",
            )
            self.assertEqual(
                config["model"],
                {
                    "class": "ReferenceCsiFeedbackAutoencoder",
                    "architecture": (
                        "example_only_klt_initialized_quantization_residual_v3"
                    ),
                    "base_channels": 24,
                    "refinement_blocks": 2,
                    "latent_activation": "identity",
                    "global_linear_domain": "frequency",
                    "use_angular_delay_transform": True,
                    "use_per_latent_adaptor": True,
                    "initialization": {
                        "kind": "training_split_klt",
                        "scale_percentile": 99.0,
                    },
                },
            )
            self.assertEqual(config["objective"]["loss"], CSI_FEEDBACK_EXAMPLE_LOSS)
            self.assertEqual(
                config["objective"]["weights"],
                {
                    "nmse": 0.5,
                    "subcarrier_direction_cosine": 0.15,
                    "mrt_rate": 0.35,
                },
            )
            self.assertEqual(config["objective"]["quantization_weight"], 0.02)
            self.assertEqual(config["objective"]["snr_db_values"], [0, 5, 10, 15, 20])
            self.assertEqual(
                config["objective"]["checkpoint_selection"],
                "maximum_validation_mean_spectral_efficiency_retention_then_minimum_nmse",
            )
            self.assertEqual(
                {
                    key: config["training"][key]
                    for key in (
                        "epochs",
                        "batch_size",
                        "learning_rate",
                        "min_learning_rate",
                        "lr_warmup_epochs",
                        "quantization_warmup_epochs",
                        "nmse_pretraining_epochs",
                        "prior_freeze_epochs",
                        "prior_learning_rate_scale",
                        "checkpoint_rate_tolerance",
                        "weight_decay",
                        "early_stopping_patience",
                        "initialization_seeds",
                        "gradient_clip_norm",
                        "optimizer",
                        "learning_rate_schedule",
                    )
                },
                {
                    "epochs": 160,
                    "batch_size": 256,
                    "learning_rate": 1e-3,
                    "min_learning_rate": 1e-5,
                    "lr_warmup_epochs": 5,
                    "quantization_warmup_epochs": 0,
                    "nmse_pretraining_epochs": 50,
                    "prior_freeze_epochs": 50,
                    "prior_learning_rate_scale": 0.1,
                    "checkpoint_rate_tolerance": 1e-5,
                    "weight_decay": 1e-4,
                    "early_stopping_patience": 35,
                    "initialization_seeds": [23, 41],
                    "gradient_clip_norm": 1.0,
                    "optimizer": "AdamW",
                    "learning_rate_schedule": "cosine_annealing",
                },
            )
            manifest = payload["project_manifest"]
            optional = manifest["external_training"]["optional_demo_scaffold"]
            self.assertFalse(optional["normative"])
            self.assertEqual(optional["exporter"], "csi-feedback")
            self.assertTrue(
                optional["post_training"]["helper_path"].endswith(
                    "reference_training/build_benchmark.py"
                )
            )

    def test_ste_forward_matches_canonical_uniform_quantizer(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        model = _load_demo_module("noema_test_csi_model", "model.py")
        values = np.asarray(
            [[-1.5, -1.0, -0.23, 0.0, 0.61, 1.0, 1.5]], dtype=np.float32
        )
        actual = (
            model.uniform_quantize(
                torch.from_numpy(values),
                bits_per_latent=4,
                clip_value=1.0,
                straight_through=True,
            )
            .detach()
            .numpy()
        )
        levels = 15
        indices = np.rint((np.clip(values, -1.0, 1.0) + 1.0) * levels / 2.0)
        expected = (indices * 2.0 / levels - 1.0).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_v2_starter_forward_shapes_and_ste_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        starter = _load_demo_module("noema_test_csi_model_shapes", "model.py")
        torch.manual_seed(19)
        network = starter.ReferenceCsiFeedbackAutoencoder(
            tx_antennas=4,
            subcarrier_count=8,
            feedback_dimension=8,
            bits_per_latent=4,
            clip_value=1.0,
            base_channels=4,
            refinement_blocks=1,
            use_angular_delay_transform=True,
        )
        inputs = torch.randn(3, 2, 4, 8)
        outputs = network(inputs, quantize=True)
        self.assertEqual(
            set(outputs), {"feedback_code", "received_code", "reconstruction"}
        )
        self.assertEqual(tuple(outputs["feedback_code"].shape), (3, 8))
        self.assertEqual(tuple(outputs["received_code"].shape), (3, 8))
        self.assertEqual(tuple(outputs["reconstruction"].shape), (3, 2, 4, 8))

        outputs["reconstruction"].square().mean().backward()
        encoder_gradients = [
            parameter.grad
            for parameter in network.encoder.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        decoder_gradients = [
            parameter.grad
            for parameter in network.decoder.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(
            encoder_gradients, "STE must carry loss gradients to the encoder"
        )
        self.assertTrue(decoder_gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in encoder_gradients))
        self.assertTrue(all(torch.isfinite(value).all() for value in decoder_gradients))
        self.assertGreater(
            sum(float(value.abs().sum()) for value in encoder_gradients), 0.0
        )

    def test_fixed_angular_delay_transform_is_unitary_round_trip(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        starter = _load_demo_module("noema_test_csi_model_transform", "model.py")
        transform = starter.FixedAngularDelayTransform(
            tx_antennas=8,
            subcarrier_count=32,
            enabled=True,
        )
        torch.manual_seed(23)
        inputs = torch.randn(2, 2, 8, 32)
        reconstructed = transform.synthesis(transform.analysis(inputs))
        torch.testing.assert_close(reconstructed, inputs, rtol=1e-5, atol=2e-5)

    def test_klt_initialization_is_exact_before_residual_training(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        starter = _load_demo_module("noema_test_csi_klt_prior", "model.py")
        torch.manual_seed(47)
        network = starter.ReferenceCsiFeedbackAutoencoder(
            tx_antennas=2,
            subcarrier_count=3,
            feedback_dimension=3,
            bits_per_latent=4,
            clip_value=1.0,
            base_channels=4,
            refinement_blocks=1,
            latent_activation="identity",
            use_angular_delay_transform=False,
        )
        dimension = 12
        mean = torch.randn(dimension)
        basis = torch.linalg.qr(torch.randn(dimension, 3)).Q.transpose(0, 1)
        scale = torch.tensor([0.7, 1.1, 1.6])
        network.initialize_klt_prior(mean=mean, basis=basis, scale=scale)
        inputs = torch.randn(5, 2, 2, 3)
        flattened = inputs.flatten(start_dim=1)
        expected_code = (flattened - mean) @ basis.transpose(0, 1) / scale
        actual_code = network.encoder(inputs)
        torch.testing.assert_close(actual_code, expected_code)
        received = starter.uniform_quantize(
            expected_code,
            bits_per_latent=4,
            clip_value=1.0,
            straight_through=False,
        )
        expected_reconstruction = (
            (received * scale) @ basis + mean
        ).reshape_as(inputs)
        actual_reconstruction = network.decoder(received)
        torch.testing.assert_close(
            actual_reconstruction,
            expected_reconstruction,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_subcarrier_direction_metric_ignores_per_subcarrier_common_phase(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        losses = _load_demo_module(
            "noema_test_csi_subcarrier_direction",
            "losses.py",
        )
        torch.manual_seed(53)
        target = torch.randn(3, 2, 4, 6)
        true_h = torch.complex(target[:, 0], target[:, 1])
        phases = torch.linspace(-1.2, 1.3, 6)
        rotated = true_h * torch.exp(1j * phases)[None, None, :]
        reconstruction = torch.stack((rotated.real, rotated.imag), dim=1)
        local = losses.subcarrier_direction_cosine_similarity(
            reconstruction, target
        )
        global_direction = losses.phase_invariant_cosine_similarity(
            reconstruction, target
        )
        self.assertAlmostEqual(float(local), 1.0, places=6)
        self.assertLess(float(global_direction), 0.9)

    def test_hybrid_objective_ranks_exact_csi_above_degraded_csi(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        losses = _load_demo_module("noema_test_csi_losses_ranking", "losses.py")
        torch.manual_seed(31)
        target = torch.randn(4, 2, 4, 8)
        feedback = torch.zeros(4, 8)
        common = {
            "feedback_code": feedback,
            "received_code": feedback,
            "snr_db_values": [0.0, 10.0, 20.0],
            "nmse_weight": 0.25,
            "direction_weight": 0.25,
            "rate_weight": 0.5,
            "quantization_weight": 0.05,
        }
        exact = losses.hybrid_csi_objective(target, target, **common)
        degraded_reconstruction = (
            target + 0.7 * torch.randn_like(target)
        ).requires_grad_()
        degraded = losses.hybrid_csi_objective(
            degraded_reconstruction, target, **common
        )
        self.assertLess(float(exact["loss"]), float(degraded["loss"]))
        self.assertAlmostEqual(float(exact["nmse"]), 0.0, places=7)
        self.assertAlmostEqual(
            float(exact["mean_spectral_efficiency_retention"]), 1.0, places=6
        )
        self.assertGreater(float(degraded["nmse"]), float(exact["nmse"]))
        self.assertLess(
            float(degraded["mean_spectral_efficiency_retention"]),
            float(exact["mean_spectral_efficiency_retention"]),
        )
        degraded["loss"].backward()
        self.assertIsNotNone(degraded_reconstruction.grad)
        self.assertTrue(torch.isfinite(degraded_reconstruction.grad).all())
        self.assertGreater(float(degraded_reconstruction.grad.abs().sum()), 0.0)

    def test_evaluation_metrics_are_invariant_to_batch_partitioning(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        evaluator = _load_demo_script(
            "noema_test_csi_evaluate_aggregation",
            "evaluate.py",
        )
        rng = np.random.RandomState(917)
        target = rng.normal(size=(7, 2, 2, 4)).astype(np.float32)
        reconstruction = target + rng.normal(scale=0.4, size=target.shape).astype(
            np.float32
        )
        common = {
            "downlink_snr_db": 10.0,
            "snr_db_values": [0.0, 10.0, 20.0],
        }
        whole = evaluator._aggregate_complete_metrics(
            [reconstruction],
            [target],
            **common,
        )
        partitioned = evaluator._aggregate_complete_metrics(
            [reconstruction[:2], reconstruction[2:6], reconstruction[6:]],
            [target[:2], target[2:6], target[6:]],
            **common,
        )
        self.assertEqual(set(whole), set(partitioned))
        for key in whole:
            self.assertAlmostEqual(whole[key], partitioned[key], places=7, msg=key)

    def test_checkpoint_metrics_are_invariant_to_validation_batch_size(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        data = _load_demo_module(
            "noema_test_csi_validation_data",
            "datamodule.py",
        )
        trainer = _load_demo_script(
            "noema_test_csi_validation_aggregation",
            "train.py",
        )

        class FixedReconstruction(torch.nn.Module):
            def forward(self, csi_ri, *, quantize=True):
                del quantize
                return {"reconstruction": csi_ri * 0.81 + 0.03}

        rng = np.random.RandomState(923)
        values = rng.normal(size=(7, 2, 2, 4)).astype(np.float32)
        dataset = data.CsiCaptureDataset(
            values,
            split="validation",
            capture_sha256=[],
            capture_dirs=[],
        )
        common = {
            "model": FixedReconstruction(),
            "device": torch.device("cpu"),
            "snr_db_values": [0.0, 10.0, 20.0],
        }
        whole = trainer._validation_metrics(
            loader=data.build_loader(
                dataset,
                batch_size=7,
                shuffle=False,
                seed=0,
            ),
            **common,
        )
        partitioned = trainer._validation_metrics(
            loader=data.build_loader(
                dataset,
                batch_size=2,
                shuffle=False,
                seed=0,
            ),
            **common,
        )
        self.assertEqual(set(whole), set(partitioned))
        for key in whole:
            self.assertAlmostEqual(whole[key], partitioned[key], places=7, msg=key)

    def test_starter_accepts_prefixed_hybrid_and_pure_nmse_ablation_modes(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        trainer = _load_demo_script("noema_test_csi_train_objective", "train.py")
        hybrid = trainer._objective_settings(
            {
                "objective": {
                    "loss": "csi.hybrid_nmse_mrt_rate",
                    "weights": {
                        "nmse": 0.2,
                        "phase_invariant_cosine": 0.3,
                        "mrt_rate": 0.5,
                    },
                    "quantization_weight": 0.04,
                    "snr_db_values": [0, 10],
                }
            }
        )
        self.assertEqual(hybrid["loss"], "hybrid_nmse_mrt_rate")
        self.assertEqual(hybrid["configured_loss"], "csi.hybrid_nmse_mrt_rate")
        self.assertEqual(hybrid["snr_db_values"], [0.0, 10.0])

        pure_nmse = trainer._objective_settings(
            {"objective": {"loss": "csi.normalized_reconstruction_mse"}}
        )
        self.assertEqual(pure_nmse["loss"], "normalized_reconstruction_mse")
        self.assertEqual(
            pure_nmse["configured_loss"], "csi.normalized_reconstruction_mse"
        )
        lower_nmse = {"nmse": 0.1, "mean_spectral_efficiency_retention": 0.5}
        higher_nmse = {"nmse": 0.2, "mean_spectral_efficiency_retention": 0.99}
        self.assertLess(
            trainer._selection_key(
                lower_nmse, loss_name="normalized_reconstruction_mse"
            ),
            trainer._selection_key(
                higher_nmse, loss_name="normalized_reconstruction_mse"
            ),
        )
        self.assertLess(
            trainer._selection_key(higher_nmse, loss_name="hybrid_nmse_mrt_rate"),
            trainer._selection_key(lower_nmse, loss_name="hybrid_nmse_mrt_rate"),
        )
        incumbent = {
            "nmse": 0.1,
            "mean_spectral_efficiency_retention": 0.99,
        }
        self.assertFalse(
            trainer._validation_is_better(
                {
                    "nmse": 0.2,
                    "mean_spectral_efficiency_retention": 0.990001,
                },
                incumbent,
                loss_name="hybrid_nmse_mrt_rate",
                rate_tolerance=1e-5,
            )
        )
        self.assertTrue(
            trainer._validation_is_better(
                {
                    "nmse": 0.09,
                    "mean_spectral_efficiency_retention": 0.990001,
                },
                incumbent,
                loss_name="hybrid_nmse_mrt_rate",
                rate_tolerance=1e-5,
            )
        )
        self.assertTrue(
            trainer._validation_is_better(
                {
                    "nmse": 0.2,
                    "mean_spectral_efficiency_retention": 0.99002,
                },
                incumbent,
                loss_name="hybrid_nmse_mrt_rate",
                rate_tolerance=1e-5,
            )
        )

    def test_v2_components_export_and_execute_with_onnxruntime(self):
        try:
            import onnx  # noqa: F401
            import onnxruntime as ort
            import torch
        except ImportError:
            self.skipTest("torch, onnx, and onnxruntime are required")
        starter = _load_demo_module("noema_test_csi_model_onnx", "model.py")
        torch.manual_seed(29)
        network = starter.ReferenceCsiFeedbackAutoencoder(
            tx_antennas=4,
            subcarrier_count=8,
            feedback_dimension=8,
            bits_per_latent=4,
            clip_value=1.0,
            base_channels=4,
            refinement_blocks=1,
            use_angular_delay_transform=True,
        ).eval()
        inputs = torch.randn(3, 2, 4, 8)
        with tempfile.TemporaryDirectory() as temporary:
            encoder_path = Path(temporary) / "encoder.onnx"
            decoder_path = Path(temporary) / "decoder.onnx"
            hashes = starter.export_onnx_components(network, encoder_path, decoder_path)
            self.assertEqual(len(hashes["encoder_sha256"]), 64)
            self.assertEqual(len(hashes["decoder_sha256"]), 64)
            encoder = ort.InferenceSession(
                str(encoder_path), providers=["CPUExecutionProvider"]
            )
            decoder = ort.InferenceSession(
                str(decoder_path), providers=["CPUExecutionProvider"]
            )
            encoded = encoder.run(None, {"csi_ri": inputs.numpy()})[0]
            decoded = decoder.run(None, {"feedback_code": encoded})[0]

        self.assertEqual(encoded.shape, (3, 8))
        self.assertEqual(decoded.shape, (3, 2, 4, 8))
        with torch.no_grad():
            expected_code = network.encoder(inputs).numpy()
            expected_reconstruction = network.decoder(torch.from_numpy(encoded)).numpy()
        np.testing.assert_allclose(encoded, expected_code, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            decoded, expected_reconstruction, rtol=1e-4, atol=1e-5
        )

    def test_dataset_disjointness_rejects_shared_csi_realization(self):
        data = _load_demo_module("noema_test_csi_data", "datamodule.py")
        train_values = np.zeros((2, 2, 2, 4), dtype=np.float32)
        train_values[1] = 1.0
        validation_values = np.full((1, 2, 2, 4), 2.0, dtype=np.float32)
        train = data.CsiCaptureDataset(
            train_values,
            split="train",
            capture_sha256=["a" * 64],
            capture_dirs=[Path("train")],
        )
        validation = data.CsiCaptureDataset(
            validation_values,
            split="validation",
            capture_sha256=["b" * 64],
            capture_dirs=[Path("validation")],
        )
        data.assert_disjoint_datasets(train, validation)
        overlap = data.CsiCaptureDataset(
            train_values[1:],
            split="test",
            capture_sha256=["c" * 64],
            capture_dirs=[Path("test")],
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            data.assert_disjoint_datasets(train, validation, overlap)

    def test_training_freezes_hashes_for_every_captured_split(self):
        data_module = _load_demo_module(
            "noema_test_csi_inventory",
            "datamodule.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_dirs = {}
            splits = []
            for split, seed in (
                ("train", 1),
                ("validation", 2),
                ("test", 3),
            ):
                capture = root / split
                shards = capture / "shards"
                shards.mkdir(parents=True)
                values = np.full((1, 2, 2, 4), seed, dtype=np.float32)
                np.savez(shards / "shard_0000.npz", true_csi=values)
                (capture / "schema.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "kind": "noema.capture_dataset",
                            "split": split,
                            "captured_samples": 1,
                            "shards": [
                                {
                                    "path": "shards/shard_0000.npz",
                                    "captured_samples": 1,
                                    "sample_start": 0,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                split_dirs[split] = capture
                splits.append({"id": split, "requested_samples": 1})
            contract_path = root / "data_contract.yaml"
            contract_path.write_text(
                yaml.safe_dump({"schema_version": 1, "splits": splits}),
                encoding="utf-8",
            )
            config_path = root / "train_config.yaml"
            config = {
                "schema_version": 1,
                "data": {
                    "contract_path": str(contract_path),
                    **{
                        "%s_capture_dirs" % split: [str(path)]
                        for split, path in split_dirs.items()
                    },
                },
            }
            config_path.write_text(
                yaml.safe_dump(config),
                encoding="utf-8",
            )

            updated = data_module.materialize_data_contract_inventory(
                config,
                config_path=config_path,
            )
            frozen = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
            for split in frozen["splits"]:
                self.assertEqual(split["captured_samples"], 1)
                self.assertEqual(len(split["files"]), 1)
                self.assertEqual(len(split["files"][0]["sha256"]), 64)
                self.assertTrue(
                    split["files"][0]["sample_id"].startswith(split["id"] + "/")
                )
            persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["data"]["contract_sha256"],
                updated["data"]["contract_sha256"],
            )
            self.assertEqual(
                persisted["data"]["contract_file_sha256"],
                updated["data"]["contract_file_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
