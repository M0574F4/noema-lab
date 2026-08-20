from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.execution_profiles import inspect_execution_profile
from noema_lab.core.executor import LocalExecutor
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.ops import build_registry
from noema_lab.ops.csi_feedback import (
    CSI_KIND,
    CsiFeedbackDecoderOperation,
    CsiFeedbackEncoderOperation,
    CsiFeedbackMetricsOperation,
    CsiMrtPrecoderOperation,
    MisoOfdmCsiOperation,
)


def _recipe(*, runtime: str, link_mode: str, feedback_dimension: int, bits: int = 8):
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "csi_feedback_%s_%s" % (runtime, link_mode),
            "execution_profile": {"id": "csi_feedback_downlink", "version": 1},
            "metadata": {"seed": 23},
            "steps": [
                {
                    "id": "channel_state",
                    "op": "wireless.miso_ofdm_csi",
                    "params": {
                        "sample_count": 6,
                        "tx_antennas": 4,
                        "ofdm_fft_size": 8,
                        "num_ofdm_symbols": 1,
                        "channel_tap_count": 4,
                        "tx_correlation_coefficient": 0.6,
                        "downlink_snr_db": 10.0,
                        "wireless_backend": "numpy",
                        "seed": 23,
                    },
                },
                {
                    "id": "feedback_encoder",
                    "op": "model.csi_feedback_encoder",
                    "inputs": {"csi": "channel_state.csi"},
                    "params": {
                        "runtime": runtime,
                        "feedback_dimension": feedback_dimension,
                    },
                },
                {
                    "id": "feedback_link",
                    "op": "channel.csi_feedback_link",
                    "inputs": {"feedback_code": "feedback_encoder.feedback_code"},
                    "params": {
                        "mode": link_mode,
                        "bits_per_latent": bits,
                        "clip_value": 1.0,
                    },
                },
                {
                    "id": "feedback_decoder",
                    "op": "model.csi_feedback_decoder",
                    "inputs": {"received_code": "feedback_link.received_code"},
                    "params": {
                        "runtime": runtime,
                        "feedback_dimension": feedback_dimension,
                    },
                },
                {
                    "id": "precoder",
                    "op": "model.csi_mrt_precoder",
                    "inputs": {"reconstruction": "feedback_decoder.reconstruction"},
                },
                {
                    "id": "evaluation",
                    "op": "metrics.csi_feedback",
                    "inputs": {
                        "true_csi": "channel_state.csi",
                        "reconstruction": "feedback_decoder.reconstruction",
                        "precoder": "precoder.precoder",
                    },
                },
            ],
        }
    )


def _run(recipe, root: Path):
    run_dir = LocalExecutor(build_registry(), LocalStore(root / ".noema")).run(recipe)
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _step(summary, step_id: str):
    return next(item for item in summary["steps"] if item["id"] == step_id)


class CsiFeedbackCoreTests(unittest.TestCase):
    def test_profile_and_identity_upper_bound_are_exact(self):
        recipe = _recipe(
            runtime="identity",
            link_mode="ideal_noiseless",
            feedback_dimension=64,
        )
        inspection = inspect_execution_profile(recipe)
        self.assertEqual(inspection.status, "conformant", inspection.issues)
        self.assertEqual(
            list(dict(inspection.stage_bindings)),
            [
                "channel_state",
                "feedback_encoder",
                "feedback_link",
                "feedback_decoder",
                "precoder",
                "evaluation",
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            summary = _run(recipe, Path(tmp))
        metrics = _step(summary, "evaluation")["metrics"]
        self.assertLess(metrics["csi_feedback.nmse"], 1e-12)
        self.assertAlmostEqual(metrics["csi_feedback.phase_invariant_cosine"], 1.0, places=6)
        self.assertAlmostEqual(metrics["csi_feedback.spectral_efficiency_retention"], 1.0, places=6)
        self.assertAlmostEqual(metrics["csi_feedback.spectral_efficiency_loss_bps_hz"], 0.0, places=6)
        self.assertEqual(metrics["csi_feedback.feedback_dimension"], 64)
        self.assertEqual(metrics["csi_feedback.original_real_dimension"], 64)
        self.assertNotIn("csi_feedback.feedback_bits_per_sample", metrics)
        preview = _step(summary, "evaluation")["metadata"]["csi_feedback_preview"]
        self.assertEqual(len(preview["samples"]), 6)
        self.assertEqual(np.asarray(preview["true_magnitude"]).shape, (4, 8))
        self.assertTrue(
            np.allclose(preview["true_magnitude"], preview["reconstructed_magnitude"])
        )

    def test_truncated_quantized_path_has_matched_and_honest_budget(self):
        recipe = _recipe(
            runtime="truncated_angular_delay",
            link_mode="uniform_quantized",
            feedback_dimension=16,
            bits=3,
        )
        with tempfile.TemporaryDirectory() as tmp:
            summary = _run(recipe, Path(tmp))
        metrics = _step(summary, "evaluation")["metrics"]
        self.assertEqual(metrics["csi_feedback.feedback_dimension"], 16)
        self.assertEqual(metrics["csi_feedback.feedback_bits_per_sample"], 48)
        self.assertAlmostEqual(metrics["csi_feedback.latent_fraction"], 0.25)
        self.assertAlmostEqual(metrics["csi_feedback.compression_factor"], 4.0)
        self.assertGreater(metrics["csi_feedback.nmse"], 0.0)
        self.assertGreaterEqual(metrics["csi_feedback.spectral_efficiency_loss_bps_hz"], -1e-6)
        self.assertLessEqual(metrics["csi_feedback.spectral_efficiency_retention"], 1.0 + 1e-6)
        link_metrics = _step(summary, "feedback_link")["metrics"]
        self.assertIn("csi_feedback.quantization_mse", link_metrics)
        self.assertIn("csi_feedback.quantization_clipped_fraction", link_metrics)
        self.assertLess(link_metrics["csi_feedback.quantization_clipped_fraction"], 0.10)
        encoder_metadata = _step(summary, "feedback_encoder")["metadata"]
        self.assertEqual(
            encoder_metadata["feedback_latent_normalization"],
            "divide_angular_delay_coefficients_by_sqrt_subcarrier_count",
        )
        self.assertAlmostEqual(encoder_metadata["angular_delay_feedback_scale"], np.sqrt(8))

    def test_full_dimension_angular_delay_round_trip_is_exact_without_quantization(self):
        recipe = _recipe(
            runtime="truncated_angular_delay",
            link_mode="ideal_noiseless",
            feedback_dimension=64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            summary = _run(recipe, Path(tmp))
        metrics = _step(summary, "evaluation")["metrics"]
        self.assertLess(metrics["csi_feedback.nmse"], 1e-12)
        self.assertAlmostEqual(metrics["csi_feedback.spectral_efficiency_retention"], 1.0, places=6)

    def test_explicit_numpy_source_is_seeded_and_sionna_does_not_silently_fallback(self):
        recipe = _recipe(
            runtime="identity",
            link_mode="ideal_noiseless",
            feedback_dimension=64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _run(recipe, root)
            second = _run(recipe, root)
            first_path = Path(_step(first, "channel_state")["outputs"]["csi"]["path"])
            second_path = Path(_step(second, "channel_state")["outputs"]["csi"]["path"])
            with np.load(first_path, allow_pickle=False) as left, np.load(
                second_path, allow_pickle=False
            ) as right:
                self.assertTrue(np.array_equal(left["csi_ri"], right["csi_ri"]))
            numpy_metadata = _step(first, "channel_state")["metadata"]
            self.assertEqual(
                numpy_metadata["channel_model_scope"],
                "abstract_static_correlated_exponential_tapped_delay",
            )
            self.assertEqual(numpy_metadata["physical_channel_controls_applied"], [])
            self.assertIsNone(numpy_metadata["tdl_model"])
            self.assertIsNone(numpy_metadata["mobility_kmh"])

            unsupported_numpy = OperationContext(
                recipe_name="numpy_physical_controls",
                step_id="channel_state",
                params={
                    "sample_count": 2,
                    "tx_antennas": 2,
                    "ofdm_fft_size": 8,
                    "num_ofdm_symbols": 1,
                    "tdl_model": "B",
                    "subcarrier_spacing_khz": 30.0,
                    "carrier_frequency_ghz": 3.5,
                    "delay_spread_ns": 100.0,
                    "mobility_kmh": 0.0,
                    "wireless_backend": "numpy",
                },
                inputs={},
                run_dir=root,
                step_dir=root / "unsupported_numpy",
            )
            with self.assertRaisesRegex(
                OperationError,
                "does not implement these physical controls: tdl_model='B'",
            ):
                MisoOfdmCsiOperation().run(unsupported_numpy)

            context = OperationContext(
                recipe_name="no_silent_fallback",
                step_id="channel_state",
                params={
                    "sample_count": 2,
                    "tx_antennas": 2,
                    "ofdm_fft_size": 8,
                    "wireless_backend": "sionna",
                },
                inputs={},
                run_dir=root,
                step_dir=root / "sionna",
            )
            with mock.patch("noema_lab.ops.csi_feedback._sionna_available", return_value=False):
                with self.assertRaisesRegex(OperationError, "explicitly select wireless_backend=numpy"):
                    MisoOfdmCsiOperation().run(context)

    def test_training_interface_is_export_only_and_artifact_abis_are_portable(self):
        registry = build_registry()
        encoder = registry.get("model.csi_feedback_encoder").describe()
        decoder = registry.get("model.csi_feedback_decoder").describe()
        self.assertEqual(encoder["trained_artifact_abi"]["inputs"]["csi_ri"]["dtype"], "float32")
        self.assertEqual(
            decoder["trained_artifact_abi"]["outputs"]["csi_hat_ri"]["shape"],
            ["batch", 2, "tx_antenna", "subcarrier"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_context = OperationContext(
                recipe_name="training_interface",
                step_id="channel_state",
                params={
                    "sample_count": 2,
                    "tx_antennas": 2,
                    "ofdm_fft_size": 8,
                    "wireless_backend": "numpy",
                    "seed": 23,
                },
                inputs={},
                run_dir=root,
                step_dir=root / "source",
            )
            csi = MisoOfdmCsiOperation().run(source_context).outputs["csi"]
            encoder_context = OperationContext(
                recipe_name="training_interface",
                step_id="feedback_encoder",
                params={"runtime": "training_interface", "feedback_dimension": 4},
                inputs={"csi": csi},
                run_dir=root,
                step_dir=root / "encoder",
            )
            with self.assertRaisesRegex(OperationError, "export-only typed interface"):
                CsiFeedbackEncoderOperation().run(encoder_context)

    def test_cosine_and_downlink_rate_are_invariant_to_common_csi_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_context = OperationContext(
                recipe_name="phase_invariant",
                step_id="channel_state",
                params={
                    "sample_count": 3,
                    "tx_antennas": 2,
                    "ofdm_fft_size": 8,
                    "wireless_backend": "numpy",
                    "seed": 23,
                },
                inputs={},
                run_dir=root,
                step_dir=root / "source",
            )
            true_csi = MisoOfdmCsiOperation().run(source_context).outputs["csi"]
            with np.load(true_csi.path, allow_pickle=False) as payload:
                true_ri = payload["csi_ri"]
            true_h = true_ri[:, 0] + 1j * true_ri[:, 1]
            rotated = true_h * np.exp(1j * 0.73)
            rotated_ri = np.stack([rotated.real, rotated.imag], axis=1).astype(np.float32)
            reconstruction_metadata = dict(true_csi.metadata)
            reconstruction_metadata.update(
                {
                    "array": "csi_hat_ri",
                    "feedback_dimension": int(np.prod(rotated_ri.shape[1:])),
                    "original_csi_shape": list(rotated_ri.shape),
                }
            )
            reconstruction_path = root / "reconstruction.npz"
            np.savez_compressed(
                reconstruction_path,
                csi_hat_ri=rotated_ri,
                metadata_json=json.dumps(reconstruction_metadata, sort_keys=True),
            )
            reconstruction = artifact(
                "channel.miso_ofdm_csi_reconstruction.numpy",
                reconstruction_path,
                reconstruction_metadata,
            )
            precoder = CsiMrtPrecoderOperation().run(
                OperationContext(
                    recipe_name="phase_invariant",
                    step_id="precoder",
                    params={},
                    inputs={"reconstruction": reconstruction},
                    run_dir=root,
                    step_dir=root / "precoder",
                )
            ).outputs["precoder"]
            result = CsiFeedbackMetricsOperation().run(
                OperationContext(
                    recipe_name="phase_invariant",
                    step_id="evaluation",
                    params={},
                    inputs={
                        "true_csi": true_csi,
                        "reconstruction": reconstruction,
                        "precoder": precoder,
                    },
                    run_dir=root,
                    step_dir=root / "metrics",
                )
            )
        self.assertAlmostEqual(result.metrics["csi_feedback.phase_invariant_cosine"], 1.0, places=6)
        self.assertAlmostEqual(result.metrics["csi_feedback.spectral_efficiency_retention"], 1.0, places=6)

    def test_schema_v2_learned_artifact_entrypoints_are_used_by_both_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "trained_artifact.yaml"
            manifest.write_text("schema_version: 2\n", encoding="utf-8")
            source_context = OperationContext(
                recipe_name="learned_pair",
                step_id="channel_state",
                params={
                    "sample_count": 2,
                    "tx_antennas": 2,
                    "ofdm_fft_size": 8,
                    "wireless_backend": "numpy",
                    "seed": 23,
                },
                inputs={},
                run_dir=root,
                step_dir=root / "source",
            )
            csi = MisoOfdmCsiOperation().run(source_context).outputs["csi"]
            calls = []

            def fake_runtime(
                _manifest,
                entrypoint,
                inputs,
                *,
                expected_package_sha256,
            ):
                calls.append((entrypoint, sorted(inputs), expected_package_sha256))
                if entrypoint == "encoder":
                    return {"feedback_code": np.ones((2, 4), dtype=np.float32)}
                return {"csi_hat_ri": np.zeros((2, 2, 2, 8), dtype=np.float32)}

            with mock.patch(
                "noema_lab.core.trained_artifact_runtime.run_trained_artifact_entrypoint",
                side_effect=fake_runtime,
            ):
                encoder_context = OperationContext(
                    recipe_name="learned_pair",
                    step_id="feedback_encoder",
                    params={
                        "runtime": "learned_artifact",
                        "feedback_dimension": 4,
                        "artifact_manifest_path": str(manifest),
                        "artifact_entrypoint": "encoder",
                        "artifact_package_sha256": "a" * 64,
                    },
                    inputs={"csi": csi},
                    run_dir=root,
                    step_dir=root / "encoder",
                )
                code = CsiFeedbackEncoderOperation().run(encoder_context).outputs[
                    "feedback_code"
                ]
                decoder_context = OperationContext(
                    recipe_name="learned_pair",
                    step_id="feedback_decoder",
                    params={
                        "runtime": "learned_artifact",
                        "feedback_dimension": 4,
                        "artifact_manifest_path": str(manifest),
                        "artifact_entrypoint": "decoder",
                        "artifact_package_sha256": "a" * 64,
                    },
                    inputs={"received_code": code},
                    run_dir=root,
                    step_dir=root / "decoder",
                )
                result = CsiFeedbackDecoderOperation().run(decoder_context)
            self.assertEqual(
                calls,
                [
                    ("encoder", ["csi_ri"], "a" * 64),
                    ("decoder", ["feedback_code"], "a" * 64),
                ],
            )
            self.assertEqual(result.outputs["reconstruction"].kind, "channel.miso_ofdm_csi_reconstruction.numpy")


if __name__ == "__main__":
    unittest.main()
