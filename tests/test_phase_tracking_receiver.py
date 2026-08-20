from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.executor import LocalExecutor
from noema_lab.core.matrix import expand_recipe_matrix
from noema_lab.core.operations import OperationContext
from noema_lab.core.recipes import load_recipe, recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.ops import build_registry
from noema_lab.ops.phase_tracking import (
    PHASE_TRUTH_KIND,
    PILOT_CONTEXT_KIND,
    PhaseTrackingReceiverAdapterOperation,
)


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "recipes" / "neural_receiver_qpsk_phase_tracking.yaml"


def _step(summary: dict, step_id: str) -> dict:
    return next(row for row in summary["steps"] if row["id"] == step_id)


def _run_recipe(recipe, root: Path) -> dict:
    # These operation tests exercise one authored coordinate. The template also
    # carries a matrix for the UI's Run All path, which is tested separately.
    recipe_payload = recipe.to_dict()
    recipe_payload.setdefault("metadata", {}).pop("matrix", None)
    run_dir = LocalExecutor(
        build_registry(), LocalStore(root / ".noema")
    ).run(recipe_from_dict(recipe_payload))
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


class PhaseTrackingReceiverTests(unittest.TestCase):
    def test_template_sweeps_awgn_noise_at_fixed_transmit_power(self) -> None:
        recipe = load_recipe(RECIPE_PATH)
        matrix = recipe.metadata["matrix"]
        self.assertEqual(
            matrix["dimensions"]["channel.snr_db"],
            [-2, 2, 6, 10],
        )
        self.assertEqual(
            matrix["step_params"]["wireless_channel"]["snr_db"],
            {"matrix": "channel.snr_db"},
        )

        expanded = expand_recipe_matrix(recipe)
        self.assertEqual(expanded["expanded_count"], 4)
        self.assertEqual(
            [
                row["metadata"]["matrix_selection"]["channel.snr_db"]
                for row in expanded["recipes"]
            ],
            [-2, 2, 6, 10],
        )
        channel_params = [
            next(
                step["params"]
                for step in row["steps"]
                if step["id"] == "wireless_channel"
            )
            for row in expanded["recipes"]
        ]
        self.assertEqual(
            [params["snr_db"] for params in channel_params],
            [-2, 2, 6, 10],
        )
        self.assertEqual(
            {params["noise_mode"] for params in channel_params},
            {"snr_at_unit_power"},
        )
        self.assertEqual(
            {params["seed"] for params in channel_params},
            {23001},
        )

    def test_template_executes_the_full_pilot_carrier_receiver_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = _run_recipe(load_recipe(RECIPE_PATH), Path(tmp))

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(
            [row["id"] for row in summary["steps"]],
            [
                "data",
                "payload_bit_boundary",
                "channel_encoder",
                "tx_bit_boundary",
                "modulator",
                "wireless_channel",
                "carrier_impairment",
                "demodulator",
                "rx_bit_boundary",
                "coded_ber",
                "coded_bler",
                "channel_bit_count_match",
                "channel_decoder",
            ],
        )
        modulator_metrics = _step(summary, "modulator")["metrics"]
        self.assertGreater(modulator_metrics["channel.pilot_symbol_count"], 0)
        self.assertGreater(
            modulator_metrics["channel.channel_use_count"],
            modulator_metrics["channel.data_symbol_count"],
        )
        bler_metrics = _step(summary, "coded_bler")["metrics"]
        self.assertEqual(bler_metrics["channel.coded.block_count"], 256)
        demodulator_outputs = _step(summary, "demodulator")["outputs"]
        for output_name in ("bits", "llr"):
            output_metadata = demodulator_outputs[output_name]["metadata"]
            self.assertEqual(output_metadata["packet_count"], 256)
            self.assertEqual(output_metadata["bit_count"], 256 * 1024)
            self.assertEqual(output_metadata["bit_count_per_example"], 1024)
            self.assertEqual(output_metadata["bits_per_packet"], 1024)
            self.assertEqual(output_metadata["capture_record_count"], 256)
            self.assertEqual(output_metadata["capture_record_shape"], [1024])
            self.assertEqual(
                output_metadata["axes"], ["flattened_packet_data_bit"]
            )
        boundary_metadata = _step(summary, "rx_bit_boundary")["outputs"][
            "bits"
        ]["metadata"]
        self.assertEqual(boundary_metadata["capture_record_count"], 256)
        self.assertEqual(boundary_metadata["capture_record_shape"], [1024])
        truth = _step(summary, "carrier_impairment")["outputs"]["phase_truth"]
        preview = truth["metadata"]["phase_truth_preview"]
        self.assertEqual(preview["packet_index"], 0)
        self.assertLessEqual(len(preview["frame_symbol_index"]), 64)
        self.assertEqual(
            len(preview["frame_symbol_index"]), len(preview["true_phase_rad"])
        )
        self.assertTrue(truth["metadata"]["simulation_truth"])
        self.assertFalse(truth["metadata"]["runtime_receiver_input"])

    def test_oracle_phase_correction_beats_an_uncompensated_receiver(self) -> None:
        authored = load_recipe(RECIPE_PATH).to_dict()
        summaries = {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ("uncompensated", "oracle"):
                payload = copy.deepcopy(authored)
                payload["name"] = "%s_%s" % (payload["name"], mode)
                demodulator = next(
                    row for row in payload["steps"] if row["id"] == "demodulator"
                )
                demodulator["params"]["mode"] = mode
                summaries[mode] = _run_recipe(recipe_from_dict(payload), root)

        uncompensated_ber = _step(summaries["uncompensated"], "coded_ber")[
            "metrics"
        ]["channel.coded.ber"]
        oracle_ber = _step(summaries["oracle"], "coded_ber")["metrics"][
            "channel.coded.ber"
        ]
        self.assertGreater(uncompensated_ber, 0.25)
        self.assertLess(oracle_ber, 0.05)
        self.assertLess(oracle_ber, uncompensated_ber)

    def test_oracle_recovers_every_bit_and_pilot_accounting_is_exact(self) -> None:
        payload = load_recipe(RECIPE_PATH).to_dict()
        payload["name"] = "qpsk_phase_tracking_noiseless_oracle"
        data = next(row for row in payload["steps"] if row["id"] == "data")
        data["params"].update({"bit_count": 128, "batch_size": 1})
        channel = next(
            row for row in payload["steps"] if row["id"] == "wireless_channel"
        )
        channel["params"]["snr_db"] = 200.0
        carrier = next(
            row for row in payload["steps"] if row["id"] == "carrier_impairment"
        )
        carrier["params"].update(
            {
                "initial_phase_min_rad": 1.1,
                "initial_phase_max_rad": 1.1,
                "cfo_min_cycles_per_symbol": 0.02,
                "cfo_max_cycles_per_symbol": 0.02,
                "phase_noise_increment_std_rad": 0.08,
            }
        )
        demodulator = next(
            row for row in payload["steps"] if row["id"] == "demodulator"
        )
        demodulator["params"]["mode"] = "oracle"

        with tempfile.TemporaryDirectory() as tmp:
            summary = _run_recipe(recipe_from_dict(payload), Path(tmp))

        metrics = _step(summary, "modulator")["metrics"]
        self.assertEqual(metrics["channel.data_symbol_count"], 64)
        self.assertEqual(metrics["channel.pilot_symbol_count"], 20)
        self.assertEqual(metrics["channel.channel_use_count"], 84)
        self.assertAlmostEqual(metrics["channel.pilot_overhead_fraction"], 20 / 84)
        ber = _step(summary, "coded_ber")["metrics"]
        self.assertEqual(ber["channel.coded.error_count"], 0)
        self.assertEqual(ber["channel.coded.ber"], 0.0)

    def test_learned_runtime_receives_observations_and_pilots_but_not_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {
                "packet_count": 1,
                "frame_symbols_per_packet": 4,
                "bit_count": 4,
                "noise_variance": 0.1,
            }
            rx = np.asarray(
                [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], dtype=np.complex64
            ) / np.sqrt(2.0)
            pilot_context = np.zeros((1, 4, 3), dtype=np.float32)
            pilot_context[0, (0, 3), 0] = 1.0
            pilot_context[0, 0, 1:] = [float(rx[0].real), float(rx[0].imag)]
            pilot_context[0, 3, 1:] = [float(rx[3].real), float(rx[3].imag)]
            rx_path = root / "rx.npz"
            context_path = root / "pilot_context.npz"
            np.savez_compressed(rx_path, symbols=rx)
            np.savez_compressed(context_path, pilot_context=pilot_context)
            # Deliberately invalid: learned inference must not even load this file.
            truth_path = root / "phase_truth.invalid"
            truth_path.write_bytes(b"simulator truth must remain private")
            manifest_path = root / "trained_artifact.yaml"
            manifest_path.write_text("schema_version: 2\n", encoding="utf-8")
            captured = {}

            def fake_runtime(
                path,
                entrypoint,
                inputs,
                *,
                expected_package_sha256,
            ):
                captured["path"] = path
                captured["entrypoint"] = entrypoint
                captured["inputs"] = inputs
                captured["package_sha256"] = expected_package_sha256
                return {
                    "residual_phase_rad": np.zeros((1, 4), dtype=np.float32),
                }

            with mock.patch(
                "noema_lab.core.trained_artifact_runtime.run_trained_artifact_entrypoint",
                side_effect=fake_runtime,
            ):
                result = PhaseTrackingReceiverAdapterOperation().run(
                    OperationContext(
                        recipe_name="learned_phase_tracking_test",
                        step_id="demodulator",
                        params={
                            "mode": "learned_artifact",
                            "artifact_manifest_path": str(manifest_path),
                            "artifact_entrypoint": "phase_tracking_receiver",
                            "artifact_package_sha256": "b" * 64,
                        },
                        inputs={
                            "rx_symbols": Artifact(
                                "channel.rx_symbols.complex_numpy", rx_path, metadata
                            ),
                            "pilot_context": Artifact(
                                PILOT_CONTEXT_KIND, context_path, metadata
                            ),
                            "phase_truth": Artifact(
                                PHASE_TRUTH_KIND,
                                truth_path,
                                {"private_simulation_truth": True},
                            ),
                        },
                        run_dir=root,
                        step_dir=root / "demodulator",
                    )
                )

        self.assertEqual(captured["entrypoint"], "phase_tracking_receiver")
        self.assertEqual(captured["package_sha256"], "b" * 64)
        self.assertEqual(set(captured["inputs"]), {"receiver_features_v3"})
        self.assertEqual(
            captured["inputs"]["receiver_features_v3"].shape,
            (1, 4, 11),
        )
        self.assertFalse(result.metadata["phase_truth_forwarded_to_learned_runtime"])
        diagnostics = result.outputs["diagnostics"].metadata
        self.assertFalse(diagnostics["phase_truth_used_for_decisions"])
        self.assertFalse(diagnostics["phase_truth_forwarded_to_learned_runtime"])
        self.assertTrue(diagnostics["phase_estimate_available"])

    def test_portable_receiver_abi_requires_only_runtime_observations(self) -> None:
        abi = PhaseTrackingReceiverAdapterOperation().describe()[
            "trained_artifact_abi"
        ]
        self.assertEqual(
            abi["required_operation_inputs"], ["rx_symbols", "pilot_context"]
        )
        self.assertEqual(
            abi["inputs"]["receiver_features_v3"]["shape"],
            ["packet", "frame_symbol", 11],
        )
        self.assertEqual(
            abi["outputs"]["residual_phase_rad"]["shape"],
            ["packet", "frame_symbol"],
        )
        self.assertNotIn("phase_truth", abi["required_operation_inputs"])


if __name__ == "__main__":
    unittest.main()
