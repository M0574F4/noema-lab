from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.core.executor import LocalExecutor
from noema_lab.core.lint import lint_recipe_invariants
from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.core.reproducibility import (
    environment_snapshot,
    installed_dependency_version,
)
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.ops import build_registry
from noema_lab.ops.channel.nr_ldpc import (
    NR_LDPC_MAX_TARGET_CODERATE,
    NrLdpcDecoderOperation,
    NrLdpcEncoderOperation,
    _nr_codec,
    _profile_sha256,
    _source_pixel_counts,
    _transport_block_configuration,
)
from noema_lab.training.differentiability import sionna_available


def _step(summary, step_id):
    return next(step for step in summary["steps"] if step["id"] == step_id)


def _nr_round_trip_recipe(snr_db: float = 30.0):
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "nr_ldpc_round_trip",
            "execution_profile": {"id": "layered_digital", "version": 1},
            "metadata": {
                "seed": 19,
                "rate_count_fixed_point": "tx_bit_boundary.channel.fixed.tx.bit_count",
            },
            "steps": [
                {
                    "id": "data",
                    "op": "source.random_bits",
                    "params": {"bit_count": 512, "batch_size": 1, "seed": 19},
                },
                {
                    "id": "payload_bit_boundary",
                    "op": "channel.bit_boundary",
                    "inputs": {"bits": "data.bits"},
                    "params": {"label": "payload", "role": "payload"},
                },
                {
                    "id": "channel_encoder",
                    "op": "channel.nr_ldpc_encoder",
                    "inputs": {"bits": "payload_bit_boundary.bits"},
                    "params": {
                        "target_coderate": 0.5,
                        "transport_block_size_bits": 256,
                        "num_bits_per_symbol": 2,
                        "num_bp_iter": 12,
                    },
                },
                {
                    "id": "tx_bit_boundary",
                    "op": "channel.bit_boundary",
                    "inputs": {"bits": "channel_encoder.coded_bits"},
                    "params": {"label": "tx", "role": "protected_link_input"},
                },
                {
                    "id": "modulator",
                    "op": "modulation.digital_modulate",
                    "inputs": {"bits": "tx_bit_boundary.bits"},
                    "params": {"modulation": "qpsk"},
                },
                {
                    "id": "wireless_channel",
                    "op": "wireless.channel",
                    "inputs": {"symbols": "modulator.symbols"},
                    "params": {
                        "channel": "awgn",
                        "snr_db": snr_db,
                        "wireless_backend": "numpy",
                        "seed": 23002,
                    },
                },
                {
                    "id": "demodulator",
                    "op": "demodulation.digital_demodulate",
                    "inputs": {"rx_symbols": "wireless_channel.rx_symbols"},
                    "params": {"modulation": "auto"},
                },
                {
                    "id": "rx_bit_boundary",
                    "op": "channel.bit_boundary",
                    "inputs": {"bits": "demodulator.bits"},
                    "params": {"label": "rx", "role": "protected_link_output"},
                },
                {
                    "id": "channel_bit_count_match",
                    "op": "channel.bit_count_match",
                    "inputs": {
                        "reference": "tx_bit_boundary.bits",
                        "candidate": "rx_bit_boundary.bits",
                    },
                    "params": {"label": "protected_link_io"},
                },
                {
                    "id": "channel_decoder",
                    "op": "channel.nr_ldpc_decoder",
                    "inputs": {"llr": "demodulator.llr"},
                    "params": {"on_tb_crc_failure": "zero_fill"},
                },
                {
                    "id": "payload_ber",
                    "op": "metrics.bit_error_rate",
                    "inputs": {
                        "reference": "payload_bit_boundary.bits",
                        "candidate": "channel_decoder.bits",
                    },
                    "params": {"label": "payload"},
                },
            ],
        }
    )


@unittest.skipUnless(
    sionna_available(),
    "NR LDPC integration requires the optional wireless extra",
)
class NrLdpcBaselineTests(unittest.TestCase):
    def test_case_study_coded_array_shape_is_consistent_in_retained_evidence(self):
        registry = build_registry()
        exact_cases = (
            (4320, 8640, 4352, 1),
            (8640, 17280, 8712, 2),
        )
        for (
            input_bit_count,
            expected_coded_bit_count,
            expected_effective_tb_size,
            expected_code_block_count,
        ) in exact_cases:
            with self.subTest(input_bit_count=input_bit_count):
                recipe = recipe_from_dict(
                    {
                        "schema_version": 1,
                        "name": "nr_ldpc_case_study_shape_%d" % input_bit_count,
                        "metadata": {"seed": 19},
                        "steps": [
                            {
                                "id": "data",
                                "op": "source.random_bits",
                                "params": {
                                    "bit_count": input_bit_count,
                                    "batch_size": 1,
                                    "seed": 19,
                                },
                            },
                            {
                                "id": "channel_encoder",
                                "op": "channel.nr_ldpc_encoder",
                                "inputs": {"bits": "data.bits"},
                                "params": {
                                    "target_coderate": 0.5,
                                    "transport_block_size_bits": 16000,
                                    "num_bits_per_symbol": 2,
                                    "num_layers": 1,
                                    "n_rnti": 1,
                                    "n_id": 1,
                                    "channel_type": "PUSCH",
                                    "codeword_index": 0,
                                    "num_bp_iter": 20,
                                },
                            },
                        ],
                    }
                )
                with tempfile.TemporaryDirectory() as tmp:
                    store = LocalStore(Path(tmp) / "workspace")
                    run_dir = LocalExecutor(registry, store).run(recipe)
                    retained = json.loads(
                        (run_dir / "summary.json").read_text(encoding="utf-8")
                    )
                    encoder = _step(retained, "channel_encoder")
                    coded_record = encoder["outputs"]["coded_bits"]
                    coded_metadata = coded_record["metadata"]
                    coded_path = Path(coded_record["path"])
                    with np.load(coded_path, allow_pickle=False) as payload:
                        file_shape = [int(value) for value in payload["bits"].shape]
                        embedded_metadata = json.loads(str(payload["metadata_json"]))

                expected_shape = [expected_coded_bit_count]
                self.assertEqual(file_shape, expected_shape)
                self.assertEqual(
                    encoder["metrics"]["channel.coded_bit_count"],
                    expected_coded_bit_count,
                )
                for metadata in (coded_metadata, embedded_metadata):
                    self.assertEqual(
                        metadata["coded_bit_count"],
                        expected_coded_bit_count,
                    )
                    self.assertEqual(metadata["bit_count"], expected_coded_bit_count)
                    self.assertEqual(metadata["shape"], expected_shape)
                    self.assertEqual(
                        metadata["arrays"]["bits"],
                        {"dtype": "uint8", "shape": expected_shape},
                    )
                block = coded_metadata["nr_transport_blocks"][0]
                self.assertEqual(
                    block["effective_tb_size_bits"],
                    expected_effective_tb_size,
                )
                self.assertEqual(
                    block["num_code_blocks"],
                    expected_code_block_count,
                )

    def test_standards_backed_transport_block_round_trip(self):
        registry = build_registry()
        recipe = _nr_round_trip_recipe()
        lint = lint_recipe_invariants(recipe, registry, strict=True)
        self.assertEqual(lint["status"], "passed", lint)
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp) / "workspace")
            run_dir = LocalExecutor(registry, store).run(recipe)
            summary = store.get_run(run_dir.name)

        encoder = _step(summary, "channel_encoder")
        decoder = _step(summary, "channel_decoder")
        self.assertEqual(encoder["metrics"]["channel.nr_ldpc.standard_profile"], 1)
        self.assertGreaterEqual(
            encoder["metrics"]["channel.nr_ldpc.transport_block_count"], 2
        )
        self.assertGreater(
            encoder["metrics"]["channel.nr_ldpc.logical_crc_bit_count"], 0
        )
        encoded_meta = encoder["outputs"]["coded_bits"]["metadata"]
        self.assertEqual(encoded_meta["nr_standard"]["version"], "18.8.0")
        self.assertEqual(
            set(encoded_meta["nr_backend_versions"]), {"sionna", "torch"}
        )
        self.assertTrue(all(encoded_meta["nr_backend_versions"].values()))
        self.assertEqual(
            decoder["metadata"]["backend_versions"],
            encoded_meta["nr_backend_versions"],
        )
        self.assertTrue(
            all(
                block["base_graph"] in {"bg1", "bg2"}
                and block["lifting_size"] > 0
                and sum(block["rate_matched_codeword_lengths"])
                == block["num_coded_bits"]
                for block in encoded_meta["nr_transport_blocks"]
            )
        )
        self.assertEqual(
            decoder["metrics"]["channel.nr_ldpc.transport_block_crc_failure_count"],
            0,
        )
        self.assertEqual(
            _step(summary, "payload_ber")["metrics"]["channel.payload.ber"], 0.0
        )

    def test_low_snr_exercises_crc_failure_and_declared_zero_fill(self):
        registry = build_registry()
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp) / "workspace")
            run_dir = LocalExecutor(registry, store).run(
                _nr_round_trip_recipe(snr_db=-20.0)
            )
            summary = store.get_run(run_dir.name)

        decoder = _step(summary, "channel_decoder")
        self.assertGreater(
            decoder["metrics"][
                "channel.nr_ldpc.transport_block_crc_failure_count"
            ],
            0,
        )
        self.assertEqual(decoder["metadata"]["failure_policy"], "zero_fill")
        self.assertGreater(
            _step(summary, "payload_ber")["metrics"]["channel.payload.ber"],
            0.0,
        )

    def test_decoder_fails_closed_when_bound_llr_length_is_wrong(self):
        recipe = _nr_round_trip_recipe()
        registry = build_registry()
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp) / "workspace")
            run_dir = LocalExecutor(registry, store).run(recipe)
            summary = store.get_run(run_dir.name)
            llr_record = _step(summary, "demodulator")["outputs"]["llr"]
            llr_path = Path(llr_record["path"])
            with np.load(llr_path, allow_pickle=False) as payload:
                llr = np.asarray(payload["llr"])
                metadata_json = str(payload["metadata_json"])
            tampered_path = Path(tmp) / "tampered_llr.npz"
            np.savez_compressed(
                tampered_path, llr=llr[:-1], metadata_json=metadata_json
            )
            decoder_dir = Path(tmp) / "decoder"
            decoder_dir.mkdir()
            with self.assertRaisesRegex(OperationError, "LLR count"):
                NrLdpcDecoderOperation().run(
                    OperationContext(
                        recipe_name="nr_ldpc_tamper",
                        step_id="decoder",
                        params={"on_tb_crc_failure": "zero_fill"},
                        inputs={
                            "llr": artifact(
                                "channel.llr.numpy",
                                tampered_path,
                                llr_record["metadata"],
                            )
                        },
                        run_dir=Path(tmp),
                        step_dir=decoder_dir,
                    )
                )

            tampered_metadata = json.loads(metadata_json)
            tampered_metadata["nr_transport_blocks"][0]["coded_offset"] = 1
            profile_tampered_path = Path(tmp) / "tampered_profile_llr.npz"
            np.savez_compressed(
                profile_tampered_path,
                llr=llr,
                metadata_json=json.dumps(tampered_metadata, sort_keys=True),
            )
            profile_decoder_dir = Path(tmp) / "profile_decoder"
            profile_decoder_dir.mkdir()
            with self.assertRaisesRegex(OperationError, "profile metadata"):
                NrLdpcDecoderOperation().run(
                    OperationContext(
                        recipe_name="nr_ldpc_profile_tamper",
                        step_id="decoder",
                        params={"on_tb_crc_failure": "zero_fill"},
                        inputs={
                            "llr": artifact(
                                "channel.llr.numpy",
                                profile_tampered_path,
                                llr_record["metadata"],
                            )
                        },
                        run_dir=Path(tmp),
                        step_dir=profile_decoder_dir,
                    )
                )

            version_tampered_metadata = json.loads(metadata_json)
            version_tampered_metadata["nr_backend_versions"]["sionna"] = "0.0.0"
            version_tampered_metadata["nr_profile_sha256"] = _profile_sha256(
                version_tampered_metadata["nr_transport_blocks"],
                version_tampered_metadata["nr_decoder_num_bp_iter"],
                version_tampered_metadata["nr_backend_versions"],
            )
            version_tampered_path = Path(tmp) / "version_tampered_llr.npz"
            np.savez_compressed(
                version_tampered_path,
                llr=llr,
                metadata_json=json.dumps(
                    version_tampered_metadata, sort_keys=True
                ),
            )
            version_decoder_dir = Path(tmp) / "version_decoder"
            version_decoder_dir.mkdir()
            with self.assertRaisesRegex(OperationError, "backend versions"):
                NrLdpcDecoderOperation().run(
                    OperationContext(
                        recipe_name="nr_ldpc_version_tamper",
                        step_id="decoder",
                        params={"on_tb_crc_failure": "zero_fill"},
                        inputs={
                            "llr": artifact(
                                "channel.llr.numpy",
                                version_tampered_path,
                                llr_record["metadata"],
                            )
                        },
                        run_dir=Path(tmp),
                        step_dir=version_decoder_dir,
                    )
                )


class NrLdpcReproducibilityGuardTests(unittest.TestCase):
    def test_encoder_schema_and_runtime_share_supported_coderate_ceiling(self):
        schema = NrLdpcEncoderOperation.params_schema["properties"][
            "target_coderate"
        ]
        self.assertEqual(schema["maximum"], NR_LDPC_MAX_TARGET_CODERATE)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(OperationError, "preflighted Sionna limit"):
                NrLdpcEncoderOperation().run(
                    OperationContext(
                        recipe_name="invalid_coderate",
                        step_id="encoder",
                        params={
                            "target_coderate": NR_LDPC_MAX_TARGET_CODERATE + 0.001
                        },
                        inputs={},
                        run_dir=Path(tmp),
                        step_dir=Path(tmp),
                    )
                )

    def test_backend_configuration_assertion_becomes_operation_error(self):
        def rejected_configuration(**kwargs):
            raise AssertionError("backend assertion")

        with mock.patch(
            "noema_lab.ops.channel.nr_ldpc._require_sionna",
            return_value=(None, None, None, rejected_configuration),
        ):
            with self.assertRaisesRegex(OperationError, "Sionna rejected"):
                _transport_block_configuration(256, 0.5, 2, 1)

    def test_backend_codec_assertion_becomes_operation_error(self):
        class RejectingEncoder:
            def __init__(self, **kwargs):
                raise AssertionError("backend assertion")

        _nr_codec.cache_clear()
        try:
            with mock.patch(
                "noema_lab.ops.channel.nr_ldpc._require_sionna",
                return_value=(None, RejectingEncoder, object, None),
            ):
                with self.assertRaisesRegex(OperationError, "Sionna rejected"):
                    _nr_codec(256, 512, 0.5, 2, 1, 1, 1, "PUSCH", 0, 20)
        finally:
            _nr_codec.cache_clear()

    def test_decoder_iteration_override_is_forbidden_before_backend_execution(self):
        self.assertNotIn(
            "num_bp_iter", NrLdpcDecoderOperation.params_schema["properties"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(OperationError, "overrides are forbidden"):
                NrLdpcDecoderOperation().run(
                    OperationContext(
                        recipe_name="decoder_override",
                        step_id="decoder",
                        params={"num_bp_iter": 12},
                        inputs={},
                        run_dir=Path(tmp),
                        step_dir=Path(tmp),
                    )
                )

    def test_environment_snapshot_captures_wireless_backend_versions(self):
        dependencies = environment_snapshot()["dependencies"]
        self.assertIn("sionna", dependencies)
        self.assertIn("sionna-no-rt", dependencies)
        self.assertIn("torch", dependencies)
        self.assertEqual(
            dependencies["sionna"],
            installed_dependency_version("sionna"),
        )
        self.assertEqual(
            dependencies["sionna-no-rt"],
            installed_dependency_version("sionna-no-rt"),
        )
        self.assertEqual(
            dependencies["torch"],
            installed_dependency_version("torch"),
        )
        # Legacy TensorFlow remains visible only for inspecting frozen
        # Sionna 1.x provenance; it is not part of the current wireless stack.
        self.assertIn("tensorflow", dependencies)


class NativeResourceAccountingTests(unittest.TestCase):
    def test_source_pixel_counts_remain_optional_only_when_undeclared(self):
        self.assertEqual(_source_pixel_counts({}), [])
        self.assertEqual(
            _source_pixel_counts({"source_item_pixel_counts": [20, 12]}),
            [20, 12],
        )
        self.assertEqual(
            _source_pixel_counts(
                {"original_shapes": [[1, 4, 5, 3], [1, 3, 4, 1]]}
            ),
            [20, 12],
        )
        self.assertEqual(
            _source_pixel_counts({"original_shape": [2, 4, 5, 3]}),
            [20, 20],
        )

    def test_source_pixel_counts_reject_malformed_declared_counts_and_shapes(self):
        malformed = [
            ("source_item_pixel_counts", {"source_item_pixel_counts": None}),
            ("source_item_pixel_counts", {"source_item_pixel_counts": []}),
            ("source_item_pixel_counts", {"source_item_pixel_counts": ["20"]}),
            ("source_item_pixel_counts", {"source_item_pixel_counts": [True]}),
            ("source_item_pixel_counts", {"source_item_pixel_counts": [0]}),
            ("original_shapes", {"original_shapes": None}),
            ("original_shapes", {"original_shapes": []}),
            ("original_shapes", {"original_shapes": [[1, 4, 5]]}),
            ("original_shapes", {"original_shapes": [[2, 4, 5, 3]]}),
            ("original_shapes", {"original_shapes": [[1, 4.0, 5, 3]]}),
            ("original_shapes", {"original_shapes": [[1, 0, 5, 3]]}),
            ("original_shape", {"original_shape": None}),
            ("original_shape", {"original_shape": [2, 4, 5]}),
            ("original_shape", {"original_shape": [2, "4", 5, 3]}),
            ("original_shape", {"original_shape": [0, 4, 5, 3]}),
        ]
        for label, metadata in malformed:
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(OperationError, label):
                    _source_pixel_counts(metadata)

    def test_source_pixel_counts_validate_all_present_declarations(self):
        with self.assertRaisesRegex(OperationError, "original_shapes"):
            _source_pixel_counts(
                {
                    "source_item_pixel_counts": [20],
                    "original_shapes": "malformed",
                }
            )
        with self.assertRaisesRegex(OperationError, "original_shape"):
            _source_pixel_counts(
                {
                    "original_shapes": [[1, 4, 5, 3]],
                    "original_shape": [],
                }
            )
        with self.assertRaisesRegex(OperationError, "disagrees"):
            _source_pixel_counts(
                {
                    "source_item_pixel_counts": [21],
                    "original_shapes": [[1, 4, 5, 3]],
                }
            )
        with self.assertRaisesRegex(OperationError, "declares 2 source items"):
            _source_pixel_counts(
                {
                    "original_shapes": [[1, 4, 5, 3], [1, 4, 5, 3]],
                    "original_shape": [1, 4, 5, 3],
                }
            )

    def test_jpeg_native_rate_is_separate_from_wrapper_and_link_overhead(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "images.npz"
            images = np.zeros((2, 16, 16, 3), dtype=np.uint8)
            images[0, 4:12, 4:12, :] = [20, 160, 230]
            images[1, 2:14, 6:10, :] = [210, 80, 25]
            np.savez_compressed(image_path, images=images)
            recipe = recipe_from_dict(
                {
                    "schema_version": 1,
                    "name": "native_resource_accounting",
                    "metadata": {"seed": 7},
                    "steps": [
                        {
                            "id": "data",
                            "op": "source.local_npz_images",
                            "params": {"path": str(image_path), "array": "images"},
                        },
                        {
                            "id": "sender",
                            "op": "model.jpeg_encode",
                            "inputs": {"images": "data.images"},
                            "params": {"quality": 75},
                        },
                        {
                            "id": "packetizer",
                            "op": "channel.packetize_crc32",
                            "inputs": {"bits": "sender.bits"},
                            "params": {"packet_payload_bits": 512},
                        },
                        {
                            "id": "channel_encoder",
                            "op": "channel.repetition_encoder",
                            "inputs": {"bits": "packetizer.bits"},
                            "params": {"factor": 3},
                        },
                        {
                            "id": "modulator",
                            "op": "modulation.digital_modulate",
                            "inputs": {"bits": "channel_encoder.coded_bits"},
                            "params": {"modulation": "qpsk"},
                        },
                        {
                            "id": "accounting",
                            "op": "channel.communication_resource_accounting",
                            "inputs": {
                                "payload_bits": "sender.bits",
                                "framed_bits": "packetizer.bits",
                                "coded_bits": "channel_encoder.coded_bits",
                                "symbols": "modulator.symbols",
                            },
                            "params": {
                                "pilot_symbol_count": 12,
                                "physical_header_symbol_count": 4,
                            },
                        },
                    ],
                }
            )
            store = LocalStore(root / "workspace")
            run_dir = LocalExecutor(build_registry(), store).run(recipe)
            summary = store.get_run(run_dir.name)

        sender = _step(summary, "sender")
        accounting = _step(summary, "accounting")
        sender_metrics = sender["metrics"]
        report = accounting["outputs"]["report"]["metadata"]
        self.assertLess(
            sender_metrics["codec.native_bit_count"],
            sender_metrics["codec.serialized_payload_bit_count"],
        )
        self.assertEqual(
            report["serialized_payload_bit_count"],
            report["native_codec_bit_count"]
            + report["safe_serialization_wrapper_bit_count"],
        )
        self.assertEqual(
            report["framed_bit_count"],
            report["serialized_payload_bit_count"]
            + report["framing_header_crc_padding_bit_count"],
        )
        self.assertEqual(
            report["coded_bit_count"],
            report["framed_bit_count"]
            + report["fec_rate_matching_overhead_bit_count"],
        )
        self.assertEqual(
            report["modulator_capacity_bit_count"],
            report["coded_bit_count"] + report["modulation_padding_bit_count"],
        )
        self.assertEqual(
            accounting["metrics"]["channel.transmitted_bit_count"],
            report["modulator_capacity_bit_count"],
        )
        self.assertAlmostEqual(
            accounting["metrics"]["rate.padded_bpp"],
            report["modulator_capacity_bit_count"] / report["pixel_count"],
        )
        self.assertEqual(report["pilot_symbol_count"], 12)
        self.assertEqual(report["physical_header_symbol_count"], 4)
        self.assertEqual(len(report["source_item_total_channel_use_counts"]), 2)
        self.assertEqual(
            report["source_item_total_channel_use_counts"],
            [value + 16 for value in report["source_item_data_symbol_counts"]],
        )
        self.assertFalse(report["source_item_use_counts_are_additive"])
        self.assertAlmostEqual(
            accounting["metrics"]["channel.max_source_item_uses_per_pixel"],
            max(report["source_item_channel_uses_per_pixel"]),
        )
        self.assertEqual(len(report["accounting_identity_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
