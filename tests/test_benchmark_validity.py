import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.core.artifacts import artifact, file_sha256
from noema_lab.core.attempt_ledger import BenchmarkAttemptLedger
from noema_lab.core.benchmark_evidence import (
    BenchmarkEvidenceError,
    snapshot_benchmark_training_evidence,
    validate_benchmark_training_evidence_snapshot,
)
from noema_lab.core.benchmarks import (
    BenchmarkError,
    BenchmarkPack,
    BenchmarkRecipe,
    _validate_recipe_training_lineage,
    benchmark_protocol_sha256,
    load_benchmark_pack,
    run_benchmark_pack,
    validate_benchmark_pack,
    validate_trained_artifact_lineage_manifest,
    write_benchmark_reports,
)
from noema_lab.core.operations import OperationContext
from noema_lab.core.publication_profile import (
    publication_verification_profile_binding,
)
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.storage import LocalStore
from noema_lab.core.verification import (
    _CheckRecorder,
    _check_benchmark_protocol,
    _check_benchmark_training_evidence_snapshot,
    verify_benchmark_result,
)
from noema_lab.ops import build_registry
from noema_lab.ops.channel.digital import (
    CapacityOracleDigitalLinkOperation,
    Crc32CheckOperation,
    Crc32PacketizeOperation,
    DigitalDemodulateOperation,
    DigitalModulateOperation,
    SymbolPowerNormalizeOperation,
    WirelessChannelOperation,
)
from noema_lab.ops.models.external import (
    DeepJsccExternalDecodeOperation,
    DeepJsccExternalEncodeOperation,
)
from noema_lab.ops.models.learned_codecs import (
    JpegCapacityOracleOperation,
    JpegDecodeOperation,
    JpegEncodeOperation,
)
from noema_lab.ops.source.kodak import KODAK_SIZE_BYTES, _verify_kodak_file


def _operation_context(root, step_id, params, inputs):
    step_dir = root / step_id
    step_dir.mkdir(parents=True, exist_ok=True)
    return OperationContext(
        recipe_name="benchmark_validity_test",
        step_id=step_id,
        params=params,
        inputs=inputs,
        run_dir=root,
        step_dir=step_dir,
    )


def _bits_artifact(root, name, bits, item_counts):
    path = root / (name + ".npz")
    metadata = {
        "bit_count": int(bits.size),
        "payload_bit_count": int(bits.size),
        "source_item_count": len(item_counts),
        "source_item_payload_bit_counts": list(item_counts),
        "source_item_ids": ["item_%d" % index for index in range(len(item_counts))],
        "original_shapes": [[1, 8, 8, 3] for _item in item_counts],
    }
    np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
    return artifact("channel.payload_bits.numpy", path, metadata)


def _images_artifact(root, name, images):
    path = root / (name + ".npz")
    metadata = {
        "image_ids": ["image_%d" % index for index in range(int(images.shape[0]))],
        "original_shape": list(images.shape),
        "storage_shape": list(images.shape),
        "original_shapes": [
            [1, int(images.shape[1]), int(images.shape[2]), int(images.shape[3])]
            for _index in range(int(images.shape[0]))
        ],
    }
    np.savez_compressed(path, images=images, metadata_json=json.dumps(metadata))
    return artifact("image.batch.numpy", path, metadata)


def _symbols_artifact(root, name, symbols, item_counts, item_ids):
    path = root / (name + ".npz")
    metadata = {
        "symbol_count": int(symbols.size),
        "source_item_symbol_counts": list(item_counts),
        "source_item_ids": list(item_ids),
        "original_shapes": [[1, 2, 2, 1] for _item in item_counts],
        "power_unit": "normalized",
    }
    np.savez_compressed(path, symbols=symbols, metadata_json=json.dumps(metadata))
    return artifact("channel.symbols.complex_numpy", path, metadata)


class BenchmarkValidityTests(unittest.TestCase):
    def test_protocol_identity_excludes_loader_checkout_path(self):
        pack = BenchmarkPack(
            id="path-independent",
            version="1",
            recipes=[BenchmarkRecipe(id="candidate", path=Path("recipe.yaml"))],
            path=Path("/checkout/one/benchmark.yaml"),
        )
        first = benchmark_protocol_sha256(pack)
        pack.path = Path("/different/runner/checkout/benchmark.yaml")
        self.assertEqual(benchmark_protocol_sha256(pack), first)

    def test_public_schemas_reject_noncanonical_publication_claims(self):
        pack_schema = json.loads(
            (ROOT / "schemas" / "benchmark_pack.schema.json").read_text(
                encoding="utf-8"
            )
        )
        pack_payload = yaml.safe_load(
            (
                ROOT
                / "benchmarks"
                / "beamforming_precoding"
                / "beam_selection_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        pack_payload["metadata"]["publication_ready"] = True

        with self.assertRaises(ValidationError):
            Draft202012Validator(pack_schema).validate(pack_payload)

        result_schema = json.loads(
            (ROOT / "schemas" / "benchmark_result.schema.json").read_text(
                encoding="utf-8"
            )
        )
        result_payload = {
            "schema_version": 1,
            "kind": "noema.benchmark_result",
            "status": "completed",
            "benchmark": {
                "id": "experimental",
                "version": "1",
                "sha256": "0" * 64,
                "benchmark_tier": "experimental",
                "publication_ready": True,
            },
            "recipes": [],
        }

        with self.assertRaises(ValidationError):
            Draft202012Validator(result_schema).validate(result_payload)

    def test_public_schemas_admit_new_profile_trigger_and_reject_alias_conflicts(
        self,
    ):
        binding = publication_verification_profile_binding()
        pack_schema = json.loads(
            (ROOT / "schemas" / "benchmark_pack.schema.json").read_text(
                encoding="utf-8"
            )
        )
        pack_payload = yaml.safe_load(
            (
                ROOT
                / "benchmarks"
                / "beamforming_precoding"
                / "beam_selection_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        pack_payload["metadata"].update(
            {
                "benchmark_tier": "canonical",
                "traceability_profile_requested": True,
                "verification_profile": binding,
            }
        )
        pack_payload["metadata"].pop("publication_ready", None)
        Draft202012Validator(pack_schema).validate(pack_payload)
        pack_payload["metadata"]["publication_ready"] = False
        with self.assertRaises(ValidationError):
            Draft202012Validator(pack_schema).validate(pack_payload)

        result_schema = json.loads(
            (ROOT / "schemas" / "benchmark_result.schema.json").read_text(
                encoding="utf-8"
            )
        )
        result_payload = {
            "schema_version": 1,
            "kind": "noema.benchmark_result",
            "status": "completed",
            "benchmark": {
                "id": "canonical",
                "version": "1",
                "sha256": "0" * 64,
                "benchmark_tier": "canonical",
                "traceability_profile_requested": True,
                "metadata": {"verification_profile": binding},
            },
            "recipes": [],
        }
        Draft202012Validator(result_schema).validate(result_payload)
        result_payload["benchmark"]["publication_ready"] = False
        with self.assertRaises(ValidationError):
            Draft202012Validator(result_schema).validate(result_payload)

    def test_benchmark_loader_rejects_duplicate_yaml_and_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            documents = {
                "duplicate.yaml": (
                    "schema_version: 1\n"
                    "id: first\n"
                    "id: silently_overwritten\n"
                    "recipes: []\n"
                ),
                "duplicate.json": (
                    '{"schema_version":1,"id":"first",'
                    '"id":"silently_overwritten","recipes":[]}'
                ),
            }
            for filename, text in documents.items():
                with self.subTest(filename=filename):
                    path = root / filename
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaisesRegex(
                        BenchmarkError,
                        "Duplicate (YAML mapping|JSON object) key",
                    ):
                        load_benchmark_pack(path)

    def test_shipped_recipes_do_not_reuse_an_explicit_seed_across_rng_steps(self):
        for recipe_path in sorted((ROOT / "recipes").glob("*.yaml")):
            with self.subTest(recipe=str(recipe_path.relative_to(ROOT))):
                document = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
                owners = {}
                for step in document.get("steps") or []:
                    params = step.get("params") or {}
                    if "seed" not in params:
                        continue
                    seed = int(params["seed"])
                    self.assertNotIn(
                        seed,
                        owners,
                        "explicit seed %d is reused by %s and %s; use independent "
                        "per-phenomenon streams"
                        % (seed, owners.get(seed), step.get("id")),
                    )
                    owners[seed] = step.get("id")

    def test_every_declared_baseline_is_linked_to_a_frozen_recipe_method(self):
        for pack_path in sorted((ROOT / "benchmarks").rglob("*.yaml")):
            with self.subTest(pack=str(pack_path.relative_to(ROOT))):
                document = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
                recipes = document.get("recipes") or []
                linked = {
                    str(recipe["id"]).strip()
                    for recipe in recipes
                    if recipe.get("id")
                }
                linked.update(
                    str(candidate).strip()
                    for recipe in recipes
                    for candidate in (
                        (recipe.get("params") or {}).get("baseline_id"),
                        (recipe.get("params") or {}).get("method_id"),
                    )
                    if candidate not in (None, "")
                )
                self.assertEqual(
                    [],
                    [
                        str(baseline)
                        for baseline in document.get("baselines") or []
                        if str(baseline).strip() not in linked
                    ],
                )

    def test_pack_validation_rejects_a_decorative_unlinked_baseline(self):
        pack = load_benchmark_pack(
            ROOT / "benchmarks" / "beamforming_precoding" / "beam_selection_v1.yaml"
        )
        pack.baselines = ["decorative_baseline_with_no_recipe"]

        with self.assertRaisesRegex(
            BenchmarkError,
            "not linked to any recipe id",
        ):
            validate_benchmark_pack(pack, build_registry(), ROOT)

    def test_experimental_pack_cannot_claim_publication_readiness(self):
        pack = load_benchmark_pack(
            ROOT / "benchmarks" / "beamforming_precoding" / "beam_selection_v1.yaml"
        )
        pack.metadata["publication_ready"] = True

        with self.assertRaisesRegex(
            BenchmarkError,
            "must use metadata.benchmark_tier=canonical",
        ):
            validate_benchmark_pack(pack, build_registry(), ROOT)

    def test_publication_readiness_must_be_a_boolean(self):
        pack = load_benchmark_pack(
            ROOT / "benchmarks" / "beamforming_precoding" / "beam_selection_v1.yaml"
        )
        pack.metadata["publication_ready"] = "false"

        with self.assertRaisesRegex(
            BenchmarkError,
            "metadata.publication_ready must be a boolean",
        ):
            validate_benchmark_pack(pack, build_registry(), ROOT)

    def test_traceability_profile_alias_disagreement_is_rejected(self):
        pack = load_benchmark_pack(
            ROOT / "benchmarks" / "beamforming_precoding" / "beam_selection_v1.yaml"
        )
        pack.metadata["traceability_profile_requested"] = True
        pack.metadata["publication_ready"] = False

        with self.assertRaisesRegex(
            BenchmarkError,
            "conflicts with deprecated alias",
        ):
            validate_benchmark_pack(pack, build_registry(), ROOT)

    def test_result_verification_rejects_an_unlinked_experimental_baseline(self):
        benchmark_json = {
            "schema_version": 1,
            "id": "experimental_unlinked_baseline",
            "version": "1",
            "baselines": ["decorative_baseline"],
            "recipes": [{"id": "actual_recipe", "role": "baseline", "params": {}}],
            "metadata": {
                "benchmark_tier": "experimental",
                "publication_ready": False,
            },
        }
        result = {
            "benchmark": {
                "id": benchmark_json["id"],
                "version": benchmark_json["version"],
                "sha256": canonical_json_sha256(benchmark_json),
                "benchmark_tier": "experimental",
                "publication_ready": False,
            },
            "recipes": [{"id": "actual_recipe", "role": "baseline"}],
        }
        recorder = _CheckRecorder()

        _check_benchmark_protocol(result, benchmark_json, recorder)

        roster_checks = [
            check
            for check in recorder.checks
            if check.id == "benchmark.protocol.baseline_roster"
        ]
        self.assertEqual(len(roster_checks), 1)
        self.assertEqual(roster_checks[0].status, "error")

    def test_result_verification_rejects_experimental_publication_claim(self):
        benchmark_json = {
            "schema_version": 1,
            "id": "experimental_publication_claim",
            "version": "1",
            "recipes": [],
            "metadata": {
                "benchmark_tier": "experimental",
                "publication_ready": True,
            },
        }
        result = {
            "benchmark": {
                "id": benchmark_json["id"],
                "version": benchmark_json["version"],
                "sha256": canonical_json_sha256(benchmark_json),
                "benchmark_tier": "experimental",
                "publication_ready": True,
            },
            "recipes": [],
        }
        recorder = _CheckRecorder()

        _check_benchmark_protocol(result, benchmark_json, recorder)

        tier_checks = [
            check
            for check in recorder.checks
            if check.id == "benchmark.protocol.publication_tier"
        ]
        self.assertEqual(len(tier_checks), 1)
        self.assertEqual(tier_checks[0].status, "error")

    def test_result_verification_rejects_traceability_alias_disagreement(self):
        benchmark_json = {
            "schema_version": 1,
            "id": "conflicting_profile_trigger",
            "version": "1",
            "recipes": [],
            "metadata": {
                "benchmark_tier": "canonical",
                "traceability_profile_requested": True,
                "publication_ready": False,
                "verification_profile": publication_verification_profile_binding(),
            },
        }
        result = {
            "benchmark": {
                "id": benchmark_json["id"],
                "version": benchmark_json["version"],
                "sha256": canonical_json_sha256(benchmark_json),
                "benchmark_tier": "canonical",
                "traceability_profile_requested": True,
                "publication_ready": True,
                "metadata": {
                    "verification_profile": publication_verification_profile_binding()
                },
            },
            "recipes": [],
        }
        recorder = _CheckRecorder()

        _check_benchmark_protocol(result, benchmark_json, recorder)

        conflict_checks = [
            check
            for check in recorder.checks
            if check.id == "benchmark.protocol.traceability_profile_request"
        ]
        self.assertEqual(len(conflict_checks), 1)
        self.assertEqual(conflict_checks[0].status, "error")

    def test_jpeg_failure_fallback_is_per_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = np.full((24, 24, 3), 35, dtype=np.uint8)
            first[5:19, 6:18, :] = [20, 190, 230]
            second = np.full((24, 24, 3), 210, dtype=np.uint8)
            second[4:20, 4:20, :] = [220, 40, 70]

            single = JpegEncodeOperation().run(
                _operation_context(
                    root,
                    "jpeg_single",
                    {"quality": 75},
                    {"images": _images_artifact(root, "single_image", first[None, ...])},
                )
            )
            batch = JpegEncodeOperation().run(
                _operation_context(
                    root,
                    "jpeg_batch",
                    {"quality": 75},
                    {
                        "images": _images_artifact(
                            root, "batch_images", np.stack([first, second], axis=0)
                        )
                    },
                )
            )
            with np.load(single.outputs["bits"].path, allow_pickle=False) as payload:
                single_bits = payload["bits"]
            with np.load(batch.outputs["bits"].path, allow_pickle=False) as payload:
                batch_bits = payload["bits"]
                batch_metadata = json.loads(str(payload["metadata_json"]))
            first_count = batch_metadata["source_item_payload_bit_counts"][0]
            np.testing.assert_array_equal(batch_bits[:first_count], single_bits)

            packetized = Crc32PacketizeOperation().run(
                _operation_context(
                    root,
                    "jpeg_packetize",
                    {"packet_payload_bits": 256},
                    {"bits": batch.outputs["bits"]},
                )
            )
            with np.load(packetized.outputs["bits"].path, allow_pickle=False) as payload:
                wire_bits = payload["bits"].copy()
                wire_metadata = json.loads(str(payload["metadata_json"]))
            second_start = (
                wire_metadata["packet_source_item_counts"][0]
                * wire_metadata["packet_total_bits"]
            )
            wire_bits[
                second_start
                + wire_metadata["packet_header_bits"]
                + wire_metadata["packet_header_crc_bits"]
                + 7
            ] ^= 1
            corrupted_path = root / "jpeg_corrupted.npz"
            np.savez_compressed(
                corrupted_path,
                bits=wire_bits,
                metadata_json=json.dumps(wire_metadata),
            )
            checked = Crc32CheckOperation().run(
                _operation_context(
                    root,
                    "jpeg_crc_check",
                    {"on_decode_failure": "gray_image"},
                    {
                        "bits": artifact(
                            "channel.payload_bits.numpy", corrupted_path, wire_metadata
                        )
                    },
                )
            )
            decoded = JpegDecodeOperation().run(
                _operation_context(
                    root,
                    "jpeg_decode",
                    {"on_error": "gray_image"},
                    {"bits": checked.outputs["bits"]},
                )
            )
            with np.load(decoded.outputs["images"].path, allow_pickle=False) as payload:
                reconstructed = payload["images"]
            self.assertFalse(np.all(reconstructed[0] == 128))
            self.assertTrue(np.all(reconstructed[1] == 128))
            self.assertEqual(
                decoded.outputs["images"].metadata["source_item_decode_success"],
                [True, False],
            )

    def test_crc_packets_are_item_owned_and_batch_size_invariant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = np.tile(np.array([0, 1, 1, 0], dtype=np.uint8), 37)
            second = np.tile(np.array([1, 0, 0, 1], dtype=np.uint8), 29)

            single_input = _bits_artifact(root, "single", first, [int(first.size)])
            single = Crc32PacketizeOperation().run(
                _operation_context(root, "packetize_single", {"packet_payload_bits": 64}, {"bits": single_input})
            )
            batch_input = _bits_artifact(
                root,
                "batch",
                np.concatenate([first, second]),
                [int(first.size), int(second.size)],
            )
            batch = Crc32PacketizeOperation().run(
                _operation_context(root, "packetize_batch", {"packet_payload_bits": 64}, {"bits": batch_input})
            )

            with np.load(single.outputs["bits"].path, allow_pickle=False) as payload:
                single_bits = payload["bits"]
            with np.load(batch.outputs["bits"].path, allow_pickle=False) as payload:
                batch_bits = payload["bits"].copy()
                batch_metadata = json.loads(str(payload["metadata_json"]))
            first_packet_bits = (
                batch_metadata["packet_source_item_counts"][0]
                * batch_metadata["packet_total_bits"]
            )
            self.assertNotIn("packet_crc32", batch_metadata)
            self.assertNotIn("packet_payload_bit_counts", batch_metadata)
            self.assertEqual(batch_metadata["packet_crc_reference"], "received_crc_only")
            self.assertEqual(
                batch_metadata["packet_protocol"], "noema.source_item_crc32.v4"
            )

            clean_single = Crc32CheckOperation().run(
                _operation_context(
                    root,
                    "crc_check_single",
                    {"on_decode_failure": "gray_image"},
                    {"bits": single.outputs["bits"]},
                )
            )
            clean_batch = Crc32CheckOperation().run(
                _operation_context(
                    root,
                    "crc_check_batch",
                    {"on_decode_failure": "gray_image"},
                    {"bits": batch.outputs["bits"]},
                )
            )
            with np.load(clean_single.outputs["bits"].path, allow_pickle=False) as payload:
                recovered_single = payload["bits"]
            with np.load(clean_batch.outputs["bits"].path, allow_pickle=False) as payload:
                recovered_batch = payload["bits"]
            np.testing.assert_array_equal(recovered_single, first)
            np.testing.assert_array_equal(recovered_batch[: first.size], first)

            second_packet_start = first_packet_bits
            batch_bits[
                second_packet_start
                + batch_metadata["packet_header_bits"]
                + batch_metadata["packet_header_crc_bits"]
                + 3
            ] ^= 1
            # Redundant sender-side fields are deliberately removed: ownership
            # and exact lengths must come from the hash-bound packet contract.
            for key in (
                "source_item_count",
                "source_item_payload_bit_counts",
                "source_item_payload_byte_counts",
                "packet_source_item_counts",
            ):
                batch_metadata.pop(key, None)
            corrupted_path = root / "corrupted.npz"
            np.savez_compressed(
                corrupted_path,
                bits=batch_bits,
                metadata_json=json.dumps(batch_metadata),
            )
            corrupted = artifact(
                "channel.payload_bits.numpy", corrupted_path, batch_metadata
            )
            checked = Crc32CheckOperation().run(
                _operation_context(
                    root,
                    "crc_check",
                    {"on_decode_failure": "gray_image"},
                    {"bits": corrupted},
                )
            )
            self.assertEqual(
                checked.outputs["bits"].metadata["source_item_outage"], [0, 1]
            )
            self.assertEqual(checked.metrics["channel.outage_rate"], 0.5)
            with np.load(checked.outputs["bits"].path, allow_pickle=False) as payload:
                recovered = payload["bits"]
            np.testing.assert_array_equal(recovered[: first.size], first)
            self.assertEqual(
                checked.outputs["bits"].metadata["source_item_length_source"],
                "immutable_packet_contract",
            )

            header_corrupted = batch_bits.copy()
            # Restore the payload flip, then corrupt only the second item's
            # ownership header. The valid first item must not be collateral loss.
            header_corrupted[
                second_packet_start
                + batch_metadata["packet_header_bits"]
                + batch_metadata["packet_header_crc_bits"]
                + 3
            ] ^= 1
            header_repeat = batch_metadata["packet_header_repetition_factor"]
            header_bit_start = second_packet_start + 5 * header_repeat
            header_corrupted[
                header_bit_start : header_bit_start + header_repeat
            ] ^= 1
            header_corrupted_path = root / "header_corrupted.npz"
            np.savez_compressed(
                header_corrupted_path,
                bits=header_corrupted,
                metadata_json=json.dumps(batch_metadata),
            )
            header_checked = Crc32CheckOperation().run(
                _operation_context(
                    root,
                    "crc_header_check",
                    {"on_decode_failure": "gray_image"},
                    {
                        "bits": artifact(
                            "channel.payload_bits.numpy",
                            header_corrupted_path,
                            batch_metadata,
                        )
                    },
                )
            )
            self.assertEqual(
                header_checked.outputs["bits"].metadata["source_item_outage"],
                [0, 1],
            )
            self.assertEqual(
                header_checked.metrics["channel.unassigned_failed_packet_count"], 0
            )
            self.assertEqual(
                header_checked.metrics[
                    "channel.unrecoverable_header_packet_count"
                ],
                1,
            )

    def test_measured_resource_budget_rejects_over_budget_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "bits.yaml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "name": "budgeted_bits",
                        "metadata": {
                            "pairing_id": 0,
                            "aggregation_cell_id": "synthetic-cell",
                            "statistical_unit": "synthetic_item",
                        },
                        "steps": [
                            {
                                "id": "data",
                                "op": "source.random_bits",
                                "params": {"bit_count": 16, "seed": 7},
                            },
                            {
                                "id": "modulator",
                                "op": "modulation.digital_modulate",
                                "inputs": {"bits": "data.bits"},
                                "params": {"modulation": "qpsk"},
                            },
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            benchmark_path = root / "benchmark.yaml"
            benchmark_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "id": "budget_rejection_test",
                        "version": "1",
                        "dataset": {"id": "synthetic_random_bits", "modality": "bits"},
                        "task": {"id": "bit_transport", "kind": "transport_integrity", "modality": "bits"},
                        "metrics": [{"id": "channel.channel_use_count"}],
                        "metadata": {
                            "resource_budget": {
                                "metric": "steps.modulator.channel.channel_use_count",
                                "maximum": 7,
                                "tolerance": 0,
                                "policy": "reject",
                                "unit": "channel_use",
                            }
                        },
                        "recipes": [{"id": "candidate", "path": str(recipe_path)}],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            pack = load_benchmark_pack(benchmark_path)
            store = LocalStore(root / "workspace")
            result_dir = run_benchmark_pack(pack, build_registry(), store, ROOT)
            result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
            self.assertFalse(
                result["benchmark"]["traceability_profile_requested"]
            )
            self.assertNotIn("publication_ready", result["benchmark"])
            self.assertFalse(
                result["benchmark"]["metadata"][
                    "traceability_profile_requested"
                ]
            )
            self.assertNotIn(
                "publication_ready",
                result["benchmark"]["metadata"],
            )
            row = result["recipes"][0]
            attempt_snapshot = BenchmarkAttemptLedger(store.benchmarks_dir).verify()
            self.assertEqual(attempt_snapshot["status"], "valid")
            self.assertEqual(attempt_snapshot["attempt_count"], 1)
            self.assertEqual(
                attempt_snapshot["attempts"][0]["attempt_id"], result["attempt_id"]
            )
            self.assertEqual(
                attempt_snapshot["attempts"][0]["terminal_status"],
                "resource_rejected",
            )
            self.assertEqual(
                attempt_snapshot["attempts"][0]["result"]["result_id"],
                result_dir.name,
            )
            self.assertEqual(row["status"], "rejected_resource_budget")
            self.assertFalse(row["resource_admission"]["admitted"])
            self.assertEqual(row["resource_admission"]["observed"], 8.0)
            self.assertEqual(row["metrics"]["benchmark.resource_budget.admitted"], 0)
            self.assertEqual(row["pairing_id"], "0")
            self.assertEqual(row["pairing_seed"], 0)
            self.assertEqual(row["aggregation_cell_id"], "synthetic-cell")
            self.assertEqual(row["statistical_unit"], "synthetic_item")
            verification = verify_benchmark_result(store, result_dir.name)
            self.assertNotEqual(verification["status"], "invalid", verification)

            # Report bytes are now part of the result evidence, not mutable
            # conveniences that can silently diverge from result.json.
            metrics_path = result_dir / "metrics.csv"
            metrics_path.write_text(
                metrics_path.read_text(encoding="utf-8") + "forged,row\n",
                encoding="utf-8",
            )
            tampered_report = verify_benchmark_result(store, result_dir.name)
            self.assertEqual(tampered_report["status"], "invalid")
            self.assertTrue(
                any(
                    check["id"] == "benchmark.reports.metrics_csv"
                    and check["status"] == "error"
                    for check in tampered_report["checks"]
                ),
                tampered_report,
            )
            write_benchmark_reports(result_dir, result)
            store.write_json(result_dir / "result.json", result)

            identity_row = result["recipes"][0]
            original_identity = {
                key: identity_row[key] for key in ("id", "label", "role")
            }
            identity_row.update(
                {"id": "forged-id", "label": "Forged label", "role": "winner"}
            )
            write_benchmark_reports(result_dir, result)
            store.write_json(result_dir / "result.json", result)
            identity_verification = verify_benchmark_result(store, result_dir.name)
            self.assertEqual(identity_verification["status"], "invalid")
            self.assertTrue(
                any(
                    check["id"] == "benchmark.protocol.recipe_identity"
                    and check["status"] == "error"
                    for check in identity_verification["checks"]
                ),
                identity_verification,
            )
            identity_row.update(original_identity)

            original_design = {
                key: identity_row[key]
                for key in (
                    "pairing_id",
                    "pairing_seed",
                    "aggregation_cell_id",
                    "statistical_unit",
                )
            }
            identity_row.update(
                {
                    "pairing_id": "forged-pair",
                    "pairing_seed": 99,
                    "aggregation_cell_id": "forged-cell",
                    "statistical_unit": "post_hoc_unit",
                }
            )
            write_benchmark_reports(result_dir, result)
            store.write_json(result_dir / "result.json", result)
            design_verification = verify_benchmark_result(store, result_dir.name)
            self.assertEqual(design_verification["status"], "invalid")
            self.assertTrue(
                any(
                    check["id"] == "benchmark.run_evidence_snapshot"
                    and check["status"] == "error"
                    for check in design_verification["checks"]
                ),
                design_verification,
            )
            identity_row.update(original_design)

            # A forged low observed value cannot launder an over-budget run:
            # admission is checked against the step-scoped backing-run metric.
            forged = result["recipes"][0]
            forged["status"] = "completed"
            forged["resource_admission"].update(
                {
                    "admitted": True,
                    "decision": "admitted",
                    "observed": 0.0,
                    "excess": 0.0,
                }
            )
            forged["metrics"].update(
                {
                    "benchmark.resource_budget.admitted": 1,
                    "benchmark.resource_budget.observed": 0.0,
                    "benchmark.resource_budget.excess": 0.0,
                }
            )
            write_benchmark_reports(result_dir, result)
            store.write_json(result_dir / "result.json", result)
            forged_verification = verify_benchmark_result(store, result_dir.name)
            self.assertEqual(forged_verification["status"], "invalid")
            self.assertTrue(
                any(
                    check["id"] == "benchmark.resource_admission.observed"
                    and check["status"] == "error"
                    for check in forged_verification["checks"]
                ),
                forged_verification,
            )

            # Removing both the rejection and its derived metrics is also
            # rejected because the frozen benchmark protocol requires admission.
            forged.pop("resource_admission")
            for key in list(forged["metrics"]):
                if key.startswith("benchmark.resource_budget."):
                    forged["metrics"].pop(key)
            write_benchmark_reports(result_dir, result)
            store.write_json(result_dir / "result.json", result)
            missing_verification = verify_benchmark_result(store, result_dir.name)
            self.assertEqual(missing_verification["status"], "invalid")
            self.assertTrue(
                any(
                    check["id"] == "benchmark.resource_admission"
                    and check["status"] == "error"
                    for check in missing_verification["checks"]
                ),
                missing_verification,
            )

    def test_capacity_oracle_accounts_and_fails_per_source_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bits = np.array([1, 1, 1], dtype=np.uint8)
            result = CapacityOracleDigitalLinkOperation().run(
                _operation_context(
                    root,
                    "capacity_oracle",
                    {"snr_db": 0.0, "ldpc_rate": 1.0},
                    {"bits": _bits_artifact(root, "oracle_bits", bits, [1, 2])},
                )
            )
            metadata = result.outputs["bits"].metadata
            self.assertEqual(metadata["capacity_oracle_accounting_scope"], "source_item")
            self.assertEqual(metadata["source_item_channel_use_counts"], [1, 1])
            self.assertEqual(
                result.metrics["channel.max_source_item_uses_per_pixel"], 1.0 / 64.0
            )
            self.assertEqual(metadata["source_item_outage"], [0, 1])
            self.assertEqual(result.metrics["channel.source_item_success_rate"], 0.5)
            with np.load(result.outputs["bits"].path, allow_pickle=False) as payload:
                received = payload["bits"]
            np.testing.assert_array_equal(received, np.array([1, 0, 0], dtype=np.uint8))

    def test_jpeg_capacity_oracle_adapts_quality_to_fixed_channel_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis = np.arange(128, dtype=np.uint8)
            yy, xx = np.meshgrid(axis, axis, indexing="ij")
            image = np.stack([xx, yy, (xx // 2 + yy // 2)], axis=-1)[
                None, ...
            ]
            source = _images_artifact(root, "capacity_images", image)

            def run(name, snr_db):
                return JpegCapacityOracleOperation().run(
                    _operation_context(
                        root,
                        name,
                        {
                            "snr_db": snr_db,
                            "channel_uses_per_pixel": 0.5,
                        },
                        {"images": source},
                    )
                )

            low = run("jpeg_capacity_low", 0.0)
            high = run("jpeg_capacity_high", 16.0)
            low_metadata = low.outputs["images"].metadata
            high_metadata = high.outputs["images"].metadata

            self.assertEqual(high.metrics["channel.uses_per_pixel"], 0.5)
            self.assertEqual(
                high.metrics["channel.max_source_item_uses_per_pixel"], 0.5
            )
            self.assertGreater(
                high.metrics["codec.jpeg.selected_quality_mean"],
                low.metrics["codec.jpeg.selected_quality_mean"],
            )
            self.assertLessEqual(
                high_metadata["source_item_native_codec_bit_counts"][0],
                high_metadata["source_item_capacity_bits"][0],
            )
            self.assertEqual(
                high_metadata["protected_digital_baseline"],
                "jpeg_capacity_oracle",
            )
            self.assertEqual(
                high.metrics["channel.source_item_success_rate"], 1.0
            )

    def test_wireless_realizations_are_stable_by_source_item_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = np.array([1 + 1j, -1 + 1j, 1 - 1j], dtype=np.complex64)
            second = np.array([-1 - 1j, 1 + 0j], dtype=np.complex64)
            params = {
                "channel": "awgn",
                "noise_mode": "fixed_variance",
                "noise_variance": 0.25,
                "wireless_backend": "numpy",
                "data_plane_backend": "numpy",
                "seed": 77,
            }

            def apply(name, rows, counts, ids):
                return WirelessChannelOperation().run(
                    _operation_context(
                        root,
                        name,
                        params,
                        {
                            "symbols": _symbols_artifact(
                                root,
                                name + "_input",
                                np.concatenate(rows),
                                counts,
                                ids,
                            )
                        },
                    )
                )

            single = apply("wireless_single", [first], [len(first)], ["first"])
            ordered = apply(
                "wireless_ordered",
                [first, second],
                [len(first), len(second)],
                ["first", "second"],
            )
            reversed_batch = apply(
                "wireless_reversed",
                [second, first],
                [len(second), len(first)],
                ["second", "first"],
            )

            def load(result):
                with np.load(result.outputs["rx_symbols"].path, allow_pickle=False) as payload:
                    return payload["symbols"]

            single_rx = load(single)
            ordered_rx = load(ordered)
            reversed_rx = load(reversed_batch)
            np.testing.assert_array_equal(single_rx, ordered_rx[: len(first)])
            np.testing.assert_array_equal(
                ordered_rx[: len(first)], reversed_rx[len(second) :]
            )
            np.testing.assert_array_equal(
                ordered_rx[len(first) :], reversed_rx[: len(second)]
            )
            self.assertTrue(
                ordered.outputs["rx_symbols"].metadata[
                    "channel_realization_order_invariant"
                ]
            )
            self.assertEqual(
                ordered.outputs["rx_symbols"].metadata[
                    "channel_realization_identity_field"
                ],
                "source_item_ids",
            )
            self.assertEqual(
                ordered.metrics["channel.max_source_item_uses_per_pixel"], 0.75
            )
            self.assertAlmostEqual(
                ordered.metrics["channel.max_source_item_tx_power.average"],
                2.0,
                places=6,
            )

    def test_modem_padding_is_isolated_and_trimmed_per_source_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bits = np.array([1, 0, 1, 0, 1, 1, 0], dtype=np.uint8)
            source = _bits_artifact(root, "odd_item_bits", bits, [3, 4])
            modulated = DigitalModulateOperation().run(
                _operation_context(
                    root,
                    "modulate_items",
                    {"modulation": "qpsk", "data_plane_backend": "numpy"},
                    {"bits": source},
                )
            )
            metadata = modulated.outputs["symbols"].metadata
            self.assertEqual(metadata["source_item_symbol_counts"], [2, 2])
            self.assertEqual(
                metadata["source_item_modulator_input_bit_counts"], [3, 4]
            )
            self.assertEqual(metadata["source_item_padded_bit_counts"], [4, 4])
            self.assertEqual(
                modulated.metrics["channel.max_source_item_uses_per_pixel"],
                2.0 / 64.0,
            )
            demodulated = DigitalDemodulateOperation().run(
                _operation_context(
                    root,
                    "demodulate_items",
                    {"modulation": "qpsk", "data_plane_backend": "numpy"},
                    {"rx_symbols": modulated.outputs["symbols"]},
                )
            )
            with np.load(demodulated.outputs["bits"].path, allow_pickle=False) as payload:
                recovered = payload["bits"]
            np.testing.assert_array_equal(recovered, bits)

    def test_deepjscc_boundary_preserves_item_ids_and_normalizes_power_per_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = np.stack(
                [
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    np.full((4, 4, 3), 255, dtype=np.uint8),
                ],
                axis=0,
            )
            image_path = root / "deepjscc_images.npz"
            image_metadata = {
                "shape": list(images.shape),
                "original_shapes": [[1, 4, 4, 3], [1, 4, 4, 3]],
                "image_ids": ["black", "white"],
            }
            np.savez_compressed(
                image_path,
                images=images,
                metadata_json=json.dumps(image_metadata),
            )

            def fake_encoder(_params, values, _metadata):
                rows = [
                    np.ones((1, 2, 2), dtype=np.complex64),
                    np.full((1, 2, 2), 2 + 0j, dtype=np.complex64),
                ]
                return {
                    "array": np.stack(rows[: int(values.shape[0])], axis=0),
                    "metadata": {"symbol_shape": [int(values.shape[0]), 1, 2, 2]},
                }

            with patch(
                "noema_lab.ops.models.external._call_external",
                side_effect=fake_encoder,
            ):
                encoded = DeepJsccExternalEncodeOperation().run(
                    _operation_context(
                        root,
                        "deepjscc_encode",
                        {"runtime": "external_callable"},
                        {
                            "images": artifact(
                                "image.batch.numpy", image_path, image_metadata
                            )
                        },
                    )
                )
            symbol_metadata = encoded.outputs["symbols"].metadata
            self.assertEqual(symbol_metadata["source_item_ids"], ["black", "white"])
            self.assertEqual(symbol_metadata["source_item_id_source"], "image_ids")
            self.assertEqual(symbol_metadata["source_item_symbol_counts"], [4, 4])

            normalized = SymbolPowerNormalizeOperation().run(
                _operation_context(
                    root,
                    "deepjscc_power",
                    {"target_power": 1.0},
                    {"symbols": encoded.outputs["symbols"]},
                )
            )
            normalized_metadata = normalized.outputs["symbols"].metadata
            self.assertEqual(
                normalized_metadata["power_normalization_scope"], "source_item"
            )
            np.testing.assert_allclose(
                normalized_metadata["source_item_power_before"], [1.0, 4.0]
            )
            np.testing.assert_allclose(
                normalized_metadata["source_item_power_after"], [1.0, 1.0], rtol=1e-6
            )
            self.assertAlmostEqual(
                normalized.metrics["channel.max_source_item_tx_power.after"],
                1.0,
                places=6,
            )

            def fake_decoder(_params, _values, _metadata):
                return np.zeros((2, 4, 4, 3), dtype=np.uint8)

            with patch(
                "noema_lab.ops.models.external._call_external",
                side_effect=fake_decoder,
            ):
                decoded = DeepJsccExternalDecodeOperation().run(
                    _operation_context(
                        root,
                        "deepjscc_decode",
                        {"runtime": "external_callable"},
                        {"symbols": normalized.outputs["symbols"]},
                    )
                )
            self.assertEqual(
                decoded.outputs["images"].metadata["image_ids"],
                ["black", "white"],
            )
            self.assertEqual(
                decoded.outputs["images"].metadata["source_item_ids"],
                ["black", "white"],
            )

    def test_returned_artifact_lineage_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "trained_artifact.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "training": {
                            "data_partitions": {
                                "train_image_ids": ["kodim01", "kodim21"],
                                "validation_image_ids": ["kodim17"],
                                "test_images_used": False,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            pack = BenchmarkPack(
                id="lineage_test",
                version="1",
                recipes=[],
                dataset={"id": "kodak", "sample_ids": ["kodim21"]},
                metadata={"require_disjoint_training_lineage": True},
            )
            entry = BenchmarkRecipe(id="candidate", path=root / "recipe.yaml")
            recipe = SimpleNamespace(
                steps=[
                    SimpleNamespace(
                        params={
                            "runtime": "learned_artifact",
                            "artifact_manifest_path": str(manifest_path),
                        }
                    )
                ]
            )
            with self.assertRaisesRegex(BenchmarkEvidenceError, "overlaps benchmark test"):
                _validate_recipe_training_lineage(
                    pack,
                    entry,
                    recipe,
                    recipe_path=root / "recipe.yaml",
                    project_root=ROOT,
                )

    def test_publication_benchmark_rejects_runtime_ready_artifact_without_selection_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.bin"
            model_path.write_bytes(b"runtime-ready-development-model")
            manifest_path = root / "trained_artifact.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "kind": "noema.trained_block_artifact",
                        "id": "development-only-artifact",
                        "name": "Development artifact",
                        "artifact": {
                            "path": model_path.name,
                            "sha256": file_sha256(model_path),
                            "format": "opaque-test-format",
                        },
                        "compatible_operations": [
                            {
                                "operation": "test.learned_operation",
                                "required_inputs": [],
                                "params": {},
                            }
                        ],
                        "training": {
                            "data_partitions": {
                                "train_image_ids": ["training-sample"],
                                "validation_image_ids": ["validation-sample"],
                                "test_images_used": False,
                            }
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            pack = BenchmarkPack(
                id="publication_artifact_gate",
                version="1",
                recipes=[],
                dataset={"id": "heldout", "sample_ids": ["publication-sample"]},
                metadata={
                    "publication_ready": True,
                    "require_disjoint_training_lineage": True,
                },
            )
            entry = BenchmarkRecipe(id="candidate", path=root / "recipe.yaml")
            recipe = SimpleNamespace(
                steps=[
                    SimpleNamespace(
                        params={
                            "runtime": "learned_artifact",
                            "artifact_manifest_path": str(manifest_path),
                        }
                    )
                ]
            )

            with self.assertRaisesRegex(
                BenchmarkEvidenceError,
                "publication readiness requires a portable schema_version=2",
            ):
                _validate_recipe_training_lineage(
                    pack,
                    entry,
                    recipe,
                    recipe_path=root / "recipe.yaml",
                    project_root=root,
                )

    def test_returned_artifact_content_alias_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            heldout_sha = "a" * 64
            data_contract_path = root / "training_data.yaml"
            data_contract_path.write_text(
                yaml.safe_dump(
                    {
                        "splits": [
                            {
                                "id": "train",
                                "files": [
                                    {
                                        "sample_id": "renamed_training_sample",
                                        "sha256": heldout_sha,
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = root / "trained_artifact.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "source": {
                            "data_contract": {
                                "path": data_contract_path.name,
                                "file_sha256": file_sha256(data_contract_path),
                            }
                        },
                        "training": {
                            "data_partitions": {
                                "train_image_ids": ["renamed_training_sample"],
                                "validation_image_ids": [],
                                "test_images_used": False,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            pack = BenchmarkPack(
                id="lineage_content_test",
                version="1",
                recipes=[],
                dataset={
                    "id": "kodak",
                    "sample_ids": ["kodim21"],
                    "manifest": {
                        "files": [
                            {"sample_id": "kodim21", "sha256": heldout_sha}
                        ]
                    },
                },
                metadata={"require_disjoint_training_lineage": True},
            )
            with self.assertRaisesRegex(
                BenchmarkEvidenceError, "overlaps benchmark test content"
            ):
                validate_trained_artifact_lineage_manifest(pack, manifest_path)

    def test_training_evidence_snapshot_copies_hashed_data_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_contract_path = root / "training_data.yaml"
            data_contract_path.write_text(
                yaml.safe_dump(
                    {
                        "splits": [
                            {
                                "id": "train",
                                "files": [
                                    {"sample_id": "train-1", "sha256": "b" * 64}
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            artifact_manifest_path = root / "artifact.yaml"
            artifact_manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 2,
                        "kind": "noema.trained_block_artifact",
                        "components": [],
                        "source": {
                            "data_contract": {
                                "path": data_contract_path.name,
                                "file_sha256": file_sha256(data_contract_path),
                            }
                        },
                        "training": {
                            "data_partitions": {
                                "train_image_ids": ["train-1"],
                                "validation_image_ids": [],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            training_evidence = [
                {
                    "series": "candidate",
                    "trained_artifact_manifest": {
                        "path": artifact_manifest_path.name,
                        "sha256": file_sha256(artifact_manifest_path),
                    },
                }
            ]
            benchmark_source = {
                "id": "snapshot_data_contract",
                "metadata": {"demo": {"training_evidence": training_evidence}},
            }
            result = {
                "benchmark": {
                    "id": "snapshot_data_contract",
                    "metadata": {
                        "demo": {"training_evidence": training_evidence}
                    },
                }
            }
            result_dir = root / "result"
            snapshot_benchmark_training_evidence(
                result_dir, result, benchmark_source, root
            )
            validated = validate_benchmark_training_evidence_snapshot(
                result_dir, result, benchmark_source=benchmark_source
            )
            self.assertTrue(validated["present"])
            references = validated["projection"]["entries"][0][
                "artifact_references"
            ]
            data_references = [
                reference
                for reference in references
                if reference["kind"] == "data_contract"
            ]
            self.assertEqual(len(data_references), 1)
            copied = result_dir / data_references[0]["path"]
            self.assertEqual(file_sha256(copied), file_sha256(data_contract_path))

    def test_external_checkpoint_adapter_without_portable_lineage_is_rejected(self):
        pack = BenchmarkPack(
            id="external_lineage_test",
            version="1",
            recipes=[],
            dataset={"id": "kodak", "sample_ids": ["kodim21"]},
            metadata={"require_disjoint_training_lineage": True},
        )
        entry = BenchmarkRecipe(id="external_candidate", path=ROOT / "recipe.yaml")
        recipe = SimpleNamespace(
            steps=[SimpleNamespace(op="external.checkpoint", params={})]
        )
        operation = SimpleNamespace(
            describe=lambda: {
                "external_adapter": {
                    "training": {"source": "opaque_remote_checkpoint"}
                }
            }
        )
        registry = SimpleNamespace(get=lambda _operation_id: operation)
        with self.assertRaisesRegex(
            BenchmarkEvidenceError, "without a portable trained artifact manifest"
        ):
            _validate_recipe_training_lineage(
                pack,
                entry,
                recipe,
                recipe_path=ROOT / "recipe.yaml",
                project_root=ROOT,
                registry=registry,
            )

    def test_required_lineage_cannot_drop_training_evidence_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = _CheckRecorder()
            benchmark = {
                "id": "lineage_required",
                "version": "1",
                "metadata": {"require_disjoint_training_lineage": True},
            }
            result = {
                "benchmark": {
                    "id": "lineage_required",
                    "version": "1",
                    "metadata": {"require_disjoint_training_lineage": True},
                }
            }
            _check_benchmark_training_evidence_snapshot(
                Path(tmp), result, benchmark, recorder
            )
            self.assertTrue(
                any(
                    check.id == "benchmark.training_evidence_snapshot"
                    and check.status == "error"
                    for check in recorder.checks
                ),
                [check.to_dict() for check in recorder.checks],
            )

    def test_kodak_cache_rejects_wrong_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kodim01.png"
            path.write_bytes(b"x" * KODAK_SIZE_BYTES["kodim01.png"])
            with self.assertRaisesRegex(RuntimeError, "sha256"):
                _verify_kodak_file(path, "kodim01.png")


if __name__ == "__main__":
    unittest.main()
