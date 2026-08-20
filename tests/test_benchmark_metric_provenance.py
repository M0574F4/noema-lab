from __future__ import annotations

import unittest
from pathlib import Path

from noema_lab.core.benchmarks import (
    BenchmarkError,
    BenchmarkPack,
    _collect_summary_metrics_with_provenance,
    _metric_definitions_for_role,
    _validate_benchmark_metadata,
    load_benchmark_pack,
    validate_benchmark_pack,
)
from noema_lab.ops import build_registry
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.publication_profile import (
    publication_verification_profile_binding,
)
from noema_lab.core.verification import (
    _CheckRecorder,
    _check_benchmark_required_metrics,
)


class BenchmarkMetricProvenanceTests(unittest.TestCase):
    def test_shipped_csi_and_aoa_packs_bind_previously_ambiguous_metrics(self):
        root = Path(__file__).resolve().parents[1]
        csi = load_benchmark_pack(
            root / "benchmarks" / "channel_estimation" / "csi_feedback_v1.yaml"
        )
        feedback = next(
            row
            for row in csi.metrics
            if row["id"] == "csi_feedback.feedback_bits_per_sample"
        )
        self.assertEqual(feedback["source_step"], "feedback_link")
        self.assertEqual(feedback["applicable_roles"], ["baseline"])
        aoa = load_benchmark_pack(
            root / "benchmarks" / "localization_sensing" / "aoa_estimation_v1.yaml"
        )
        snr = next(row for row in aoa.metrics if row["id"] == "channel.snr_db")
        self.assertEqual(snr["source_step"], "array_observation")
        for pack in (csi, aoa):
            report = validate_benchmark_pack(pack, build_registry(), root)
            self.assertEqual(report["recipe_count"], len(pack.recipes), report)

    def _summary(self):
        return {
            "metrics": {},
            "steps": [
                {
                    "id": "candidate",
                    "op": "models.untrusted_candidate",
                    "metrics": {"quality.psnr_db": 1_000_000.0},
                    "outputs": {},
                    "execution_binding": {
                        "implementation": "candidate",
                        "implementation_metadata": {"source_sha256": "a" * 64},
                    },
                },
                {
                    "id": "evaluation",
                    "op": "metrics.image_reconstruction",
                    "metrics": {"quality.psnr_db": 31.25},
                    "outputs": {"report": {"kind": "metrics.report"}},
                    "execution_binding": {
                        "implementation": "evaluator",
                        "implementation_metadata": {"source_sha256": "b" * 64},
                    },
                },
            ],
        }

    def _validated_evidence(self, summary, report_sha256="c" * 64):
        return {
            "entries": [
                {
                    "entry_index": 0,
                    "entry_id": "candidate",
                    "run_id": "run-1",
                    "descriptor": {"files_sha256": report_sha256},
                    "summary": summary,
                }
            ]
        }

    def test_ambiguous_required_metric_fails_closed(self):
        with self.assertRaisesRegex(BenchmarkError, "has 2 producers"):
            _collect_summary_metrics_with_provenance(
                self._summary(),
                [{"id": "quality.psnr_db", "definition_version": 1}],
            )

    def test_declared_evaluator_step_controls_unqualified_metric(self):
        definition = {
            "id": "quality.psnr_db",
            "definition_version": 1,
            "source_step": "evaluation",
            "source_operation": "metrics.image_reconstruction",
        }
        metrics, provenance = _collect_summary_metrics_with_provenance(
            self._summary(),
            [definition],
            source_run_evidence_sha256="c" * 64,
        )
        self.assertEqual(metrics["quality.psnr_db"], 31.25)
        self.assertEqual(
            metrics["steps.candidate.quality.psnr_db"],
            1_000_000.0,
        )
        self.assertEqual(provenance["quality.psnr_db"]["source_step"], "evaluation")
        self.assertEqual(
            provenance["quality.psnr_db"]["source_operation"],
            "metrics.image_reconstruction",
        )
        self.assertEqual(provenance["quality.psnr_db"]["definition_version"], 1)
        self.assertEqual(
            provenance["quality.psnr_db"]["source_implementation_sha256"],
            "b" * 64,
        )
        self.assertEqual(
            provenance["quality.psnr_db"]["source_run_evidence_sha256"],
            "c" * 64,
        )

    def test_role_scoped_metric_is_not_required_for_inapplicable_method(self):
        definition = {
            "id": "rate.native_codec_bpp",
            "definition_version": 1,
            "source_step": "sender",
            "source_operation": "model.jpeg_encode",
            "applicable_roles": ["digital_baseline"],
        }
        self.assertEqual(
            _metric_definitions_for_role([definition], "learned_baseline"), []
        )
        self.assertEqual(
            _metric_definitions_for_role([definition], "digital_baseline"),
            [definition],
        )
        result = {
            "benchmark": {"metrics": [definition]},
            "recipes": [
                {
                    "id": "learned",
                    "role": "learned_baseline",
                    "status": "completed",
                    "metrics": {},
                    "metric_provenance": {},
                }
            ],
        }
        recorder = _CheckRecorder()
        _check_benchmark_required_metrics(
            result,
            recorder,
            validated_run_evidence=self._validated_evidence(self._summary()),
        )
        errors = [check for check in recorder.checks if check.status == "error"]
        self.assertEqual(errors, [])

        result["recipes"][0]["role"] = "digital_baseline"
        recorder = _CheckRecorder()
        _check_benchmark_required_metrics(result, recorder)
        self.assertTrue(
            any(
                check.status == "error"
                and "missing required metrics" in check.message
                for check in recorder.checks
            )
        )

    def test_verifier_rejects_forged_unqualified_value(self):
        definition = {
            "id": "quality.psnr_db",
            "definition_version": 1,
            "source_step": "evaluation",
            "source_operation": "metrics.image_reconstruction",
        }
        summary = self._summary()
        metrics, provenance = _collect_summary_metrics_with_provenance(
            summary,
            [definition],
            source_run_evidence_sha256="c" * 64,
        )
        metrics["quality.psnr_db"] = 99.0
        result = {
            "benchmark": {"metrics": [definition]},
            "recipes": [
                {
                    "id": "candidate",
                    "status": "completed",
                    "metrics": metrics,
                    "metric_provenance": provenance,
                }
            ],
        }
        recorder = _CheckRecorder()
        _check_benchmark_required_metrics(
            result,
            recorder,
            validated_run_evidence=self._validated_evidence(summary),
        )
        errors = [check.message for check in recorder.checks if check.status == "error"]
        self.assertTrue(any("differs from its authoritative step value" in error for error in errors))

    def test_verifier_rejects_forged_producer_identity_fields(self):
        definition = {
            "id": "quality.psnr_db",
            "definition_version": 1,
            "source_step": "evaluation",
            "source_operation": "metrics.image_reconstruction",
        }
        summary = self._summary()
        metrics, provenance = _collect_summary_metrics_with_provenance(
            summary,
            [definition],
            source_run_evidence_sha256="c" * 64,
        )
        for field in (
            "source_implementation_sha256",
            "source_outputs_sha256",
            "source_execution_binding_sha256",
            "source_run_evidence_sha256",
        ):
            with self.subTest(field=field):
                forged = {
                    metric_id: dict(entry)
                    for metric_id, entry in provenance.items()
                }
                forged["quality.psnr_db"][field] = "f" * 64
                result = {
                    "benchmark": {"metrics": [definition]},
                    "recipes": [
                        {
                            "id": "candidate",
                            "status": "completed",
                            "metrics": dict(metrics),
                            "metric_provenance": forged,
                        }
                    ],
                }
                recorder = _CheckRecorder()
                _check_benchmark_required_metrics(
                    result,
                    recorder,
                    validated_run_evidence=self._validated_evidence(summary),
                )
                errors = [
                    check.message
                    for check in recorder.checks
                    if check.status == "error"
                ]
                self.assertTrue(any(field in error for error in errors), errors)

    def test_verifier_rejects_unversioned_definition_and_missing_run_evidence(self):
        versioned = {
            "id": "quality.psnr_db",
            "definition_version": 1,
            "source_step": "evaluation",
            "source_operation": "metrics.image_reconstruction",
        }
        summary = self._summary()
        metrics, provenance = _collect_summary_metrics_with_provenance(
            summary,
            [versioned],
            source_run_evidence_sha256="c" * 64,
        )
        unversioned = dict(versioned)
        unversioned.pop("definition_version")
        result = {
            "benchmark": {"metrics": [unversioned]},
            "recipes": [
                {
                    "id": "candidate",
                    "status": "completed",
                    "metrics": metrics,
                    "metric_provenance": provenance,
                }
            ],
        }
        recorder = _CheckRecorder()
        _check_benchmark_required_metrics(result, recorder)
        errors = [check.message for check in recorder.checks if check.status == "error"]
        self.assertTrue(any("unversioned protocol definition" in error for error in errors))
        self.assertTrue(any("no validated result-local run evidence" in error for error in errors))

    def test_publication_ready_pack_requires_explicit_common_conditions(self):
        pack = BenchmarkPack(
            id="publication.protocol",
            version="1",
            recipes=[],
            dataset={
                "id": "sealed",
                "sample_ids": ["sealed-1"],
                "selection_role": "publication_test",
                "preprocessing": {
                    "spatial_policy": "fixed",
                    "operation_params": {
                        "resize_shorter_side": 256,
                        "crop_size": 256,
                        "repeat_count": 1,
                    },
                },
                "source_bindings": [
                    {
                        "operation": "source.image_dataset",
                        "selection_param": "image_ids",
                        "params": {
                            "manifest_path": "sealed.yaml",
                            "manifest_sha256": "d" * 64,
                            "split": "publication_test",
                        },
                    }
                ],
                "manifest": {
                    "files": [
                        {
                            "sample_id": "sealed-1",
                            "sha256": "b" * 64,
                            "source_id": "source-1",
                            "group_id": "group-1",
                            "source_sha256": "c" * 64,
                            "ancestry_ids": ["source-1"],
                            "transform": {"name": "identity"},
                            "transform_fingerprint_sha256": canonical_json_sha256(
                                {"name": "identity"}
                            ),
                        }
                    ]
                },
                "access_policy": {
                    "state": "sealed_single_access",
                    "publication_test": True,
                    "access_ledger": "ledger",
                    "access_budget": 1,
                    "seal_sha256": "a" * 64,
                },
            },
            metrics=[
                {
                    "id": "quality.psnr_db",
                    "definition_version": 1,
                    "source_step": "evaluation",
                    "source_operation": "metrics.image_reconstruction",
                }
            ],
            metadata={
                "benchmark_tier": "canonical",
                "publication_ready": True,
                "verification_profile": publication_verification_profile_binding(),
                "protocol_id": "publication.protocol",
                "protocol_version": "1",
                "dataset_split": {"id": "sealed-test"},
                "channel": {"model": "awgn"},
                "rate_accounting": {"boundary": "channel_input"},
                "expected_outputs": ["quality.psnr_db"],
                "frozen": True,
                "require_identical_source_transform": True,
                "require_disjoint_training_lineage": True,
                "resource_budget": {
                    "metric": "steps.channel.channel.uses_per_pixel",
                    "maximum": 1.0,
                    "tolerance": 0.0,
                    "policy": "reject",
                },
            },
        )
        with self.assertRaisesRegex(BenchmarkError, "metadata.common_conditions"):
            _validate_benchmark_metadata(pack)

        pack.metadata["common_conditions"] = {
            "power": {
                "coordinate": "average symbol power",
                "normalization_scope": "source_item",
                "target": 1.0,
            },
            "randomness": {
                "pairing_keys": ["source_item_id", "snr_db", "replicate"],
                "seed_derivation": "sha256(protocol, pairing keys)",
                "paired_operation_ids": ["wireless.channel"],
            },
            "receiver": {
                "channel_state_information": "none",
                "receiver_processing": "identity",
            },
            "failure": {
                "outage_definition": "crc failure",
                "decode_failure_policy": "fixed fallback",
                "denominator_policy": "all attempted source items",
            },
        }
        _validate_benchmark_metadata(pack)


if __name__ == "__main__":
    unittest.main()
