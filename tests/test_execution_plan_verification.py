from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from noema_lab.core.executor import LocalExecutor
from noema_lab.core.artifacts import file_sha256
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationRegistry,
    OperationResult,
    object_schema,
)
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.storage import LocalStore
from noema_lab.core.verification import (
    _CheckRecorder,
    _benchmark_flat_metrics,
    _check_metric_plausibility,
    _check_runtime_backend_namespaces,
    verify_run_bundle,
)


class _VerificationOperation(Operation):
    id = "test.execution_plan_verification"
    name = "Execution-plan verification fixture"
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": [],
        "differentiable_export": [],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "verification_fixture",
            "status": "implemented",
            "parameter_bindings": {"mode": "fixture"},
        }
    ]
    params_schema = object_schema(
        {
            "mode": {
                "type": "string",
                "default": "fixture",
                "enum": ["fixture", "tampered"],
            }
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult(metrics={"fixture.count": 1})


class ExecutionPlanVerificationTests(unittest.TestCase):
    def test_wireless_runtime_backend_namespace_must_match_plan(self):
        plan_steps = {
            "channel": {
                "operation_id": "wireless.channel",
                "backend": "numpy",
            }
        }
        contradictory = [
            {
                "id": "channel",
                "metadata": {
                    "wireless_backend": "cpp_native",
                    "data_plane_backend": "cpp_native",
                },
                "outputs": {
                    "rx_symbols": {
                        "metadata": {
                            "wireless_backend": "cpp_native",
                            "data_plane_backend": "cpp_native",
                        }
                    }
                },
            }
        ]
        recorder = _CheckRecorder()
        _check_runtime_backend_namespaces(
            contradictory,
            plan_steps,
            recorder,
            require_all=True,
        )
        bad_report = recorder.report(
            target_type="run",
            target_id="backend-fixture",
            path=Path("."),
        )

        consistent = json.loads(json.dumps(contradictory))
        consistent[0]["metadata"]["wireless_backend"] = "numpy"
        consistent[0]["outputs"]["rx_symbols"]["metadata"][
            "wireless_backend"
        ] = "numpy"
        recorder = _CheckRecorder()
        _check_runtime_backend_namespaces(
            consistent,
            plan_steps,
            recorder,
            require_all=True,
        )
        good_report = recorder.report(
            target_type="run",
            target_id="backend-fixture",
            path=Path("."),
        )

        self.assertEqual(bad_report["status"], "invalid", bad_report)
        self.assertEqual(good_report["status"], "valid", good_report)

    def test_all_wireless_auto_surfaces_verify_the_planned_namespace(self):
        plan_steps = {
            "link": {
                "operation_id": "wireless.digital_link",
                "backend": "numpy",
            },
            "observation": {
                "operation_id": "wireless.pilot_observation",
                "backend": "numpy",
            },
            "legacy_source": {
                "operation_id": "source.ai_phy_pilot_channel",
                "backend": "numpy",
            },
        }
        runtime = [
            {
                "id": "link",
                "metadata": {
                    "wireless_backend": "numpy",
                    "data_plane_backend": "python_numpy",
                },
            },
            {
                "id": "observation",
                "metadata": {"wireless_backend": "numpy"},
            },
            {
                "id": "legacy_source",
                "metadata": {"wireless_backend": "numpy"},
            },
        ]
        recorder = _CheckRecorder()
        _check_runtime_backend_namespaces(
            runtime,
            plan_steps,
            recorder,
            require_all=True,
        )
        valid = recorder.report(
            target_type="run",
            target_id="all-wireless-auto",
            path=Path("."),
        )

        runtime[1]["metadata"]["wireless_backend"] = "python_numpy"
        recorder = _CheckRecorder()
        _check_runtime_backend_namespaces(
            runtime,
            plan_steps,
            recorder,
            require_all=True,
        )
        invalid = recorder.report(
            target_type="run",
            target_id="all-wireless-auto",
            path=Path("."),
        )

        self.assertEqual(valid["status"], "valid", valid)
        self.assertEqual(invalid["status"], "invalid", invalid)

    def test_summary_metric_mutation_breaks_manifest_content_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            summary_path = store.runs_dir / run_id / "summary.json"
            summary = self._read_json(summary_path)
            summary["steps"][0]["metrics"]["fixture.count"] = 0.99
            self._write_json(summary_path, summary)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid", report)
        self.assertTrue(
            any(
                "summary.json does not match its manifest content binding"
                in message
                for message in report["errors"]
            ),
            report,
        )

    def test_rebound_summary_rejects_string_metric_values(self):
        for invalid_value in ("not-a-number", "0.5"):
            with self.subTest(value=invalid_value), tempfile.TemporaryDirectory() as tmp:
                store, run_id = self._write_run(Path(tmp))
                pristine = verify_run_bundle(store, run_id)
                self.assertEqual(pristine["errors"], [], pristine)

                run_dir = store.runs_dir / run_id
                summary_path = run_dir / "summary.json"
                summary = self._read_json(summary_path)
                summary["metrics"]["quality.ber"] = invalid_value
                self._write_json(summary_path, summary)
                self._rebind_summary(run_dir)

                report = verify_run_bundle(store, run_id)

            self.assertEqual(report["status"], "invalid", report)
            self.assertTrue(
                any(
                    check["id"] == "metrics.type"
                    and check["status"] == "error"
                    and "quality.ber" in check["message"]
                    for check in report["checks"]
                ),
                report,
            )
            self.assertFalse(
                any(
                    "summary.json does not match its manifest content binding"
                    in message
                    for message in report["errors"]
                ),
                report,
            )

    def test_benchmark_flat_metrics_reject_all_non_number_json_types(self):
        invalid_values = (
            "0.5",
            "not-a-number",
            True,
            None,
            [],
            {},
            float("nan"),
            float("inf"),
        )
        for invalid_value in invalid_values:
            with self.subTest(value=repr(invalid_value)):
                recorder = _CheckRecorder()
                result = {
                    "recipes": [
                        {
                            "id": "candidate",
                            "metrics": {"quality.ber": invalid_value},
                        }
                    ]
                }
                _check_metric_plausibility(
                    _benchmark_flat_metrics(result),
                    recorder,
                    prefix="benchmark.metrics",
                )
                report = recorder.report(
                    target_type="benchmark_result",
                    target_id="metric-types",
                    path=Path("."),
                )

                self.assertEqual(report["status"], "invalid", report)
                self.assertTrue(
                    any(
                        check["id"]
                        in {"benchmark.metrics.type", "benchmark.metrics.finite"}
                        and check["status"] == "error"
                        for check in report["checks"]
                    ),
                    report,
                )

    def test_benchmark_metric_plausibility_ignores_recipe_and_step_names(self):
        recorder = _CheckRecorder()
        _check_metric_plausibility(
            {
                (
                    "recipes.calibrated_iq_oracle_snrm2_seed71001."
                    "channel.snr_db"
                ): -2.0,
                (
                    "recipes.calibrated_iq_oracle_snrm2_seed71001."
                    "steps.wireless_channel.channel.effective_snr_db"
                ): -2.1,
                (
                    "recipes.calibrated_iq_oracle_snrm2_seed71001."
                    "channel.coded.error_count"
                ): 12,
            },
            recorder,
            prefix="benchmark.metrics",
        )
        report = recorder.report(
            target_type="benchmark_result",
            target_id="qualified-metric-plausibility",
            path=Path("."),
        )
        self.assertEqual(report["status"], "valid", report)

    def test_executor_plan_evidence_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["errors"], [], report)
        checks = {check["id"]: check["status"] for check in report["checks"]}
        self.assertEqual(checks["execution_plan.sha256"], "pass")
        self.assertEqual(checks["execution_plan.operation_contracts_link"], "pass")
        self.assertEqual(checks["execution_plan.summary_binding_link"], "pass")
        self.assertEqual(checks["execution_plan.cache_plan_sha256"], "pass")
        self.assertEqual(checks["execution_plan.cache_summary_link"], "pass")
        self.assertEqual(checks["execution.summary_manifest_link"], "pass")
        self.assertEqual(checks["execution.mode"], "pass")

    def test_cache_bypass_evidence_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp), use_plan_cache=False)
            run_dir = store.runs_dir / run_id
            manifest = self._read_json(run_dir / "manifest.json")

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["errors"], [], report)
        cache = manifest["execution_plan"]["cache"]
        self.assertEqual(cache["outcome"], "bypass")
        self.assertFalse(cache["enabled"])

    def test_cache_and_execution_evidence_tampering_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            run_dir = store.runs_dir / run_id
            manifest_path = run_dir / "manifest.json"
            manifest = self._read_json(manifest_path)
            manifest["execution_plan"]["cache"]["plan_sha256"] = "0" * 64
            manifest["execution"]["mode"] = "parallel"
            self._write_json(manifest_path, manifest)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid", report)
        checks = {check["id"]: check["status"] for check in report["checks"]}
        self.assertEqual(checks["execution_plan.cache_plan_sha256"], "error")
        self.assertEqual(checks["execution_plan.cache_summary_link"], "error")
        self.assertEqual(checks["execution.summary_manifest_link"], "error")
        self.assertEqual(checks["execution.mode"], "error")

    def test_missing_execution_plan_file_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            (store.runs_dir / run_id / "execution-plan.json").unlink()

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any("execution-plan.json is missing" in message for message in report["errors"]),
            report,
        )

    def test_plan_payload_tampering_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            plan_path = store.runs_dir / run_id / "execution-plan.json"
            plan = self._read_json(plan_path)
            plan["steps"][0]["backend"] = "tampered"
            self._write_json(plan_path, plan)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any("execution-plan SHA-256" in message for message in report["errors"]),
            report,
        )
        self.assertTrue(
            any("binding SHA" in message for message in report["errors"]),
            report,
        )

    def test_summary_plan_digest_and_binding_tampering_are_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            summary_path = store.runs_dir / run_id / "summary.json"
            summary = self._read_json(summary_path)
            summary["execution_plan"]["sha256"] = "0" * 64
            summary["steps"][0]["execution_binding"]["implementation"] = "tampered"
            self._write_json(summary_path, summary)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any("summary execution-plan SHA does not match" in message for message in report["errors"]),
            report,
        )
        self.assertTrue(
            any("summary step work binding differs" in message for message in report["errors"]),
            report,
        )

    def test_rehashed_false_materialization_identity_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            run_dir = store.runs_dir / run_id
            plan = self._read_json(run_dir / "execution-plan.json")
            plan["steps"][0]["materialization_id"] = "totally-false"
            self._rewrite_plan_evidence(run_dir, plan)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any(
                "materialization_id does not match its binding tuple" in message
                for message in report["errors"]
            ),
            report,
        )
        self.assertFalse(
            any("execution-plan SHA-256 does not match" in message for message in report["errors"]),
            report,
        )

    def test_rehashed_operation_contract_id_key_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            run_dir = store.runs_dir / run_id
            plan = self._read_json(run_dir / "execution-plan.json")
            plan["operation_contracts"]["operations"][
                _VerificationOperation.id
            ]["id"] = "test.fabricated_contract_id"
            self._rewrite_plan_evidence(run_dir, plan)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any(
                "embedded operation contract id differs from its operation_contracts key"
                in message
                for message in report["errors"]
            ),
            report,
        )

    def test_rehashed_false_materialization_parameter_bindings_are_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            run_dir = store.runs_dir / run_id
            plan = self._read_json(run_dir / "execution-plan.json")
            metadata = plan["steps"][0]["implementation_metadata"]
            metadata["selection"]["parameter_bindings"] = {"mode": "tampered"}
            metadata["parameter_overrides"] = {"mode": "tampered"}
            self._rewrite_plan_evidence(run_dir, plan)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any(
                "planned parameter bindings do not match" in message
                for message in report["errors"]
            ),
            report,
        )
        self.assertFalse(
            any("binding SHA does not match" in message for message in report["errors"]),
            report,
        )

    def test_rehashed_parameter_override_that_cannot_activate_binding_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            run_dir = store.runs_dir / run_id
            plan = self._read_json(run_dir / "execution-plan.json")
            metadata = plan["steps"][0]["implementation_metadata"]
            metadata["parameter_overrides"] = {"mode": "tampered"}
            self._rewrite_plan_evidence(run_dir, plan)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any(
                "parameter overrides do not activate" in message
                for message in report["errors"]
            ),
            report,
        )
        self.assertFalse(
            any("binding SHA does not match" in message for message in report["errors"]),
            report,
        )

    def test_rehashed_boolean_candidate_index_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            run_dir = store.runs_dir / run_id
            plan = self._read_json(run_dir / "execution-plan.json")
            plan["steps"][0]["implementation_metadata"]["selection"][
                "candidate_index"
            ] = False
            self._rewrite_plan_evidence(run_dir, plan)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any(
                "planned materialization metadata is inconsistent" in message
                for message in report["errors"]
            ),
            report,
        )

    def test_rehashed_unsupported_runner_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            run_dir = store.runs_dir / run_id
            plan = self._read_json(run_dir / "execution-plan.json")
            plan["runner"] = "fabricated_runner"
            step = plan["steps"][0]
            step["runner"] = "fabricated_runner"
            step["materialization_id"] = (
                _VerificationOperation.id
                + "@fabricated_runner/numpy/verification_fixture"
            )
            plan["operation_contracts"]["operations"][
                _VerificationOperation.id
            ]["materializations"][0]["runner"] = "fabricated_runner"
            self._rewrite_plan_evidence(run_dir, plan)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any("execution-plan runner is missing, unsupported" in message for message in report["errors"]),
            report,
        )

    def test_rehashed_falsey_invalid_declared_bindings_are_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            run_dir = store.runs_dir / run_id
            plan = self._read_json(run_dir / "execution-plan.json")
            plan["operation_contracts"]["operations"][
                _VerificationOperation.id
            ]["materializations"][0]["parameter_bindings"] = []
            self._rewrite_plan_evidence(run_dir, plan)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any(
                "embedded materialization parameter_bindings must be an object"
                in message
                for message in report["errors"]
            ),
            report,
        )

    def test_rehashed_fabricated_materialization_is_invalid_with_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            run_dir = store.runs_dir / run_id
            plan = self._read_json(run_dir / "execution-plan.json")
            plan["operation_contracts"]["operations"][
                _VerificationOperation.id
            ]["materializations"][0]["implementation"] = "fabricated"
            step = plan["steps"][0]
            step["implementation"] = "fabricated"
            step["materialization_id"] = (
                _VerificationOperation.id
                + "@benchmark_run/numpy/fabricated"
            )
            self._rewrite_plan_evidence(run_dir, plan)

            internal_report = verify_run_bundle(store, run_id)
            registry = OperationRegistry()
            registry.register(_VerificationOperation())
            registry_report = verify_run_bundle(store, run_id, registry=registry)

        self.assertEqual(internal_report["errors"], [], internal_report)
        self.assertEqual(registry_report["status"], "invalid")
        self.assertTrue(
            any(
                "embedded operation contract differs from the supplied registry"
                in message
                for message in registry_report["errors"]
            ),
            registry_report,
        )

    def test_rehashed_nonimplemented_materialization_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            run_dir = store.runs_dir / run_id
            plan = self._read_json(run_dir / "execution-plan.json")
            operation = plan["operation_contracts"]["operations"][
                _VerificationOperation.id
            ]
            operation["materializations"][0]["status"] = "planned"
            plan["steps"][0]["implementation_metadata"]["status"] = "planned"
            self._rewrite_plan_evidence(run_dir, plan)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any(
                "materialization is not declared implemented" in message
                for message in report["errors"]
            ),
            report,
        )
        self.assertFalse(
            any("operation-contract SHA does not match" in message for message in report["errors"]),
            report,
        )

    def test_operation_contract_and_effective_recipe_links_detect_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            run_dir = store.runs_dir / run_id
            manifest_path = run_dir / "manifest.json"
            manifest = self._read_json(manifest_path)
            operation_id = _VerificationOperation.id
            manifest["operation_contracts"]["operations"][operation_id]["name"] = "tampered"
            manifest["recipe"]["effective_sha256"] = "f" * 64
            self._write_json(manifest_path, manifest)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any("operation-contract SHA" in message for message in report["errors"]),
            report,
        )
        self.assertTrue(
            any("manifest effective recipe SHA does not match" in message for message in report["errors"]),
            report,
        )

    def test_authored_recipe_link_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            authored_path = store.runs_dir / run_id / "recipe.authored.json"
            authored = self._read_json(authored_path)
            authored["description"] = "tampered after execution"
            self._write_json(authored_path, authored)

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any("authored recipe SHA does not match" in message for message in report["errors"]),
            report,
        )

    def test_bundle_without_plan_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_id = self._write_run(Path(tmp))
            run_dir = store.runs_dir / run_id
            manifest_path = run_dir / "manifest.json"
            manifest = self._read_json(manifest_path)
            manifest.pop("execution_plan")
            self._write_json(manifest_path, manifest)
            summary_path = run_dir / "summary.json"
            summary = self._read_json(summary_path)
            summary.pop("execution_plan")
            self._write_json(summary_path, summary)
            (run_dir / "execution-plan.json").unlink()

            report = verify_run_bundle(store, run_id)

        self.assertEqual(report["status"], "invalid", report)
        self.assertTrue(
            any("execution-plan SHA-256 is missing" in message for message in report["errors"]),
            report,
        )
        checks = {check["id"]: check["status"] for check in report["checks"]}
        self.assertEqual(checks["execution_plan.required"], "error")

    def _write_run(self, root: Path, **run_options):
        registry = OperationRegistry()
        registry.register(_VerificationOperation())
        store = LocalStore(root / ".noema")
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "execution_plan_verification",
                "steps": [
                    {
                        "id": "work",
                        "op": _VerificationOperation.id,
                        "inputs": {},
                        "params": {},
                    }
                ],
            }
        )
        run_dir = LocalExecutor(registry, store).run(recipe, **run_options)
        return store, run_dir.name

    @staticmethod
    def _read_json(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, payload) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def _rebind_summary(cls, run_dir: Path) -> None:
        summary_path = run_dir / "summary.json"
        manifest_path = run_dir / "manifest.json"
        manifest = cls._read_json(manifest_path)
        manifest["summary"] = {
            "kind": "noema.run_summary",
            "relative_path": "summary.json",
            "sha256": file_sha256(summary_path),
            "size_bytes": summary_path.stat().st_size,
        }
        cls._write_json(manifest_path, manifest)

    @classmethod
    def _rewrite_plan_evidence(cls, run_dir: Path, plan) -> None:
        operations = plan["operation_contracts"]["operations"]
        plan["operation_contracts"]["sha256"] = canonical_json_sha256(operations)
        for step in plan["steps"]:
            step["operation_contract_sha256"] = canonical_json_sha256(
                operations[step["operation_id"]]
            )
            binding_payload = dict(step)
            binding_payload.pop("binding_sha256", None)
            step["binding_sha256"] = canonical_json_sha256(binding_payload)
        plan_payload = dict(plan)
        plan_payload.pop("sha256", None)
        plan["sha256"] = canonical_json_sha256(plan_payload)
        cls._write_json(run_dir / "execution-plan.json", plan)

        manifest_path = run_dir / "manifest.json"
        manifest = cls._read_json(manifest_path)
        manifest["execution_plan"]["sha256"] = plan["sha256"]
        manifest["execution_plan"]["runner"] = plan["runner"]
        manifest["execution_plan"]["steps"] = json.loads(
            json.dumps(plan["steps"])
        )
        if isinstance(manifest["execution_plan"].get("cache"), dict):
            manifest["execution_plan"]["cache"]["plan_sha256"] = plan["sha256"]
        manifest["operation_contracts"] = json.loads(
            json.dumps(plan["operation_contracts"])
        )
        manifest["steps"][0]["execution_binding"] = json.loads(
            json.dumps(plan["steps"][0])
        )
        summary_path = run_dir / "summary.json"
        summary = cls._read_json(summary_path)
        summary["execution_plan"]["sha256"] = plan["sha256"]
        summary["execution_plan"]["runner"] = plan["runner"]
        if isinstance(summary["execution_plan"].get("cache"), dict):
            summary["execution_plan"]["cache"]["plan_sha256"] = plan["sha256"]
        summary["steps"][0]["execution_binding"] = json.loads(
            json.dumps(plan["steps"][0])
        )
        cls._write_json(summary_path, summary)
        cls._write_json(manifest_path, manifest)
        cls._rebind_summary(run_dir)


if __name__ == "__main__":
    unittest.main()
