from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.core.attempt_ledger import BenchmarkAttemptLedger
from noema_lab.core.artifacts import file_sha256
from noema_lab.cli.main import build_parser
from noema_lab.core.benchmark_run_evidence import (
    validate_benchmark_run_evidence_snapshots,
)
from noema_lab.core.benchmarks import (
    BenchmarkError,
    load_benchmark_pack,
    run_benchmark_pack,
    write_benchmark_resource_guard_evidence,
)
from noema_lab.core.operations import OperationRegistry
from noema_lab.core.storage import LocalStore
from noema_lab.core.verification import verify_benchmark_result
from noema_lab.core.benchmark_plots import plot_benchmark_result
from noema_lab.ops.channel.digital import DigitalModulateOperation
from noema_lab.ops.source.random_bits import RandomBitsOperation


class BenchmarkAttemptIntegrationTests(unittest.TestCase):
    @staticmethod
    def _registry():
        registry = OperationRegistry()
        registry.register(RandomBitsOperation())
        registry.register(DigitalModulateOperation())
        return registry

    def test_benchmark_resume_and_storage_controls_are_explicit_cli_options(self):
        args = build_parser().parse_args(
            [
                "benchmark",
                "run",
                "benchmark.yaml",
                "--resume",
                "failed-result-id",
                "--retain-backing-runs",
            ]
        )

        self.assertEqual(args.resume, "failed-result-id")
        self.assertTrue(args.retain_backing_runs)

    def test_second_sealed_test_access_fails_and_remains_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "recipe.yaml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "name": "sealed_test_fixture",
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
            pack_path = root / "benchmark.yaml"
            pack_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "id": "sealed_test_budget_fixture",
                        "version": "1",
                        "dataset": {
                            "id": "synthetic_random_bits",
                            "selection_role": "publication_test",
                            "access_policy": {
                                "state": "sealed_single_access",
                                "publication_test": True,
                                "access_ledger": ".attempt-ledger",
                                "access_budget": 1,
                                "seal_sha256": "a" * 64,
                            },
                        },
                        "task": {
                            "id": "bit_transport",
                            "kind": "transport_integrity",
                            "modality": "bits",
                        },
                        "metrics": [
                            {
                                "id": "channel.channel_use_count",
                                "source_step": "modulator",
                                "source_operation": "modulation.digital_modulate",
                            }
                        ],
                        "recipes": [{"id": "candidate", "path": str(recipe_path)}],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            pack = load_benchmark_pack(pack_path)
            store = LocalStore(root / "workspace")
            first = run_benchmark_pack(pack, self._registry(), store, ROOT)
            sealed_result_path = first / "result.json"
            sealed_sha256 = file_sha256(sealed_result_path)
            write_benchmark_resource_guard_evidence(
                store,
                first.name,
                {"kind": "fixture-resource-guard"},
            )
            self.assertEqual(file_sha256(sealed_result_path), sealed_sha256)
            pristine_report = verify_benchmark_result(
                store,
                first.name,
                registry=self._registry(),
            )
            self.assertNotEqual(pristine_report["status"], "invalid", pristine_report)

            resource_sidecar = first / "resource-guard.json"
            resource_sidecar.unlink()
            resource_sidecar.symlink_to(root / "missing-resource-guard.json")
            unsafe_sidecar_report = verify_benchmark_result(
                store,
                first.name,
                registry=self._registry(),
            )
            self.assertEqual(
                unsafe_sidecar_report["status"],
                "invalid",
                unsafe_sidecar_report,
            )
            self.assertTrue(
                any(
                    "resource-guard sidecar must be a regular file"
                    in message
                    for message in unsafe_sidecar_report["errors"]
                ),
                unsafe_sidecar_report,
            )
            resource_sidecar.unlink()
            write_benchmark_resource_guard_evidence(
                store,
                first.name,
                {"kind": "fixture-resource-guard"},
            )

            mutated = store.get_benchmark_result(first.name)
            mutated["resource_guard"] = {"forged": True}
            store.write_json(sealed_result_path, mutated)
            mutated_report = verify_benchmark_result(
                store,
                first.name,
                registry=self._registry(),
            )
            self.assertEqual(mutated_report["status"], "invalid", mutated_report)
            self.assertTrue(
                any(
                    "immutable attempt-ledger identity" in message
                    for message in mutated_report["errors"]
                ),
                mutated_report,
            )
            with self.assertRaisesRegex(BenchmarkError, "access budget is exhausted"):
                run_benchmark_pack(pack, self._registry(), store, ROOT)

            snapshot = BenchmarkAttemptLedger(store.benchmarks_dir).verify()

        self.assertEqual(snapshot["status"], "valid")
        self.assertEqual(snapshot["attempt_count"], 2)
        self.assertEqual(snapshot["status_counts"], {"completed": 1, "failed": 1})
        self.assertEqual(snapshot["attempts"][0]["result"]["result_id"], first.name)
        self.assertEqual(
            [row["invocation"]["access_sequence"] for row in snapshot["attempts"]],
            [1, 2],
        )

    def test_benchmark_created_event_failure_finalizes_result_and_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "recipe.yaml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "name": "event_sink_failure_fixture",
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
            pack_path = root / "benchmark.yaml"
            pack_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "id": "event_sink_failure_pack",
                        "version": "1",
                        "dataset": {"id": "synthetic_random_bits"},
                        "task": {
                            "id": "bit_transport",
                            "kind": "transport_integrity",
                            "modality": "bits",
                        },
                        "metrics": [
                            {
                                "id": "channel.channel_use_count",
                                "source_step": "modulator",
                                "source_operation": "modulation.digital_modulate",
                            }
                        ],
                        "recipes": [
                            {"id": "candidate", "path": str(recipe_path)}
                        ],
                        "metadata": {"benchmark_tier": "experimental"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            store = LocalStore(root / "workspace")

            def failing_sink(_event):
                raise RuntimeError("synthetic benchmark event sink failure")

            with self.assertRaisesRegex(RuntimeError, "event sink failure"):
                run_benchmark_pack(
                    load_benchmark_pack(pack_path),
                    self._registry(),
                    store,
                    ROOT,
                    event_sink=failing_sink,
                )

            result_dirs = [
                path
                for path in store.benchmarks_dir.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ]
            self.assertEqual(len(result_dirs), 1)
            result = store.get_benchmark_result(result_dirs[0].name)
            snapshot = BenchmarkAttemptLedger(store.benchmarks_dir).verify()

        self.assertEqual(result["status"], "failed")
        self.assertIn("event sink failure", result["error"])
        self.assertEqual(
            result["attempt_ledger"]["expected_terminal_status"],
            "failed",
        )
        self.assertEqual(snapshot["status"], "valid")
        self.assertEqual(snapshot["status_counts"], {"failed": 1})
        self.assertEqual(snapshot["attempts"][0]["terminal_status"], "failed")
        self.assertEqual(
            snapshot["attempts"][0]["result"]["result_id"],
            result_dirs[0].name,
        )

    def test_failed_benchmark_explicit_resume_reuses_only_validated_completed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "recipe.yaml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "name": "resume_fixture",
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
            pack_path = root / "benchmark.yaml"
            pack_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "id": "explicit_resume_fixture",
                        "version": "1",
                        "dataset": {"id": "synthetic_random_bits"},
                        "task": {
                            "id": "bit_transport",
                            "kind": "transport_integrity",
                            "modality": "bits",
                        },
                        "metrics": [
                            {
                                "id": "channel.channel_use_count",
                                "source_step": "modulator",
                                "source_operation": "modulation.digital_modulate",
                            }
                        ],
                        "recipes": [
                            {"id": "first", "path": str(recipe_path)},
                            {"id": "second", "path": str(recipe_path)},
                        ],
                        "metadata": {"benchmark_tier": "experimental"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            pack = load_benchmark_pack(pack_path)
            registry = self._registry()
            store = LocalStore(root / "workspace")

            from noema_lab.core import benchmarks as benchmarks_module

            original_snapshot = benchmarks_module.snapshot_benchmark_run_evidence
            snapshot_calls = 0

            def fail_second_snapshot(*args, **kwargs):
                nonlocal snapshot_calls
                snapshot_calls += 1
                if snapshot_calls == 2:
                    raise OSError("synthetic evidence storage failure")
                return original_snapshot(*args, **kwargs)

            with patch(
                "noema_lab.core.benchmarks.snapshot_benchmark_run_evidence",
                side_effect=fail_second_snapshot,
            ):
                with self.assertRaisesRegex(
                    OSError, "synthetic evidence storage failure"
                ):
                    run_benchmark_pack(pack, registry, store, ROOT)

            failed_dirs = [
                path
                for path in store.benchmarks_dir.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ]
            self.assertEqual(len(failed_dirs), 1)
            failed_dir = failed_dirs[0]
            failed = store.get_benchmark_result(failed_dir.name)
            self.assertEqual(
                [row["status"] for row in failed["recipes"]],
                ["completed", "failed"],
            )
            first_run_id = failed["recipes"][0]["run_id"]
            self.assertFalse((store.runs_dir / first_run_id).exists())
            self.assertEqual(len(store.list_runs()), 1)

            validated_source = validate_benchmark_run_evidence_snapshots(
                failed_dir,
                failed,
            )
            source_summary_record = next(
                row
                for row in validated_source["entries"][0][
                    "snapshot_manifest"
                ]["files"]
                if row["role"] == "summary"
            )
            source_summary_path = failed_dir / source_summary_record["path"]
            source_summary_bytes = source_summary_path.read_bytes()
            source_summary_path.write_bytes(source_summary_bytes + b"\n")
            try:
                with self.assertRaisesRegex(
                    BenchmarkError, "run evidence is invalid"
                ):
                    run_benchmark_pack(
                        pack,
                        registry,
                        store,
                        ROOT,
                        resume_result_id=failed_dir.name,
                    )
            finally:
                source_summary_path.write_bytes(source_summary_bytes)
            self.assertEqual(len(store.list_runs()), 1)

            pack.metadata["resume_protocol_mutation"] = True
            try:
                with self.assertRaisesRegex(
                    BenchmarkError, "pack/protocol identity does not match"
                ):
                    run_benchmark_pack(
                        pack,
                        registry,
                        store,
                        ROOT,
                        resume_result_id=failed_dir.name,
                    )
            finally:
                pack.metadata.pop("resume_protocol_mutation")
            self.assertEqual(len(store.list_runs()), 1)

            with self.assertRaisesRegex(
                BenchmarkError, "execution settings do not match"
            ):
                run_benchmark_pack(
                    pack,
                    registry,
                    store,
                    ROOT,
                    parallel_workers=2,
                    resume_result_id=failed_dir.name,
                )
            self.assertEqual(len(store.list_runs()), 1)

            resumed_dir = run_benchmark_pack(
                pack,
                registry,
                store,
                ROOT,
                resume_result_id=failed_dir.name,
            )
            resumed = store.get_benchmark_result(resumed_dir.name)
            verification = verify_benchmark_result(
                store,
                resumed_dir.name,
                registry=registry,
            )
            ledger = BenchmarkAttemptLedger(store.benchmarks_dir).verify()
            run_count = len(store.list_runs(read_only=True))

        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(
            [row["status"] for row in resumed["recipes"]],
            ["completed", "completed"],
        )
        self.assertEqual(resumed["recipes"][0]["run_id"], first_run_id)
        self.assertIn("resume_reuse", resumed["recipes"][0])
        self.assertNotIn("resume_reuse", resumed["recipes"][1])
        self.assertEqual(run_count, 1)
        self.assertEqual(resumed["storage"]["pruned_backing_run_count"], 1)
        self.assertEqual(resumed["storage"]["backing_run_prune_failures"], [])
        self.assertEqual(resumed["resume"]["source_result_id"], failed_dir.name)
        self.assertEqual(resumed["resume"]["eligible_recipe_count"], 1)
        self.assertEqual(resumed["resume"]["reused_recipe_count"], 1)
        self.assertNotEqual(verification["status"], "invalid", verification)
        self.assertEqual(ledger["status"], "valid")
        self.assertEqual(
            ledger["status_counts"],
            {"failed": 4, "completed": 1},
        )
        completed_attempt = ledger["attempts"][-1]
        self.assertEqual(
            completed_attempt["details"]["resume"]["source_result_id"],
            failed_dir.name,
        )

    def test_completed_benchmark_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "recipe.yaml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "name": "completed_resume_fixture",
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
            pack_path = root / "benchmark.yaml"
            pack_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "id": "completed_resume_fixture",
                        "version": "1",
                        "dataset": {"id": "synthetic_random_bits"},
                        "task": {
                            "id": "bit_transport",
                            "kind": "transport_integrity",
                            "modality": "bits",
                        },
                        "metrics": [
                            {
                                "id": "channel.channel_use_count",
                                "source_step": "modulator",
                                "source_operation": "modulation.digital_modulate",
                            }
                        ],
                        "recipes": [
                            {"id": "candidate", "path": str(recipe_path)}
                        ],
                        "metadata": {"benchmark_tier": "experimental"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            pack = load_benchmark_pack(pack_path)
            store = LocalStore(root / "workspace")
            completed_dir = run_benchmark_pack(
                pack,
                self._registry(),
                store,
                ROOT,
                retain_backing_runs=True,
            )
            completed = store.get_benchmark_result(completed_dir.name)
            self.assertTrue(
                (store.runs_dir / completed["recipes"][0]["run_id"]).is_dir()
            )
            self.assertTrue(completed["storage"]["retain_backing_runs"])

            with self.assertRaisesRegex(
                BenchmarkError, "requires a failed terminal result"
            ):
                run_benchmark_pack(
                    pack,
                    self._registry(),
                    store,
                    ROOT,
                    resume_result_id=completed_dir.name,
                )

    def test_placeholder_method_makes_result_and_attempt_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "recipe.yaml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "name": "placeholder_fixture",
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
            pack_path = root / "benchmark.yaml"
            pack_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "id": "placeholder_pack",
                        "version": "1",
                        "dataset": {"id": "synthetic_random_bits"},
                        "task": {
                            "id": "bit_transport",
                            "kind": "transport_integrity",
                            "modality": "bits",
                        },
                        "metrics": [
                            {
                                "id": "channel.channel_use_count",
                                "source_step": "modulator",
                                "source_operation": "modulation.digital_modulate",
                            }
                        ],
                        "recipes": [
                            {
                                "id": "baseline",
                                "path": str(recipe_path),
                                "role": "baseline",
                            },
                            {
                                "id": "candidate",
                                "path": str(recipe_path),
                                "role": "candidate_placeholder",
                                "params": {
                                    "skip": True,
                                    "skip_reason": "adapter not supplied",
                                },
                            },
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            store = LocalStore(root / "workspace")
            result_dir = run_benchmark_pack(
                load_benchmark_pack(pack_path),
                self._registry(),
                store,
                ROOT,
            )
            result = store.get_benchmark_result(result_dir.name)
            snapshot = BenchmarkAttemptLedger(store.benchmarks_dir).verify()

            with self.assertRaisesRegex(BenchmarkError, "status is incomplete"):
                plot_benchmark_result(
                    store,
                    result_dir.name,
                    "channel-uses",
                    Path("plots/invalid.png"),
                )

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(
            [row["status"] for row in result["recipes"]],
            ["completed", "skipped"],
        )
        self.assertEqual(
            snapshot["attempts"][0]["terminal_status"],
            "incomplete",
        )

    def test_shipped_packs_use_canonical_benchmark_tier_key(self):
        pack_paths = sorted((ROOT / "benchmarks").rglob("*.yaml"))
        self.assertTrue(pack_paths)
        legacy_paths = []
        missing_paths = []
        for path in pack_paths:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata") if isinstance(payload, dict) else None
            metadata = metadata if isinstance(metadata, dict) else {}
            if "tier" in metadata:
                legacy_paths.append(str(path.relative_to(ROOT)))
            if "benchmark_tier" not in metadata:
                missing_paths.append(str(path.relative_to(ROOT)))

        self.assertEqual(legacy_paths, [])
        self.assertEqual(missing_paths, [])

    def test_resource_allocation_baseline_roster_links_to_method_ids(self):
        path = (
            ROOT
            / "benchmarks"
            / "resource_allocation"
            / "power_allocation_v1.yaml"
        )
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        declared = {str(value) for value in payload.get("baselines") or []}
        linked = {
            str(value)
            for row in payload.get("recipes") or []
            if isinstance(row, dict)
            for value in (
                row.get("id"),
                (row.get("params") or {}).get("method_id"),
                (row.get("params") or {}).get("baseline_id"),
            )
            if value not in (None, "")
        }

        self.assertTrue(declared)
        self.assertEqual(declared.difference(linked), set())


if __name__ == "__main__":
    unittest.main()
