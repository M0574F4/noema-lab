from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.core import benchmark_run_evidence as evidence_module
from noema_lab.core import benchmarks as benchmarks_module
from noema_lab.core.benchmark_run_evidence import (
    BenchmarkRunEvidenceError,
    snapshot_benchmark_run_evidence,
    validate_benchmark_run_evidence_snapshots,
)
from noema_lab.core.benchmarks import load_benchmark_pack, run_benchmark_pack
from noema_lab.core.operations import OperationRegistry
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.storage import LocalStore
from noema_lab.ops.channel.digital import DigitalModulateOperation
from noema_lab.ops.source.random_bits import RandomBitsOperation


class _TrackedPayload(dict):
    """Weak-referenceable mapping used to detect retained decoded payloads."""


class BenchmarkEvidenceScalingTests(unittest.TestCase):
    @staticmethod
    def _registry() -> OperationRegistry:
        registry = OperationRegistry()
        registry.register(RandomBitsOperation())
        registry.register(DigitalModulateOperation())
        return registry

    @staticmethod
    def _write_tiny_pack(root: Path, recipe_count: int) -> Path:
        recipe_path = root / "recipe.yaml"
        recipe_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "benchmark_evidence_scaling_fixture",
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
                    "id": "benchmark_evidence_scaling_fixture",
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
                        {"id": "cell-%03d" % index, "path": str(recipe_path)}
                        for index in range(recipe_count)
                    ],
                    "metadata": {"benchmark_tier": "experimental"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return pack_path

    @staticmethod
    def _write_snapshot_fixture(result_dir: Path, entry_index: int) -> dict:
        run_id = "synthetic-run-%03d" % entry_index
        entry_id = "cell-%03d" % entry_index
        run_dir = result_dir.parent / ("source-" + run_id)
        run_dir.mkdir()
        recipe = {
            "schema_version": 1,
            "name": "synthetic_streaming_audit_fixture",
            "metadata": {},
            "steps": [],
        }
        recipe_sha256 = canonical_json_sha256(recipe)
        execution_plan = {
            "schema_version": 1,
            "kind": "noema.execution_plan",
            "runner": "local",
            "recipe": {"sha256": recipe_sha256},
        }
        execution_plan["sha256"] = canonical_json_sha256(execution_plan)
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "recipe_name": recipe["name"],
            "recipe_sha256": recipe_sha256,
            "status": "completed",
            "metrics": {},
            "steps": [],
        }
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "recipe_name": recipe["name"],
            "status": "completed",
            "recipe": {
                "sha256": recipe_sha256,
                "authored_sha256": recipe_sha256,
            },
            "execution_plan": {"sha256": execution_plan["sha256"]},
            "artifacts": [],
        }
        for filename, payload in (
            ("recipe.json", recipe),
            ("recipe.authored.json", recipe),
            ("summary.json", summary),
            ("manifest.json", manifest),
            ("execution-plan.json", execution_plan),
        ):
            (run_dir / filename).write_text(
                json.dumps(payload, sort_keys=True),
                encoding="utf-8",
            )
        descriptor = snapshot_benchmark_run_evidence(
            result_dir,
            entry_id=entry_id,
            entry_index=entry_index,
            run_dir=run_dir,
            run_id=run_id,
            semantic_recipe_sha256=recipe_sha256,
            metric_producer_steps=set(),
        )
        shutil.rmtree(run_dir)
        return {
            "id": entry_id,
            "recipe_name": recipe["name"],
            "run_id": run_id,
            "status": "completed",
            "recipe_sha256": recipe_sha256,
            "semantic_recipe_sha256": recipe_sha256,
            "metrics": {},
            "metric_provenance": {},
            "pairing_id": None,
            "pairing_seed": None,
            "aggregation_cell_id": None,
            "research": None,
            "run_evidence_snapshot": descriptor,
        }

    def test_runner_validates_each_current_cell_once_and_runs_one_final_audit(self):
        recipe_count = 4
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = load_benchmark_pack(self._write_tiny_pack(root, recipe_count))
            store = LocalStore(root / "workspace")
            original_single = evidence_module.validate_benchmark_run_evidence_snapshot
            original_audit = evidence_module.audit_benchmark_run_evidence_snapshots
            original_reports = benchmarks_module.write_benchmark_reports

            with (
                patch.object(
                    benchmarks_module,
                    "validate_benchmark_run_evidence_snapshot",
                    wraps=original_single,
                ) as single_validator,
                patch.object(
                    benchmarks_module,
                    "audit_benchmark_run_evidence_snapshots",
                    wraps=original_audit,
                ) as streaming_audit,
                patch.object(
                    benchmarks_module,
                    "write_benchmark_reports",
                    wraps=original_reports,
                ) as report_writer,
            ):
                run_benchmark_pack(pack, self._registry(), store, ROOT)

        self.assertEqual(single_validator.call_count, recipe_count)
        self.assertEqual(streaming_audit.call_count, 1)
        self.assertEqual(
            report_writer.call_count,
            1,
            "full reports should be generated only after the benchmark reaches a terminal state",
        )

    def test_final_streaming_audit_detects_tampering_after_an_earlier_row_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = load_benchmark_pack(self._write_tiny_pack(root, 2))
            store = LocalStore(root / "workspace")
            original_single = evidence_module.validate_benchmark_run_evidence_snapshot
            original_reports = benchmarks_module.write_benchmark_reports

            def validate_current_then_tamper_first(
                result_dir, entry, *, entry_index, **kwargs
            ):
                validated = original_single(
                    result_dir,
                    entry,
                    entry_index=entry_index,
                    **kwargs,
                )
                if entry_index == 1:
                    first_root = next((Path(result_dir) / "run_evidence").glob("000-*"))
                    first_summary = next(first_root.glob("summary.json*"))
                    first_summary.write_bytes(first_summary.read_bytes() + b"tamper")
                return validated

            with (
                patch.object(
                    benchmarks_module,
                    "validate_benchmark_run_evidence_snapshot",
                    side_effect=validate_current_then_tamper_first,
                ),
                patch.object(
                    benchmarks_module,
                    "write_benchmark_reports",
                    wraps=original_reports,
                ) as report_writer,
            ):
                with self.assertRaisesRegex(
                    BenchmarkRunEvidenceError,
                    "run-evidence snapshot (size|hash) mismatch",
                ):
                    run_benchmark_pack(pack, self._registry(), store, ROOT)

        self.assertEqual(report_writer.call_count, 1)

    def test_compact_streaming_audit_matches_existing_aggregate_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "benchmark"
            result_dir.mkdir()
            result = {
                "recipes": [
                    self._write_snapshot_fixture(result_dir, index)
                    for index in range(3)
                ]
            }

            full = validate_benchmark_run_evidence_snapshots(result_dir, result)
            compact = validate_benchmark_run_evidence_snapshots(
                result_dir,
                result,
                include_payloads=False,
            )

        self.assertEqual(full["schema_version"], compact["schema_version"])
        self.assertEqual(full["kind"], compact["kind"])
        self.assertEqual(len(full["entries"]), len(compact["entries"]))
        for full_entry, compact_entry in zip(full["entries"], compact["entries"]):
            expected = {
                key: value
                for key, value in full_entry.items()
                if key not in {"recipe", "summary", "manifest"}
            }
            self.assertEqual(compact_entry, expected)
            self.assertTrue(
                {"descriptor", "snapshot_manifest", "verification"}.issubset(
                    compact_entry
                )
            )

    def test_compact_audit_loads_linearly_and_does_not_retain_decoded_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "benchmark"
            result_dir.mkdir()
            all_rows = [
                self._write_snapshot_fixture(result_dir, index)
                for index in range(200)
            ]
            original_loader = evidence_module._load_json_mapping
            observed_total_loads = []

            for recipe_count in (50, 100, 200):
                with self.subTest(recipe_count=recipe_count):
                    required_payload_refs = []
                    total_loads = 0
                    maximum_live_required_payloads = 0

                    def tracked_loader(path, label):
                        nonlocal total_loads, maximum_live_required_payloads
                        total_loads += 1
                        payload = _TrackedPayload(original_loader(path, label))
                        if label in {
                            "run-evidence manifest",
                            "run-evidence recipe",
                            "run-evidence authored_recipe",
                            "run-evidence summary",
                            "run-evidence execution_plan",
                        }:
                            required_payload_refs.append(weakref.ref(payload))
                            maximum_live_required_payloads = max(
                                maximum_live_required_payloads,
                                sum(ref() is not None for ref in required_payload_refs),
                            )
                        return payload

                    with patch.object(
                        evidence_module,
                        "_load_json_mapping",
                        side_effect=tracked_loader,
                    ):
                        compact = validate_benchmark_run_evidence_snapshots(
                            result_dir,
                            {"recipes": all_rows[:recipe_count]},
                            include_payloads=False,
                        )

                    gc.collect()
                    observed_total_loads.append(total_loads)
                    self.assertEqual(len(compact["entries"]), recipe_count)
                    self.assertEqual(total_loads, 6 * recipe_count)
                    self.assertLessEqual(maximum_live_required_payloads, 5)
                    self.assertFalse(
                        any(ref() is not None for ref in required_payload_refs)
                    )
                    self.assertTrue(
                        all(
                            not {"recipe", "summary", "manifest"}.intersection(
                                entry
                            )
                            for entry in compact["entries"]
                        )
                    )

        self.assertEqual(observed_total_loads, [300, 600, 1200])


if __name__ == "__main__":
    unittest.main()
