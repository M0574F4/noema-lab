from __future__ import annotations

import errno
import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.benchmark_run_evidence import (
    ARTIFACT_PROJECTION_SCHEMA_VERSION,
    BenchmarkRunEvidenceError,
    _load_json_mapping,
    _project_declared_artifacts,
    _stable_compress_json,
    _stable_copy_binary,
    _validate_artifact_projection,
    copy_file_independent,
    snapshot_benchmark_run_evidence,
    validate_benchmark_run_evidence_snapshot,
)
from noema_lab.core.reproducibility import canonical_json_sha256


class BenchmarkRunEvidenceStorageTests(unittest.TestCase):
    def _write_completed_run(
        self,
        run_dir: Path,
        *,
        run_id: str,
        artifacts,
    ):
        recipe = {
            "schema_version": 1,
            "name": "retained-coded-evidence",
            "metadata": {},
            "steps": [],
        }
        recipe_sha = canonical_json_sha256(recipe)
        execution_plan = {
            "schema_version": 1,
            "kind": "noema.execution_plan",
            "runner": "local",
            "recipe": {"sha256": recipe_sha},
        }
        execution_plan["sha256"] = canonical_json_sha256(execution_plan)
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "recipe_name": recipe["name"],
            "recipe_sha256": recipe_sha,
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
                "sha256": recipe_sha,
                "authored_sha256": recipe_sha,
            },
            "execution_plan": {"sha256": execution_plan["sha256"]},
            "artifacts": artifacts,
        }
        for filename, payload in (
            ("recipe.json", recipe),
            ("recipe.authored.json", recipe),
            ("summary.json", summary),
            ("manifest.json", manifest),
            ("execution-plan.json", execution_plan),
        ):
            (run_dir / filename).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
        return recipe, recipe_sha

    def test_v2_json_compression_is_deterministic_bounded_and_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "summary.json"
            source.write_text(
                json.dumps({"status": "completed", "padding": "x" * 8192}),
                encoding="utf-8",
            )
            first = root / "first.json.gz"
            second = root / "second.json.gz"

            first_record = _stable_compress_json(
                source,
                first,
                role="summary",
                result_relative=Path("run_evidence/000-case/summary.json.gz"),
            )
            second_record = _stable_compress_json(
                source,
                second,
                role="summary",
                result_relative=Path("run_evidence/001-case/summary.json.gz"),
            )

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_record["sha256"], second_record["sha256"])
            self.assertEqual(first.read_bytes()[9], 255)
            self.assertEqual(_load_json_mapping(first, "summary")["status"], "completed")

            concatenated = root / "concatenated.json.gz"
            concatenated.write_bytes(
                gzip.compress(b'{"status":"completed"}', mtime=0)
                + gzip.compress(b'{"extra":true}', mtime=0)
            )
            with self.assertRaisesRegex(
                BenchmarkRunEvidenceError,
                "one bounded complete gzip member",
            ):
                _load_json_mapping(concatenated, "concatenated summary")

    def test_projection_omits_large_tensors_and_keeps_only_declared_metric_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            staging = root / "staging"
            run_dir.mkdir()
            staging.mkdir()
            tensor = run_dir / "artifacts" / "channel" / "state.npz"
            report = run_dir / "artifacts" / "metrics" / "report.json"
            other_report = run_dir / "artifacts" / "other_metrics" / "report.json"
            tensor.parent.mkdir(parents=True)
            report.parent.mkdir(parents=True)
            other_report.parent.mkdir(parents=True)
            tensor.write_bytes(os.urandom(4 * 1024 * 1024))
            report.write_text('{"metric": 1.0}', encoding="utf-8")
            other_report.write_text('{"other": 2.0}', encoding="utf-8")
            manifest = {
                "artifacts": [
                    {
                        "step_id": "channel",
                        "output_name": "state",
                        "kind": "channel.state.numpy",
                        "relative_path": "artifacts/channel/state.npz",
                        "sha256": file_sha256(tensor),
                    },
                    {
                        "step_id": "metrics",
                        "output_name": "report",
                        "kind": "metrics.report",
                        "relative_path": "artifacts/metrics/report.json",
                        "sha256": file_sha256(report),
                    },
                    {
                        "step_id": "other_metrics",
                        "output_name": "report",
                        "kind": "metrics.report",
                        "relative_path": "artifacts/other_metrics/report.json",
                        "sha256": file_sha256(other_report),
                    },
                ]
            }

            records, projection = _project_declared_artifacts(
                run_dir,
                staging,
                Path("run_evidence/000-case"),
                manifest,
                metric_producer_steps={"metrics"},
            )

            self.assertEqual(
                [record["path"] for record in records],
                ["run_evidence/000-case/artifacts/metrics/report.json"],
            )
            self.assertFalse((staging / "artifacts/channel/state.npz").exists())
            self.assertFalse(
                (staging / "artifacts/other_metrics/report.json").exists()
            )
            self.assertEqual(projection["source_artifact_count"], 3)
            self.assertEqual(projection["retained_artifact_count"], 1)
            self.assertEqual(projection["omitted_artifact_count"], 2)
            self.assertEqual(projection["explicit_artifact_paths"], [])
            legacy_projection = dict(projection)
            legacy_projection["schema_version"] = 1
            legacy_projection.pop("explicit_artifact_paths")
            self.assertEqual(
                _validate_artifact_projection(legacy_projection)[
                    "explicit_artifact_paths"
                ],
                [],
            )
            self.assertGreater(
                projection["omitted_artifact_bytes"],
                4 * 1024 * 1024 - 1,
            )

    def test_snapshot_exact_allowlist_retains_one_non_metric_artifact_and_binds_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "benchmark"
            run_dir = root / "run-001"
            result_dir.mkdir()
            run_dir.mkdir()
            selected = run_dir / "artifacts/channel_encoder/coded_bits.npz"
            unselected = run_dir / "artifacts/receiver/coded_bits.npz"
            metric = run_dir / "artifacts/evaluation/report.json"
            selected.parent.mkdir(parents=True)
            unselected.parent.mkdir(parents=True)
            metric.parent.mkdir(parents=True)
            selected.write_bytes(b"selected coded evidence")
            unselected.write_bytes(b"unselected coded evidence")
            metric.write_text('{"quality.psnr_db": 30.0}', encoding="utf-8")
            artifacts = [
                {
                    "step_id": "channel_encoder",
                    "output_name": "coded_bits",
                    "kind": "channel.coded_bits.numpy",
                    "relative_path": "artifacts/channel_encoder/coded_bits.npz",
                    "sha256": file_sha256(selected),
                },
                {
                    "step_id": "receiver",
                    "output_name": "coded_bits",
                    "kind": "channel.coded_bits.numpy",
                    "relative_path": "artifacts/receiver/coded_bits.npz",
                    "sha256": file_sha256(unselected),
                },
                {
                    "step_id": "evaluation",
                    "output_name": "report",
                    "kind": "metrics.report",
                    "relative_path": "artifacts/evaluation/report.json",
                    "sha256": file_sha256(metric),
                },
            ]
            recipe, recipe_sha = self._write_completed_run(
                run_dir,
                run_id=run_dir.name,
                artifacts=artifacts,
            )

            descriptor = snapshot_benchmark_run_evidence(
                result_dir,
                entry_id="publication-case",
                entry_index=0,
                run_dir=run_dir,
                run_id=run_dir.name,
                semantic_recipe_sha256=recipe_sha,
                metric_producer_steps={"evaluation"},
                retained_artifact_paths={
                    "artifacts/channel_encoder/coded_bits.npz"
                },
            )

            snapshot_root = result_dir / descriptor["root"]
            self.assertEqual(
                (snapshot_root / "artifacts/channel_encoder/coded_bits.npz").read_bytes(),
                b"selected coded evidence",
            )
            self.assertTrue(
                (snapshot_root / "artifacts/evaluation/report.json").is_file()
            )
            self.assertFalse(
                (snapshot_root / "artifacts/receiver/coded_bits.npz").exists()
            )
            projection = json.loads(
                (snapshot_root / "snapshot.json").read_text(encoding="utf-8")
            )["artifact_projection"]
            self.assertEqual(
                projection["schema_version"],
                ARTIFACT_PROJECTION_SCHEMA_VERSION,
            )
            self.assertEqual(
                projection["selection"],
                "declared_metric_producer_reports",
            )
            self.assertEqual(
                projection["explicit_artifact_paths"],
                ["artifacts/channel_encoder/coded_bits.npz"],
            )
            self.assertEqual(projection["source_artifact_count"], 3)
            self.assertEqual(projection["retained_artifact_count"], 2)
            self.assertEqual(projection["omitted_artifact_count"], 1)

            entry = {
                "id": "publication-case",
                "recipe_name": recipe["name"],
                "run_id": run_dir.name,
                "status": "completed",
                "recipe_sha256": recipe_sha,
                "semantic_recipe_sha256": recipe_sha,
                "metrics": {},
                "metric_provenance": {
                    "quality.psnr_db": {
                        "source_scope": "step",
                        "source_step": "evaluation",
                    }
                },
                "pairing_id": None,
                "pairing_seed": None,
                "aggregation_cell_id": None,
                "research": None,
                "run_evidence_snapshot": descriptor,
            }
            validated = validate_benchmark_run_evidence_snapshot(
                result_dir,
                entry,
                entry_index=0,
            )
            self.assertEqual(
                validated["snapshot_manifest"]["artifact_projection"],
                projection,
            )

    def test_explicit_allowlist_rejects_absent_or_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            staging = root / "staging"
            run_dir.mkdir()
            staging.mkdir()
            manifest = {"artifacts": []}

            for retained_path, message in (
                ("artifacts/missing/coded_bits.npz", "absent from"),
                ("../coded_bits.npz", "safe relative path"),
                ("coded_bits.npz", "below artifacts/"),
            ):
                with self.subTest(retained_path=retained_path):
                    with self.assertRaisesRegex(BenchmarkRunEvidenceError, message):
                        _project_declared_artifacts(
                            run_dir,
                            staging,
                            Path("run_evidence/000-case"),
                            manifest,
                            metric_producer_steps=set(),
                            retained_artifact_paths={retained_path},
                        )

    def test_allowlisted_artifact_enforces_digest_size_symlink_and_toctou_checks(self):
        def project_one(root: Path, declared_sha: str):
            return _project_declared_artifacts(
                root / "run",
                root / "staging",
                Path("run_evidence/000-case"),
                {
                    "artifacts": [
                        {
                            "step_id": "channel_encoder",
                            "output_name": "coded_bits",
                            "kind": "channel.coded_bits.numpy",
                            "relative_path": "artifacts/channel_encoder/coded_bits.npz",
                            "sha256": declared_sha,
                        }
                    ]
                },
                metric_producer_steps=set(),
                retained_artifact_paths={
                    "artifacts/channel_encoder/coded_bits.npz"
                },
            )

        with self.subTest(protection="digest"), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "run/artifacts/channel_encoder/coded_bits.npz"
            artifact.parent.mkdir(parents=True)
            (root / "staging").mkdir()
            artifact.write_bytes(b"coded evidence")
            with self.assertRaisesRegex(BenchmarkRunEvidenceError, "hash disagrees"):
                project_one(root, "0" * 64)

        with self.subTest(protection="size"), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "run/artifacts/channel_encoder/coded_bits.npz"
            artifact.parent.mkdir(parents=True)
            (root / "staging").mkdir()
            artifact.write_bytes(b"coded evidence")
            with patch(
                "noema_lab.core.benchmark_run_evidence._MAX_ARTIFACT_BYTES",
                len(artifact.read_bytes()) - 1,
            ):
                with self.assertRaisesRegex(
                    BenchmarkRunEvidenceError,
                    "exceeds the snapshot limit",
                ):
                    project_one(root, file_sha256(artifact))

        with self.subTest(protection="symlink"), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "outside.npz"
            target.write_bytes(b"outside coded evidence")
            artifact = root / "run/artifacts/channel_encoder/coded_bits.npz"
            artifact.parent.mkdir(parents=True)
            (root / "staging").mkdir()
            artifact.symlink_to(target)
            with self.assertRaisesRegex(BenchmarkRunEvidenceError, "symlink"):
                project_one(root, file_sha256(target))

        with self.subTest(protection="toctou"), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "run/artifacts/channel_encoder/coded_bits.npz"
            artifact.parent.mkdir(parents=True)
            (root / "staging").mkdir()
            artifact.write_bytes(b"original evidence")
            original_sha = file_sha256(artifact)
            stable_copy = _stable_copy_binary

            def replace_before_retention(source, destination, **kwargs):
                source.write_bytes(b"mutated evidence!")
                return stable_copy(source, destination, **kwargs)

            with patch(
                "noema_lab.core.benchmark_run_evidence._stable_copy_binary",
                side_effect=replace_before_retention,
            ):
                with self.assertRaisesRegex(
                    BenchmarkRunEvidenceError,
                    "hash disagrees with manifest",
                ):
                    project_one(root, original_sha)

    def test_unsupported_reflink_falls_back_to_an_independent_byte_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            destination = root / "snapshot.bin"
            source.write_bytes(b"portable fallback")

            with patch(
                "noema_lab.core.benchmark_run_evidence._try_linux_reflink",
                return_value=False,
            ):
                method = copy_file_independent(source, destination)

            self.assertEqual(method, "copy")
            self.assertEqual(destination.read_bytes(), b"portable fallback")
            self.assertNotEqual(source.stat().st_ino, destination.stat().st_ino)
            source.write_bytes(b"changed source")
            self.assertEqual(destination.read_bytes(), b"portable fallback")

    def test_stable_copy_is_hash_preserving_and_independent_of_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            destination = root / "snapshot.bin"
            source.write_bytes((b"immutable benchmark evidence\n" * 4096) + b"tail")
            expected = file_sha256(source)

            record = _stable_copy_binary(
                source,
                destination,
                result_relative=Path("run_evidence/000-case/artifacts/data.bin"),
            )

            self.assertNotEqual(source.stat().st_ino, destination.stat().st_ino)
            self.assertEqual(record["sha256"], expected)
            self.assertEqual(record["size_bytes"], destination.stat().st_size)
            source.write_bytes(b"the mutable source changed after the snapshot")
            self.assertEqual(file_sha256(destination), expected)
            source.unlink()
            self.assertEqual(file_sha256(destination), expected)
            self.assertTrue(destination.read_bytes().endswith(b"tail"))

    def test_snapshot_removes_partial_staging_tree_after_copy_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "benchmark"
            run_dir = root / "run"
            result_dir.mkdir()
            run_dir.mkdir()

            def fail_after_partial_copy(source, destination, **kwargs):
                destination.write_bytes(b"partial")
                raise OSError(errno.ENOSPC, "No space left on device")

            with patch(
                "noema_lab.core.benchmark_run_evidence._stable_compress_json",
                side_effect=fail_after_partial_copy,
            ):
                with self.assertRaisesRegex(OSError, "No space left"):
                    snapshot_benchmark_run_evidence(
                        result_dir,
                        entry_id="method",
                        entry_index=0,
                        run_dir=run_dir,
                        run_id="run-001",
                        semantic_recipe_sha256="a" * 64,
                    )

            self.assertEqual(list(result_dir.glob(".run-evidence-*")), [])
            self.assertFalse((result_dir / "run_evidence").exists())

    def test_reflink_uses_little_incremental_space_when_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            destination = root / "snapshot.bin"
            with source.open("wb") as handle:
                for _ in range(16):
                    handle.write(os.urandom(1024 * 1024))

            source_allocated = source.stat().st_blocks * 512
            before = os.statvfs(root)
            before_free = before.f_bfree * before.f_frsize
            method = copy_file_independent(source, destination)
            after = os.statvfs(root)
            after_free = after.f_bfree * after.f_frsize

            if method != "reflink":
                self.skipTest("test filesystem does not support copy-on-write reflinks")
            consumed = max(0, before_free - after_free)
            tolerance = max(1024 * 1024, source_allocated // 4)
            self.assertLess(
                consumed,
                tolerance,
                "copy-on-write snapshot consumed space comparable to a byte copy",
            )
            self.assertEqual(file_sha256(destination), file_sha256(source))


if __name__ == "__main__":
    unittest.main()
