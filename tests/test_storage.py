from __future__ import annotations

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from noema_lab.core.storage import LocalStore


class _CountingStore(LocalStore):
    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace)
        self.summary_reads = 0

    def read_json(self, path: Path):
        if path.name == "summary.json":
            self.summary_reads += 1
        return super().read_json(path)


class LocalStoreRunIndexTests(unittest.TestCase):
    def _write_summary(
        self,
        workspace: Path,
        run_id: str,
        *,
        status: str = "queued",
        recipe_name: str = "example",
        suite=None,
    ) -> Path:
        summary_path = workspace / "runs" / run_id / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "recipe_name": recipe_name,
                    "recipe": {"suite": suite or {}},
                    "steps": [{"large_ignored_value": "x" * 1024}],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return summary_path

    def test_list_runs_reuses_validated_index_across_store_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_summary(
                workspace,
                "run_a",
                status="completed",
                recipe_name="suite_recipe",
                suite={"id": "resource_allocation", "name": "Resource Allocation"},
            )
            first_store = _CountingStore(workspace)
            expected = [
                {
                    "run_id": "run_a",
                    "status": "completed",
                    "recipe_name": "suite_recipe",
                    "suite": {"id": "resource_allocation", "name": "Resource Allocation"},
                }
            ]
            self.assertEqual(first_store.list_runs(), expected)
            self.assertEqual(first_store.summary_reads, 1)
            returned = first_store.list_runs()
            self.assertEqual(returned, expected)
            self.assertEqual(first_store.summary_reads, 1)
            returned[0]["status"] = "caller mutation"
            returned[0]["suite"]["id"] = "caller mutation"
            self.assertEqual(first_store.list_runs(), expected)
            self.assertEqual(first_store.summary_reads, 1)

            restarted_store = _CountingStore(workspace)
            self.assertEqual(restarted_store.list_runs(), expected)
            self.assertEqual(restarted_store.summary_reads, 0)

    def test_list_runs_invalidates_entries_for_mtime_size_and_path_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            summary_path = self._write_summary(workspace, "run_a", status="queued")
            store = _CountingStore(workspace)
            self.assertEqual(store.list_runs()[0]["status"], "queued")
            original_stat = summary_path.stat()

            # queued and failed have equal length, so this isolates mtime_ns.
            self._write_summary(workspace, "run_a", status="failed")
            changed_stat = summary_path.stat()
            os.utime(
                summary_path,
                ns=(changed_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000),
            )
            self.assertEqual(summary_path.stat().st_size, original_stat.st_size)
            self.assertEqual(store.list_runs()[0]["status"], "failed")
            self.assertEqual(store.summary_reads, 2)

            # Restore the indexed mtime while changing size; size must invalidate.
            indexed_mtime = summary_path.stat().st_mtime_ns
            self._write_summary(workspace, "run_a", status="failed", recipe_name="longer_recipe")
            changed_stat = summary_path.stat()
            os.utime(summary_path, ns=(changed_stat.st_atime_ns, indexed_mtime))
            self.assertEqual(store.list_runs()[0]["recipe_name"], "longer_recipe")
            self.assertEqual(store.summary_reads, 3)

            # A rename retains file metadata but changes the authoritative path.
            (workspace / "runs" / "run_a").rename(workspace / "runs" / "run_b")
            rows = store.list_runs()
            self.assertEqual([row["run_id"] for row in rows], ["run_b"])
            self.assertEqual(store.summary_reads, 4)

    def test_same_size_rewrite_with_restored_mtime_is_not_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            summary_path = self._write_summary(workspace, "run_a", status="queued")
            store = _CountingStore(workspace)
            self.assertEqual(store.list_runs()[0]["status"], "queued")
            indexed_stat = summary_path.stat()

            # Keep both fields traditionally used by simple caches identical.
            # The rewrite/change-time identity must still force a fresh parse.
            replacement_path = self._write_summary(workspace, "replacement", status="failed")
            os.replace(replacement_path, summary_path)
            rewritten_stat = summary_path.stat()
            os.utime(
                summary_path,
                ns=(rewritten_stat.st_atime_ns, indexed_stat.st_mtime_ns),
            )
            current_stat = summary_path.stat()
            self.assertEqual(current_stat.st_size, indexed_stat.st_size)
            self.assertEqual(current_stat.st_mtime_ns, indexed_stat.st_mtime_ns)
            self.assertTrue(
                current_stat.st_ctime_ns != indexed_stat.st_ctime_ns
                or current_stat.st_ino != indexed_stat.st_ino
            )

            self.assertEqual(store.list_runs()[0]["status"], "failed")
            self.assertEqual(store.summary_reads, 2)

    def test_list_runs_reports_corrupt_summaries_and_recovers_after_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            first_path = self._write_summary(workspace, "run_a")
            corrupt_path = workspace / "runs" / "run_corrupt" / "summary.json"
            corrupt_path.parent.mkdir(parents=True)
            corrupt_path.write_text("{bad json", encoding="utf-8")
            store = _CountingStore(workspace)

            with self.assertRaisesRegex(ValueError, "Invalid run summary.*run_corrupt"):
                store.list_runs()
            self.assertEqual(store.summary_reads, 2)

            first_path.unlink()
            self._write_summary(workspace, "run_new", status="completed")
            self._write_summary(workspace, "run_corrupt", status="failed")
            rows = store.list_runs()
            self.assertEqual([row["run_id"] for row in rows], ["run_corrupt", "run_new"])
            self.assertEqual([row["status"] for row in rows], ["failed", "completed"])

            index = json.loads(
                (workspace / "runs" / ".run-list-index.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("run_a/summary.json", index["entries"])

    def test_corrupt_derived_index_falls_back_to_legacy_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_summary(workspace, "legacy_run", status="completed")
            index_path = workspace / "runs" / ".run-list-index.json"
            index_path.write_text("not json", encoding="utf-8")

            store = _CountingStore(workspace)
            rows = store.list_runs()
            self.assertEqual(rows[0]["run_id"], "legacy_run")
            self.assertEqual(store.summary_reads, 1)
            repaired = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(repaired["schema_version"], 1)

    def test_concurrent_run_and_benchmark_directory_claims_are_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp))
            with ThreadPoolExecutor(max_workers=16) as pool:
                run_dirs = list(pool.map(store.create_run_dir, ["same recipe"] * 64))
                benchmark_dirs = list(
                    pool.map(store.create_benchmark_dir, ["same benchmark"] * 64)
                )

            self.assertEqual(len({path.name for path in run_dirs}), 64)
            self.assertEqual(len({path.name for path in benchmark_dirs}), 64)
            self.assertTrue(all((path / "artifacts").is_dir() for path in run_dirs))


if __name__ == "__main__":
    unittest.main()
