from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from noema_lab.core.reproducibility import (
    _artifact_manifest_record,
    _git_worktree_content_identity,
    canonical_json_sha256,
    git_snapshot,
)


class ReproducibilityFailClosedTests(unittest.TestCase):
    def test_git_snapshot_records_command_failures_instead_of_omitting_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()

            def fake_git(_root, arguments):
                command = " ".join(arguments)
                if command == "rev-parse HEAD":
                    return {"ok": True, "stdout": "abc123", "stderr": ""}
                return {
                    "ok": False,
                    "stdout": "",
                    "stderr": "%s failed" % command,
                }

            with mock.patch(
                "noema_lab.core.reproducibility._git",
                side_effect=fake_git,
            ):
                snapshot = git_snapshot(root)

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["commit"], "abc123")
        self.assertIsNone(snapshot["dirty"])
        self.assertFalse(snapshot["status_available"])
        self.assertIn("branch", snapshot["command_errors"])
        self.assertIn("status", snapshot["command_errors"])
        self.assertIn("diff_stat", snapshot["command_errors"])

    def test_dirty_content_identity_records_git_failure(self):
        failure = {"ok": False, "stdout": "", "stderr": "identity unavailable"}
        with mock.patch(
            "noema_lab.core.reproducibility._git",
            return_value=failure,
        ):
            identity = _git_worktree_content_identity(Path.cwd())
        payload = identity["dirty_content_identity"]
        self.assertFalse(payload["available"])
        self.assertIn("tracked_diff", payload["errors"])
        self.assertIn("untracked_files", payload["errors"])

    def test_manifest_rejects_artifact_outside_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            inside = run_dir / "artifacts" / "value.json"
            inside.parent.mkdir()
            inside.write_text("{}", encoding="utf-8")
            producer_metrics_sha256 = canonical_json_sha256({})
            record = _artifact_manifest_record(
                "step",
                "inside",
                {"path": str(inside), "metadata": {}},
                run_dir,
                producer_metrics_sha256=producer_metrics_sha256,
            )
            self.assertEqual(record["relative_path"], "artifacts/value.json")
            self.assertEqual(
                record["producer_metrics_sha256"],
                producer_metrics_sha256,
            )

            outside = root / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside run directory"):
                _artifact_manifest_record(
                    "step",
                    "outside",
                    {"path": str(outside), "metadata": {}},
                    run_dir,
                    producer_metrics_sha256=producer_metrics_sha256,
                )


if __name__ == "__main__":
    unittest.main()
