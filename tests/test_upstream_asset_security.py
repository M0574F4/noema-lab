import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from noema_lab.ops.models import upstream_lic


class _Context:
    step_id = "asset_setup"

    def report_progress(self, *_args, **_kwargs):
        return None


class UpstreamAssetSecurityTests(unittest.TestCase):
    def test_pinned_repository_rejects_dirty_or_untracked_python_sources(self):
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "source"
            revision = _create_git_repo(repo)
            (repo / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "tracked changes"):
                upstream_lic._require_repo_revision(repo, revision)
            (repo / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
            (repo / "shadow.py").write_text("VALUE = 3\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "untracked executable"):
                upstream_lic._require_repo_revision(repo, revision)

    def test_clone_uses_private_staging_without_deleting_legacy_part_path(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            revision = _create_git_repo(source)
            target = root / "checkout"
            legacy_part = root / "checkout.part"
            legacy_part.mkdir()
            sentinel = legacy_part / "owned-by-another-process"
            sentinel.write_text("keep", encoding="utf-8")

            upstream_lic._clone_repo(str(source), target, 30, revision)

            self.assertEqual((target / "model.py").read_text(encoding="utf-8"), "VALUE = 1\n")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual(list(root.glob(".checkout.*.clone-part")), [])

    def test_auto_setup_is_disabled_by_default(self):
        with mock.patch.object(upstream_lic, "_clone_repo") as clone, mock.patch.object(
            upstream_lic, "_download_checkpoint"
        ) as download:
            upstream_lic._prepare_upstream_assets(
                _Context(),
                "tcm",
                {"repo_path": "", "checkpoint": ""},
            )
        clone.assert_not_called()
        download.assert_not_called()

    def test_repository_fetch_requires_full_commit(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(RuntimeError, "pinned repo_revision"):
                upstream_lic._prepare_upstream_assets(
                    _Context(),
                    "tcm",
                    {
                        "auto_setup": True,
                        "repo_path": str(Path(raw) / "repo"),
                        "repo_url": "https://example.invalid/repo.git",
                        "checkpoint": "",
                    },
                )

    def test_checkpoint_download_requires_digest(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(RuntimeError, "expected_checkpoint_sha256"):
                upstream_lic._prepare_upstream_assets(
                    _Context(),
                    "tcm",
                    {
                        "auto_setup": True,
                        "repo_path": "",
                        "checkpoint": str(Path(raw) / "model.pth"),
                        "checkpoint_url": "https://example.invalid/model.pth",
                    },
                )

    def test_downloaded_checkpoint_is_verified_before_install(self):
        payload = b"reviewed checkpoint bytes"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "model.pth"

            def fake_download(_url, path, _timeout, **_kwargs):
                Path(path).write_bytes(payload)

            with mock.patch.object(
                upstream_lic, "_download_file", side_effect=fake_download
            ):
                upstream_lic._download_checkpoint(
                    "https://example.invalid/model.pth",
                    target,
                    30,
                    expected_sha256=digest,
                )
            self.assertEqual(target.read_bytes(), payload)


def _create_git_repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "model.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Noema test",
            "-c",
            "user.email=noema-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    unittest.main()
