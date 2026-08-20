import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from noema_lab.ops.models import eflic


class EfLicAssetSecurityTests(unittest.TestCase):
    def test_executable_download_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "EF-LIC"
            with self.assertRaisesRegex(RuntimeError, "downloads are disabled"):
                eflic._prepare_eflic_assets({"repo_path": str(repo)})

    def test_opt_in_download_requires_digest(self):
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "EF-LIC"
            with self.assertRaisesRegex(RuntimeError, "requires an expected SHA-256"):
                eflic._prepare_eflic_assets(
                    {"repo_path": str(repo), "auto_setup": True}
                )

    def test_opt_in_download_is_verified_before_atomic_install(self):
        payload = b"def model():\n    return None\n"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "EF-LIC"

            class Response(io.BytesIO):
                headers = {"Content-Length": str(len(payload))}

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    self.close()

                def geturl(self):
                    return "https://example.invalid/EF_LIC.py"

            with mock.patch.object(
                eflic, "urlopen", side_effect=lambda *_args, **_kwargs: Response(payload)
            ):
                eflic._prepare_eflic_assets(
                    {
                        "repo_path": str(repo),
                        "auto_setup": True,
                        "expected_model_sha256": digest,
                    }
                )
            self.assertEqual((repo / "EF_LIC.py").read_bytes(), payload)
            self.assertFalse((repo / "EF_LIC.py.part").exists())

    def test_existing_model_is_rejected_on_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "EF-LIC"
            repo.mkdir()
            (repo / "EF_LIC.py").write_text("def model(): pass\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                eflic._prepare_eflic_assets(
                    {
                        "repo_path": str(repo),
                        "expected_model_sha256": "0" * 64,
                    }
                )


if __name__ == "__main__":
    unittest.main()
