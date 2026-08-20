from __future__ import annotations

import hashlib
import io
from pathlib import Path
import tempfile
import unittest

from noema_lab.core.downloads import DownloadSecurityError, download_verified_https


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, *, final_url: str, content_length=None):
        super().__init__(payload)
        self._final_url = final_url
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def geturl(self):
        return self._final_url


class SecureDownloadTests(unittest.TestCase):
    def test_verified_download_is_atomic_and_exact(self):
        payload = b"reviewed asset bytes"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "asset.bin"
            size = download_verified_https(
                "https://example.invalid/asset.bin",
                target,
                expected_sha256=digest,
                expected_size=len(payload),
                max_bytes=len(payload),
                timeout_s=1,
                opener=lambda *_args, **_kwargs: _Response(
                    payload,
                    final_url="https://cdn.example.invalid/asset.bin",
                    content_length=len(payload),
                ),
            )
            self.assertEqual(size, len(payload))
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(list(target.parent.glob("*.part")), [])

    def test_stream_without_content_length_is_still_bounded(self):
        payload = b"x" * 33
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "asset.bin"
            with self.assertRaisesRegex(DownloadSecurityError, "download limit"):
                download_verified_https(
                    "https://example.invalid/asset.bin",
                    target,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    max_bytes=32,
                    timeout_s=1,
                    opener=lambda *_args, **_kwargs: _Response(
                        payload,
                        final_url="https://example.invalid/asset.bin",
                    ),
                )
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.glob("*.part")), [])

    def test_https_download_rejects_downgrade_redirect(self):
        payload = b"asset"
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "asset.bin"
            with self.assertRaisesRegex(DownloadSecurityError, "remain on HTTPS"):
                download_verified_https(
                    "https://example.invalid/asset.bin",
                    target,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    max_bytes=len(payload),
                    timeout_s=1,
                    opener=lambda *_args, **_kwargs: _Response(
                        payload,
                        final_url="http://example.invalid/asset.bin",
                    ),
                )
            self.assertFalse(target.exists())

    def test_existing_or_symlink_destination_is_never_followed(self):
        payload = b"asset"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            victim = root / "victim.bin"
            victim.write_bytes(b"do not overwrite")
            link = root / "asset.bin"
            try:
                link.symlink_to(victim)
            except OSError as exc:
                self.skipTest("symbolic links are unavailable: %s" % exc)
            with self.assertRaisesRegex(DownloadSecurityError, "refusing to overwrite"):
                download_verified_https(
                    "https://example.invalid/asset.bin",
                    link,
                    expected_sha256=digest,
                    max_bytes=len(payload),
                    timeout_s=1,
                    opener=lambda *_args, **_kwargs: _Response(
                        payload,
                        final_url="https://example.invalid/asset.bin",
                    ),
                )
            self.assertEqual(victim.read_bytes(), b"do not overwrite")

    def test_destination_created_during_download_is_not_overwritten(self):
        payload = b"downloaded"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "asset.bin"

            def racing_opener(*_args, **_kwargs):
                target.write_bytes(b"concurrent owner")
                return _Response(
                    payload,
                    final_url="https://example.invalid/asset.bin",
                )

            with self.assertRaisesRegex(DownloadSecurityError, "appeared during"):
                download_verified_https(
                    "https://example.invalid/asset.bin",
                    target,
                    expected_sha256=digest,
                    max_bytes=len(payload),
                    timeout_s=1,
                    opener=racing_opener,
                )
            self.assertEqual(target.read_bytes(), b"concurrent owner")
            self.assertEqual(list(target.parent.glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
