from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional


DOWNLOAD_CHUNK_BYTES = 1024 * 1024


class DownloadSecurityError(RuntimeError):
    """Raised when a remote asset violates a declared download boundary."""


def download_verified_https(
    url: str,
    target: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
    timeout_s: float,
    expected_size: Optional[int] = None,
    opener: Optional[Callable[..., Any]] = None,
) -> int:
    """Download one HTTPS asset through a bounded, private, atomic temp file.

    The caller must bind the bytes to a SHA-256 digest.  ``max_bytes`` is
    enforced while streaming, including when a server omits or lies about its
    Content-Length header.  The final path is installed only after every check
    succeeds.
    """

    parsed = urllib.parse.urlsplit(str(url or ""))
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise DownloadSecurityError("remote asset URL must use HTTPS")
    digest_text = str(expected_sha256 or "").strip().lower()
    if len(digest_text) != 64 or any(
        char not in "0123456789abcdef" for char in digest_text
    ):
        raise DownloadSecurityError(
            "remote asset requires a 64-character hexadecimal SHA-256"
        )
    byte_limit = int(max_bytes)
    if byte_limit <= 0:
        raise DownloadSecurityError("remote asset byte limit must be positive")
    exact_size = None if expected_size is None else int(expected_size)
    if exact_size is not None and (exact_size <= 0 or exact_size > byte_limit):
        raise DownloadSecurityError(
            "remote asset expected size must be positive and within its byte limit"
        )

    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        raise DownloadSecurityError(
            "refusing to overwrite an existing remote-asset destination: %s"
            % destination
        )

    fd, temporary_name = tempfile.mkstemp(
        prefix=".%s." % destination.name,
        suffix=".part",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    open_url = opener or urllib.request.urlopen
    written = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as output:
            fd = -1
            with open_url(url, timeout=timeout_s) as response:
                final_url = getattr(response, "geturl", lambda: url)()
                if isinstance(final_url, str):
                    final = urllib.parse.urlsplit(final_url)
                    if final.scheme.lower() != "https" or not final.hostname:
                        raise DownloadSecurityError(
                            "remote asset redirect must remain on HTTPS"
                        )
                headers = getattr(response, "headers", None)
                raw_length = headers.get("Content-Length") if headers is not None else None
                if raw_length not in (None, ""):
                    try:
                        declared_length = int(raw_length)
                    except (TypeError, ValueError) as exc:
                        raise DownloadSecurityError(
                            "remote asset returned an invalid Content-Length"
                        ) from exc
                    if declared_length < 0 or declared_length > byte_limit:
                        raise DownloadSecurityError(
                            "remote asset Content-Length exceeds the %d-byte limit"
                            % byte_limit
                        )
                    if exact_size is not None and declared_length != exact_size:
                        raise DownloadSecurityError(
                            "remote asset Content-Length mismatch: expected %d, got %d"
                            % (exact_size, declared_length)
                        )
                while True:
                    remaining = byte_limit - written
                    chunk = response.read(min(DOWNLOAD_CHUNK_BYTES, remaining + 1))
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > byte_limit:
                        raise DownloadSecurityError(
                            "remote asset exceeded the %d-byte download limit"
                            % byte_limit
                        )
                    output.write(chunk)
                    digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if written <= 0:
            raise DownloadSecurityError("remote asset download was empty")
        if exact_size is not None and written != exact_size:
            raise DownloadSecurityError(
                "remote asset size mismatch: expected %d bytes, got %d"
                % (exact_size, written)
            )
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != digest_text:
            raise DownloadSecurityError(
                "remote asset SHA-256 mismatch: expected %s, got %s"
                % (digest_text, actual_sha256)
            )
        try:
            # A same-directory hard link is an atomic no-clobber publish: unlike
            # os.replace(), it fails if another worker created the destination
            # after our initial check.
            os.link(str(temporary), str(destination))
        except FileExistsError as exc:
            raise DownloadSecurityError(
                "remote-asset destination appeared during download: %s"
                % destination
            ) from exc
        except OSError as exc:
            raise DownloadSecurityError(
                "could not atomically publish remote asset %s: %s"
                % (destination, exc)
            ) from exc
        temporary.unlink()
        return written
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
