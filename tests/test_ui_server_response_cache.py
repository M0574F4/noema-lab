from __future__ import annotations

import gzip
import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from noema_lab.ui.server import (
    _accepts_gzip,
    _content_addressed_static_version,
    start_ui_server_in_thread,
)

ROOT = Path(__file__).resolve().parents[1]


class _Registry:
    def __init__(self) -> None:
        self.describe_calls = 0

    def describe(self):
        self.describe_calls += 1
        return [
            {
                "id": "test.operation",
                "name": "Test operation",
                "description": "x" * 4096,
            }
        ]


class UiServerResponseCacheTests(unittest.TestCase):
    def test_accept_encoding_respects_explicit_gzip_disable(self):
        self.assertTrue(_accepts_gzip("br, gzip"))
        self.assertTrue(_accepts_gzip("*;q=0.5"))
        self.assertFalse(_accepts_gzip("gzip;q=0, *;q=1"))
        self.assertFalse(_accepts_gzip("br"))

    def test_only_full_content_hash_is_an_immutable_static_version(self):
        data = b"current asset"
        digest = hashlib.sha256(data).hexdigest()
        self.assertTrue(_content_addressed_static_version("v=" + digest, data))
        self.assertTrue(
            _content_addressed_static_version("sha256=sha256-" + digest, data)
        )
        self.assertFalse(
            _content_addressed_static_version("v=release-name", data)
        )
        self.assertFalse(
            _content_addressed_static_version("v=" + digest[:16], data)
        )

    def test_registry_payload_is_built_once_but_artifacts_are_rediscovered(self):
        registry = _Registry()
        artifact_discovery_calls = 0

        def discover(*args, **kwargs):
            nonlocal artifact_discovery_calls
            artifact_discovery_calls += 1
            return [{"id": "artifact-%d" % artifact_discovery_calls}]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("noema_lab.ui.server.build_registry", return_value=registry), mock.patch(
                "noema_lab.ui.server.discover_trained_artifacts",
                side_effect=discover,
            ):
                server, thread = start_ui_server_in_thread(
                    "127.0.0.1",
                    0,
                    root / ".noema",
                    root,
                )
                try:
                    status, headers, body = _request(
                        server,
                        "/api/ops",
                        {"Accept-Encoding": "gzip"},
                    )
                    self.assertEqual(status, 200)
                    self.assertEqual(headers.get("content-encoding"), "gzip")
                    self.assertIn("no-cache", headers.get("cache-control", ""))
                    payload = json.loads(gzip.decompress(body).decode("utf-8"))
                    self.assertEqual(payload["operations"][0]["id"], "test.operation")
                    etag = headers.get("etag")
                    self.assertTrue(etag)

                    status, headers, body = _request(
                        server,
                        "/api/ops",
                        {
                            "Accept-Encoding": "gzip",
                            "If-None-Match": str(etag),
                        },
                    )
                    self.assertEqual(status, 304)
                    self.assertEqual(body, b"")
                    self.assertEqual(registry.describe_calls, 1)

                    first = _request_json(server, "/api/trained-artifacts")
                    second = _request_json(server, "/api/trained-artifacts")
                    self.assertEqual(first["artifacts"][0]["id"], "artifact-1")
                    self.assertEqual(second["artifacts"][0]["id"], "artifact-2")
                    self.assertEqual(artifact_discovery_calls, 2)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

    def test_index_is_never_stored_and_static_assets_revalidate(self):
        registry = _Registry()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("noema_lab.ui.server.build_registry", return_value=registry):
                server, thread = start_ui_server_in_thread(
                    "127.0.0.1",
                    0,
                    root / ".noema",
                    root,
                )
                try:
                    status, headers, body = _request(server, "/")
                    self.assertEqual(status, 200)
                    self.assertIn("Noema", body.decode("utf-8"))
                    self.assertEqual(headers.get("cache-control"), "no-store, max-age=0")
                    self.assertIsNone(headers.get("etag"))

                    status, headers, body = _request(
                        server,
                        "/static/app.js",
                        {"Accept-Encoding": "gzip"},
                    )
                    self.assertEqual(status, 200)
                    self.assertEqual(headers.get("content-encoding"), "gzip")
                    self.assertEqual(
                        headers.get("cache-control"),
                        "private, no-cache, must-revalidate",
                    )
                    self.assertIn(b"const", gzip.decompress(body))
                    etag = str(headers.get("etag") or "")
                    digest = etag.removeprefix('W/"sha256-').removesuffix('"')
                    self.assertEqual(len(digest), 64)

                    status, headers, _ = _request(
                        server,
                        "/static/app.js?v=" + digest,
                    )
                    self.assertEqual(status, 200)
                    self.assertIn("immutable", headers.get("cache-control", ""))

                    status, headers, body = _request(
                        server,
                        "/static/app.js",
                        {"If-None-Match": etag},
                    )
                    self.assertEqual(status, 304)
                    self.assertEqual(body, b"")
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

    def test_recipe_template_endpoint_reinspects_project_files(self):
        source = ROOT / "recipes" / "text_semantic_utf8_clean.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template_path = root / "recipes" / source.name
            template_path.parent.mkdir(parents=True)
            template_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                root / ".noema",
                root,
            )
            try:
                first = _request_json(server, "/api/recipe-templates")
                first_row = next(
                    item
                    for item in first["templates"]
                    if item["recipe_path"] == "recipes/text_semantic_utf8_clean.yaml"
                )
                self.assertEqual(first_row["validation"]["status"], "valid")

                template_path.write_text("not: a valid recipe\n", encoding="utf-8")
                second = _request_json(server, "/api/recipe-templates")
                second_row = next(
                    item
                    for item in second["templates"]
                    if item["recipe_path"] == "recipes/text_semantic_utf8_clean.yaml"
                )
                self.assertEqual(second_row["validation"]["status"], "invalid")
                self.assertTrue(second_row["available"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


def _request(server, path: str, headers=None):
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=10)
    try:
        connection.request("GET", path, headers=dict(headers or {}))
        response = connection.getresponse()
        body = response.read()
        return (
            response.status,
            {key.lower(): value for key, value in response.getheaders()},
            body,
        )
    finally:
        connection.close()


def _request_json(server, path: str):
    status, headers, body = _request(server, path)
    if status != 200:
        raise AssertionError("GET %s returned %s" % (path, status))
    if headers.get("content-encoding") == "gzip":
        body = gzip.decompress(body)
    return json.loads(body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
