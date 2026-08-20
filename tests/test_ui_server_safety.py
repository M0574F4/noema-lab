from __future__ import annotations

import http.client
import json
import tempfile
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from noema_lab.core.plan_cache import ExecutionPlanCache
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.ops import build_registry
from noema_lab.ui.server import (
    RunJob,
    _MAX_JSON_REQUEST_BYTES,
    start_ui_server_in_thread,
)


def _post_json(url: str, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(
        urllib.request.urlopen(request, timeout=5).read().decode("utf-8")
    )


def _capture_recipe():
    return {
        "schema_version": 1,
        "name": "bounded_capture",
        "dataset_capture": {
            "split": "train",
            "samples": 1,
            "taps": [{"id": "bits", "from": "data.bits"}],
        },
        "steps": [
            {
                "id": "data",
                "op": "source.random_bits",
                "params": {"bit_count": 8},
            }
        ],
    }


class UiServerSafetyTests(unittest.TestCase):
    def test_artifact_endpoints_reject_outside_files_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".noema"
            managed = workspace / "runs" / "example" / "artifacts"
            managed.mkdir(parents=True)
            secret = root / "secret.json"
            secret.write_text('{"secret":true}', encoding="utf-8")
            link = managed / "escape.json"
            link.symlink_to(secret)
            safe = managed / "safe.json"
            safe.write_text('{"safe":true}', encoding="utf-8")
            server, thread = start_ui_server_in_thread(
                "127.0.0.1", 0, workspace, root
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                preview = json.loads(
                    urllib.request.urlopen(
                        base
                        + "/api/artifact?path="
                        + urllib.parse.quote(str(safe)),
                        timeout=5,
                    )
                    .read()
                    .decode("utf-8")
                )
                self.assertTrue(preview["payload"]["safe"])
                for path in (secret, link):
                    with self.subTest(path=path), self.assertRaises(
                        urllib.error.HTTPError
                    ) as raised:
                        urllib.request.urlopen(
                            base
                            + "/api/artifact?path="
                            + urllib.parse.quote(str(path)),
                            timeout=5,
                        )
                    self.assertEqual(raised.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

    def test_json_body_limit_and_malformed_benchmark_are_client_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".noema"
            server, thread = start_ui_server_in_thread(
                "127.0.0.1", 0, workspace, root
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                connection = http.client.HTTPConnection(host, port, timeout=5)
                connection.putrequest("POST", "/api/recipe/validate")
                connection.putheader(
                    "Content-Length", str(_MAX_JSON_REQUEST_BYTES + 1)
                )
                connection.putheader("Content-Type", "application/json")
                connection.endheaders()
                response = connection.getresponse()
                body = json.loads(response.read().decode("utf-8"))
                connection.close()
                self.assertEqual(response.status, 413)
                self.assertIn("byte limit", body["error"])

                with self.assertRaises(urllib.error.HTTPError) as raised:
                    _post_json(
                        base + "/api/benchmark-jobs",
                        {"path": "missing-pack.yaml"},
                    )
                self.assertEqual(raised.exception.code, 400)
                error = json.loads(raised.exception.read().decode("utf-8"))
                self.assertNotIn("traceback", error)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

    def test_inline_capture_is_contained_and_force_requires_owned_dataset(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as external:
            root = Path(tmp)
            workspace = root / ".noema"
            unowned = workspace / "dataset_captures" / "unowned"
            unowned.mkdir(parents=True)
            (unowned / "keep.txt").write_text("keep", encoding="utf-8")
            server, thread = start_ui_server_in_thread(
                "127.0.0.1", 0, workspace, root
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                cases = (
                    {
                        "recipe": _capture_recipe(),
                        "out": str(Path(external) / "capture"),
                    },
                    {
                        "recipe": _capture_recipe(),
                        "out": str(unowned),
                        "force": True,
                    },
                )
                for payload in cases:
                    with self.subTest(payload=payload), self.assertRaises(
                        urllib.error.HTTPError
                    ) as raised:
                        _post_json(
                            base + "/api/dataset-capture-jobs", payload
                        )
                    self.assertEqual(raised.exception.code, 400)
                self.assertEqual(
                    (unowned / "keep.txt").read_text(encoding="utf-8"),
                    "keep",
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

    def test_run_job_event_cursor_returns_bounded_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = RunJob(
                recipe_from_dict(
                    {
                        "schema_version": 1,
                        "name": "cursor",
                        "steps": [
                            {
                                "id": "data",
                                "op": "source.random_bits",
                                "params": {"bit_count": 8},
                            }
                        ],
                    }
                ),
                build_registry(),
                LocalStore(root / ".noema"),
                ExecutionPlanCache(),
                {},
                root,
            )
            job.add_event("one", "one")
            job.add_event("two", "two")
            job.add_event("three", "three")
            first = job.snapshot((0, 2))
            second = job.snapshot((first["next_event_seq"], 2))
            self.assertEqual([row["seq"] for row in first["events"]], [1, 2])
            self.assertTrue(first["events_truncated"])
            self.assertEqual([row["seq"] for row in second["events"]], [3])
            self.assertFalse(second["events_truncated"])


if __name__ == "__main__":
    unittest.main()
