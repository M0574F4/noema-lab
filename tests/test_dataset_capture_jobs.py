import hashlib
import json
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.ui.server import start_ui_server_in_thread


def _capture_recipe_payload():
    return {
        "schema_version": 1,
        "name": "dataset_capture_job_smoke",
        "metadata": {"seed": 123},
        "dataset_capture": {
            "split": "validation",
            "samples": 2,
            "taps": [
                {"id": "received_embedding", "from": "data.clip_embeddings"},
                {"id": "target_mask", "from": "data.segmentation_reference"},
            ],
        },
        "steps": [
            {
                "id": "data",
                "op": "source.semantic_artifacts_smoke",
                "params": {"embedding_dim": 4},
            }
        ],
    }


def _get_json(url):
    return json.loads(urllib.request.urlopen(url, timeout=5).read().decode("utf-8"))


def _post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(request, timeout=5).read().decode("utf-8"))


class DatasetCaptureJobTests(unittest.TestCase):
    def test_background_capture_job_reports_progress_events_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".noema"
            recipe_path = root / "capture_recipe.yaml"
            recipe_path.write_text(
                yaml.safe_dump(_capture_recipe_payload(), sort_keys=False),
                encoding="utf-8",
            )
            capture_out = workspace / "dataset_captures" / "validation"
            server, thread = start_ui_server_in_thread("127.0.0.1", 0, workspace, root)
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                job = _post_json(
                    base + "/api/dataset-capture-jobs",
                    {
                        "path": str(recipe_path.relative_to(root)),
                        "out": str(capture_out.relative_to(root)),
                        "force": True,
                    },
                )
                self.assertIn(job["status"], {"queued", "running", "completed"})
                self.assertEqual(job["split"], "validation")
                self.assertEqual(job["progress"]["unit"], "samples")
                for _ in range(100):
                    if job["status"] not in {"queued", "running"}:
                        break
                    time.sleep(0.05)
                    job = _get_json(
                        base
                        + "/api/dataset-capture-jobs/"
                        + urllib.parse.quote(job["job_id"])
                    )
                self.assertEqual(job["status"], "completed", job.get("error"))
                self.assertEqual(job["progress"]["percent"], 100.0)
                self.assertEqual(job["progress"]["phase"], "completed")
                self.assertEqual(job["dataset_capture"]["captured_samples"], 2)
                self.assertTrue((capture_out / "schema.json").is_file())
                kinds = [item["kind"] for item in job["events"]]
                self.assertIn("capture_started", kinds)
                self.assertIn("step_completed", kinds)
                self.assertIn("job_completed", kinds)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

    def test_path_based_capture_job_reuses_ui_managed_output_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".noema"
            recipe_path = root / "capture_recipe.yaml"
            recipe_path.write_text(
                yaml.safe_dump(_capture_recipe_payload(), sort_keys=False),
                encoding="utf-8",
            )
            server, thread = start_ui_server_in_thread("127.0.0.1", 0, workspace, root)
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    _post_json(
                        base + "/api/dataset-capture-jobs",
                        {
                            "path": str(recipe_path.relative_to(root)),
                            "out": "unsafe_capture_output",
                            "force": True,
                        },
                    )
                self.assertEqual(raised.exception.code, 400)
                body = json.loads(raised.exception.read().decode("utf-8"))
                self.assertIn("Noema dataset_captures directory", body["error"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

    def test_manifest_backed_capture_uses_only_declared_bundle_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".noema"
            bundle = root / "training_bundle"
            bundle.mkdir()
            recipe_path = bundle / "capture_validation_recipe.yaml"
            recipe_path.write_text(
                yaml.safe_dump(_capture_recipe_payload(), sort_keys=False),
                encoding="utf-8",
            )
            capture_out = bundle / "data" / "validation"
            manifest = {
                "schema_version": 1,
                "kind": "noema.training_interface_bundle@1",
                "capture_jobs": [
                    {
                        "split": "validation",
                        "recipe_path": str(recipe_path.relative_to(root)),
                        "output_dir": str(capture_out.relative_to(root)),
                        "requested_samples": 2,
                        "expected_taps": _capture_recipe_payload()[
                            "dataset_capture"
                        ]["taps"],
                        "recipe_file_sha256": hashlib.sha256(
                            recipe_path.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
            manifest_path = bundle / "project_manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
            )
            server, thread = start_ui_server_in_thread(
                "127.0.0.1", 0, workspace, root
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                job = _post_json(
                    base + "/api/dataset-capture-jobs",
                    {
                        "project_path": str(bundle.relative_to(root)),
                        "split": "validation",
                        "force": True,
                    },
                )
                for _ in range(100):
                    if job["status"] not in {"queued", "running"}:
                        break
                    time.sleep(0.05)
                    job = _get_json(
                        base
                        + "/api/dataset-capture-jobs/"
                        + urllib.parse.quote(job["job_id"])
                    )
                self.assertEqual(job["status"], "completed", job.get("error"))
                self.assertEqual(Path(job["out_dir"]), capture_out)
                self.assertTrue((capture_out / "schema.json").is_file())

                manifest["capture_jobs"][0]["output_dir"] = str(
                    (workspace / "dataset_captures" / "escaped").relative_to(root)
                )
                manifest_path.write_text(
                    yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
                )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    _post_json(
                        base + "/api/dataset-capture-jobs",
                        {
                            "project_path": str(bundle.relative_to(root)),
                            "split": "validation",
                            "force": True,
                        },
                    )
                self.assertEqual(raised.exception.code, 400)
                body = json.loads(raised.exception.read().decode("utf-8"))
                self.assertIn("inside exported project", body["error"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)


if __name__ == "__main__":
    unittest.main()
