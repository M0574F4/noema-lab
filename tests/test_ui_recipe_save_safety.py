import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error
import urllib.parse
import urllib.request

from noema_lab.ui.server import _atomic_write_text, start_ui_server_in_thread


def _post_json(url: str, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class UiRecipeSaveSafetyTests(unittest.TestCase):
    def test_save_reports_legacy_matrix_normalization_and_precedence_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                root / "workspace",
                project_root,
            )
            host, port = server.server_address
            try:
                status, response = _post_json(
                    "http://%s:%d/api/recipe/save" % (host, port),
                    {
                        "filename": "normalized.yaml",
                        "recipe": {
                            "schema_version": 1,
                            "name": "normalized_matrix",
                            "metadata": {
                                "sweeps": {"source.bit_count": "8,16"},
                                "ui_sweeps": {"source.bit_count": "32,64"},
                            },
                            "steps": [
                                {
                                    "id": "source",
                                    "op": "source.random_bits",
                                    "inputs": {},
                                    "params": {"bit_count": 8},
                                }
                            ],
                        },
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

            self.assertEqual(status, 200)
            diagnostic_codes = {
                item["code"] for item in response["validation"]["diagnostics"]
            }
            self.assertIn("legacy_sweep_precedence", diagnostic_codes)
            self.assertIn("legacy_sweep_normalized", diagnostic_codes)
            self.assertNotIn("sweeps", response["recipe"]["metadata"])
            self.assertNotIn("ui_sweeps", response["recipe"]["metadata"])

    def test_get_and_post_matrix_expansion_share_validation_and_400_errors(self):
        valid_recipe = {
            "schema_version": 1,
            "name": "matrix_api",
            "metadata": {
                "matrix": {
                    "dimensions": {"count": [8, 16]},
                    "step_params": {
                        "source": {"bit_count": {"matrix": "count"}}
                    },
                }
            },
            "steps": [
                {
                    "id": "source",
                    "op": "source.random_bits",
                    "inputs": {},
                    "params": {"bit_count": 8},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            recipes_dir = project_root / "recipes"
            recipes_dir.mkdir(parents=True)
            recipe_path = recipes_dir / "matrix.json"
            recipe_path.write_text(json.dumps(valid_recipe), encoding="utf-8")
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                root / "workspace",
                project_root,
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            get_url = base + "/api/recipe/matrix?path=" + urllib.parse.quote(
                "recipes/matrix.json"
            )
            try:
                get_status, get_response = _get_json(get_url)
                post_status, post_response = _post_json(
                    base + "/api/recipe/matrix/expand",
                    {"recipe": valid_recipe},
                )

                invalid_recipe = json.loads(json.dumps(valid_recipe))
                invalid_recipe["metadata"]["matrix"]["step_params"]["source"][
                    "bit_count"
                ] = {"matrix": "missing"}
                recipe_path.write_text(json.dumps(invalid_recipe), encoding="utf-8")
                invalid_get_status, invalid_get = _get_json(get_url)
                invalid_post_status, invalid_post = _post_json(
                    base + "/api/recipe/matrix/expand",
                    {"recipe": invalid_recipe},
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

            self.assertEqual(get_status, 200)
            self.assertEqual(post_status, 200)
            self.assertEqual(get_response, post_response)
            self.assertEqual(get_response["expanded_count"], 2)
            self.assertEqual(invalid_get_status, 400)
            self.assertEqual(invalid_post_status, 400)
            self.assertEqual(invalid_get["error"], invalid_post["error"])
            self.assertTrue(invalid_get["error"].startswith("Recipe matrix is invalid:"))

    def test_strict_save_rejects_unknown_fields_without_creating_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                root / "workspace",
                project_root,
            )
            host, port = server.server_address
            try:
                status, response = _post_json(
                    "http://%s:%d/api/recipe/save" % (host, port),
                    {
                        "filename": "typo.yaml",
                        "recipe": {
                            "schema_version": 1,
                            "name": "strict_save_typo",
                            "metdata": {"seed": 7},
                            "steps": [
                                {
                                    "id": "source",
                                    "op": "source.random_bits",
                                    "inputs": {},
                                    "params": {"bit_count": 8},
                                }
                            ],
                        },
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

            self.assertEqual(status, 400)
            self.assertEqual(response["status"], "error")
            self.assertIn("Unknown recipe field `metdata`", response["error"])
            self.assertFalse((project_root / "recipes" / "typo.yaml").exists())

    def test_invalid_recipe_is_rejected_before_existing_file_is_touched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            recipes_dir = project_root / "recipes"
            recipes_dir.mkdir(parents=True)
            recipe_path = recipes_dir / "existing.yaml"
            original = "sentinel: original\n"
            recipe_path.write_text(original, encoding="utf-8")

            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                root / "workspace",
                project_root,
            )
            host, port = server.server_address
            try:
                status, response = _post_json(
                    "http://%s:%d/api/recipe/save" % (host, port),
                    {
                        "filename": "existing.yaml",
                        "recipe": {
                            "schema_version": 1,
                            "name": "invalid_replacement",
                            "steps": [
                                {
                                    "id": "bad",
                                    "op": "operation.does_not_exist",
                                    "inputs": {},
                                    "params": {},
                                }
                            ],
                        },
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

            self.assertEqual(status, 400)
            self.assertEqual(response["status"], "error")
            self.assertIn("Recipe is invalid", response["error"])
            self.assertEqual(recipe_path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(recipes_dir.glob(".existing.yaml.*.tmp")), [])

    def test_atomic_write_replaces_content_and_preserves_existing_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipe.yaml"
            path.write_text("old\n", encoding="utf-8")
            path.chmod(0o640)

            _atomic_write_text(path, "new\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "new\n")
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            self.assertEqual(list(path.parent.glob(".recipe.yaml.*.tmp")), [])

    def test_atomic_write_failure_leaves_original_and_cleans_staging_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipe.yaml"
            path.write_text("old\n", encoding="utf-8")

            with mock.patch.object(Path, "replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    _atomic_write_text(path, "new\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "old\n")
            self.assertEqual(list(path.parent.glob(".recipe.yaml.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
