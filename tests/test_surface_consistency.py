from __future__ import annotations

import contextlib
import io
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

from noema_lab.cli.main import build_parser, main
from noema_lab.core.benchmark_run_evidence import (
    validate_benchmark_run_evidence_snapshots,
)
from noema_lab.core.executor import MAX_PARALLEL_WORKERS
from noema_lab.ui.server import (
    _is_client_disconnect,
    _make_handler,
    start_ui_server_in_thread,
)


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _random_bits_recipe(name: str = "surface_consistency"):
    return {
        "schema_version": 1,
        "name": name,
        "steps": [
            {
                "id": "data",
                "op": "source.random_bits",
                "params": {"bit_count": 16, "seed": 7},
            }
        ],
    }


class CliSurfaceConsistencyTests(unittest.TestCase):
    def test_benchmark_run_exposes_the_complete_execution_policy(self):
        args = build_parser().parse_args(
            [
                "benchmark",
                "run",
                "benchmark.yaml",
                "--strict-lint",
                "--backend",
                "numpy",
                "--implementation",
                "default",
                "--parallel-workers",
                str(MAX_PARALLEL_WORKERS),
                "--no-plan-cache",
                "--json",
            ]
        )

        self.assertTrue(args.strict_lint)
        self.assertEqual(args.backend, "numpy")
        self.assertEqual(args.implementation, "default")
        self.assertEqual(args.parallel_workers, MAX_PARALLEL_WORKERS)
        self.assertTrue(args.no_plan_cache)
        self.assertTrue(args.json)

    def test_benchmark_run_applies_policy_to_each_recipe_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".noema"
            recipe_path = root / "recipe.yaml"
            recipe_path.write_text(
                yaml.safe_dump(
                    _random_bits_recipe("benchmark_policy_recipe"),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            pack_path = root / "benchmark.yaml"
            pack_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "id": "execution_policy_benchmark",
                        "version": "1",
                        "dataset": {},
                        "task": {},
                        "metrics": [],
                        "recipes": [
                            {
                                "id": "candidate",
                                "path": str(recipe_path),
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "--workspace",
                        str(workspace),
                        "benchmark",
                        "run",
                        str(pack_path),
                        "--backend",
                        "numpy",
                        "--implementation",
                        "default",
                        "--parallel-workers",
                        "2",
                        "--no-plan-cache",
                        "--json",
                    ]
                )
            response = json.loads(stdout.getvalue())
            result = json.loads(
                (
                    workspace
                    / "benchmarks"
                    / response["result_id"]
                    / "result.json"
                ).read_text(encoding="utf-8")
            )
            result_dir = (
                workspace / "benchmarks" / response["result_id"]
            )
            summary = validate_benchmark_run_evidence_snapshots(
                result_dir,
                result,
            )["entries"][0]["summary"]
            backing_run_pruned = not (
                workspace / "runs" / result["recipes"][0]["run_id"]
            ).exists()

        self.assertEqual(code, 0)
        self.assertEqual(response["status"], "completed")
        self.assertEqual(
            result["execution"],
            {
                "strict_lint": False,
                "backend": "numpy",
                "implementation": "default",
                "parallel_workers": 2,
                "use_plan_cache": False,
            },
        )
        self.assertTrue(backing_run_pruned)
        self.assertEqual(summary["execution"]["parallel_workers"], 2)
        self.assertEqual(summary["execution_plan"]["cache"]["outcome"], "bypass")
        self.assertEqual(
            summary["steps"][0]["execution_binding"]["implementation"],
            "default",
        )

    def test_runs_list_is_read_only_and_missing_workspace_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "mistyped-workspace"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                missing_code = main(
                    ["--workspace", str(missing), "runs", "list"]
                )

            workspace = root / "existing-workspace"
            runs_dir = workspace / "runs"
            run_dir = runs_dir / "run-one"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-one",
                        "status": "completed",
                        "recipe_name": "read_only_fixture",
                        "recipe": {"suite": {}},
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                list_code = main(
                    ["--workspace", str(workspace), "runs", "list", "--json"]
                )

        self.assertEqual(missing_code, 1)
        self.assertIn("Workspace does not exist", stderr.getvalue())
        self.assertFalse(missing.exists())
        self.assertEqual(list_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["runs"][0]["run_id"], "run-one")
        self.assertFalse((runs_dir / ".run-list-index.json").exists())
        self.assertFalse((workspace / "benchmarks").exists())

    def test_json_failure_is_one_typed_envelope_with_command_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stdout = io.StringIO()
            stderr = io.StringIO()
            missing_recipe = root / "missing-recipe.yaml"
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                code = main(
                    [
                        "dataset-capture",
                        "run",
                        str(missing_recipe),
                        "--out",
                        str(root / "capture"),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(code, 1)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(payload["status"], "error")
        self.assertTrue(payload["error"]["type"])
        self.assertIn("missing-recipe.yaml", payload["error"]["message"])
        self.assertEqual(payload["error"]["context"]["command"], "dataset-capture")
        self.assertEqual(payload["error"]["context"]["subcommand"], "run")
        self.assertEqual(payload["error"]["context"]["path"], str(missing_recipe))

    def test_malformed_discovered_benchmark_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "packs"
            directory.mkdir()
            broken = directory / "broken.yaml"
            broken.write_text(
                "schema_version: 1\nid: broken\nrecipes: not-a-list\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "benchmark",
                        "list",
                        "--directory",
                        str(directory),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(code, 1)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["type"], "BenchmarkError")
        self.assertIn(str(broken), payload["error"]["message"])
        self.assertIn("non-empty recipes list", payload["error"]["message"])

    def test_graph_uses_the_same_strict_schema_gate_as_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe_path = Path(tmp) / "unknown-field.yaml"
            payload = _random_bits_recipe()
            payload["unknown_root_field"] = True
            recipe_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["recipe", "graph", str(recipe_path)])

        self.assertEqual(code, 1)
        self.assertIn("unknown_root_field", stderr.getvalue())


class HttpSurfaceConsistencyTests(unittest.TestCase):
    def test_health_contract_and_run_api_apply_complete_execution_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                Path(tmp) / ".noema",
                ROOT,
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                _, health = _get_json(base + "/api/health")
                controls = health["execution_controls"]
                self.assertEqual(
                    controls["parallel_workers"]["maximum"],
                    MAX_PARALLEL_WORKERS,
                )
                execution = {
                    "strict_lint": False,
                    "backend": "numpy",
                    "implementation": "default",
                    "parallel_workers": 2,
                    "use_plan_cache": False,
                }
                status, run = _post_json(
                    base + "/api/recipe/run-payload",
                    {
                        "recipe": _random_bits_recipe("http_full_policy"),
                        "execution": execution,
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(run["execution"], execution)
                _, summary = _get_json(
                    base
                    + "/api/runs/"
                    + urllib.parse.quote(run["run_id"])
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual(summary["execution"]["parallel_workers"], 2)
        self.assertEqual(summary["execution_plan"]["cache"]["outcome"], "bypass")
        self.assertEqual(
            summary["steps"][0]["execution_binding"]["backend"],
            "numpy",
        )
        self.assertEqual(
            summary["steps"][0]["execution_binding"]["implementation"],
            "default",
        )

    def test_recipe_list_and_get_reject_unknown_fields_strictly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_dir = root / "recipes"
            recipe_dir.mkdir()
            recipe_path = recipe_dir / "bad.yaml"
            payload = _random_bits_recipe("bad_ui_recipe")
            payload["unknown_root_field"] = True
            recipe_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                root / ".noema",
                root,
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                _, listing = _get_json(base + "/api/recipes")
                request_url = (
                    base
                    + "/api/recipe?path="
                    + urllib.parse.quote("recipes/bad.yaml")
                )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    _get_json(request_url)
                error = json.loads(
                    raised.exception.read().decode("utf-8")
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual(listing["recipes"][0]["status"], "error")
        self.assertIn(
            "unknown_root_field",
            listing["recipes"][0]["description"],
        )
        self.assertEqual(raised.exception.code, 400)
        self.assertIn("unknown_root_field", error["error"])

    def test_benchmark_job_api_applies_the_same_execution_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".noema"
            recipe_dir = root / "recipes"
            benchmark_dir = root / "benchmarks"
            recipe_dir.mkdir()
            benchmark_dir.mkdir()
            recipe_path = recipe_dir / "policy.yaml"
            recipe_path.write_text(
                yaml.safe_dump(
                    _random_bits_recipe("http_benchmark_policy"),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            pack_path = benchmark_dir / "policy.yaml"
            pack_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "id": "http_execution_policy",
                        "version": "1",
                        "dataset": {},
                        "task": {},
                        "metrics": [],
                        "recipes": [
                            {
                                "id": "candidate",
                                "path": str(recipe_path),
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                workspace,
                root,
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            execution = {
                "strict_lint": False,
                "backend": "numpy",
                "implementation": "default",
                "parallel_workers": 2,
                "use_plan_cache": False,
            }
            try:
                _, job = _post_json(
                    base + "/api/benchmark-jobs",
                    {
                        "path": "benchmarks/policy.yaml",
                        "execution": execution,
                    },
                )
                deadline = time.monotonic() + 20
                while job["status"] not in {
                    "completed",
                    "incomplete",
                    "failed",
                    "resource_exhausted",
                }:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.05)
                    _, job = _get_json(
                        base
                        + "/api/benchmark-jobs/"
                        + urllib.parse.quote(job["job_id"])
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            result = json.loads(
                (
                    workspace
                    / "benchmarks"
                    / job["result_id"]
                    / "result.json"
                ).read_text(encoding="utf-8")
            )
            result_dir = workspace / "benchmarks" / job["result_id"]
            summary = validate_benchmark_run_evidence_snapshots(
                result_dir,
                result,
            )["entries"][0]["summary"]
            backing_run_pruned = not (
                workspace / "runs" / result["recipes"][0]["run_id"]
            ).exists()

        self.assertEqual(job["status"], "completed", job.get("error"))
        self.assertEqual(job["execution"], execution)
        self.assertEqual(result["execution"], execution)
        self.assertEqual(summary["execution"]["parallel_workers"], 2)
        self.assertEqual(summary["execution_plan"]["cache"]["outcome"], "bypass")
        self.assertTrue(backing_run_pruned)


class ServerDisconnectTests(unittest.TestCase):
    def test_disconnect_detection_covers_common_closed_socket_errors(self):
        self.assertTrue(_is_client_disconnect(BrokenPipeError()))
        self.assertTrue(_is_client_disconnect(ConnectionResetError()))
        self.assertFalse(_is_client_disconnect(ValueError("not a socket error")))

    def test_get_disconnect_does_not_attempt_exception_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            handler_type = _make_handler(Path(tmp) / ".noema", ROOT)
            handler = object.__new__(handler_type)
            handler.path = "/api/health"
            handler.headers = {}
            handler.close_connection = False
            exception_calls = []

            def disconnected_json(*_args, **_kwargs):
                raise BrokenPipeError("browser closed")

            handler._json = disconnected_json
            handler._exception = lambda exc: exception_calls.append(exc)

            handler.do_GET()

        self.assertTrue(handler.close_connection)
        self.assertEqual(exception_calls, [])

    def test_json_response_swallows_disconnect_without_nested_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            handler_type = _make_handler(Path(tmp) / ".noema", ROOT)
            handler = object.__new__(handler_type)
            handler.close_connection = False
            handler._response_bytes = (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    BrokenPipeError("browser closed")
                )
            )

            result = handler_type._json(handler, {"status": "ok"})

        self.assertIsNone(result)
        self.assertTrue(handler.close_connection)


if __name__ == "__main__":
    unittest.main()
