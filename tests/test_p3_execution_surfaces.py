import contextlib
import io
import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.cli.main import build_parser
from noema_lab.core.executor import MAX_PARALLEL_WORKERS
from noema_lab.ui.server import (
    _execution_options_from_request,
    start_ui_server_in_thread,
)


def _post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _get_json(url, *, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _recipe_payload(name="p3_execution_surface"):
    return {
        "schema_version": 1,
        "name": name,
        "steps": [
            {
                "id": "left",
                "op": "source.random_bits",
                "params": {"bit_count": 16, "seed": 11},
            },
            {
                "id": "right",
                "op": "source.random_bits",
                "params": {"bit_count": 16, "seed": 17},
            },
        ],
    }


class P3CliExecutionSurfaceTests(unittest.TestCase):
    def test_recipe_run_and_matrix_expose_bounded_execution_controls(self):
        parser = build_parser()
        run = parser.parse_args(
            [
                "recipe",
                "run",
                "recipe.yaml",
                "--parallel-workers",
                "4",
                "--no-plan-cache",
            ]
        )
        self.assertEqual(run.parallel_workers, 4)
        self.assertTrue(run.no_plan_cache)

        matrix = parser.parse_args(["recipe", "run-matrix", "recipe.yaml"])
        self.assertEqual(matrix.parallel_workers, 1)
        self.assertFalse(matrix.no_plan_cache)

    def test_parallel_worker_cli_bound_is_rejected_by_argparse(self):
        parser = build_parser()
        for value in ("0", str(MAX_PARALLEL_WORKERS + 1), "true"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parser.parse_args(
                        [
                            "recipe",
                            "run",
                            "recipe.yaml",
                            "--parallel-workers",
                            value,
                        ]
                    )
            self.assertEqual(raised.exception.code, 2)


class P3ApiExecutionSurfaceTests(unittest.TestCase):
    def test_execution_options_are_strict_and_normalized(self):
        self.assertEqual(
            _execution_options_from_request({}),
            {
                "strict_lint": False,
                "backend": None,
                "implementation": None,
                "parallel_workers": 1,
                "use_plan_cache": True,
            },
        )
        self.assertEqual(
            _execution_options_from_request(
                {
                    "execution": {
                        "parallel_workers": 3,
                        "use_plan_cache": False,
                    }
                }
            ),
            {
                "strict_lint": False,
                "backend": None,
                "implementation": None,
                "parallel_workers": 3,
                "use_plan_cache": False,
            },
        )
        invalid = [
            [],
            {"execution": None},
            {"execution": {"parallel_workers": True}},
            {"execution": {"parallel_workers": 1.0}},
            {"execution": {"parallel_workers": 0}},
            {"execution": {"parallel_workers": MAX_PARALLEL_WORKERS + 1}},
            {"execution": {"use_plan_cache": 1}},
            {"execution": {"strict_lint": 1}},
            {"execution": {"backend": ""}},
            {"execution": {"implementation": False}},
            {"execution": {"unknown": True}},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                _execution_options_from_request(payload)

    def test_sync_and_job_apis_apply_and_report_execution_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                Path(tmp),
                ROOT,
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                execution = {
                    "strict_lint": False,
                    "backend": None,
                    "implementation": None,
                    "parallel_workers": 2,
                    "use_plan_cache": False,
                }
                status, run = _post_json(
                    base + "/api/recipe/run-payload",
                    {"recipe": _recipe_payload(), "execution": execution},
                )
                self.assertEqual(status, 200)
                self.assertEqual(run["execution"], execution)
                summary = _get_json(
                    base + "/api/runs/" + urllib.parse.quote(run["run_id"])
                )
                self.assertEqual(summary["execution"]["parallel_workers"], 2)
                self.assertEqual(summary["execution"]["mode"], "parallel")
                self.assertEqual(
                    summary["execution_plan"]["cache"]["outcome"],
                    "bypass",
                )

                status, job = _post_json(
                    base + "/api/run-jobs",
                    {
                        "recipe": _recipe_payload("p3_execution_job"),
                        "execution": execution,
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(job["execution"], execution)
                job = _get_json(
                    base
                    + "/api/run-jobs/"
                    + urllib.parse.quote(job["job_id"])
                    + "/wait?timeout_seconds=30",
                    timeout=35,
                )
                self.assertEqual(job["status"], "completed", job.get("error"))
                self.assertEqual(job["execution"], execution)
                self.assertEqual(job["resource_guard"]["isolation"], "subprocess_process_group")
                guarded_summary = _get_json(
                    base + "/api/runs/" + urllib.parse.quote(job["run_id"])
                )
                self.assertEqual(
                    guarded_summary["resource_guard"]["kind"],
                    "noema.execution_resource_guard",
                )

                cache_outcomes = []
                isolated_worker_pids = []
                for _ in range(2):
                    _status, cached_run = _post_json(
                        base + "/api/recipe/run-payload",
                        {"recipe": _recipe_payload("p3_shared_plan_cache")},
                    )
                    isolated_worker_pids.append(
                        cached_run["resource_guard"]["launcher_pid"]
                    )
                    cached_summary = _get_json(
                        base
                        + "/api/runs/"
                        + urllib.parse.quote(cached_run["run_id"])
                    )
                    cache_outcomes.append(
                        cached_summary["execution_plan"]["cache"]["outcome"]
                    )
                # Legacy synchronous routes now wait on the same guarded job
                # path as /api/run-jobs. Each request gets a fresh subprocess,
                # so an in-memory execution-plan cache cannot cross the safety
                # boundary and each worker truthfully records its own miss.
                self.assertEqual(cache_outcomes, ["miss", "miss"])
                self.assertEqual(len(set(isolated_worker_pids)), 2)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

    def test_path_recipe_api_applies_execution_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            recipe_path = project_root / "p3_path_recipe.json"
            recipe_path.write_text(
                json.dumps(_recipe_payload("p3_path_execution")),
                encoding="utf-8",
            )
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                project_root / ".noema",
                project_root,
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                status, run = _post_json(
                    base + "/api/recipe/run",
                    {
                        "path": recipe_path.name,
                        "execution": {
                            "parallel_workers": 2,
                            "use_plan_cache": False,
                        },
                    },
                )
                self.assertEqual(status, 200)
                summary = _get_json(
                    base + "/api/runs/" + urllib.parse.quote(run["run_id"])
                )
                self.assertEqual(
                    summary["execution"],
                    {"mode": "parallel", "parallel_workers": 2},
                )
                self.assertEqual(
                    summary["execution_plan"]["cache"]["outcome"],
                    "bypass",
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)

    def test_api_rejects_bool_parallel_workers_before_starting_a_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                Path(tmp),
                ROOT,
            )
            host, port = server.server_address
            try:
                request = urllib.request.Request(
                    "http://%s:%d/api/run-jobs" % (host, port),
                    data=json.dumps(
                        {
                            "recipe": _recipe_payload(),
                            "execution": {"parallel_workers": True},
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=10)
                self.assertEqual(raised.exception.code, 400)
                error = json.loads(raised.exception.read().decode("utf-8"))
                self.assertIn("must be an integer", error["error"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(1)


if __name__ == "__main__":
    unittest.main()
