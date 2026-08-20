import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.cli.main import main
from noema_lab.cli.progress import CliProgressRenderer


class _TerminalStream(io.StringIO):
    def isatty(self):
        return True


class _BrokenStream(io.StringIO):
    def write(self, _value):
        raise BrokenPipeError("closed consumer")


class _Clock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def _write_capture_recipe(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "name": "cli_progress_capture_smoke",
        "metadata": {"seed": 123},
        "dataset_capture": {
            "split": "train",
            "samples": 2,
            "taps": [
                {"id": "received_embedding", "from": "data.clip_embeddings"},
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
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class CliProgressRendererTests(unittest.TestCase):
    def test_interactive_stream_is_enabled_automatically_and_finishes_line(self):
        stream = _TerminalStream()
        with CliProgressRenderer(stream=stream, label="Dataset capture") as progress:
            progress.update(
                {
                    "percent": 10,
                    "phase": "preparing",
                    "message": "Preparing capture",
                    "completed_samples": 0,
                    "total_samples": 2,
                }
            )
            progress.update(
                {
                    "percent": 100,
                    "phase": "completed",
                    "completed_samples": 2,
                    "total_samples": 2,
                }
            )

        rendered = stream.getvalue()
        self.assertTrue(progress.enabled)
        self.assertIn("\rDataset capture", rendered)
        self.assertIn("2/2 samples", rendered)
        self.assertTrue(rendered.endswith("\n"))

    def test_redirected_stream_is_disabled_by_default(self):
        stream = io.StringIO()
        with CliProgressRenderer(stream=stream) as progress:
            progress.update({"percent": 50, "phase": "capturing"})
        self.assertFalse(progress.enabled)
        self.assertEqual(stream.getvalue(), "")

    def test_forced_redirected_progress_is_line_oriented(self):
        stream = io.StringIO()
        with CliProgressRenderer(
            stream=stream,
            enabled=True,
            clock=_Clock(100.0, 105.0, 110.0),
        ) as progress:
            progress.update(
                {
                    "percent": 1,
                    "phase": "preparing",
                    "completed_samples": 0,
                    "total_samples": 10,
                }
            )
            progress.update(
                {
                    "percent": 50,
                    "phase": "capturing",
                    "completed_samples": 5,
                    "total_samples": 10,
                }
            )
            progress.update(
                {
                    "percent": 100,
                    "phase": "completed",
                    "completed_samples": 10,
                    "total_samples": 10,
                }
            )
        rendered = stream.getvalue()
        self.assertNotIn("\r", rendered)
        self.assertEqual(len(rendered.splitlines()), 3)
        self.assertIn("[############------------]", rendered)
        self.assertIn("elapsed 5.0s", rendered)
        self.assertIn("1.0 samples/s", rendered)
        self.assertIn("ETA 5.0s", rendered)
        self.assertIn("100.0%", rendered)

    def test_progress_output_failures_do_not_break_the_workflow(self):
        progress = CliProgressRenderer(stream=_BrokenStream(), enabled=True)
        progress.update({"percent": 20, "phase": "capturing"})
        progress.close()
        self.assertFalse(progress.enabled)


class DatasetCaptureCliProgressTests(unittest.TestCase):
    def test_json_defaults_to_silent_stderr_even_for_a_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "recipe.yaml"
            _write_capture_recipe(recipe_path)
            stdout = io.StringIO()
            stderr = _TerminalStream()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "--workspace",
                        str(root / ".noema"),
                        "dataset-capture",
                        "run",
                        str(recipe_path),
                        "--out",
                        str(root / "capture"),
                        "--json",
                    ]
                )

            self.assertEqual(code, 0)
            json.loads(stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")

    def test_forced_progress_uses_stderr_and_keeps_json_stdout_pristine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "recipe.yaml"
            _write_capture_recipe(recipe_path)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "--workspace",
                        str(root / ".noema"),
                        "dataset-capture",
                        "run",
                        str(recipe_path),
                        "--out",
                        str(root / "capture"),
                        "--json",
                        "--progress",
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "captured")
            self.assertIn("Train capture", stderr.getvalue())
            self.assertIn("[", stderr.getvalue())
            self.assertNotIn("Train capture", stdout.getvalue())

    def test_no_progress_suppresses_progress_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "recipe.yaml"
            _write_capture_recipe(recipe_path)
            stdout = io.StringIO()
            stderr = _TerminalStream()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "--workspace",
                        str(root / ".noema"),
                        "dataset-capture",
                        "run",
                        str(recipe_path),
                        "--out",
                        str(root / "capture"),
                        "--json",
                        "--no-progress",
                    ]
                )

            self.assertEqual(code, 0)
            json.loads(stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")

    def test_human_summary_reports_complete_capture_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "recipe.yaml"
            _write_capture_recipe(recipe_path)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "--workspace",
                        str(root / ".noema"),
                        "dataset-capture",
                        "run",
                        str(recipe_path),
                        "--out",
                        str(root / "capture"),
                        "--no-progress",
                    ]
                )

            self.assertEqual(code, 0)
            rendered = stdout.getvalue()
            self.assertIn("records: 2/2", rendered)
            self.assertIn("runs: 1", rendered)
            self.assertIn("shards: 1", rendered)
            self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
