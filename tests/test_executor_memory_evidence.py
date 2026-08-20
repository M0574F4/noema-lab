from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from noema_lab.core import executor as executor_module
from noema_lab.core.executor import LocalExecutor
from noema_lab.core.operations import OperationRegistry
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.ops.source.random_bits import RandomBitsOperation


def _single_step_recipe():
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "executor_memory_evidence",
            "steps": [
                {
                    "id": "source",
                    "op": "source.random_bits",
                    "inputs": {},
                    "params": {"bit_count": 8, "seed": 17},
                }
            ],
        }
    )


def _registry():
    registry = OperationRegistry()
    registry.register(RandomBitsOperation())
    return registry


class ExecutorMemoryEvidenceTests(unittest.TestCase):
    def _run_with_rss_probe(self, probe):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp))
            with mock.patch.object(
                executor_module,
                "_current_rss_bytes",
                side_effect=probe,
            ):
                run_dir = LocalExecutor(_registry(), store).run(_single_step_recipe())
            return store.get_run(run_dir.name)

    def test_procfs_read_failure_is_not_converted_to_zero(self):
        with mock.patch("builtins.open", side_effect=OSError("forced procfs read failure")):
            with self.assertRaisesRegex(OSError, "forced procfs read failure"):
                executor_module._current_rss_bytes()
        with mock.patch("builtins.open", mock.mock_open(read_data="1 0\n")):
            with self.assertRaisesRegex(ValueError, "non-positive resident-page"):
                executor_module._current_rss_bytes()

    def test_unavailable_probe_records_error_and_omits_numeric_rss_metrics(self):
        summary = self._run_with_rss_probe(OSError("forced procfs read failure"))

        measurement = summary["measurement_evidence"]["memory.run.rss"]
        self.assertEqual(measurement["status"], "unavailable")
        self.assertFalse(measurement["available"])
        self.assertEqual(measurement["backend"], "linux_procfs_statm")
        self.assertEqual(measurement["source"], "/proc/self/statm")
        self.assertEqual(measurement["successful_samples"], 0)
        self.assertGreaterEqual(measurement["failed_samples"], 2)
        self.assertEqual(
            measurement["error"],
            {"type": "OSError", "message": "forced procfs read failure"},
        )
        self.assertFalse(
            any(key.startswith("memory.run.") for key in summary["metrics"]),
            summary["metrics"],
        )

    def test_successful_probe_preserves_numeric_metrics_and_backend_evidence(self):
        summary = self._run_with_rss_probe(lambda: 8192)

        measurement = summary["measurement_evidence"]["memory.run.rss"]
        self.assertEqual(measurement["status"], "measured")
        self.assertTrue(measurement["available"])
        self.assertEqual(measurement["backend"], "linux_procfs_statm")
        self.assertGreaterEqual(measurement["successful_samples"], 2)
        self.assertEqual(measurement["failed_samples"], 0)
        self.assertIsNone(measurement["error"])
        self.assertEqual(
            summary["metrics"],
            {
                "memory.run.start_rss_bytes": 8192,
                "memory.run.peak_rss_bytes": 8192,
                "memory.run.end_rss_bytes": 8192,
                "memory.run.peak_rss_delta_bytes": 0,
                "memory.run.sampler_interval_s": 0.05,
            },
        )


if __name__ == "__main__":
    unittest.main()
