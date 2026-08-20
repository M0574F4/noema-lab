from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"


class UiP3ExecutionControlsTests(unittest.TestCase):
    def test_parallel_execution_is_explicit_and_sequential_by_default(self):
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn("applyExecutionControlsContract(health.execution_controls)", source)
        self.assertIn("executionParallelWorkerOptions()", source)
        self.assertIn("executionParallelWorkers: initialExecutionParallelWorkers", source)
        self.assertIn('data-execution-parallel-workers', source)
        self.assertIn('workers === 1 ? "Sequential"', source)

    def test_run_job_payload_sends_the_complete_user_selected_policy(self):
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn("execution: executionRequestOptions()", source)
        self.assertIn("strict_lint: Boolean(state.executionStrictLint)", source)
        self.assertIn("use_plan_cache: Boolean(state.executionUsePlanCache)", source)
        self.assertIn("data-execution-backend", source)
        self.assertIn("data-execution-implementation", source)
        self.assertIn("data-execution-strict-lint", source)
        self.assertIn("data-execution-use-plan-cache", source)
        self.assertNotIn("use_plan_cache: true", source)
        self.assertIn("dependency-independent DAG branches", source)


if __name__ == "__main__":
    unittest.main()
