from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
INDEX_HTML = ROOT / "src" / "noema_lab" / "ui" / "static" / "index.html"


class UiProgressiveStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_js = APP_JS.read_text(encoding="utf-8")
        cls.index_html = INDEX_HTML.read_text(encoding="utf-8")

    def test_recipe_bootstrap_does_not_wait_for_history_or_artifacts(self):
        start = self.app_js.index("async function refreshAll()")
        end = self.app_js.index("function scheduleDeferredArtifactDiscovery", start)
        bootstrap = self.app_js[start:end]

        self.assertIn('api("/api/health")', bootstrap)
        self.assertIn('api("/api/ops")', bootstrap)
        self.assertIn('api("/api/recipes")', bootstrap)
        self.assertNotIn('api("/api/runs")', bootstrap)
        self.assertNotIn('api("/api/benchmarks/results")', bootstrap)
        self.assertNotIn('api("/api/trained-artifacts")', bootstrap)
        self.assertIn("setBusy(false)", bootstrap)
        self.assertIn("scheduleDeferredArtifactDiscovery(generation)", bootstrap)

    def test_history_is_lazy_deduplicated_and_loaded_by_results_view(self):
        self.assertIn("function ensureResultHistoryLoaded()", self.app_js)
        self.assertIn("if (runHistoryLoadPromise && runHistoryGeneration === generation)", self.app_js)
        self.assertIn("if (benchmarkHistoryLoadPromise && benchmarkHistoryGeneration === generation)", self.app_js)
        self.assertIn("void ensureResultHistoryLoaded();", self.app_js)

        view_start = self.app_js.index("function setActiveView(view")
        view_end = self.app_js.index("function defaultParamsFor", view_start)
        results_view = self.app_js[view_start:view_end]
        self.assertIn("else if (resultsActive)", results_view)
        self.assertIn("void ensureResultHistoryLoaded();", results_view)

    def test_deferred_responses_are_generation_guarded_and_merge_local_updates(self):
        self.assertIn("if (generation !== startupLoadGeneration) return false;", self.app_js)
        self.assertIn("function mergeDeferredRows(incomingRows, currentRows, identity)", self.app_js)
        self.assertIn("merged[index] = { ...merged[index], ...row };", self.app_js)
        self.assertIn("if (runHistoryLoaded && runHistoryGeneration === startupLoadGeneration)", self.app_js)

    def test_bootstrap_failure_remains_visible_until_successful_retry(self):
        start = self.app_js.index("async function refreshAll()")
        end = self.app_js.index("function scheduleDeferredArtifactDiscovery", start)
        bootstrap = self.app_js[start:end]
        self.assertIn("showStartupFailure(error)", bootstrap)
        self.assertIn("clearStartupFailure()", bootstrap)
        self.assertIn("return recipeReady;", bootstrap)
        self.assertNotIn("notify(error.message)", bootstrap)
        self.assertIn("async function retryStartup()", self.app_js)
        self.assertIn("showStartupRetrying();", self.app_js)
        self.assertIn('id="startupStatusBanner"', self.index_html)
        self.assertIn('id="startupRetryButton"', self.index_html)


if __name__ == "__main__":
    unittest.main()
