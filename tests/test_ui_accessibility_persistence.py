from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
INDEX_HTML = ROOT / "src" / "noema_lab" / "ui" / "static" / "index.html"
STYLES_CSS = ROOT / "src" / "noema_lab" / "ui" / "static" / "styles.css"
FOUNDATION_PY = ROOT / "src" / "noema_lab" / "ops" / "foundation.py"


class UiAccessibilityPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = APP_JS.read_text(encoding="utf-8")
        cls.index = INDEX_HTML.read_text(encoding="utf-8")
        cls.styles = STYLES_CSS.read_text(encoding="utf-8")
        cls.foundation = FOUNDATION_PY.read_text(encoding="utf-8")

    def test_primary_views_are_real_keyboard_tabs_with_url_state(self):
        for tab_id, panel_id in (
            ("graphTabButton", "graphView"),
            ("trainingTabButton", "trainingView"),
            ("resultsTabButton", "resultsView"),
        ):
            marker = f'id="{tab_id}"'
            line = next(line for line in self.index.splitlines() if marker in line)
            self.assertIn('role="tab"', line)
            self.assertIn(f'aria-controls="{panel_id}"', line)
            panel_line = next(
                line for line in self.index.splitlines() if f'id="{panel_id}"' in line
            )
            self.assertIn('role="tabpanel"', panel_line)
            self.assertIn(f'aria-labelledby="{tab_id}"', panel_line)
        self.assertIn("function handleTopLevelTabKeydown(event)", self.app)
        self.assertIn('window.addEventListener("popstate"', self.app)
        self.assertIn('window.addEventListener("hashchange"', self.app)
        self.assertIn("syncViewLocation(view, locationMode)", self.app)
        self.assertIn("function handleRecipeTabKeydown(event)", self.app)
        self.assertIn('aria-controls="graphView"', self.app)
        self.assertIn('id="results-facet-tab-${escapeAttr(facet.id)}"', self.app)
        self.assertIn('role="tabpanel" aria-labelledby="results-facet-tab-', self.app)
        self.assertIn('button.addEventListener("keydown"', self.app)

    def test_dialogs_trap_focus_and_make_background_inert(self):
        self.assertIn('aria-modal="true"', self.index)
        self.assertIn('role="dialog"', self.index)
        self.assertIn("function trapModalFocus(event)", self.app)
        self.assertIn("modalFocusableElements(overlay)", self.app)
        self.assertIn("els.appShell.inert = modalOpen", self.app)
        self.assertIn('els.appShell.setAttribute("aria-hidden", "true")', self.app)
        self.assertIn("modalReturnFocus", self.app)

    def test_graph_topology_controls_are_keyboard_operable(self):
        self.assertIn('role="group" tabindex="0"', self.app)
        self.assertIn('role="button" tabindex="-1" aria-pressed=', self.app)
        self.assertIn("function bindGraphCompositeNavigation()", self.app)
        self.assertIn("function graphCompositeControls(svg)", self.app)
        self.assertIn('or Tab to skip the graph.', self.app)
        self.assertIn('data-node-input="${escapeAttr(node.id)}"', self.app)
        self.assertIn('data-node-output="${escapeAttr(node.id)}"', self.app)
        self.assertIn("function activateGraphControlOnKeyboard(event, action)", self.app)
        self.assertIn('port.addEventListener("keydown"', self.app)
        self.assertIn('edge.addEventListener("keydown"', self.app)

    def test_narrow_layout_does_not_force_desktop_document_width(self):
        body_start = self.styles.index("body {\n  position: relative")
        body = self.styles[body_start : self.styles.index("}", body_start)]
        self.assertIn("min-width: 0", body)
        self.assertNotIn("min-width: 1120px", self.styles)
        responsive = self.styles[self.styles.index("@media (max-width: 1100px)") :]
        self.assertIn("grid-template-columns: minmax(0, 1fr)", responsive)
        self.assertIn("overflow-y: auto", responsive)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", responsive)
        self.assertIn("overflow: visible", responsive)

    def test_startup_failure_is_persistent_actionable_and_accessible(self):
        banner_line = next(
            line
            for line in self.index.splitlines()
            if 'id="startupStatusBanner"' in line
        )
        banner_block_start = self.index.index(banner_line)
        banner_block_end = self.index.index("</section>", banner_block_start)
        banner = self.index[banner_block_start:banner_block_end]
        self.assertIn('role="alert"', banner)
        self.assertIn('aria-live="assertive"', banner)
        self.assertIn('aria-atomic="true"', banner)
        self.assertIn('id="startupRetryButton"', banner)
        self.assertIn(">Retry</button>", banner)
        self.assertIn("function showStartupFailure(error)", self.app)
        self.assertIn("function retryStartup()", self.app)
        self.assertIn('showStartupFailure(error);', self.app)
        self.assertIn('clearStartupFailure();', self.app)
        bootstrap_start = self.app.index("async function refreshAll()")
        bootstrap_end = self.app.index(
            "function scheduleDeferredArtifactDiscovery",
            bootstrap_start,
        )
        self.assertNotIn(
            "notify(error.message);",
            self.app[bootstrap_start:bootstrap_end],
        )
        self.assertIn(".startup-status-banner[hidden]", self.styles)

    def test_faithfulness_metric_uses_precise_unsupported_assertion_name(self):
        shipped_ui = "\n".join((self.app, self.index, self.styles))
        self.assertNotIn("hallucination_rate", shipped_ui)
        self.assertNotIn("Hallucination rate", shipped_ui)
        self.assertIn("faithfulness.unsupported_assertion_rate", shipped_ui)
        self.assertIn(
            "Reference-relative unsupported assertion rate",
            shipped_ui,
        )

    def test_dead_external_foundation_modes_are_not_shipped(self):
        dead_modes = ("external_" + "llm", "external_" + "vlm")
        for source in (self.app, self.foundation):
            for mode in dead_modes:
                self.assertNotIn(mode, source)
        self.assertNotIn("External LLM adapter", self.app)
        self.assertNotIn("External VLM backend", self.foundation)

    def test_unsupported_saved_text_modes_are_exposed_not_silently_rewritten(self):
        normalize_start = self.app.index("function normalizeTextRecipeConfig(config)")
        normalize_end = self.app.index(
            "\nfunction buildTextRecipeFromConfig", normalize_start
        )
        normalization = self.app[normalize_start:normalize_end]
        self.assertNotIn(
            'config.foundation.generator = "local_template"',
            normalization,
        )
        self.assertIn("function unsupportedSavedSelectOption", self.app)
        self.assertIn("Unsupported saved value:", self.app)
        self.assertIn("selected disabled", self.app)

    def test_kb_reconstruction_copy_describes_received_semantic_overlap(self):
        modes_start = self.app.index("const TEXT_GENERATOR_MODES")
        modes_end = self.app.index("];", modes_start) + 2
        hint_start = self.app.index("function textGeneratorHint(generator)")
        hint_end = self.app.index("\n}", hint_start) + 2
        copy = self.app[modes_start:modes_end] + self.app[hint_start:hint_end]
        self.assertIn("Knowledge-base semantic overlap", copy)
        self.assertIn("token overlap with the received semantic evidence", copy)
        copy_lower = copy.lower()
        for misleading in (
            "oracle",
            "source id",
            "source-id",
            "canonical sentence",
            "exact canonical",
            "exact answer",
            "exact text",
        ):
            self.assertNotIn(misleading, copy_lower)

    def test_project_recipes_are_listed_and_working_copies_can_be_saved(self):
        library_start = self.app.index("function recipeLibraryRows()")
        library_end = self.app.index("async function defaultImageWorkingRecipePayload", library_start)
        library = self.app[library_start:library_end]
        self.assertIn("projectByPath.forEach", library)
        self.assertIn('libraryKind: "project-recipe"', library)
        self.assertIn('id="saveRecipeButton"', self.index)
        self.assertIn('api("/api/recipe/save"', self.app)
        self.assertIn('api("/api/recipes")', self.app)

    def test_dataset_browsing_is_read_only_until_explicit_download(self):
        loader_start = self.app.index("async function loadDatasetImagesForConfig")
        loader_end = self.app.index("function datasetCacheKey", loader_start)
        loader = self.app[loader_start:loader_end]
        self.assertIn('ensure: ensure ? "1" : "0"', loader)
        self.assertIn("data-materialize-image-dataset", self.app)
        self.assertIn("Download Kodak dataset", self.app)

    def test_matrix_variants_fail_fast_and_cancel_is_deduplicated(self):
        run_start = self.app.index("async function runRecipeJob(recipe)")
        run_end = self.app.index("async function runSingleRecipeVariant", run_start)
        run_source = self.app[run_start:run_end]
        failure = run_source.index('if (job.status !== "completed")')
        self.assertIn("break;", run_source[failure : failure + 260])

        stop_start = self.app.index("async function stopActiveRuns()")
        stop_end = self.app.index("async function cancelRecipeJob", stop_start)
        stop_source = self.app[stop_start:stop_end]
        self.assertLess(
            stop_source.index("cancel_requested: true"),
            stop_source.index("await cancelRecipeJob"),
        )
        single_start = self.app.index("async function runSingleRecipeVariant")
        single_end = self.app.index("async function pollRunJobWithRetry", single_start)
        self.assertIn("recordedJob.cancel_requested", self.app[single_start:single_end])


if __name__ == "__main__":
    unittest.main()
