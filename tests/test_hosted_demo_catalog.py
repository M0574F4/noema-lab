from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DEMO = DOCS / "demo"
CATALOG = DEMO / "catalog.json"
CATALOG_JS = DEMO / "catalog.js"
CHART_RE = re.compile(r'data-noema-chart="([^"]+)"')
RECIPE_RE = re.compile(r"(recipes/[A-Za-z0-9_./-]+\.ya?ml)")


class HostedDemoCatalogTests(unittest.TestCase):
    def test_catalog_covers_every_chart_bearing_tutorial(self):
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        entries = {item["id"]: item for item in catalog["experiments"]}
        chart_tutorials = {}
        for path in (DOCS / "tutorials").glob("*.md"):
            chart_ids = CHART_RE.findall(path.read_text(encoding="utf-8"))
            if chart_ids:
                chart_tutorials[path.stem] = chart_ids

        self.assertEqual(set(entries), set(chart_tutorials))
        self.assertEqual(catalog["experiment_count"], len(chart_tutorials))
        for tutorial_id, chart_ids in chart_tutorials.items():
            self.assertEqual(entries[tutorial_id]["chart_ids"], chart_ids)
            self.assertTrue(entries[tutorial_id]["chart_scripts"])

    def test_catalog_evidence_hashes_match_the_documentation_assets(self):
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        for experiment in catalog["experiments"]:
            for evidence in experiment["evidence"]:
                path = DEMO / evidence["path"]
                self.assertTrue(path.is_file(), path)
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    evidence["sha256"],
                    path,
                )
                self.assertEqual(path.stat().st_size, evidence["size_bytes"], path)

    def test_catalog_graphs_use_the_recipes_referenced_by_each_tutorial(self):
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        for experiment in catalog["experiments"]:
            tutorial = ROOT / experiment["source_path"]
            recipe_references = RECIPE_RE.findall(tutorial.read_text(encoding="utf-8"))
            self.assertTrue(recipe_references, tutorial)

            recipe = experiment["recipe"]
            self.assertEqual(recipe["source_path"], recipe_references[0], tutorial)
            recipe_path = ROOT / recipe["source_path"]
            self.assertTrue(recipe_path.is_file(), recipe_path)
            self.assertEqual(
                recipe["sha256"],
                hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
                recipe_path,
            )
            self.assertTrue(recipe["steps"], recipe_path)
            self.assertTrue(
                all(step.get("id") and step.get("op") for step in recipe["steps"]),
                recipe_path,
            )

    def test_published_result_registry_remains_discoverable(self):
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        registry = json.loads(
            (DEMO / "experiments" / "index.json").read_text(encoding="utf-8")
        )
        catalog_paths = {
            item["path"]
            for experiment in catalog["experiments"]
            for item in experiment["published_results"]
        }
        registry_paths = {
            f"experiments/{item['path']}"
            for item in registry["demos"]
        }
        self.assertEqual(catalog_paths, registry_paths)

    def test_hosted_page_uses_shared_result_assets_instead_of_embedded_values(self):
        page = (DEMO / "index.html").read_text(encoding="utf-8")
        app = (DEMO / "app.js").read_text(encoding="utf-8")
        browser_catalog = CATALOG_JS.read_text(encoding="utf-8")
        configuration = (DOCS / "conf.py").read_text(encoding="utf-8")
        chart_runtime = (
            DOCS / "_static" / "noema-demo-charts-v2.js"
        ).read_text(encoding="utf-8")

        self.assertIn('href="dashboard.css"', page)
        self.assertIn('src="catalog.js"', page)
        self.assertIn('src="../_static/noema-demo-charts-v2.js"', page)
        self.assertLess(page.index(">Graph</button>"), page.index(">Results</button>"))
        self.assertIn('id="graphView"', page)
        self.assertIn('id="resultsView"', page)
        topbar = page[page.index("<header"):page.index("</header>")]
        self.assertIn('id="experimentSelect"', topbar)
        self.assertNotIn("Same evidence as the documentation", app)
        self.assertIn('fetch("catalog.json")', app)
        self.assertIn("NOEMA_HOSTED_DEMO_CATALOG", app)
        self.assertIn("NOEMA_HOSTED_DEMO_TABLES", app)
        self.assertIn(
            'const DEFAULT_EXPERIMENT_ID = "learned_qpsk_demapper_demo"',
            app,
        )
        self.assertIn("renderRecipeGraph(experiment.recipe)", app)
        self.assertIn("buildRecipeLayout(recipe.steps)", app)
        self.assertIn("window.NOEMA_HOSTED_DEMO_CATALOG", browser_catalog)
        self.assertIn("window.NOEMA_HOSTED_DEMO_TABLES", browser_catalog)
        self.assertNotIn("demo_result.json", page + app)
        self.assertIn("styles.css", configuration)
        self.assertIn("destination / \"dashboard.css\"", configuration)
        self.assertIn("NOEMA_DEMO_CHART_RUNTIME", chart_runtime)
        self.assertIn("initialize: initializeCharts", chart_runtime)


if __name__ == "__main__":
    unittest.main()
