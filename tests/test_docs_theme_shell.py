from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class DocumentationThemeShellTests(unittest.TestCase):
    def test_global_header_uses_logo_without_duplicate_document_links(self):
        configuration = (DOCS / "conf.py").read_text(encoding="utf-8")
        self.assertIn('"navbar_center": []', configuration)
        self.assertIn('"image_light": "_static/noema-logo.svg"', configuration)
        self.assertIn('"image_dark": "_static/noema-logo.svg"', configuration)
        self.assertIn('"text": "Noema"', configuration)
        self.assertTrue((DOCS / "_static" / "noema-logo.svg").is_file())

    def test_sidebar_links_to_the_documentation_homepage(self):
        template = (
            DOCS / "_templates" / "noema-global-sidebar.html"
        ).read_text(encoding="utf-8")
        self.assertIn("Noema Documentation", template)
        self.assertIn('href="{{ pathto(root_doc) }}"', template)
        self.assertIn('aria-current="page"', template)

    def test_wide_layout_uses_the_available_browser_width(self):
        stylesheet = (DOCS / "_static" / "noema-docs.css").read_text(
            encoding="utf-8"
        )
        self.assertIn(".bd-page-width", stylesheet)
        self.assertIn("max-width: none;", stylesheet)
        self.assertIn("width: clamp(17rem, 18vw, 20rem);", stylesheet)
        self.assertIn(".bd-article-container", stylesheet)
        self.assertIn(".navbar-header-items__end", stylesheet)
        self.assertIn("margin-left: auto;", stylesheet)

    def test_overview_title_uses_the_tightly_cropped_noema_mark(self):
        overview = (DOCS / "index.md").read_text(encoding="utf-8")
        self.assertIn(
            '<img src="_static/noema-logo.svg" alt="" '
            'class="noema-title-logo"> Noema Documentation',
            overview,
        )
        self.assertNotIn("noema-logo-lockup.svg", overview)

        logo = (DOCS / "_static" / "noema-logo.svg").read_text(
            encoding="utf-8"
        )
        self.assertIn('viewBox="8 8 80 80"', logo)
        self.assertNotIn("Semantic communication research toolkit", logo)

        stylesheet = (DOCS / "_static" / "noema-docs.css").read_text(
            encoding="utf-8"
        )
        self.assertIn(".noema-title-logo", stylesheet)
        self.assertIn("width: 1.05em;", stylesheet)

    def test_overview_page_shows_the_editable_training_loop(self):
        overview = (DOCS / "index.md").read_text(encoding="utf-8")
        self.assertIn("_static/noema-training-loop.svg", overview)
        self.assertNotIn("_static/noema-workbench-graph.png", overview)
        self.assertIn(":figclass: noema-overview-workflow-figure", overview)
        self.assertTrue(
            (DOCS / "_static" / "noema-training-loop.svg").is_file()
        )

        source = (
            DOCS / "_static" / "diagrams" / "noema-training-loop.drawio"
        )
        self.assertTrue(source.is_file())
        diagram = source.read_text(encoding="utf-8")
        self.assertIn("Open a scenario", diagram)
        self.assertIn("Choose trainable blocks", diagram)
        self.assertIn("Export a training bundle", diagram)
        self.assertIn("Train your own model", diagram)
        self.assertIn("Bind it into the recipe", diagram)
        self.assertIn("Benchmark and verify", diagram)
        self.assertIn("image=data:image/png,", diagram)
        badge_style = (
            "style=\"ellipse;whiteSpace=wrap;html=1;aspect=fixed;"
            "fillColor=#54C58F;"
        )
        self.assertEqual(diagram.count(badge_style), 6)

        stylesheet = (DOCS / "_static" / "noema-docs.css").read_text(
            encoding="utf-8"
        )
        self.assertIn(".noema-overview-workflow-figure", stylesheet)
        self.assertIn("float: right;", stylesheet)
        self.assertIn("width: 70%;", stylesheet)

    def test_overview_page_is_the_single_website_and_documentation_entry(self):
        overview = (DOCS / "index.md").read_text(encoding="utf-8")
        for required in (
            "_static/launch/f0-launch-hero.svg",
            'class="noema-home-actions"',
            'class="noema-home-routes"',
            "break_the_comparison.html",
            "tutorials.html",
            "demos.html",
            "reference/index.html",
            "_static/noema-training-loop.svg",
            "_static/launch/f3-receiver-ber-vs-snr.svg",
            "## Browse the documentation",
        ):
            self.assertIn(required, overview)
        self.assertEqual(overview.count("# <img"), 1)
        self.assertFalse((DOCS / "website.md").exists())
        self.assertFalse((DOCS / "getting_started.md").exists())

        stylesheet = (DOCS / "_static" / "noema-docs.css").read_text(
            encoding="utf-8"
        )
        for selector in (
            ".noema-home-hero",
            ".noema-home-actions",
            ".noema-home-routes",
            ".noema-home-route:focus-visible",
        ):
            self.assertIn(selector, stylesheet)


if __name__ == "__main__":
    unittest.main()
