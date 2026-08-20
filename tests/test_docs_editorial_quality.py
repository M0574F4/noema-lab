from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXCLUDED_TOP_LEVEL = {
    "readme_strategy.md",
    "release_readiness.md",
    "template_training_roadmap.md",
}


def public_narrative_pages() -> tuple[Path, ...]:
    pages: list[Path] = []
    for path in DOCS.rglob("*.md"):
        relative = path.relative_to(DOCS)
        if relative.parts[0] in {"_build", "readme_drafts"}:
            continue
        if relative.parts[:2] == ("reference", "generated"):
            continue
        if relative.name in EXCLUDED_TOP_LEVEL:
            continue
        if relative.name.startswith(("paper_", "adversarial_")):
            continue
        pages.append(path)
    return tuple(sorted(pages))


class DocumentationEditorialQualityTests(unittest.TestCase):
    def test_public_pages_do_not_expose_internal_record_metadata(self):
        forbidden = (
            r"(?m)^Status:\s",
            r"(?m)^Date:\s",
            r"(?im)^##\s+(Decision|Consequences|Product Rule|Rationale|Still Planned)\s*$",
            r"\baccepted ADR\b",
            r"\bThis ADR\b",
            r"Recommended public language",
            r"Avoid language such as",
            r"\bhero (pack|baseline|workflow)\b",
        )
        pages = public_narrative_pages()
        self.assertGreater(len(pages), 30)
        for path in pages:
            source = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                self.assertNotRegex(source, pattern, str(path.relative_to(ROOT)))

    def test_tutorials_do_not_define_validity_by_a_desired_outcome(self):
        forbidden = (
            r"(?im)^##\s+Expected result\s*$",
            r"(?m)^#\s+Demo:\s",
            r"A sound demonstration should show",
            r"A useful (?:demo )?result should",
            r"valid result has.+materially better",
            r"(?m)^\s+--allow-warnings(?:\s|$)",
        )
        for path in sorted((DOCS / "tutorials").glob("*.md")):
            source = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                self.assertNotRegex(source, pattern, str(path.relative_to(ROOT)))

    def test_navigation_separates_overview_from_advanced_evidence(self):
        source = (DOCS / "index.md").read_text(encoding="utf-8")
        for caption in (
            "Overview",
            "Suites",
            "Core Architecture",
            "Workflows",
            "Evidence and Submissions",
            "Extending Noema",
            "Reference",
        ):
            self.assertIn(f":caption: {caption}", source)
        self.assertNotIn(":caption: Start Here", source)
        self.assertNotIn("## Documentation Sources", source)


if __name__ == "__main__":
    unittest.main()
