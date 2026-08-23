from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from noema_lab import __version__


PUBLIC_DEMO_EXCLUSIONS = {
    Path("data") / "digital_vs_deepjscc" / "reconstructions",
    Path("data") / "deepjscc_slow_rayleigh" / "reconstructions",
}

project = "Noema"
author = "Mostafa Naseri"
copyright = "2026, Mostafa Naseri and Noema contributors"
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
master_doc = "index"
exclude_patterns = [
    "_build",
    "_includes/*.md",
    "Thumbs.db",
    ".DS_Store",
    "readme_drafts/*.md",
    "paper_strategy.md",
    "paper_adversarial_review.md",
    "paper_adversarial_review_round2_15_lenses.md",
    "paper_adversarial_review_remediation_2026-07-22.md",
    "adversarial_code_certification_review_*.md",
    "adversarial_code_certification_remediation_*.md",
    "readme_strategy.md",
    "release_readiness.md",
    "publication_analysis.md",
    "template_training_roadmap.md",
]
templates_path = ["_templates"]

myst_enable_extensions = ["colon_fence", "deflist"]
autodoc_typehints = "description"
autodoc_member_order = "bysource"

html_theme = "pydata_sphinx_theme"
html_title = "Noema — Learned-Communication Experiment Contracts"
html_meta = {
    "description": (
        "Noema links learned-communication experiment protocols to execution plans, "
        "resource accounting, results, and plot evidence."
    )
}
html_favicon = "_static/noema-logo.svg"
html_static_path = ["_static"]
html_extra_path = []
html_css_files = ["noema-docs.css", "noema-break-the-comparison.css"]
html_js_files = [
    "noema-demo-chart-data.js",
    "noema-ofdm-resource-allocation-chart-data.js",
    "noema-delayed-csi-demo-chart-data.js",
    "noema-mimo-channel-estimation-chart-data.js",
    "noema-csi-feedback-chart-data.js",
    "noema-qpsk-phase-tracking-chart-data.js",
    "noema-modulation-recognition-chart-data.js",
    "noema-deepjscc-chart-data.js",
    "noema-deepjscc-slow-rayleigh-chart-data-v2.js",
    "noema-range-localization-chart-data.js",
    "noema-aoa-estimation-chart-data.js",
    "noema-miso-beam-selection-chart-data.js",
    "noema-isac-joint-allocation-chart-data.js",
    "noema-near-field-xl-mimo-chart-data.js",
    "noema-leo-ntn-tracking-chart-data.js",
    "noema-demo-charts-v2.js",
    "noema-break-the-comparison.js",
]
html_sidebars = {
    "**": ["sidebar-collapse", "noema-global-sidebar"],
}
html_theme_options = {
    "collapse_navigation": True,
    "show_nav_level": 2,
    "show_toc_level": 3,
    "navigation_depth": 2,
    "navbar_center": [],
    "logo": {
        "image_light": "_static/noema-logo.svg",
        "image_dark": "_static/noema-logo.svg",
        "alt_text": "Noema home",
        "text": "Noema",
    },
    "github_url": "https://github.com/M0574F4/noema-lab",
}


def setup(app):
    app.connect("builder-inited", _verify_demo_snapshots)
    app.connect("builder-inited", _verify_launch_assets)
    app.connect("builder-inited", _generate_hosted_demo_catalog)
    app.connect("builder-inited", _generate_reference_pages)
    app.connect("build-finished", _copy_static_demo)


def _verify_demo_snapshots(app) -> None:
    generator_path = ROOT / "tools" / "generate_deepjscc_demo_assets.py"
    spec = importlib.util.spec_from_file_location(
        "noema_deepjscc_demo_assets",
        generator_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load DeepJSCC docs verifier: %s" % generator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_snapshot_assets(
        ROOT / "docs" / "demo" / "data" / "digital_vs_deepjscc"
        / "snapshot_manifest.json"
    )


def _verify_launch_assets(app) -> None:
    generator_path = ROOT / "tools" / "generate_launch_assets.py"
    spec = importlib.util.spec_from_file_location(
        "noema_launch_asset_generator",
        generator_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load launch-asset verifier: %s" % generator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.main(["--check"])
    if result:
        raise RuntimeError("launch-asset verification failed with exit code %s" % result)


def _generate_reference_pages(app) -> None:
    generator_path = ROOT / "tools" / "generate_docs_reference.py"
    spec = importlib.util.spec_from_file_location("noema_docs_reference_generator", generator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load documentation reference generator: %s" % generator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.main([])
    if result:
        raise RuntimeError("documentation reference generation failed with exit code %s" % result)


def _generate_hosted_demo_catalog(app) -> None:
    generator_path = ROOT / "tools" / "generate_hosted_demo_catalog.py"
    spec = importlib.util.spec_from_file_location(
        "noema_hosted_demo_catalog_generator",
        generator_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load hosted demo catalog generator: %s" % generator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.main([])
    if result:
        raise RuntimeError("hosted demo catalog generation failed with exit code %s" % result)


def _copy_static_demo(app, exception) -> None:
    if exception is not None:
        return
    source = ROOT / "docs" / "demo"
    destination = Path(app.outdir) / "demo"
    if destination.exists():
        shutil.rmtree(destination)

    def ignore_restricted_assets(directory: str, names: list[str]) -> set[str]:
        relative = Path(directory).resolve().relative_to(source.resolve())
        return {
            name
            for name in names
            if relative / name in PUBLIC_DEMO_EXCLUSIONS
        }

    shutil.copytree(source, destination, ignore=ignore_restricted_assets)
    shutil.copy2(
        ROOT / "src" / "noema_lab" / "ui" / "static" / "styles.css",
        destination / "dashboard.css",
    )
    shutil.copy2(ROOT / "launch_evidence.json", Path(app.outdir) / "launch_evidence.json")
