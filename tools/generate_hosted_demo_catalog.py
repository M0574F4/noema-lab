from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
TUTORIALS = DOCS / "tutorials"
TUTORIAL_INDEX = DOCS / "tutorials.md"
STATIC = DOCS / "_static"
OUTPUT = DOCS / "demo" / "catalog.json"
OUTPUT_JS = DOCS / "demo" / "catalog.js"
PUBLISHED_REGISTRY = DOCS / "demo" / "experiments" / "index.json"

CHART_RE = re.compile(r'data-noema-chart="([^"]+)"')
DATA_FILE_RE = re.compile(
    r"\.\./demo/data/([A-Za-z0-9_-]+)/"
    r"([A-Za-z0-9_./-]+\.(?:csv|json))"
)
TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
TUTORIAL_LINK_RE = re.compile(r"tutorials/([A-Za-z0-9_-]+)\.md")
RECIPE_FILE_RE = re.compile(r"(recipes/[A-Za-z0-9_./-]+\.ya?ml)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown_text(value: str) -> str:
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"[`*_]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _description(source: str) -> str:
    goal_match = re.search(
        r"^##\s+Goal\s*$\n+(.*?)(?=\n\n(?:[-*]\s|##\s|```|:::\s|\{))",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    if goal_match:
        return _markdown_text(goal_match.group(1))

    title_match = TITLE_RE.search(source)
    if not title_match:
        return ""
    remainder = source[title_match.end() :]
    for paragraph in re.split(r"\n\s*\n", remainder):
        candidate = paragraph.strip()
        if not candidate or candidate.startswith(("#", "```", ":::", "{")):
            continue
        if candidate.startswith(("-", "*", "1.")):
            continue
        return _markdown_text(candidate)
    return ""


def _html_js_files() -> list[str]:
    tree = ast.parse((DOCS / "conf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "html_js_files"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("docs/conf.py html_js_files must be a literal string list")
        return value
    raise ValueError("docs/conf.py does not define html_js_files")


def _chart_script_sources() -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    for filename in _html_js_files():
        if "chart-data" not in filename:
            continue
        path = STATIC / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append((filename, path.read_text(encoding="utf-8")))
    return sources


def _tutorial_order() -> list[str]:
    source = TUTORIAL_INDEX.read_text(encoding="utf-8")
    ordered: list[str] = []
    for stem in TUTORIAL_LINK_RE.findall(source):
        if stem not in ordered:
            ordered.append(stem)
    return ordered


def _ordered_tutorial_paths() -> list[Path]:
    chart_pages = {
        path.stem: path
        for path in TUTORIALS.glob("*.md")
        if CHART_RE.search(path.read_text(encoding="utf-8"))
    }
    paths = [chart_pages.pop(stem) for stem in _tutorial_order() if stem in chart_pages]
    paths.extend(chart_pages[stem] for stem in sorted(chart_pages))
    return paths


def _data_files(source: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for directory, filename in DATA_FILE_RE.findall(source):
        relative = Path("data") / directory / filename
        public_path = relative.as_posix()
        if public_path in seen:
            continue
        source_path = DOCS / "demo" / relative
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        seen.add(public_path)
        files.append(
            {
                "label": filename.replace("_", " ").replace("-", " "),
                "path": public_path,
                "sha256": _sha256(source_path),
                "size_bytes": source_path.stat().st_size,
                "type": source_path.suffix.lstrip("."),
            }
        )
    return files


def _recipe_graph(source: str, tutorial_path: Path) -> dict[str, Any]:
    recipe_matches = RECIPE_FILE_RE.findall(source)
    if not recipe_matches:
        raise ValueError(
            f"{tutorial_path.relative_to(ROOT)} does not reference a recipe YAML file"
        )

    public_path = recipe_matches[0]
    recipe_path = ROOT / public_path
    if not recipe_path.is_file():
        raise FileNotFoundError(recipe_path)

    payload = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{public_path} must contain a recipe mapping")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{public_path} must contain at least one recipe step")

    graph_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not step.get("id") or not step.get("op"):
            raise ValueError(f"{public_path} step {index + 1} requires id and op")
        graph_steps.append(
            {
                "id": str(step["id"]),
                "op": str(step["op"]),
                "inputs": step.get("inputs") if isinstance(step.get("inputs"), dict) else {},
                "params": step.get("params") if isinstance(step.get("params"), dict) else {},
            }
        )

    execution_profile = payload.get("execution_profile")
    suite = payload.get("suite")
    return {
        "source_path": public_path,
        "sha256": _sha256(recipe_path),
        "name": str(payload.get("name") or recipe_path.stem),
        "description": str(payload.get("description") or ""),
        "execution_profile": execution_profile if isinstance(execution_profile, dict) else {},
        "suite": suite if isinstance(suite, dict) else {},
        "steps": graph_steps,
    }


def _snapshot_summary(files: list[dict[str, Any]]) -> dict[str, Any]:
    manifests = [
        item for item in files
        if item["type"] == "json" and "manifest" in Path(item["path"]).name
    ]
    if not manifests:
        return {}

    manifest_file = manifests[0]
    manifest = json.loads((DOCS / "demo" / manifest_file["path"]).read_text(encoding="utf-8"))
    projection = manifest.get("projection") if isinstance(manifest, dict) else {}
    runs = manifest.get("runs") if isinstance(manifest, dict) else []
    summary: dict[str, Any] = {
        "kind": str(manifest.get("kind") or "snapshot manifest"),
        "manifest_path": manifest_file["path"],
        "manifest_sha256": manifest_file["sha256"],
    }
    if isinstance(runs, list):
        summary["run_count"] = len(runs)
    if isinstance(projection, dict) and isinstance(projection.get("rows"), int):
        summary["projection_rows"] = projection["rows"]
    return summary


def _published_results_by_tutorial() -> dict[str, list[dict[str, str]]]:
    if not PUBLISHED_REGISTRY.is_file():
        return {}
    registry = json.loads(PUBLISHED_REGISTRY.read_text(encoding="utf-8"))
    results: dict[str, list[dict[str, str]]] = {}
    for item in registry.get("demos") or []:
        relative_page = Path(str(item.get("path") or ""))
        payload_path = PUBLISHED_REGISTRY.parent / relative_page.parent / "data" / "demo.json"
        if not relative_page.name or not payload_path.is_file():
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        tutorial = str((payload.get("demo") or {}).get("tutorial") or "")
        tutorial_id = Path(tutorial).stem
        if not tutorial_id:
            continue
        results.setdefault(tutorial_id, []).append(
            {
                "title": str(item.get("title") or item.get("slug") or tutorial_id),
                "path": f"experiments/{relative_page.as_posix()}",
                "result_id": str(item.get("result_id") or ""),
                "source_bundle_sha256": str(item.get("source_bundle_sha256") or ""),
            }
        )
    return results


def build_catalog() -> dict[str, Any]:
    chart_sources = _chart_script_sources()
    published_results = _published_results_by_tutorial()
    experiments: list[dict[str, Any]] = []

    for path in _ordered_tutorial_paths():
        source = path.read_text(encoding="utf-8")
        chart_ids = CHART_RE.findall(source)
        files = _data_files(source)
        scripts = [
            f"../_static/{filename}"
            for filename, javascript in chart_sources
            if any(f'"{chart_id}"' in javascript for chart_id in chart_ids)
        ]
        missing_charts = [
            chart_id
            for chart_id in chart_ids
            if not any(f'"{chart_id}"' in javascript for _, javascript in chart_sources)
        ]
        if missing_charts:
            raise ValueError(
                f"{path.relative_to(ROOT)} has charts without configured data scripts: "
                + ", ".join(missing_charts)
            )

        title_match = TITLE_RE.search(source)
        if not title_match:
            raise ValueError(f"{path.relative_to(ROOT)} has no level-one title")

        data_directories = sorted(
            {
                str(Path(item["path"]).parent)
                for item in files
            }
        )
        experiments.append(
            {
                "id": path.stem,
                "title": _markdown_text(title_match.group(1)),
                "description": _description(source),
                "documentation_path": f"../tutorials/{path.stem}.html",
                "source_path": path.relative_to(ROOT).as_posix(),
                "recipe": _recipe_graph(source, path),
                "chart_ids": chart_ids,
                "chart_scripts": scripts,
                "data_directories": data_directories,
                "tables": [item for item in files if item["type"] == "csv"],
                "evidence": files,
                "snapshot": _snapshot_summary(files),
                "published_results": published_results.get(path.stem, []),
            }
        )

    return {
        "kind": "noema.hosted_demo_catalog",
        "version": 2,
        "source": "chart-bearing tutorials, their representative recipes, and checked-in demo evidence",
        "experiment_count": len(experiments),
        "experiments": experiments,
    }


def _serialized_browser_catalog(catalog: dict[str, Any]) -> str:
    table_payload = {
        table["path"]: (DOCS / "demo" / table["path"]).read_text(encoding="utf-8")
        for experiment in catalog["experiments"]
        for table in experiment["tables"]
    }
    return (
        "(function () {\n"
        '  "use strict";\n'
        "  window.NOEMA_HOSTED_DEMO_CATALOG = "
        + json.dumps(catalog, separators=(",", ":"), sort_keys=True)
        + ";\n"
        "  window.NOEMA_HOSTED_DEMO_TABLES = Object.freeze("
        + json.dumps(table_payload, separators=(",", ":"), sort_keys=True)
        + ");\n"
        "})();\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the hosted static demo catalog from documentation tutorials."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if docs/demo/catalog.json is not current",
    )
    args = parser.parse_args(argv)
    catalog = build_catalog()
    rendered = json.dumps(catalog, indent=2, sort_keys=True) + "\n"
    rendered_js = _serialized_browser_catalog(catalog)

    if args.check:
        stale = [
            path.relative_to(ROOT).as_posix()
            for path, expected in ((OUTPUT, rendered), (OUTPUT_JS, rendered_js))
            if not path.is_file() or path.read_text(encoding="utf-8") != expected
        ]
        if stale:
            print(f"{', '.join(stale)} is stale; regenerate it")
            return 1
        print(
            f"{OUTPUT.relative_to(ROOT)} and {OUTPUT_JS.relative_to(ROOT)} are current"
        )
        return 0

    OUTPUT.write_text(rendered, encoding="utf-8")
    OUTPUT_JS.write_text(rendered_js, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} and {OUTPUT_JS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
