from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from noema_lab import __version__
from noema_lab.cli.progress import CliProgressRenderer
from noema_lab.core.benchmarks import (
    BenchmarkError,
    finalize_resource_exhausted_benchmark,
    list_benchmark_packs,
    load_benchmark_pack,
    reproduce_benchmark_report_artifacts,
    validate_benchmark_pack,
    write_benchmark_resource_guard_evidence,
)
from noema_lab.core.benchmark_plots import (
    load_benchmark_plot_records,
    plot_benchmark_result,
)
from noema_lab.core.demo_export import publish_benchmark_demo
from noema_lab.core.capture import run_dataset_capture_recipe
from noema_lab.core.executor import MAX_PARALLEL_WORKERS, LocalExecutor
from noema_lab.core.execution_controls import (
    executor_options,
    normalize_execution_controls,
    require_strict_lint,
)
from noema_lab.core.external_adapters import scaffold_adapter, validate_adapter_manifest
from noema_lab.core.graph import format_graph_dot, format_graph_text, recipe_graph
from noema_lab.core.lint import lint_recipe_invariants
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import RecipeValidationError, load_recipe
from noema_lab.core.recipe_templates import (
    inspect_recipe_template_catalog,
    instantiate_recipe_template,
)
from noema_lab.core.research import research_specs_from_recipe
from noema_lab.core.research_catalog import load_research_catalog, validate_research_specs_against_catalog
from noema_lab.core.resource_guard import IsolatedJobSupervisor, ResourceExhausted
from noema_lab.core.storage import LocalStore
from noema_lab.core.structured_input import (
    StructuredInputError,
    decode_strict_yaml_or_json,
)
from noema_lab.core.submissions import validate_submission_bundle
from noema_lab.core.suites_catalog import SuiteDefinition, load_suites_catalog
from noema_lab.core.training import format_training_inspection_human, inspect_training_feasibility
from noema_lab.core.training_plans import apply_training_plan, load_training_plan
from noema_lab.core.verification import (
    format_verification_human,
    verify_benchmark_result,
    verify_run_bundle,
)
from noema_lab.core.variants import plan_recipe_variants, prepare_single_run_recipe
from noema_lab.ops import build_registry
from noema_lab.ops.source.kodak import download_kodak_dataset
from noema_lab.training.exporter import export_differentiable_scenario
from noema_lab.ui.server import serve_ui


def _parallel_workers_argument(value: str) -> int:
    """Parse a bounded executor worker count for recipe CLI commands."""

    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("parallel workers must be an integer") from exc
    if workers < 1 or workers > MAX_PARALLEL_WORKERS:
        raise argparse.ArgumentTypeError(
            "parallel workers must be between 1 and %d" % MAX_PARALLEL_WORKERS
        )
    return workers


def _kodak_limit_argument(value: str) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Kodak limit must be an integer") from exc
    if limit < 1 or limit > 24:
        raise argparse.ArgumentTypeError("Kodak limit must be between 1 and 24")
    return limit


def _dataset_capture_progress_label(split: Any) -> str:
    normalized = str(split or "").strip().lower().replace("-", "_").replace(" ", "_")
    split_labels = {
        "train": "Train",
        "validation": "Validation",
        "test": "Test",
        "held_out_test": "Held-out test",
    }
    label = split_labels.get(normalized)
    return "%s capture" % label if label else "Dataset capture"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="noema",
        description=(
            "Run learned-communication experiment contracts and inspect "
            "protocol-to-plot evidence."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + __version__,
    )
    parser.add_argument(
        "--workspace",
        default=".noema",
        help="Workspace for datasets, artifacts, and run records. Defaults to .noema.",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="External adapter manifest or directory to register. May be repeated. NOEMA_ADAPTER_PATHS is also honored.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ops_parser = subparsers.add_parser("ops", help="Inspect operation contracts")
    ops_subparsers = ops_parser.add_subparsers(dest="ops_command", required=True)
    ops_list = ops_subparsers.add_parser("list", help="List registered operations")
    ops_list.add_argument("--json", action="store_true", help="Emit JSON")
    ops_show = ops_subparsers.add_parser("show", help="Show one operation contract")
    ops_show.add_argument("operation_id")
    ops_show.add_argument("--json", action="store_true", help="Emit JSON")

    data_parser = subparsers.add_parser("data", help="Fetch or inspect sample datasets")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_fetch = data_subparsers.add_parser("fetch", help="Download a sample dataset")
    data_fetch.add_argument("dataset", choices=["kodak"])
    data_fetch.add_argument("--limit", type=_kodak_limit_argument, default=24)
    data_fetch.add_argument("--directory", default=None)
    data_fetch.add_argument("--json", action="store_true", help="Emit JSON")

    recipe_parser = subparsers.add_parser("recipe", help="Validate, graph, or run recipes")
    recipe_subparsers = recipe_parser.add_subparsers(dest="recipe_command", required=True)
    recipe_validate = recipe_subparsers.add_parser("validate", help="Validate a recipe")
    recipe_validate.add_argument("path")
    recipe_lint = recipe_subparsers.add_parser(
        "lint",
        help="Run strict shared-platform recipe invariant checks",
    )
    recipe_lint.add_argument("path")
    recipe_lint.add_argument("--json", action="store_true", help="Emit JSON")
    recipe_lint.add_argument(
        "--relaxed",
        action="store_true",
        help="Keep catalog status warnings as warnings instead of strict errors.",
    )
    recipe_specs = recipe_subparsers.add_parser(
        "specs",
        help="Show normalized dataset/task/benchmark specs for a recipe",
    )
    recipe_specs.add_argument("path")
    recipe_graph_parser = recipe_subparsers.add_parser("graph", help="Show recipe graph")
    recipe_graph_parser.add_argument("path")
    recipe_graph_parser.add_argument(
        "--format", choices=["text", "json", "dot"], default="text"
    )
    recipe_expand = recipe_subparsers.add_parser(
        "expand-matrix",
        help="Expand a recipe metadata.matrix into concrete recipe JSON documents",
    )
    recipe_expand.add_argument("path")
    recipe_run = recipe_subparsers.add_parser("run", help="Run a recipe locally")
    recipe_run.add_argument("path")
    recipe_run.add_argument(
        "--strict-lint",
        action="store_true",
        help="Run strict shared-platform lint checks before execution.",
    )
    recipe_run.add_argument(
        "--backend",
        default=None,
        help="Require this materialization backend for every recipe step.",
    )
    recipe_run.add_argument(
        "--implementation",
        default=None,
        help="Require this materialization implementation for every recipe step.",
    )
    recipe_run.add_argument(
        "--parallel-workers",
        type=_parallel_workers_argument,
        default=1,
        metavar="N",
        help=(
            "Execute independent DAG branches with up to N workers "
            "(1-%d, default: 1)." % MAX_PARALLEL_WORKERS
        ),
    )
    recipe_run.add_argument(
        "--no-plan-cache",
        action="store_true",
        help="Bypass execution-plan caching for this run.",
    )
    recipe_run_matrix = recipe_subparsers.add_parser(
        "run-matrix",
        help="Validate and run every concrete metadata.matrix variant in order",
    )
    recipe_run_matrix.add_argument("path")
    recipe_run_matrix.add_argument(
        "--strict-lint",
        action="store_true",
        help="Run strict shared-platform lint checks on every concrete variant.",
    )
    recipe_run_matrix.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining variants after a variant fails.",
    )
    recipe_run_matrix.add_argument(
        "--backend",
        default=None,
        help="Require this materialization backend for every recipe step.",
    )
    recipe_run_matrix.add_argument(
        "--implementation",
        default=None,
        help="Require this materialization implementation for every recipe step.",
    )
    recipe_run_matrix.add_argument(
        "--parallel-workers",
        type=_parallel_workers_argument,
        default=1,
        metavar="N",
        help=(
            "Execute independent DAG branches with up to N workers per variant "
            "(1-%d, default: 1)." % MAX_PARALLEL_WORKERS
        ),
    )
    recipe_run_matrix.add_argument(
        "--no-plan-cache",
        action="store_true",
        help="Bypass execution-plan caching for every matrix variant.",
    )

    template_parser = subparsers.add_parser(
        "template",
        help="Inspect or instantiate cataloged recipe starters",
    )
    template_subparsers = template_parser.add_subparsers(
        dest="template_command",
        required=True,
    )
    template_list = template_subparsers.add_parser(
        "list",
        help="List validated recipe starters",
    )
    template_list.add_argument("--json", action="store_true", help="Emit JSON")
    template_show = template_subparsers.add_parser(
        "show",
        help="Show one recipe starter contract",
    )
    template_show.add_argument("template_id")
    template_show.add_argument("--json", action="store_true", help="Emit JSON")
    template_instantiate = template_subparsers.add_parser(
        "instantiate",
        help="Instantiate a standalone recipe from a stable template ID",
    )
    template_instantiate.add_argument("template_id")
    template_instantiate.add_argument("--name", default=None)
    template_instantiate.add_argument("--description", default=None)
    template_instantiate.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="STEP.PARAM=JSON",
        help="Apply a typed step parameter override; may be repeated.",
    )
    template_instantiate.add_argument(
        "--json",
        action="store_true",
        help="Emit the recipe and provenance as JSON instead of recipe YAML.",
    )

    research_parser = subparsers.add_parser("research", help="Inspect research datasets, tasks, and metrics")
    research_subparsers = research_parser.add_subparsers(dest="research_command", required=True)
    research_subparsers.add_parser("catalog", help="Show the full research catalog")
    research_subparsers.add_parser("datasets", help="List cataloged datasets")
    research_subparsers.add_parser("tasks", help="List cataloged tasks")
    research_subparsers.add_parser("metrics", help="List cataloged metrics")
    research_show = research_subparsers.add_parser("show", help="Show one catalog entry")
    research_show.add_argument("kind", choices=["dataset", "task", "metric"])
    research_show.add_argument("id")
    research_validate = research_subparsers.add_parser(
        "validate-recipe",
        help="Validate a recipe against the research catalog",
    )
    research_validate.add_argument("path")
    research_validate.add_argument("--strict", action="store_true")

    suite_parser = subparsers.add_parser("suite", help="Inspect benchmark suites")
    suite_subparsers = suite_parser.add_subparsers(dest="suite_command", required=True)
    suite_list = suite_subparsers.add_parser("list", help="List benchmark suites")
    suite_list.add_argument("--json", action="store_true", help="Emit JSON")
    suite_show = suite_subparsers.add_parser("show", help="Show one benchmark suite")
    suite_show.add_argument("suite_id")
    suite_show.add_argument("--json", action="store_true", help="Emit JSON")
    suite_benchmarks = suite_subparsers.add_parser("benchmarks", help="List benchmark packs in one suite")
    suite_benchmarks.add_argument("suite_id")
    suite_benchmarks.add_argument("--json", action="store_true", help="Emit JSON")

    benchmark_parser = subparsers.add_parser("benchmark", help="Inspect or run benchmark packs")
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_command", required=True)
    benchmark_list = benchmark_subparsers.add_parser("list", help="List benchmark packs grouped by suite")
    benchmark_list.add_argument("--directory", default="benchmarks")
    benchmark_list.add_argument("--json", action="store_true", help="Emit JSON")
    benchmark_show = benchmark_subparsers.add_parser("show", help="Show one benchmark pack")
    benchmark_show.add_argument("path")
    benchmark_validate = benchmark_subparsers.add_parser("validate", help="Validate one benchmark pack")
    benchmark_validate.add_argument("path")
    benchmark_validate.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact human-readable validation summary instead of the full JSON report.",
    )
    benchmark_run = benchmark_subparsers.add_parser("run", help="Run every recipe in a benchmark pack")
    benchmark_run.add_argument("path")
    benchmark_run.add_argument(
        "--strict-lint",
        action="store_true",
        help="Require every benchmark recipe to pass strict shared-platform lint.",
    )
    benchmark_run.add_argument(
        "--backend",
        default=None,
        help="Require this materialization backend for every benchmark recipe step.",
    )
    benchmark_run.add_argument(
        "--implementation",
        default=None,
        help="Require this materialization implementation for every benchmark recipe step.",
    )
    benchmark_run.add_argument(
        "--parallel-workers",
        type=_parallel_workers_argument,
        default=1,
        metavar="N",
        help=(
            "Execute independent DAG branches with up to N workers per recipe "
            "(1-%d, default: 1)." % MAX_PARALLEL_WORKERS
        ),
    )
    benchmark_run.add_argument(
        "--no-plan-cache",
        action="store_true",
        help="Bypass execution-plan caching for every benchmark recipe.",
    )
    benchmark_run.add_argument(
        "--resume",
        metavar="FAILED_RESULT_ID",
        default=None,
        help=(
            "Explicitly resume a failed terminal result. Validated completed "
            "recipe evidence is reused; the failed row and remaining rows run "
            "with identical pack and execution settings."
        ),
    )
    benchmark_run.add_argument(
        "--retain-backing-runs",
        action="store_true",
        help=(
            "Keep redundant .noema/runs backing directories after their "
            "result-local evidence snapshots are verified. By default they "
            "are pruned to control benchmark disk usage."
        ),
    )
    benchmark_run.add_argument("--json", action="store_true", help="Emit JSON")
    benchmark_results = benchmark_subparsers.add_parser("results", help="List benchmark result bundles")
    benchmark_result_show = benchmark_subparsers.add_parser("result", help="Show one benchmark result bundle")
    benchmark_result_show.add_argument("result_id")
    benchmark_export = benchmark_subparsers.add_parser(
        "export",
        help="Regenerate CSV and Markdown reports for a benchmark result",
    )
    benchmark_export.add_argument("result_id")
    benchmark_publish = benchmark_subparsers.add_parser(
        "publish",
        help="Publish a verified benchmark result as a deterministic static demo",
    )
    benchmark_publish.add_argument("result_id")
    benchmark_publish.add_argument("--slug", required=True, help="Stable demo slug; must match metadata.demo.slug when declared.")
    benchmark_publish.add_argument("--out", required=True, help="Output directory for the static demo publication.")
    benchmark_publish.add_argument("--force", action="store_true", help="Replace an existing publication directory.")
    benchmark_publish.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Publish verifier warnings visibly; invalid or incomplete bundles are always rejected.",
    )
    benchmark_publish.add_argument("--json", action="store_true", help="Emit JSON")
    benchmark_verify = benchmark_subparsers.add_parser("verify", help="Verify one benchmark result bundle")
    benchmark_verify.add_argument("result_id")
    benchmark_verify.add_argument("--json", action="store_true", help="Emit JSON")
    benchmark_plot = benchmark_subparsers.add_parser("plot", help="Export a paper-ready figure from a benchmark result")
    benchmark_plot.add_argument("result_id")
    benchmark_plot.add_argument(
        "--plot",
        required=True,
        choices=["graceful-degradation", "packet-success", "channel-uses"],
        help="Plot type to export.",
    )
    benchmark_plot.add_argument("--out", required=True, help="Output image path. Relative paths are written inside the result bundle.")
    benchmark_plot.add_argument(
        "--y-metric",
        default=None,
        help="Optional metric id for the y-axis, for example quality.psnr_db or task.accuracy.",
    )
    benchmark_plot.add_argument("--x", dest="x_metric", default=None, help="Metric id for the x-axis, for example channel.snr_db.")
    benchmark_plot.add_argument("--y", dest="y_metric_explicit", default=None, help="Metric id for the y-axis, for example quality.psnr_db.")
    benchmark_plot.add_argument(
        "--group",
        default="method",
        help="Grouping key for curves. Use method, recipe, role, or metric:<metric_id>.",
    )
    benchmark_plot.add_argument(
        "--method-order",
        default=None,
        help="Comma-separated method/series order for the legend and curves.",
    )
    benchmark_plot.add_argument(
        "--style",
        default="noema",
        choices=["noema", "paper", "compact"],
        help="Built-in plot style preset.",
    )
    benchmark_plot.add_argument("--style-config", default=None, help="Optional JSON/YAML style override file.")
    benchmark_plot.add_argument(
        "--packet-success-panel",
        action="store_true",
        help="For graceful-degradation plots, add a secondary packet-success panel when the metric exists.",
    )
    benchmark_plot.add_argument("--outage-markers", dest="outage_markers", action="store_true", default=True, help="Mark outage points on digital cliff plots.")
    benchmark_plot.add_argument("--no-outage-markers", dest="outage_markers", action="store_false", help="Disable outage markers.")
    benchmark_plot.add_argument("--json", action="store_true", help="Emit JSON")

    submission_parser = subparsers.add_parser("submission", help="Validate benchmark submission bundles")
    submission_subparsers = submission_parser.add_subparsers(dest="submission_command", required=True)
    submission_validate = submission_subparsers.add_parser("validate", help="Validate a submission JSON/YAML file")
    submission_validate.add_argument("path")
    submission_validate.add_argument("--json", action="store_true", help="Emit JSON")

    adapter_parser = subparsers.add_parser("adapter", help="Create and validate external adapter SDK manifests")
    adapter_subparsers = adapter_parser.add_subparsers(dest="adapter_command", required=True)
    adapter_validate = adapter_subparsers.add_parser("validate", help="Validate an external adapter manifest")
    adapter_validate.add_argument("path")
    adapter_validate.add_argument("--json", action="store_true", help="Emit JSON")
    adapter_validate.add_argument(
        "--no-import",
        action="store_true",
        help="Validate manifest structure without importing adapter callables.",
    )
    adapter_scaffold = adapter_subparsers.add_parser("scaffold", help="Create a starter adapter manifest and Python module")
    adapter_scaffold.add_argument("directory")
    adapter_scaffold.add_argument("--name", default="example_external_codec")
    adapter_scaffold.add_argument(
        "--kind",
        choices=[
            "bits",
            "indices",
            "latents",
            "deepjscc_symbols",
            "classification_dataset",
            "classification_metric",
        ],
        default="bits",
    )
    adapter_scaffold.add_argument("--force", action="store_true")
    adapter_scaffold.add_argument("--json", action="store_true", help="Emit JSON")

    runs_parser = subparsers.add_parser("runs", help="Inspect run records")
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_subparsers.add_parser("list", help="List runs")
    runs_list.add_argument("--json", action="store_true", help="Emit JSON")
    runs_show = runs_subparsers.add_parser("show", help="Show one run summary")
    runs_show.add_argument("run_id")
    runs_show.add_argument("--json", action="store_true", help="Emit JSON")
    runs_manifest = runs_subparsers.add_parser("manifest", help="Show one run manifest")
    runs_manifest.add_argument("run_id")
    runs_manifest.add_argument("--json", action="store_true", help="Emit JSON")
    runs_verify = runs_subparsers.add_parser("verify", help="Verify one run/result bundle")
    runs_verify.add_argument("run_id")
    runs_verify.add_argument("--json", action="store_true", help="Emit JSON")

    diff_parser = subparsers.add_parser(
        "differentiable",
        help="Inspect replacement-to-loss training paths and export training contracts",
    )
    diff_subparsers = diff_parser.add_subparsers(dest="differentiable_command", required=True)
    diff_inspect = diff_subparsers.add_parser(
        "inspect",
        help="Inspect replacement targets, downstream gradient support, or capture-only training",
    )
    diff_inspect.add_argument("path")
    diff_inspect.add_argument(
        "--replacement",
        "--optimizable",
        dest="optimizable",
        metavar="STEP_IDS",
        default="",
        help=(
            "Optional comma-separated replacement step ids to inspect against the selected loss. "
            "--optimizable is retained as a compatibility alias."
        ),
    )
    diff_inspect.add_argument(
        "--loss",
        default="",
        help="Optional loss/evaluation step id. Defaults to metrics/evaluation steps or terminal steps.",
    )
    diff_inspect.add_argument(
        "--training-plan",
        default="",
        help="Optional separate training-plan YAML/JSON overlay; the source recipe remains unchanged.",
    )
    diff_inspect.add_argument("--json", action="store_true", help="Emit JSON")
    diff_export = diff_subparsers.add_parser(
        "export",
        help="Export an architecture-neutral training contract from a recipe",
    )
    diff_export.add_argument("path")
    diff_export.add_argument(
        "--training-plan",
        default="",
        help="Optional separate training-plan YAML/JSON overlay; the source recipe remains unchanged.",
    )
    diff_export.add_argument(
        "--replacement",
        "--optimizable",
        dest="optimizable",
        metavar="STEP_IDS",
        default="",
        help=(
            "Comma-separated replacement-target step ids, for example sender,receiver. "
            "May instead be supplied as selected_steps in --training-plan; --optimizable is a "
            "compatibility alias."
        ),
    )
    diff_export.add_argument(
        "--route-loss",
        default="",
        help=(
            "Optional comma-separated recipe loss/evaluation step ids that define the live "
            "downstream support DAG. May instead come from training-plan loss_steps."
        ),
    )
    diff_export.add_argument(
        "--framework",
        default="",
        choices=["torch-sionna", "torch"],
        help="Framework requested for support blocks; may instead come from --training-plan.",
    )
    diff_export.add_argument("--out", required=True, help="Output directory for generated differentiable-export files.")
    diff_export.add_argument("--force", action="store_true", help="Overwrite generated files in an existing directory.")
    diff_export.add_argument("--json", action="store_true", help="Emit JSON")

    dataset_capture_parser = subparsers.add_parser(
        "dataset-capture",
        help="Generate dataset-capture shards from researcher-selected recipe taps",
    )
    dataset_capture_subparsers = dataset_capture_parser.add_subparsers(dest="dataset_capture_command", required=True)
    dataset_capture_run = dataset_capture_subparsers.add_parser("run", help="Run a recipe and write dataset-capture tap outputs")
    dataset_capture_run.add_argument("path")
    dataset_capture_run.add_argument(
        "--training-plan",
        default="",
        help="Separate training-plan YAML/JSON containing the capture taps and split settings.",
    )
    dataset_capture_run.add_argument("--out", required=True, help="Output dataset-capture bundle directory")
    dataset_capture_run.add_argument("--force", action="store_true", help="Overwrite files in an existing dataset-capture directory")
    dataset_capture_run.add_argument(
        "--json",
        action="store_true",
        help="Emit final JSON and suppress live progress unless --progress is set.",
    )
    dataset_capture_progress = dataset_capture_run.add_mutually_exclusive_group()
    dataset_capture_progress.add_argument(
        "--progress",
        dest="progress",
        action="store_const",
        const=True,
        help="Show capture progress on stderr even when it is not a terminal.",
    )
    dataset_capture_progress.add_argument(
        "--no-progress",
        dest="progress",
        action="store_const",
        const=False,
        help="Disable capture progress.",
    )
    dataset_capture_run.set_defaults(progress=None)

    agentic_parser = subparsers.add_parser(
        "agentic",
        help="Validate, run, replay, or verify agentic supervisory experiments",
    )
    agentic_subparsers = agentic_parser.add_subparsers(
        dest="agentic_command",
        required=True,
    )
    agentic_validate = agentic_subparsers.add_parser(
        "validate",
        help="Validate an agentic supervisory experiment contract",
    )
    agentic_validate.add_argument("path")
    agentic_validate.add_argument("--json", action="store_true", help="Emit JSON")

    agentic_run = agentic_subparsers.add_parser(
        "run",
        help="Run an agent and its declared paired comparators",
    )
    agentic_run.add_argument("path")
    agentic_run.add_argument(
        "--provider",
        default=None,
        help="Override the configured decision provider for this recorded campaign",
    )
    agentic_run.add_argument("--model", default=None, help="Override the model identifier")
    agentic_run.add_argument(
        "--model-revision",
        default=None,
        help="Override the immutable local-model revision",
    )
    agentic_run.add_argument(
        "--endpoint",
        default=None,
        help="Override the Ollama or compatible HTTP endpoint",
    )
    agentic_run.add_argument(
        "--base-url",
        default=None,
        help="Alias for --endpoint for OpenAI-compatible providers",
    )
    agentic_run.add_argument(
        "--api-key-env",
        default=None,
        help="Dedicated NOEMA_AGENT_* environment variable containing the API key",
    )
    agentic_run.add_argument(
        "--prompt",
        default=None,
        help="Override the recorded prompt file",
    )
    agentic_run.add_argument(
        "--device",
        default=None,
        help="Override the local Transformers device, for example cpu",
    )
    agentic_run.add_argument(
        "--out",
        default=None,
        help="Campaign output directory; defaults below WORKSPACE/agentic",
    )
    agentic_run.add_argument("--json", action="store_true", help="Emit JSON")

    agentic_verify = agentic_subparsers.add_parser(
        "verify",
        help="Verify a completed agentic campaign and its evidence bindings",
    )
    agentic_verify.add_argument("campaign")
    agentic_verify.add_argument("--json", action="store_true", help="Emit JSON")

    agentic_replay = agentic_subparsers.add_parser(
        "replay",
        help="Replay recorded effective actions without calling the model",
    )
    agentic_replay.add_argument("campaign")
    agentic_replay.add_argument(
        "--out",
        default=None,
        help="Replay output directory; defaults below WORKSPACE/agentic",
    )
    agentic_replay.add_argument("--json", action="store_true", help="Emit JSON")

    ui_parser = subparsers.add_parser("ui", help="Run the dashboard server")
    ui_subparsers = ui_parser.add_subparsers(dest="ui_command", required=True)
    ui_serve = ui_subparsers.add_parser("serve", help="Serve the local dashboard")
    ui_serve.add_argument("--host", default="127.0.0.1")
    ui_serve.add_argument("--port", type=int, default=8765)
    ui_serve.add_argument("--project-root", default=".")
    return parser



def _benchmark_pack_row(pack: Any, catalog: Any) -> Dict[str, Any]:
    suite = _resolve_pack_suite(pack, catalog)
    if suite is not None:
        suite_payload = {
            "id": suite.id,
            "name": suite.name,
            "status": suite.status,
            "version": suite.version,
        }
    elif pack.suite:
        suite_payload = {
            "id": str(pack.suite.get("id") or "uncategorized"),
            "name": str(pack.suite.get("name") or pack.suite.get("id") or "Uncategorized"),
            "status": str(pack.suite.get("status") or "unknown"),
            "version": str(pack.suite.get("version") or ""),
        }
    else:
        suite_payload = {
            "id": "uncategorized",
            "name": "Uncategorized",
            "status": "unknown",
            "version": "",
        }
    return {
        "id": pack.id,
        "version": pack.version,
        "name": pack.name,
        "path": str(pack.path),
        "recipe_count": len(pack.recipes),
        "suite": suite_payload,
    }


def _resolve_pack_suite(pack: Any, catalog: Any) -> Optional[SuiteDefinition]:
    suite_id = str((pack.suite or {}).get("id") or "")
    if suite_id and catalog.suite(suite_id) is not None:
        return catalog.suite(suite_id)
    for suite in catalog.suites.values():
        for benchmark in suite.benchmark_packs:
            if benchmark.id == pack.id:
                return suite
            if pack.path is not None and benchmark.path == str(pack.path):
                return suite
    return None


def _group_benchmark_rows(rows: List[Dict[str, Any]], catalog: Any) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        suite = row["suite"]
        suite_id = suite["id"]
        if suite_id not in grouped:
            grouped[suite_id] = {"suite": suite, "benchmarks": []}
        grouped[suite_id]["benchmarks"].append(row)
    ordered_ids = [suite_id for suite_id in catalog.suites if suite_id in grouped]
    ordered_ids.extend(sorted(suite_id for suite_id in grouped if suite_id not in set(ordered_ids)))
    return [grouped[suite_id] for suite_id in ordered_ids]


def _benchmark_list_payload(packs: List[Any], catalog: Any) -> Dict[str, Any]:
    rows = [_benchmark_pack_row(pack, catalog) for pack in packs]
    return {
        "schema_version": 1,
        "benchmarks": rows,
        "suites": _group_benchmark_rows(rows, catalog),
    }


def _format_benchmark_list_human(payload: Dict[str, Any]) -> str:
    if not payload["benchmarks"]:
        return "no benchmark packs found"
    lines: List[str] = []
    for group in payload["suites"]:
        suite = group["suite"]
        status = suite.get("status") or "unknown"
        lines.append("%s [%s, %s]" % (suite.get("name") or suite["id"], suite["id"], status))
        for benchmark in group["benchmarks"]:
            name = benchmark.get("name") or benchmark["id"]
            lines.append(
                "  %s\t%s\t%d recipes\t%s"
                % (benchmark["id"], benchmark["version"], benchmark["recipe_count"], benchmark["path"])
            )
            if name != benchmark["id"]:
                lines.append("    %s" % name)
    return "\n".join(lines)


def _format_benchmark_validation_human(payload: Dict[str, Any]) -> str:
    recipes = list(payload.get("recipes") or [])
    role_counts: Dict[str, int] = {}
    lint_counts: Dict[str, int] = {}
    warning_count = 0
    error_count = 0
    for recipe in recipes:
        role = str(recipe.get("role") or "unspecified")
        role_counts[role] = role_counts.get(role, 0) + 1
        lint = dict(recipe.get("lint") or {})
        status = str(lint.get("status") or "unknown")
        lint_counts[status] = lint_counts.get(status, 0) + 1
        warning_count += int(lint.get("warning_count") or 0)
        error_count += int(lint.get("error_count") or 0)

    suite = dict(payload.get("suite") or {})
    catalog = dict(payload.get("catalog_validation") or {})
    lines = [
        "validated benchmark: %s" % payload.get("name", payload.get("id", "unknown")),
        "id: %s" % payload.get("id", "unknown"),
        "suite: %s (%s)"
        % (suite.get("name", suite.get("id", "unknown")), suite.get("version", "unknown")),
        "recipes: %d%s"
        % (
            int(payload.get("recipe_count") or len(recipes)),
            " [%s]"
            % ", ".join(
                "%s=%d" % (role, count) for role, count in sorted(role_counts.items())
            )
            if role_counts
            else "",
        ),
        "catalog: %s" % catalog.get("status", "unknown"),
        "lint: %s; warnings=%d; errors=%d"
        % (
            ", ".join(
                "%s=%d" % (status, count)
                for status, count in sorted(lint_counts.items())
            )
            or "not reported",
            warning_count,
            error_count,
        ),
        "sha256: %s" % payload.get("sha256", "unknown"),
    ]
    return "\n".join(lines)


def _format_suite_list_human(catalog: Any) -> str:
    if not catalog.suites:
        return "no suites registered"
    lines = []
    for suite in catalog.suites.values():
        lines.append("%s\t%s\t%s\t%d benchmark packs" % (suite.status, suite.id, suite.name, len(suite.benchmark_packs)))
    return "\n".join(lines)


def _format_suite_show_human(suite: SuiteDefinition) -> str:
    lines = [
        "suite: %s" % suite.id,
        "name: %s" % suite.name,
        "status: %s" % suite.status,
        "version: %s" % suite.version,
    ]
    if suite.summary:
        lines.append("summary: %s" % suite.summary)
    if suite.docs:
        lines.append("docs:")
        for label, path in sorted(suite.docs.items()):
            lines.append("  %s: %s" % (label, path))
    if suite.supported_tasks:
        lines.append("supported tasks: %s" % ", ".join(suite.supported_tasks))
    if suite.planned_tasks:
        lines.append("planned tasks: %s" % ", ".join(suite.planned_tasks))
    lines.append("benchmark packs: %d" % len(suite.benchmark_packs))
    for benchmark in suite.benchmark_packs:
        lines.append("  %s\t%s\t%s\t%s" % (benchmark.status, benchmark.id, benchmark.task, benchmark.path))
    return "\n".join(lines)


def _format_suite_benchmarks_human(suite: SuiteDefinition) -> str:
    if not suite.benchmark_packs:
        return "%s has no benchmark packs yet" % suite.id
    lines = ["%s benchmark packs:" % suite.id]
    for benchmark in suite.benchmark_packs:
        lines.append("  %s\t%s\t%s\t%s" % (benchmark.status, benchmark.id, benchmark.task, benchmark.path))
    return "\n".join(lines)


def _template_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.name is not None:
        overrides["name"] = args.name
    if args.description is not None:
        overrides["description"] = args.description
    step_params: Dict[str, Dict[str, Any]] = {}
    for expression in args.set:
        target, separator, raw_value = str(expression).partition("=")
        step_id, dot, param_name = target.partition(".")
        if not separator or not dot or not step_id or not param_name:
            raise ValueError(
                "template --set must use STEP.PARAM=JSON; got %s" % expression
            )
        params = step_params.setdefault(step_id, {})
        if param_name in params:
            raise ValueError(
                "template --set repeats %s.%s" % (step_id, param_name)
            )
        try:
            params[param_name] = decode_strict_yaml_or_json(
                raw_value,
                input_format="json",
            )
        except StructuredInputError as exc:
            raise ValueError(
                "template --set value for %s.%s must be valid JSON: %s"
                % (step_id, param_name, exc)
            ) from exc
    if step_params:
        overrides["step_params"] = step_params
    return overrides


def _execution_controls_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return normalize_execution_controls(
        {
            "strict_lint": bool(getattr(args, "strict_lint", False)),
            "backend": getattr(args, "backend", None),
            "implementation": getattr(args, "implementation", None),
            "parallel_workers": getattr(args, "parallel_workers", 1),
            "use_plan_cache": not bool(getattr(args, "no_plan_cache", False)),
        }
    )


def _inspection_store(workspace: Path) -> LocalStore:
    """Open an existing workspace for inspection without creating anything."""

    if not workspace.is_dir():
        raise FileNotFoundError(
            "Workspace does not exist or is not a directory: %s" % workspace
        )
    store = LocalStore(workspace)
    if not store.runs_dir.is_dir():
        raise FileNotFoundError(
            "Workspace has no runs directory: %s" % store.runs_dir
        )
    return store


def _cli_json_error_envelope(
    exc: Exception,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "command": getattr(args, "command", None),
    }
    command = str(getattr(args, "command", "") or "")
    subcommand_name = "%s_command" % command.replace("-", "_")
    subcommand = getattr(args, subcommand_name, None)
    if subcommand is not None:
        context["subcommand"] = subcommand
    for name in (
        "path",
        "directory",
        "out",
        "run_id",
        "result_id",
        "operation_id",
        "template_id",
    ):
        value = getattr(args, name, None)
        if value not in (None, ""):
            context[name] = _json_safe_error_value(value)

    for attribute in (
        "noema_run_id",
        "noema_run_dir",
        "result_id",
        "result_path",
        "payload",
        "evidence",
    ):
        value = getattr(exc, attribute, None)
        if value not in (None, "", {}, []):
            context[attribute] = _json_safe_error_value(value)
    if isinstance(exc, OSError):
        if exc.errno is not None:
            context["errno"] = exc.errno
        if exc.filename:
            context["filename"] = exc.filename
    cause = exc.__cause__
    if cause is not None:
        context["cause"] = {
            "type": type(cause).__name__,
            "message": str(cause),
        }
    return {
        "status": "error",
        "error": {
            "type": type(exc).__name__,
            "module": type(exc).__module__,
            "message": str(exc),
            "context": context,
        },
    }


def _json_safe_error_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_error_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_error_value(item) for item in value]
    return str(value)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = Path(args.workspace)

    try:
        adapter_paths = [] if args.command == "adapter" else (args.adapter or None)
        registry = build_registry(adapter_paths)
        if args.command == "adapter":
            if args.adapter_command == "validate":
                payload = validate_adapter_manifest(
                    Path(args.path),
                    registry,
                    import_callables=not args.no_import,
                )
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print("valid adapter: %s (%d operations)" % (payload["name"], payload["operation_count"]))
                    for operation in payload["operations"]:
                        print("%s\twraps %s\t%s" % (operation["id"], operation["wraps"], operation["callable"]))
                return 0
            if args.adapter_command == "scaffold":
                payload = scaffold_adapter(
                    Path(args.directory),
                    name=args.name,
                    kind=args.kind,
                    force=args.force,
                )
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print("created adapter scaffold: %s" % payload["directory"])
                    print("manifest: %s" % payload["manifest"])
                    print("module: %s" % payload["module"])
                    print("validate: noema adapter validate %s" % payload["manifest"])
                return 0

        if args.command == "ops":
            if args.ops_command == "list":
                operations = registry.describe()
                if args.json:
                    print(json.dumps({"operations": operations}, indent=2, sort_keys=True))
                else:
                    for operation in operations:
                        print(
                            "%s\t%s\t%s"
                            % (operation["id"], operation["status"], operation["name"])
                        )
                return 0
            if args.ops_command == "show":
                operation = registry.get(args.operation_id).describe()
                if args.json:
                    print(json.dumps(operation, indent=2, sort_keys=True))
                else:
                    print("%s\t%s" % (operation["id"], operation["name"]))
                    print("status\t%s" % operation["status"])
                    print("inputs\t%s" % json.dumps(operation["input_kinds"], sort_keys=True))
                    print("outputs\t%s" % json.dumps(operation["output_kinds"], sort_keys=True))
                    print("params\t%s" % json.dumps(operation["params_schema"], sort_keys=True))
                    if "differentiability" in operation:
                        print("differentiability\t%s" % json.dumps(operation["differentiability"], sort_keys=True))
                    if "backends" in operation:
                        print("backends\t%s" % json.dumps(operation["backends"], sort_keys=True))
                    if "equivalence" in operation:
                        print("equivalence\t%s" % json.dumps(operation["equivalence"], sort_keys=True))
                    if "formats" in operation:
                        print("formats\t%s" % json.dumps(operation["formats"], sort_keys=True))
                    if "materializations" in operation:
                        print("materializations\t%s" % json.dumps(operation["materializations"], sort_keys=True))
                return 0

        if args.command == "data" and args.data_command == "fetch":
            destination = Path(args.directory) if args.directory else workspace / "datasets" / args.dataset
            if args.dataset == "kodak":
                paths = download_kodak_dataset(destination, limit=args.limit)
                payload = {
                    "dataset": "kodak",
                    "directory": str(destination),
                    "count": len(paths),
                    "files": [str(path) for path in paths],
                }
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print("downloaded kodak\t%s\t%d files" % (destination, len(paths)))
                return 0

        if args.command == "template":
            if args.template_command == "instantiate":
                result = instantiate_recipe_template(
                    args.template_id,
                    Path.cwd(),
                    registry,
                    overrides=_template_overrides_from_args(args),
                )
                if args.json:
                    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
                else:
                    print(yaml.safe_dump(result.recipe.to_dict(), sort_keys=False).rstrip())
                return 0
            inspection = inspect_recipe_template_catalog(Path.cwd(), registry)
            if args.template_command == "list":
                payload = inspection.to_dict()
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    for row in inspection.rows:
                        print(
                            "%s\t%s\t%s\t%s"
                            % (
                                row.validation_status,
                                row.template.id,
                                row.template.task_id,
                                row.template.label,
                            )
                        )
                return 0 if inspection.status == "valid" else 1
            if args.template_command == "show":
                row = next(
                    (
                        item
                        for item in inspection.rows
                        if item.template.id == args.template_id
                    ),
                    None,
                )
                if row is None:
                    raise ValueError("Unknown recipe template id: %s" % args.template_id)
                payload = row.to_dict()
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print(yaml.safe_dump(payload, sort_keys=False).rstrip())
                return 0 if row.valid else 1
        if args.command == "recipe":
            recipe = load_recipe(
                Path(args.path),
                mode="strict",
                registry=registry,
                effective=False,
            )
            if args.recipe_command == "validate":
                variant_plan = plan_recipe_variants(recipe, registry)
                suffix = (
                    ", %d matrix variants" % variant_plan.expanded_count
                    if variant_plan.canonical_matrix.enabled
                    else ""
                )
                print(
                    "valid recipe: %s (%d steps%s)"
                    % (recipe.name, len(recipe.steps), suffix)
                )
                return 0
            if args.recipe_command == "lint":
                report = lint_recipe_invariants(recipe, registry, strict=not args.relaxed)
                if args.json:
                    print(json.dumps(report, indent=2, sort_keys=True))
                else:
                    if report["status"] == "passed":
                        print(
                            "lint passed: %s (%s, %d warnings)"
                            % (
                                report["recipe"],
                                ", ".join(report["profiles"]),
                                report["warning_count"],
                            )
                        )
                    else:
                        print(
                            "lint failed: %s (%d errors, %d warnings)"
                            % (report["recipe"], report["error_count"], report["warning_count"])
                        )
                        for issue in report["issues"]:
                            location = " [%s]" % issue["step_id"] if issue.get("step_id") else ""
                            print(
                                "%s %s%s: %s"
                                % (
                                    issue["severity"],
                                    issue["code"],
                                    location,
                                    issue["message"],
                                )
                            )
                return 0 if report["status"] == "passed" else 1
            if args.recipe_command == "specs":
                validate_recipe_against_registry(recipe, registry)
                print(json.dumps(research_specs_from_recipe(recipe), indent=2, sort_keys=True))
                return 0
            if args.recipe_command == "graph":
                validate_recipe_against_registry(recipe, registry)
                graph = recipe_graph(recipe, registry)
                if args.format == "json":
                    print(json.dumps(graph, indent=2, sort_keys=True))
                elif args.format == "dot":
                    print(format_graph_dot(graph))
                else:
                    print(format_graph_text(graph))
                return 0
            if args.recipe_command == "expand-matrix":
                expansion = plan_recipe_variants(recipe, registry).to_expansion_dict()
                print(json.dumps(expansion, indent=2, sort_keys=True))
                return 0
            if args.recipe_command == "run":
                recipe = prepare_single_run_recipe(recipe, registry)
                execution = _execution_controls_from_args(args)
                if execution["strict_lint"]:
                    require_strict_lint(recipe, registry, context=recipe.name)
                store = LocalStore(workspace)
                run_dir = LocalExecutor(registry, store).run(
                    recipe,
                    **executor_options(execution),
                )
                print("completed run: %s" % run_dir.name)
                print("summary: %s" % (run_dir / "summary.json"))
                return 0
            if args.recipe_command == "run-matrix":
                variant_plan = plan_recipe_variants(recipe, registry)
                if not variant_plan.canonical_matrix.enabled:
                    raise RecipeValidationError(
                        "recipe run-matrix requires an enabled metadata.matrix"
                    )
                execution = _execution_controls_from_args(args)
                if execution["strict_lint"]:
                    for variant in variant_plan.variants:
                        require_strict_lint(
                            variant.recipe,
                            registry,
                            context=variant.matrix_variant_id,
                        )
                store = LocalStore(workspace)
                executor = LocalExecutor(registry, store)
                failures = []
                for variant in variant_plan.variants:
                    try:
                        run_dir = executor.run(
                            variant.recipe,
                            **executor_options(execution),
                        )
                        print(
                            "completed variant %d/%d: %s -> %s"
                            % (
                                variant.matrix_index + 1,
                                variant_plan.expanded_count,
                                variant.matrix_variant_id,
                                run_dir.name,
                            )
                        )
                    except Exception as exc:
                        failures.append((variant.matrix_variant_id, str(exc)))
                        if not args.continue_on_error:
                            raise
                        print(
                            "failed variant %s: %s"
                            % (variant.matrix_variant_id, exc),
                            file=sys.stderr,
                        )
                if failures:
                    raise RuntimeError(
                        "%d/%d matrix variants failed"
                        % (len(failures), variant_plan.expanded_count)
                    )
                return 0

        if args.command == "research":
            catalog = load_research_catalog()
            if args.research_command == "catalog":
                print(json.dumps(catalog.to_dict(), indent=2, sort_keys=True))
                return 0
            if args.research_command == "datasets":
                print(json.dumps({"datasets": [item.to_dict() for item in catalog.datasets.values()]}, indent=2, sort_keys=True))
                return 0
            if args.research_command == "tasks":
                print(json.dumps({"tasks": [item.to_dict() for item in catalog.tasks.values()]}, indent=2, sort_keys=True))
                return 0
            if args.research_command == "metrics":
                print(json.dumps({"metrics": [item.to_dict() for item in catalog.metrics.values()]}, indent=2, sort_keys=True))
                return 0
            if args.research_command == "show":
                collection = {
                    "dataset": catalog.datasets,
                    "task": catalog.tasks,
                    "metric": catalog.metrics,
                }[args.kind]
                item = collection.get(args.id)
                if not item:
                    raise FileNotFoundError("%s is not in the research catalog: %s" % (args.kind, args.id))
                print(json.dumps(item.to_dict(), indent=2, sort_keys=True))
                return 0
            if args.research_command == "validate-recipe":
                recipe = load_recipe(Path(args.path))
                validate_recipe_against_registry(recipe, registry)
                specs = research_specs_from_recipe(recipe)
                validation = validate_research_specs_against_catalog(specs, catalog, strict=args.strict)
                print(
                    json.dumps(
                        {
                            "recipe": recipe.name,
                            "research": specs,
                            "catalog_validation": validation,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return (
                    1
                    if args.strict and validation.get("status") != "valid"
                    else 0
                )

        if args.command == "suite":
            catalog = load_suites_catalog()
            if args.suite_command == "list":
                if args.json:
                    print(json.dumps(catalog.to_dict(), indent=2, sort_keys=True))
                else:
                    print(_format_suite_list_human(catalog))
                return 0
            suite = catalog.suite(args.suite_id)
            if suite is None:
                message = "unknown suite: %s" % args.suite_id
                if args.json:
                    print(
                        json.dumps(
                            _cli_json_error_envelope(
                                FileNotFoundError(message),
                                args,
                            ),
                            indent=2,
                            sort_keys=True,
                        )
                    )
                else:
                    print(message, file=sys.stderr)
                return 1
            if args.suite_command == "show":
                if args.json:
                    print(json.dumps(suite.to_dict(), indent=2, sort_keys=True))
                else:
                    print(_format_suite_show_human(suite))
                return 0
            if args.suite_command == "benchmarks":
                payload = {
                    "schema_version": catalog.schema_version,
                    "suite": {
                        "id": suite.id,
                        "name": suite.name,
                        "status": suite.status,
                        "version": suite.version,
                    },
                    "benchmarks": [benchmark.to_dict() for benchmark in suite.benchmark_packs],
                }
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print(_format_suite_benchmarks_human(suite))
                return 0

        if args.command == "benchmark":
            store = LocalStore(workspace)
            project_root = Path(".").resolve()
            if args.benchmark_command == "list":
                packs = list_benchmark_packs(Path(args.directory))
                payload = _benchmark_list_payload(packs, load_suites_catalog())
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print(_format_benchmark_list_human(payload))
                return 0
            if args.benchmark_command == "show":
                pack = load_benchmark_pack(Path(args.path))
                print(json.dumps(pack.to_dict(), indent=2, sort_keys=True))
                return 0
            if args.benchmark_command == "validate":
                pack = load_benchmark_pack(Path(args.path))
                validation = validate_benchmark_pack(pack, registry, project_root)
                if args.summary:
                    print(_format_benchmark_validation_human(validation))
                else:
                    print(json.dumps(validation, indent=2, sort_keys=True))
                return 0
            if args.benchmark_command == "run":
                pack_path = Path(args.path).resolve()
                pack = load_benchmark_pack(pack_path)
                execution = _execution_controls_from_args(args)
                validate_benchmark_pack(
                    pack,
                    registry,
                    project_root,
                    strict_lint=bool(execution["strict_lint"]),
                )
                supervisor = IsolatedJobSupervisor(
                    job_id=uuid.uuid4().hex,
                    request={
                        "kind": "benchmark",
                        "workspace": str(workspace.resolve()),
                        "project_root": str(project_root),
                        "adapter_paths": list(args.adapter or []),
                        "pack_path": str(pack_path),
                        "execution": dict(execution),
                        "resume_result_id": args.resume,
                        "retain_backing_runs": bool(args.retain_backing_runs),
                    },
                    workspace=workspace,
                    project_root=project_root,
                )
                try:
                    outcome = supervisor.run()
                except ResourceExhausted as exc:
                    finalize_resource_exhausted_benchmark(
                        store,
                        str(exc.payload.get("result_id") or ""),
                        str(exc),
                        exc.evidence,
                    )
                    raise
                result_id = str(outcome.payload["result_id"])
                benchmark_dir = store.get_benchmark_result_dir(result_id)
                result = store.get_benchmark_result(result_id)
                write_benchmark_resource_guard_evidence(
                    store,
                    result_id,
                    outcome.evidence,
                )
                if str(result.get("status") or "").lower() != "completed":
                    message = (
                        result.get("incomplete_reason")
                        or "benchmark did not complete every declared method"
                    )
                    if args.json:
                        error = BenchmarkError(str(message))
                        error.result_id = benchmark_dir.name
                        error.result_path = str(benchmark_dir / "result.json")
                        raise error
                    print(
                        "incomplete benchmark: %s" % benchmark_dir.name,
                        file=sys.stderr,
                    )
                    print(
                        "result: %s" % (benchmark_dir / "result.json"),
                        file=sys.stderr,
                    )
                    print(
                        "error: %s"
                        % (
                            message
                        ),
                        file=sys.stderr,
                    )
                    return 1
                if args.json:
                    print(
                        json.dumps(
                            {
                                "status": "completed",
                                "result_id": benchmark_dir.name,
                                "result": str(benchmark_dir / "result.json"),
                                "metrics": str(benchmark_dir / "metrics.csv"),
                                "recipes": str(benchmark_dir / "recipes.csv"),
                                "summary": str(benchmark_dir / "summary.md"),
                                "execution": execution,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                    )
                    return 0
                print("completed benchmark: %s" % benchmark_dir.name)
                print("result: %s" % (benchmark_dir / "result.json"))
                print("metrics: %s" % (benchmark_dir / "metrics.csv"))
                print("recipes: %s" % (benchmark_dir / "recipes.csv"))
                print("summary: %s" % (benchmark_dir / "summary.md"))
                return 0
            if args.benchmark_command == "results":
                print(json.dumps({"results": store.list_benchmark_results(read_only=True)}, indent=2, sort_keys=True))
                return 0
            if args.benchmark_command == "result":
                result_dir = store.get_benchmark_result_dir(args.result_id)
                result = store.get_benchmark_result(args.result_id)
                plots = load_benchmark_plot_records(result_dir)
                if plots:
                    result["plots"] = plots
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0
            if args.benchmark_command == "export":
                result_dir = store.get_benchmark_result_dir(args.result_id)
                result = store.get_benchmark_result(args.result_id)
                reproduce_benchmark_report_artifacts(result_dir, result)
                reports = result.get("reports") or {}
                print(json.dumps({"result_id": args.result_id, "reports": reports}, indent=2, sort_keys=True))
                return 0
            if args.benchmark_command == "publish":
                payload = publish_benchmark_demo(
                    store,
                    args.result_id,
                    Path(args.out),
                    args.slug,
                    registry=registry,
                    project_root=project_root,
                    force=args.force,
                    allow_warnings=args.allow_warnings,
                )
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print("published demo: %s" % payload["index"])
                    print("publication sha256: %s" % payload["publication_sha256"])
                    print("manifest: %s" % payload["manifest"])
                return 0
            if args.benchmark_command == "verify":
                report = verify_benchmark_result(store, args.result_id, registry=registry)
                if args.json:
                    print(json.dumps(report, indent=2, sort_keys=True))
                else:
                    print(format_verification_human(report))
                return 0 if report["status"] == "valid" else 1
            if args.benchmark_command == "plot":
                payload = plot_benchmark_result(
                    store,
                    args.result_id,
                    args.plot,
                    Path(args.out),
                    y_metric=args.y_metric_explicit or args.y_metric,
                    x_metric=args.x_metric,
                    group_by=args.group,
                    method_order=args.method_order,
                    style=args.style,
                    style_config=Path(args.style_config) if args.style_config else None,
                    outage_markers=args.outage_markers,
                    packet_success_panel=args.packet_success_panel,
                )
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print("exported plot: %s" % payload["plot"]["path"])
                    print("data: %s" % payload["plot"]["data_csv_path"])
                    print("summary: %s" % payload["reports"]["summary_markdown"]["path"])
                return 0

        if args.command == "submission":
            if args.submission_command == "validate":
                payload = validate_submission_bundle(Path(args.path), registry=registry)
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print("%s submission: %s" % (payload["status"], payload["submission"]))
                    for message in payload["errors"]:
                        print("error: %s" % message)
                    for message in payload["warnings"]:
                        print("warning: %s" % message)
                return 0 if payload["status"] == "valid" else 1

        if args.command == "runs":
            store = _inspection_store(workspace)
            if args.runs_command == "list":
                runs = store.list_runs(read_only=True)
                if args.json:
                    print(json.dumps({"runs": runs}, indent=2, sort_keys=True))
                    return 0
                for run in runs:
                    print("%s\t%s\t%s" % (run["run_id"], run["status"], run["recipe_name"]))
                return 0
            if args.runs_command == "show":
                print(json.dumps(store.get_run(args.run_id), indent=2, sort_keys=True))
                return 0
            if args.runs_command == "manifest":
                print(json.dumps(store.get_manifest(args.run_id), indent=2, sort_keys=True))
                return 0
            if args.runs_command == "verify":
                report = verify_run_bundle(store, args.run_id, registry=registry)
                if args.json:
                    print(json.dumps(report, indent=2, sort_keys=True))
                else:
                    print(format_verification_human(report))
                return 0 if report["status"] == "valid" else 1

        if args.command == "differentiable":
            if args.differentiable_command == "inspect":
                recipe = load_recipe(Path(args.path))
                training_plan = None
                if args.training_plan:
                    training_plan = load_training_plan(args.training_plan)
                    recipe = apply_training_plan(recipe, training_plan)
                report = inspect_training_feasibility(
                    recipe,
                    registry,
                    optimizable_steps=(
                        args.optimizable
                        or (",".join(training_plan.selected_steps) if training_plan else "")
                        or None
                    ),
                    loss=(
                        args.loss
                        or (
                            ",".join(training_plan.loss_steps)
                            if training_plan
                            else ""
                        )
                        or None
                    ),
                )
                if args.json:
                    print(json.dumps(report, indent=2, sort_keys=True))
                else:
                    print(format_training_inspection_human(report))
                return 0
            if args.differentiable_command == "export":
                recipe_path = Path(args.path)
                recipe = load_recipe(recipe_path)
                training_plan = None
                if args.training_plan:
                    training_plan = load_training_plan(args.training_plan)
                    recipe = apply_training_plan(recipe, training_plan)
                optimizable_steps = (
                    args.optimizable
                    or (",".join(training_plan.selected_steps) if training_plan else "")
                )
                if not optimizable_steps:
                    raise ValueError(
                        "Choose replacement targets with --replacement or selected_steps in --training-plan"
                    )
                payload = export_differentiable_scenario(
                    recipe,
                    registry,
                    optimizable_steps=optimizable_steps,
                    route_loss_steps=(
                        args.route_loss
                        or (
                            ",".join(training_plan.loss_steps)
                            if training_plan
                            else ""
                        )
                        or None
                    ),
                    loss="",
                    framework=args.framework or (training_plan.framework if training_plan else "") or "torch",
                    out_dir=Path(args.out),
                    source_path=recipe_path,
                    project_root=Path.cwd(),
                    force=args.force,
                    exporter="auto",
                    include_starter=False,
                    training_objective=(
                        training_plan.objective if training_plan else ""
                    ),
                )
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print("exported training contract: %s" % payload["out_dir"])
                    print("recipe: %s" % payload["recipe"])
                    print("recipe sha256: %s" % payload["recipe_sha256"])
                    print("replacement targets: %s" % ", ".join(payload["optimizable_steps"]))
                    print("validate: cd %s && python validate_contract.py" % payload["out_dir"])
                return 0

        if args.command == "dataset-capture" and args.dataset_capture_command == "run":
            recipe = load_recipe(Path(args.path))
            if args.training_plan:
                recipe = apply_training_plan(recipe, load_training_plan(args.training_plan))
            store = LocalStore(workspace)
            progress_enabled = args.progress
            if progress_enabled is None and args.json:
                progress_enabled = False
            with CliProgressRenderer(
                stream=sys.stderr,
                enabled=progress_enabled,
                label=_dataset_capture_progress_label(
                    (recipe.dataset_capture or {}).get("split")
                ),
            ) as progress:
                payload = run_dataset_capture_recipe(
                    recipe,
                    registry,
                    store,
                    Path(args.out),
                    force=args.force,
                    progress_sink=progress.update if progress.enabled else None,
                )
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print("captured dataset: %s" % payload["out_dir"])
                print("recipe: %s" % payload["recipe"])
                print("split: %s" % payload["split"])
                print("taps: %d" % payload["tap_count"])
                requested_samples = payload.get("requested_samples")
                records = str(payload["captured_samples"])
                if requested_samples is not None:
                    records += "/%d" % requested_samples
                print("records: %s" % records)
                print("runs: %d" % len(payload.get("run_ids") or []))
                print("shards: %d" % payload["number_of_shards"])
            return 0

        if args.command == "agentic":
            from noema_lab.agentic.contracts import load_agentic_contract
            from noema_lab.agentic.harness import (
                replay_agentic_campaign,
                run_agentic_campaign,
                verify_agentic_campaign,
            )

            if args.agentic_command == "validate":
                contract_path = Path(args.path)
                contract = load_agentic_contract(
                    contract_path,
                    project_root=Path.cwd(),
                    verify_base_recipe=True,
                )
                payload = {
                    "status": "valid",
                    "contract": contract.to_dict(),
                    "contract_sha256": contract.sha256,
                    "source": str(contract_path),
                }
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print("valid agentic contract: %s" % contract.id)
                    print("contract sha256: %s" % contract.sha256)
                    print("provider: %s (%s)" % (contract.provider.kind, contract.provider.model))
                    print(
                        "episodes: %d x %d decisions"
                        % (
                            contract.episodes.count,
                            contract.episodes.decisions_per_episode,
                        )
                    )
                return 0
            if args.agentic_command == "run":
                payload = run_agentic_campaign(
                    Path(args.path),
                    workspace,
                    project_root=Path.cwd(),
                    provider=args.provider,
                    model=args.model,
                    model_revision=args.model_revision,
                    endpoint=args.endpoint,
                    base_url=args.base_url,
                    api_key_env=args.api_key_env,
                    prompt=Path(args.prompt) if args.prompt else None,
                    device=args.device,
                    out=Path(args.out) if args.out else None,
                )
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print("completed agentic campaign: %s" % payload["campaign_dir"])
                    print("runs: %d" % payload["run_count"])
                    print("decisions: %d" % payload["decision_count"])
                    print("manifest sha256: %s" % payload["manifest_sha256"])
                return 0
            if args.agentic_command == "verify":
                payload = verify_agentic_campaign(Path(args.campaign))
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print("verified agentic campaign: %s" % payload["campaign_id"])
                    print("runs: %d" % payload["run_count"])
                    print("decisions: %d" % payload["decision_count"])
                return 0
            if args.agentic_command == "replay":
                payload = replay_agentic_campaign(
                    Path(args.campaign),
                    workspace,
                    out=Path(args.out) if args.out else None,
                )
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print("completed agentic replay: %s" % payload["replay_dir"])
                    print("runs: %d" % payload["replayed_run_count"])
                    print("metric mismatches: %d" % payload["mismatch_count"])
                return 0 if payload.get("status") == "passed" else 1

        if args.command == "ui" and args.ui_command == "serve":
            serve_ui(
                host=args.host,
                port=args.port,
                workspace=workspace,
                project_root=Path(args.project_root),
                adapter_paths=args.adapter or None,
            )
            return 0
    except Exception as exc:
        if bool(getattr(args, "json", False)):
            print(
                json.dumps(
                    _cli_json_error_envelope(exc, args),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print("error: %s" % exc, file=sys.stderr)
        failed_run_id = str(getattr(exc, "noema_run_id", "") or "")
        failed_run_dir = str(getattr(exc, "noema_run_dir", "") or "")
        if failed_run_id and failed_run_dir:
            print("failed run: %s" % failed_run_id, file=sys.stderr)
            print(
                "summary: %s" % (Path(failed_run_dir) / "summary.json"),
                file=sys.stderr,
            )
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
