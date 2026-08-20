from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DOCS = ROOT / "docs"
GENERATED = DOCS / "reference" / "generated"


class StableHelpFormatter(argparse.HelpFormatter):
    """Render checked-in CLI help independently of terminals and Python minors."""

    def __init__(self, prog: str) -> None:
        super().__init__(prog, width=78)

    def _format_action_invocation(self, action: argparse.Action) -> str:
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            return " ".join(self._metavar_formatter(action, default)(1))
        if action.nargs == 0:
            return ", ".join(action.option_strings)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ", ".join("%s %s" % (option, args_string) for option in action.option_strings)

    def _format_usage(self, usage, actions, groups, prefix):
        """Use the stable pre-3.13 wrapping algorithm with a fixed width."""

        if prefix is None:
            prefix = "usage: "
        if usage is not None:
            usage = usage % {"prog": self._prog}
        elif not actions:
            usage = self._prog
        else:
            prog = self._prog
            optionals = [action for action in actions if action.option_strings]
            positionals = [action for action in actions if not action.option_strings]
            format_actions = self._format_actions_usage
            action_usage = format_actions(optionals + positionals, groups)
            usage = " ".join(part for part in (prog, action_usage) if part)

            text_width = self._width - self._current_indent
            if len(prefix) + len(usage) > text_width:
                part_pattern = r"\(.*?\)+(?=\s|$)|\[.*?\]+(?=\s|$)|\S+"
                optional_usage = format_actions(optionals, groups)
                positional_usage = format_actions(positionals, groups)
                optional_parts = re.findall(part_pattern, optional_usage)
                positional_parts = re.findall(part_pattern, positional_usage)
                if " ".join(optional_parts) != optional_usage:
                    raise RuntimeError("could not deterministically wrap optional CLI usage")
                if " ".join(positional_parts) != positional_usage:
                    raise RuntimeError("could not deterministically wrap positional CLI usage")

                def wrapped_lines(parts, indent, first_prefix=None):
                    lines: list[str] = []
                    line: list[str] = []
                    indent_length = len(indent)
                    line_length = len(first_prefix) - 1 if first_prefix is not None else indent_length - 1
                    for part in parts:
                        if line_length + 1 + len(part) > text_width and line:
                            lines.append(indent + " ".join(line))
                            line = []
                            line_length = indent_length - 1
                        line.append(part)
                        line_length += len(part) + 1
                    if line:
                        lines.append(indent + " ".join(line))
                    if first_prefix is not None:
                        lines[0] = lines[0][indent_length:]
                    return lines

                if len(prefix) + len(prog) <= 0.75 * text_width:
                    indent = " " * (len(prefix) + len(prog) + 1)
                    if optional_parts:
                        lines = wrapped_lines([prog] + optional_parts, indent, prefix)
                        lines.extend(wrapped_lines(positional_parts, indent))
                    elif positional_parts:
                        lines = wrapped_lines([prog] + positional_parts, indent, prefix)
                    else:
                        lines = [prog]
                else:
                    indent = " " * len(prefix)
                    parts = optional_parts + positional_parts
                    lines = wrapped_lines(parts, indent)
                    if len(lines) > 1:
                        lines = wrapped_lines(optional_parts, indent)
                        lines.extend(wrapped_lines(positional_parts, indent))
                    lines = [prog] + lines
                usage = "\n".join(lines)
        return "%s%s\n\n" % (prefix, usage)


CLI_COMMANDS: Sequence[Sequence[str]] = (
    (),
    ("ops",),
    ("ops", "list"),
    ("ops", "show"),
    ("data",),
    ("data", "fetch"),
    ("template",),
    ("template", "list"),
    ("template", "show"),
    ("template", "instantiate"),
    ("recipe",),
    ("recipe", "validate"),
    ("recipe", "lint"),
    ("recipe", "specs"),
    ("recipe", "graph"),
    ("recipe", "expand-matrix"),
    ("recipe", "run"),
    ("recipe", "run-matrix"),
    ("research",),
    ("research", "catalog"),
    ("research", "datasets"),
    ("research", "tasks"),
    ("research", "metrics"),
    ("research", "show"),
    ("research", "validate-recipe"),
    ("suite",),
    ("suite", "list"),
    ("suite", "show"),
    ("suite", "benchmarks"),
    ("benchmark",),
    ("benchmark", "list"),
    ("benchmark", "show"),
    ("benchmark", "validate"),
    ("benchmark", "run"),
    ("benchmark", "results"),
    ("benchmark", "result"),
    ("benchmark", "export"),
    ("benchmark", "verify"),
    ("benchmark", "plot"),
    ("benchmark", "publish"),
    ("submission",),
    ("submission", "validate"),
    ("adapter",),
    ("adapter", "validate"),
    ("adapter", "scaffold"),
    ("runs",),
    ("runs", "list"),
    ("runs", "show"),
    ("runs", "manifest"),
    ("runs", "verify"),
    ("differentiable",),
    ("differentiable", "inspect"),
    ("differentiable", "export"),
    ("dataset-capture",),
    ("dataset-capture", "run"),
    ("ui",),
    ("ui", "serve"),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Noema documentation reference pages.")
    parser.add_argument("--out", default=str(GENERATED), help="Output directory for generated Markdown files.")
    parser.add_argument("--check", action="store_true", help="Fail if generated files differ from the checked-in files.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    payloads = {
        "cli.md": generate_cli_reference(),
        "operations.md": generate_operations_reference(),
    }

    if args.check:
        mismatches = []
        for name, content in payloads.items():
            target = out_dir / name
            if not target.is_file() or target.read_text(encoding="utf-8") != content:
                mismatches.append(str(target.relative_to(ROOT)))
        if mismatches:
            print("generated documentation is stale:", file=sys.stderr)
            for item in mismatches:
                print("  - %s" % item, file=sys.stderr)
            print("run: uv run python tools/generate_docs_reference.py", file=sys.stderr)
            return 1
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        (out_dir / name).write_text(content, encoding="utf-8")
    return 0


def generate_cli_reference() -> str:
    lines = [
        "# CLI Reference",
        "",
        "<!-- This file is generated by tools/generate_docs_reference.py. Do not edit by hand. -->",
        "",
        "The command help below is captured from the installed Noema CLI entry point.",
        "",
    ]
    for command in CLI_COMMANDS:
        title = "noema" if not command else "noema " + " ".join(command)
        lines.extend(["## `%s`" % title, "", "```text", _cli_help(command).rstrip(), "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def generate_operations_reference() -> str:
    sys.path.insert(0, str(SRC))
    from noema_lab.ops import build_registry

    registry = build_registry()
    operations = registry.describe()
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for operation in operations:
        group = str(operation["id"]).split(".", 1)[0]
        groups.setdefault(group, []).append(operation)

    lines = [
        "# Operation Reference",
        "",
        "<!-- This file is generated by tools/generate_docs_reference.py. Do not edit by hand. -->",
        "",
        "This reference is generated from `noema_lab.ops.build_registry()` and each operation's `describe()` contract.",
        "",
        "Operation contracts are the implementation-adjacent source of truth for recipe validation, graph rendering, training inspection, and UI parameter panels.",
        "",
        "`differentiability.trainable_params` is legacy compatibility metadata. The independent `training_capabilities` values below govern built-in fine-tuning and portable replacement.",
        "",
    ]
    for group in sorted(groups):
        lines.extend(["## %s" % group, ""])
        for operation in sorted(groups[group], key=lambda item: str(item["id"])):
            lines.extend(_operation_lines(operation))
    return "\n".join(lines).rstrip() + "\n"


def _operation_lines(operation: Mapping[str, Any]) -> list[str]:
    op_id = str(operation.get("id") or "")
    lines = [
        "### `%s`" % op_id,
        "",
        "**Name:** %s" % operation.get("name", op_id),
        "",
        "**Status:** `%s`" % operation.get("status", "implemented"),
        "",
    ]
    lines.extend(_mapping_table("Inputs", operation.get("input_kinds") or {}))
    lines.extend(_mapping_table("Optional inputs", operation.get("optional_input_kinds") or {}))
    lines.extend(_mapping_table("Outputs", operation.get("output_kinds") or {}))
    lines.extend(_params_lines(operation.get("params_schema") or {}))
    lines.extend(_differentiability_lines(operation.get("differentiability") or {}))
    lines.extend(_training_capability_lines(operation))
    lines.extend(_backend_contract_lines(operation))
    entropy_info = operation.get("entropy_info") or {}
    if isinstance(entropy_info, Mapping) and entropy_info:
        lines.extend(["**Entropy/payload info:**", "", "```json", _json_block(entropy_info), "```", ""])
    return lines


def _mapping_table(title: str, value: Mapping[str, Any]) -> list[str]:
    lines = ["**%s:**" % title, ""]
    if not value:
        return lines + ["None.", ""]
    lines.extend(["| Name | Kind |", "| --- | --- |"])
    for key in sorted(value):
        kind = value[key]
        if isinstance(kind, list):
            rendered = ", ".join("`%s`" % item for item in kind)
        else:
            rendered = "`%s`" % kind
        lines.append("| `%s` | %s |" % (key, rendered))
    lines.append("")
    return lines


def _params_lines(schema: Mapping[str, Any]) -> list[str]:
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    required = set(schema.get("required") or []) if isinstance(schema, Mapping) else set()
    lines = ["**Parameters:**", ""]
    if not isinstance(properties, Mapping) or not properties:
        return lines + ["None.", ""]
    lines.extend(["| Name | Type | Required | Default / values | Description |", "| --- | --- | --- | --- | --- |"])
    for key in sorted(properties):
        spec = properties[key]
        if not isinstance(spec, Mapping):
            spec = {}
        type_name = _param_type(spec)
        default = _param_default(spec)
        description = _table_cell(str(spec.get("description") or "").replace("\n", " "))
        lines.append(
            "| `%s` | `%s` | %s | %s | %s |"
            % (key, _table_cell(type_name), "yes" if key in required else "no", _table_cell(default), description)
        )
    lines.append("")
    return lines


def _differentiability_lines(metadata: Mapping[str, Any]) -> list[str]:
    keys = ["framework", "gradient", "trainable_params", "exportable"]
    summary = ", ".join("%s=`%s`" % (key, metadata.get(key, "")) for key in keys)
    lines = ["**Differentiability (legacy `trainable_params`):** %s" % summary, ""]
    reason = metadata.get("reason")
    if reason:
        lines.extend([str(reason), ""])
    return lines


def _training_capability_lines(operation: Mapping[str, Any]) -> list[str]:
    capabilities = operation.get("training_capabilities") or {}
    lines = [
        "**Training capabilities:** built_in_fine_tuning=`%s`, portable_replacement=`%s`"
        % (
            bool(capabilities.get("built_in_fine_tuning", False)),
            bool(capabilities.get("portable_replacement", False)),
        ),
        "",
    ]
    artifact_abi = operation.get("trained_artifact_abi") or {}
    if artifact_abi:
        lines.extend(
            [
                "**Portable trained-artifact ABI:**",
                "",
                "```json",
                _json_block(artifact_abi),
                "```",
                "",
            ]
        )
    else:
        lines.extend(["**Portable trained-artifact ABI:** None.", ""])
    return lines


def _backend_contract_lines(operation: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    backends = operation.get("backends") or {}
    if isinstance(backends, Mapping) and backends:
        lines.extend(["**Backends:**", "", "| Runner | Backends |", "| --- | --- |"])
        for runner in sorted(backends):
            values = backends.get(runner) or []
            if isinstance(values, list) and values:
                rendered = ", ".join("`%s`" % item for item in values)
            else:
                rendered = "None"
            lines.append("| `%s` | %s |" % (runner, rendered))
        lines.append("")
    equivalence = operation.get("equivalence") or {}
    if isinstance(equivalence, Mapping) and equivalence:
        lines.extend(["**Equivalence:**", "", "```json", _json_block(equivalence), "```", ""])
    formats = operation.get("formats") or {}
    if isinstance(formats, Mapping) and formats:
        lines.extend(["**Formats:**", "", "```json", _json_block(formats), "```", ""])
    materializations = operation.get("materializations") or []
    if isinstance(materializations, list) and materializations:
        lines.extend(["**Materializations:**", "", "```json", _json_block(materializations), "```", ""])
    return lines


def _param_type(spec: Mapping[str, Any]) -> str:
    if "type" in spec:
        value = spec["type"]
        if isinstance(value, list):
            return " | ".join(str(item) for item in value)
        return str(value)
    if "enum" in spec:
        return "enum"
    return "any"


def _param_default(spec: Mapping[str, Any]) -> str:
    parts = []
    if "default" in spec:
        parts.append("default `%s`" % spec["default"])
    enum = spec.get("enum")
    if isinstance(enum, Iterable) and not isinstance(enum, (str, bytes)):
        parts.append("values %s" % ", ".join("`%s`" % item for item in enum))
    return "<br>".join(parts) if parts else ""


def _json_block(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _table_cell(value: str) -> str:
    return str(value).replace("|", "\\|")


def _cli_help(command: Sequence[str]) -> str:
    sys.path.insert(0, str(SRC))
    from noema_lab.cli.main import build_parser

    parser = build_parser()
    for part in command:
        subparser_action = _subparser_action(parser)
        if subparser_action is None or part not in subparser_action.choices:
            raise RuntimeError("could not find CLI parser for command: noema %s" % " ".join(command))
        parser = subparser_action.choices[part]
    parser.formatter_class = StableHelpFormatter
    return parser.format_help()


def _subparser_action(parser: argparse.ArgumentParser):
    for action in parser._actions:  # argparse exposes subparsers only through actions.
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


if __name__ == "__main__":
    raise SystemExit(main())
