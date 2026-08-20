from __future__ import annotations

from typing import Any, Dict, List

from noema_lab.core.operations import OperationRegistry
from noema_lab.core.recipes import Recipe, require_strict_recipe
from noema_lab.core.runner_contracts import operation_runner_supports

JsonDict = Dict[str, Any]


def recipe_graph(recipe: Recipe, registry: OperationRegistry) -> JsonDict:
    require_strict_recipe(recipe)
    nodes: List[JsonDict] = []
    edges: List[JsonDict] = []
    for step in recipe.steps:
        operation = registry.get(step.op)
        description = operation.describe()
        inputs = {
            **dict(description["input_kinds"]),
            **dict(description.get("optional_input_kinds") or {}),
        }
        training_capabilities = dict(description.get("training_capabilities") or {})
        nodes.append(
            {
                "id": step.id,
                "op": step.op,
                "name": description["name"],
                "status": description["status"],
                "inputs": inputs,
                "optional_inputs": sorted((description.get("optional_input_kinds") or {}).keys()),
                "outputs": dict(description["output_kinds"]),
                "params": dict(step.params),
                "differentiability": dict(description.get("differentiability") or {}),
                "training_capabilities": training_capabilities,
                "fine_tuning_supported": bool(
                    training_capabilities.get("built_in_fine_tuning", False)
                ),
                "replacement_ready": bool(
                    training_capabilities.get("portable_replacement", False)
                ),
                "backends": dict(description.get("backends") or {}),
                "equivalence": dict(description.get("equivalence") or {}),
                "formats": dict(description.get("formats") or {}),
                "materializations": list(description.get("materializations") or []),
                "runner_support": operation_runner_supports(description),
            }
        )
        for input_name, reference in step.inputs.items():
            source_step, source_output = reference.split(".", 1)
            edges.append(
                {
                    "from_step": source_step,
                    "from_output": source_output,
                    "to_step": step.id,
                    "to_input": input_name,
                }
            )
    return {"schema_version": 1, "recipe": recipe.name, "nodes": nodes, "edges": edges}


def format_graph_text(graph: JsonDict) -> str:
    lines = ["recipe: %s" % graph["recipe"]]
    for node in graph["nodes"]:
        lines.append("node\t%(id)s\t%(op)s" % node)
    for edge in graph["edges"]:
        lines.append(
            "edge\t%(from_step)s.%(from_output)s\t->\t%(to_step)s.%(to_input)s" % edge
        )
    return "\n".join(lines)


def format_graph_dot(graph: JsonDict) -> str:
    lines = ["digraph noema {", "  rankdir=LR;"]
    for node in graph["nodes"]:
        label = "%s\\n%s" % (node["id"], node["op"])
        lines.append('  "%s" [shape=box,label="%s"];' % (_dot(node["id"]), _dot(label)))
    for edge in graph["edges"]:
        label = "%s -> %s" % (edge["from_output"], edge["to_input"])
        lines.append(
            '  "%s" -> "%s" [label="%s"];'
            % (_dot(edge["from_step"]), _dot(edge["to_step"]), _dot(label))
        )
    lines.append("}")
    return "\n".join(lines)


def _dot(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')
