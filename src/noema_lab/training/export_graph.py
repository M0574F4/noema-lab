from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import math

import yaml

from noema_lab.core.operations import OperationRegistry
from noema_lab.core.recipes import Recipe
from noema_lab.core.reproducibility import (
    derive_seed,
    master_seed_from_recipe,
    seed_namespace_from_recipe,
)
from noema_lab.core.training import inspect_training_feasibility
from noema_lab.training.differentiability import (
    TrainingDependencyError,
    require_sionna_available,
    require_torch_available,
)
from noema_lab.training.standalone_input import write_standalone_structured_input
from noema_lab.training.sionna_blocks import (
    AwgnChannelBlock,
    ExportableBlock,
    FlatRayleighChannelBlock,
    IdentityExportBlock,
    PowerNormalizationBlock,
    ReceiverIqImpairmentBlock,
    SionnaAwgnChannelBlock,
    SionnaFlatFadingChannelBlock,
    SymbolPowerAllocatorBlock,
)

JsonDict = Dict[str, Any]


# The legacy export surface returns one ``torch.nn.Sequential``.  Keep its
# contract deliberately narrow: each materializer below has one proven tensor
# input and one proven tensor output.  Richer port wiring belongs to the typed
# scenario graph.
_LEGACY_SEQUENTIAL_PORTS = {
    "channel.identity_symbol_link": ("symbols", "symbols"),
    "channel.symbol_boundary": ("symbols", "symbols"),
    "channel.symbol_power_identity": ("symbols", "symbols"),
    "channel.symbol_power_normalize": ("symbols", "symbols"),
    "model.symbol_power_allocator": ("symbols", "symbols"),
    "wireless.channel": ("symbols", "rx_symbols"),
    "hardware.receiver_iq_imbalance": ("rx_symbols", "rx_symbols"),
}


def _channel_reference_snr_db(params: JsonDict) -> float:
    if str(params.get("noise_mode") or "snr_at_unit_power") == "fixed_variance":
        variance = float(params.get("noise_variance", 0.0))
        if variance <= 0.0:
            raise TrainingDependencyError("fixed_variance channel export requires noise_variance > 0")
        return -10.0 * math.log10(variance)
    return float(params.get("snr_db", 12.0))


def _export_step_seed(recipe: Recipe, step, stream: str) -> int:
    if step.params.get("seed") is not None:
        return int(step.params["seed"])
    master_seed = master_seed_from_recipe(recipe)
    if master_seed is None:
        return 0
    return derive_seed(
        master_seed,
        seed_namespace_from_recipe(recipe),
        step.id,
        stream,
    )


@dataclass
class ExportGraph:
    recipe_name: str
    blocks: List[ExportableBlock] = field(default_factory=list)
    feasibility: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": 1,
            "recipe": self.recipe_name,
            "block_count": len(self.blocks),
            "blocks": [block.export_description() for block in self.blocks],
            "feasibility": dict(self.feasibility),
            "metadata": dict(self.metadata),
        }

    def torch_module(self):
        issue = str(self.metadata.get("execution_issue") or "").strip()
        if issue:
            raise TrainingDependencyError(issue)
        torch = require_torch_available()
        return torch.nn.Sequential(*self.blocks)


def build_export_graph_from_recipe(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    backend: str = "torch",
    include_power_normalization: bool = True,
    replacement_steps: Optional[Sequence[str] | str] = None,
    loss_steps: Optional[Sequence[str] | str] = None,
) -> ExportGraph:
    backend = str(backend or "torch").lower().replace("_", "-")
    if backend == "torch-sionna":
        backend = "sionna"
    if backend not in {"torch", "sionna"}:
        raise ValueError("Differentiable export backend must be 'torch' or 'sionna'.")
    if backend == "sionna":
        require_sionna_available()
    require_torch_available()

    feasibility = inspect_training_feasibility(
        recipe,
        registry,
        optimizable_steps=replacement_steps,
        loss=loss_steps,
    )
    if feasibility["recommended_mode"] not in {"differentiable_export", "receiver_only", "dataset_capture"}:
        raise TrainingDependencyError(
            "Recipe %s does not expose a trainable or capture-capable path; run `noema differentiable inspect` for details."
            % recipe.name
        )

    selected_replacements = set(
        str(item)
        for item in list(feasibility.get("selected_replacement_steps") or [])
    )
    support_step_ids: Optional[set[str]] = None
    active_route_step_ids: Optional[set[str]] = None
    if selected_replacements:
        support_step_ids = set()
        active_route_step_ids = set()
        loss_step_ids = set(str(item) for item in list(feasibility.get("loss_steps") or []))
        for path in list(feasibility.get("paths") or []):
            route_steps = {
                str(item)
                for item in list(
                    path.get("downstream_route_steps")
                    or path.get("replacement_to_loss_path")
                    or path.get("optimizable_to_loss_path")
                    or []
                )
            }
            active_route_step_ids.update(route_steps)
            support_step_ids.update(route_steps)
        support_step_ids.difference_update(selected_replacements)
        support_step_ids.difference_update(loss_step_ids)

    blocks: List[ExportableBlock] = []
    materialized_step_ids: set[str] = set()
    for step in recipe.steps:
        if support_step_ids is not None and step.id not in support_step_ids:
            continue
        before = len(blocks)
        if step.op == "channel.identity_symbol_link":
            blocks.append(IdentityExportBlock())
        elif step.op == "channel.symbol_boundary":
            blocks.append(IdentityExportBlock())
        elif step.op == "channel.symbol_power_identity":
            blocks.append(IdentityExportBlock())
        elif step.op == "channel.symbol_power_normalize":
            blocks.append(
                PowerNormalizationBlock(
                    target_power=float(step.params.get("target_power", 1.0)),
                    eps=float(step.params.get("eps", 1e-8)),
                    normalization_scope=str(
                        step.params.get("normalization_scope") or "source_item"
                    ),
                )
            )
        elif step.op == "model.symbol_power_allocator":
            policy = str(step.params.get("policy") or "snr_sigmoid")
            if policy in {"water_filling", "learned_checkpoint"}:
                raise TrainingDependencyError(
                    "CSI-conditioned power policy %s cannot be represented by the scalar-SNR sequential export graph; "
                    "use a separate resource-allocation training plan with channel-state captures."
                    % policy
                )
            blocks.append(
                SymbolPowerAllocatorBlock(
                    snr_db=float(step.params.get("snr_db", 12.0)),
                    target_power=float(step.params.get("target_power", 1.0)),
                    min_power=float(step.params.get("min_power", 0.25)),
                    max_power=float(step.params.get("max_power", 2.0)),
                    midpoint_snr_db=float(step.params.get("midpoint_snr_db", 12.0)),
                    slope_db=float(step.params.get("slope_db", 4.0)),
                    policy=policy,
                    granularity=str(step.params.get("granularity", "global")),
                    budget_mode=str(step.params.get("budget_mode", "fixed_average")),
                    allocation_contrast=float(step.params.get("allocation_contrast", 0.6)),
                    subcarrier_count=int(step.params.get("subcarrier_count", 64)),
                    stream_count=int(step.params.get("stream_count", 1)),
                    eps=float(step.params.get("eps", 1e-8)),
                )
            )
        elif step.op == "hardware.receiver_iq_imbalance":
            blocks.append(
                ReceiverIqImpairmentBlock(
                    gain_imbalance_db=float(
                        step.params.get("gain_imbalance_db", 5.0)
                    ),
                    quadrature_error_deg=float(
                        step.params.get("quadrature_error_deg", 12.0)
                    ),
                    phase_offset_deg=float(
                        step.params.get("phase_offset_deg", 20.0)
                    ),
                    dc_offset_i=float(step.params.get("dc_offset_i", 0.18)),
                    dc_offset_q=float(step.params.get("dc_offset_q", -0.12)),
                )
            )
        elif step.op == "wireless.channel":
            channel = str(step.params.get("channel") or "awgn").lower()
            reference_snr_db = _channel_reference_snr_db(step.params)
            receiver_processing = str(
                step.params.get("receiver_processing") or "matched"
            )
            channel_state_mode = str(
                step.params.get("channel_state_mode") or "none"
            )
            if channel_state_mode != "none":
                raise TrainingDependencyError(
                    "The legacy sequential Torch export does not materialize explicit channel-state side inputs; use the typed scenario graph."
                )
            channel_seed = _export_step_seed(
                recipe, step, "wireless_channel"
            )
            if channel == "awgn":
                channel_block = (
                    SionnaAwgnChannelBlock
                    if backend == "sionna"
                    else AwgnChannelBlock
                )
                blocks.append(
                    channel_block(
                        reference_snr_db,
                        seed=channel_seed,
                        receiver_processing=receiver_processing,
                    )
                )
            elif channel == "flat_rayleigh":
                channel_block = (
                    SionnaFlatFadingChannelBlock
                    if backend == "sionna"
                    else FlatRayleighChannelBlock
                )
                blocks.append(
                    channel_block(
                        reference_snr_db,
                        receiver_processing=receiver_processing,
                        seed=channel_seed,
                    )
                )
            else:
                raise TrainingDependencyError(
                    "Differentiable export MVP supports AWGN and optional flat Rayleigh only; %s requested %s."
                    % (step.id, channel)
                )
        if len(blocks) > before:
            materialized_step_ids.add(step.id)

    execution_issues: List[str] = []
    support_topology = "generic"
    if support_step_ids is not None:
        support_topology, topology_issue = _sequential_support_topology(
            recipe,
            support_step_ids,
        )
        if feasibility.get("recommended_mode") != "differentiable_export":
            execution_issues.append(
                "No clean live replacement-to-loss route is available; use the typed "
                "training contract and captured-data workflow."
            )
        missing_materializations = sorted(
            support_step_ids - materialized_step_ids
        )
        if missing_materializations:
            execution_issues.append(
                "The legacy sequential graph has no runtime materializer for downstream "
                "support step(s): %s. Use scenario_graph.json from the typed training contract."
                % ", ".join(missing_materializations)
            )
        elif active_route_step_ids is not None:
            interface_issue = _sequential_support_interface_issue(
                recipe,
                support_step_ids,
                active_route_step_ids,
            )
            if interface_issue:
                execution_issues.append(interface_issue)
        if topology_issue:
            execution_issues.append(topology_issue)
    execution_issue = " ".join(execution_issues)

    return ExportGraph(
        recipe_name=recipe.name,
        blocks=blocks,
        feasibility=feasibility,
        metadata={
            "backend": backend,
            "include_power_normalization": False,
            "power_normalization_source": "recipe_step",
            "scope": (
                "replacement_downstream_support"
                if support_step_ids is not None
                else "minimal_differentiable_phy_mvp"
            ),
            "selected_replacement_steps": sorted(selected_replacements),
            "selected_loss_steps": sorted(
                str(item) for item in list(feasibility.get("selected_loss_steps") or [])
            ),
            "support_step_ids": sorted(support_step_ids or []),
            "materialized_support_step_ids": sorted(materialized_step_ids),
            "support_topology": support_topology,
            "executable_sequential": not bool(execution_issue),
            "execution_issue": execution_issue,
        },
    )


def _sequential_support_topology(
    recipe: Recipe,
    support_step_ids: set[str],
) -> tuple[str, str]:
    """Classify whether an induced support DAG is representable by Sequential."""

    if len(support_step_ids) <= 1:
        return ("empty" if not support_step_ids else "linear", "")
    incoming = {step_id: 0 for step_id in support_step_ids}
    outgoing = {step_id: 0 for step_id in support_step_ids}
    edge_count = 0
    for step in recipe.steps:
        if step.id not in support_step_ids:
            continue
        producers = {
            str(reference).split(".", 1)[0]
            for reference in step.inputs.values()
            if str(reference).split(".", 1)[0] in support_step_ids
        }
        for producer_id in producers:
            incoming[step.id] += 1
            outgoing[producer_id] += 1
            edge_count += 1
    if any(value > 1 for value in incoming.values()) or any(
        value > 1 for value in outgoing.values()
    ):
        return (
            "branched",
            "The downstream support route is a branched DAG and cannot be flattened into "
            "torch.nn.Sequential. Use scenario_graph.json from the typed training contract.",
        )
    if edge_count != len(support_step_ids) - 1:
        return (
            "disconnected",
            "The downstream support route contains disconnected paths and cannot be represented "
            "by one torch.nn.Sequential. Use scenario_graph.json from the typed training contract.",
        )
    return "linear", ""


def _sequential_support_interface_issue(
    recipe: Recipe,
    support_step_ids: set[str],
    active_route_step_ids: set[str],
) -> str:
    """Reject active port wiring that one unary Sequential cannot preserve."""

    if not support_step_ids:
        return ""
    active_route = set(active_route_step_ids)
    issues: List[str] = []
    active_consumers: Dict[str, List[tuple[str, str, str]]] = {
        step_id: [] for step_id in support_step_ids
    }
    for consumer in recipe.steps:
        if consumer.id not in active_route:
            continue
        for input_name, reference in consumer.inputs.items():
            producer_id, output_name = str(reference).split(".", 1)
            if producer_id in active_consumers:
                active_consumers[producer_id].append(
                    (output_name, consumer.id, str(input_name))
                )

    for step in recipe.steps:
        if step.id not in support_step_ids:
            continue
        expected = _LEGACY_SEQUENTIAL_PORTS.get(step.op)
        if expected is None:
            issues.append(
                "`%s` (%s) has no proven unary input/output contract"
                % (step.id, step.op)
            )
            continue
        expected_input, expected_output = expected
        connected_inputs = [
            (str(input_name), str(reference).split(".", 1)[0])
            for input_name, reference in step.inputs.items()
        ]
        active_inputs = [
            (input_name, producer_id)
            for input_name, producer_id in connected_inputs
            if producer_id in active_route
        ]
        if (
            len(connected_inputs) != 1
            or len(active_inputs) != 1
            or active_inputs[0][0] != expected_input
        ):
            connected_names = ", ".join(
                sorted(input_name for input_name, _producer_id in connected_inputs)
            ) or "none"
            issues.append(
                "`%s` (%s) connects recipe input port(s) %s; its legacy materializer "
                "supports only unary `%s`"
                % (step.id, step.op, connected_names, expected_input)
            )

        consumers = active_consumers.get(step.id, [])
        active_outputs = {output_name for output_name, _consumer_id, _input_name in consumers}
        if len(consumers) != 1 or active_outputs != {expected_output}:
            output_names = ", ".join(sorted(active_outputs)) or "none"
            issues.append(
                "`%s` (%s) exposes active recipe output port(s) %s across %d edge(s); "
                "its legacy materializer returns only one `%s` tensor"
                % (
                    step.id,
                    step.op,
                    output_names,
                    len(consumers),
                    expected_output,
                )
            )

    if not issues:
        return ""
    return (
        "The legacy sequential graph cannot preserve the active recipe port wiring: %s. "
        "Use scenario_graph.json from the typed training contract."
        % "; ".join(issues)
    )


def export_differentiable_graph_bundle(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    backend: str = "torch",
    include_power_normalization: bool = True,
    replacement_steps: Optional[Sequence[str] | str] = None,
    loss_steps: Optional[Sequence[str] | str] = None,
    out_dir: Path,
    force: bool = False,
) -> JsonDict:
    graph = build_export_graph_from_recipe(
        recipe,
        registry,
        backend=backend,
        include_power_normalization=include_power_normalization,
        replacement_steps=replacement_steps,
        loss_steps=loss_steps,
    )
    execution_issue = str(graph.metadata.get("execution_issue") or "").strip()
    if execution_issue:
        raise TrainingDependencyError(execution_issue)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise ValueError("Export graph directory already exists and is not empty: %s. Use force to overwrite generated files." % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_payload = graph.to_dict()
    files = [
        "export_graph.json",
        "noema_recipe.yaml",
        "scenario.py",
        "structured_input.py",
        "README.md",
    ]
    (out_dir / "export_graph.json").write_text(json.dumps(graph_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "noema_recipe.yaml").write_text(yaml.safe_dump(recipe.to_dict(), sort_keys=False), encoding="utf-8")
    (out_dir / "scenario.py").write_text(
        _export_graph_scenario_py(
            backend,
            include_power_normalization,
            list(graph.metadata.get("selected_replacement_steps") or []) or None,
            list(graph.metadata.get("selected_loss_steps") or []) or None,
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    write_standalone_structured_input(out_dir)
    (out_dir / "README.md").write_text(_export_graph_readme_md(recipe, graph).strip() + "\n", encoding="utf-8")
    return {
        "status": "exported",
        "kind": "differentiable_graph",
        "recipe": recipe.name,
        "out_dir": str(out_dir),
        "backend": str(backend or "torch"),
        "block_count": int(graph_payload.get("block_count") or 0),
        "export_graph": graph_payload,
        "files": files,
    }


def _export_graph_scenario_py(
    backend: str,
    include_power_normalization: bool,
    replacement_steps: Optional[Sequence[str]] = None,
    loss_steps: Optional[Sequence[str]] = None,
) -> str:
    return f'''
from __future__ import annotations

from pathlib import Path

from noema_lab.core.recipes import recipe_from_dict
from noema_lab.ops import build_registry
from noema_lab.training.export_graph import build_export_graph_from_recipe
try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree one-file inspection.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def build_export_graph():
    recipe_payload = load_strict_yaml_or_json(Path("noema_recipe.yaml"))
    if not isinstance(recipe_payload, dict):
        raise ValueError("noema_recipe.yaml must contain a mapping")
    recipe = recipe_from_dict(recipe_payload)
    return build_export_graph_from_recipe(
        recipe,
        build_registry(),
        backend={backend!r},
        include_power_normalization={bool(include_power_normalization)!r},
        replacement_steps={list(replacement_steps) if replacement_steps is not None else None!r},
        loss_steps={list(loss_steps) if loss_steps is not None else None!r},
    )


def build_module():
    return build_export_graph().torch_module()


if __name__ == "__main__":
    graph = build_export_graph()
    print(graph.to_dict())
'''


def _export_graph_readme_md(recipe: Recipe, graph: ExportGraph) -> str:
    blocks = ", ".join(block.export_description().get("block", "block") for block in graph.blocks) or "none"
    return f'''
# Noema Differentiable Graph Export: {recipe.name}

This bundle exports the differentiable support graph that Noema can materialize from the recipe.
When replacement steps are selected, their current implementations are omitted; only unchanged
downstream support between replacement outputs and the selected loss is included.

It is not the same as a full trainable scenario. Frozen differentiable blocks can be used inside an
external training pipeline, but Noema does not optimize them or the replaced implementations.

Materialized blocks: {blocks}

## Files

- `export_graph.json`: machine-readable graph description.
- `noema_recipe.yaml`: source recipe copy.
- `scenario.py`: helper that rebuilds the exported graph and returns `torch.nn.Sequential` for currently materialized blocks.
- `README.md`: this note.
'''
