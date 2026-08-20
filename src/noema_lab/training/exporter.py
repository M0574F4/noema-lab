from __future__ import annotations

import hashlib
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import yaml

from noema_lab.core.matrix import RecipeMatrixError, matrix_values_for_step_param
from noema_lab.core.operations import OperationRegistry
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import Recipe, RecipeStep
from noema_lab.core.structured_input import load_strict_yaml_or_json
from noema_lab.core.training import inspect_training_feasibility
from noema_lab.core.training_plans import (
    TrainingPlan,
    scenario_recipe_fingerprint,
    training_plan_fingerprint,
)
from noema_lab.training.capture_plan import (
    TrainingCapturePlan,
    TrainingCapturePlanError,
    capture_plan_inspection,
    capture_tap_candidates,
    replacement_boundary_capture_requirements,
    resolve_training_capture_plan,
)
from noema_lab.training.package_layout import find_demo_training_dir
from noema_lab.training.contracts import (
    ConstraintSpec,
    NamedValueSpec,
    SignalSpec,
    SlotGroupSpec,
    TensorSpec,
    TrainingContractError,
    TrainingContractOptions,
    compile_training_contract,
    write_training_contract_bundle,
)
from noema_lab.training.csi_feedback_export import (
    CSI_FEEDBACK_DECODER_OP,
    CSI_FEEDBACK_ENCODER_OP,
    CSI_FEEDBACK_LOSS,
    CsiFeedbackExportError,
    CsiFeedbackExportPlan,
    build_csi_feedback_export_plan,
    is_csi_feedback_slot_pair,
    write_csi_feedback_starter,
)
from noema_lab.training.channel_estimation_export import (
    CHANNEL_ESTIMATION_LOSS,
    CHANNEL_ESTIMATOR_OP,
    ChannelEstimationExportError,
    ChannelEstimationExportPlan,
    build_channel_estimation_export_plan,
    write_channel_estimation_starter,
)
from noema_lab.training.image_data_export import (
    split_image_ids,
)
from noema_lab.training.generic_capture_export import (
    GenericCaptureDataContractError,
    write_generic_capture_data_contract,
)
from noema_lab.training.neural_receiver_export import (
    write_neural_receiver_starter,
)
from noema_lab.training.phase_tracking_receiver_export import (
    write_phase_tracking_receiver_starter,
)
from noema_lab.training.modulation_recognition_export import (
    MODULATION_CLASSIFICATION_LOSS,
    MODULATION_CLASSIFIER_OP,
    ModulationRecognitionExportError,
    ModulationRecognitionExportPlan,
    build_modulation_recognition_export_plan,
    write_modulation_recognition_starter,
)
from noema_lab.training.resource_allocation_export import (
    RESOURCE_ALLOCATION_LOSS,
    ResourceAllocationExportError,
    ResourceAllocationExportPlan,
    build_resource_allocation_export_plan,
    write_resource_allocation_export,
)
from noema_lab.training.delayed_csi_resource_allocation_export import (
    DELAYED_CSI_RESOURCE_ALLOCATION_LOSS,
    DelayedCsiResourceAllocationExportError,
    DelayedCsiResourceAllocationExportPlan,
    build_delayed_csi_resource_allocation_export_plan,
    write_delayed_csi_resource_allocation_starter,
)
from noema_lab.training.starter_refresh import prepare_demo_starter_directory
from noema_lab.training.standalone_input import write_standalone_structured_input

JsonDict = Dict[str, Any]


class DifferentiableExportError(ValueError):
    pass


@dataclass(frozen=True)
class ExportOptions:
    optimizable_steps: List[str]
    loss: str
    framework: str
    exporter: str
    source_path: Optional[Path] = None
    project_root: Optional[Path] = None
    include_starter: bool = False


@dataclass(frozen=True)
class NeuralReceiverExportPlan:
    recipe: Recipe
    recipe_sha256: str
    receiver_step: RecipeStep
    feature_step: RecipeStep
    target_step: RecipeStep
    feature_input: str
    feature_reference: str
    target_reference: str
    framework: str
    loss: str
    feature_kind: str
    target_kind: str
    capture_plan: TrainingCapturePlan
    project_root: Optional[Path] = None


@dataclass(frozen=True)
class PhaseTrackingReceiverExportPlan:
    recipe: Recipe
    recipe_sha256: str
    receiver_step: RecipeStep
    feature_step: RecipeStep
    pilot_context_step: RecipeStep
    target_step: RecipeStep
    feature_reference: str
    pilot_context_reference: str
    target_reference: str
    framework: str
    loss: str
    feature_kind: str
    pilot_context_kind: str
    target_kind: str
    capture_plan: TrainingCapturePlan
    project_root: Optional[Path] = None


class DifferentiableExporter(ABC):
    id: str
    name: str

    @abstractmethod
    def supports(self, recipe: Recipe, registry: OperationRegistry) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_plan(self, recipe: Recipe, registry: OperationRegistry, options: ExportOptions):
        raise NotImplementedError

    @abstractmethod
    def write_export(self, plan, out_dir: Path, *, force: bool = False) -> JsonDict:
        raise NotImplementedError

    def explain_unsupported(self, recipe: Recipe, registry: OperationRegistry, options: ExportOptions) -> str:
        try:
            self.build_plan(recipe, registry, options)
        except DifferentiableExportError as exc:
            return str(exc)
        return "Recipe is not supported by exporter %s." % self.id


class DatasetCaptureOnlyExporter(DifferentiableExporter):
    id = "dataset-capture-only"
    name = "Dataset-capture exporter"

    def supports(self, recipe: Recipe, registry: OperationRegistry) -> bool:
        return bool(getattr(recipe, "dataset_capture", None))

    def build_plan(self, recipe: Recipe, registry: OperationRegistry, options: ExportOptions):
        raise DifferentiableExportError("dataset-capture exporter is represented by `noema dataset-capture run`; no training harness is generated yet.")

    def write_export(self, plan, out_dir: Path, *, force: bool = False) -> JsonDict:
        raise DifferentiableExportError("dataset-capture exporter is represented by `noema dataset-capture run`; no training harness is generated yet.")


class TextSemanticJSCCExporter(DifferentiableExporter):
    id = "text-semantic-jscc"
    name = "Text semantic JSCC exporter"

    def supports(self, recipe: Recipe, registry: OperationRegistry) -> bool:
        return False

    def build_plan(self, recipe: Recipe, registry: OperationRegistry, options: ExportOptions):
        raise DifferentiableExportError("text-semantic-jscc exporter is planned but not implemented yet.")

    def write_export(self, plan, out_dir: Path, *, force: bool = False) -> JsonDict:
        raise DifferentiableExportError("text-semantic-jscc exporter is planned but not implemented yet.")


class TaskHeadExporter(DifferentiableExporter):
    id = "task-head"
    name = "Task head exporter"

    def supports(self, recipe: Recipe, registry: OperationRegistry) -> bool:
        return False

    def build_plan(self, recipe: Recipe, registry: OperationRegistry, options: ExportOptions):
        raise DifferentiableExportError("task-head exporter is planned but not implemented yet.")

    def write_export(self, plan, out_dir: Path, *, force: bool = False) -> JsonDict:
        raise DifferentiableExportError("task-head exporter is planned but not implemented yet.")


@dataclass(frozen=True)
class DeepJsccExportPlan:
    recipe: Recipe
    recipe_sha256: str
    optimizable_steps: List[str]
    sender_step: RecipeStep
    receiver_step: RecipeStep
    data_step: RecipeStep
    channel_step: RecipeStep
    framework: str
    loss: str
    snr_db: List[float]
    channel: str
    power_normalization_step: Optional[RecipeStep] = None
    power_normalization_target: float = 1.0
    source_path: Optional[Path] = None
    project_root: Optional[Path] = None
    template_id: str = "deepjscc_image_reconstruction.reference_cnn_awgn"


class DeepJSCCImageExporter(DifferentiableExporter):
    id = "deepjscc-image"
    name = "DeepJSCC image differentiable exporter"

    def supports(self, recipe: Recipe, registry: OperationRegistry) -> bool:
        try:
            validate_recipe_against_registry(recipe, registry)
            image_sources = [step for step in recipe.steps if step.op == "source.image_dataset"]
            senders = [
                step
                for step in recipe.steps
                if "image.batch.numpy" in _operation_kinds(registry, step)["inputs"]
                and "channel.symbols.complex_numpy" in _operation_kinds(registry, step)["outputs"]
            ]
            receivers = [
                step
                for step in recipe.steps
                if {"channel.symbols.complex_numpy", "channel.rx_symbols.complex_numpy"} & _operation_kinds(registry, step)["inputs"]
                and "image.batch.numpy" in _operation_kinds(registry, step)["outputs"]
            ]
            channels = [step for step in recipe.steps if step.op == "wireless.channel"]
        except Exception:
            return False
        return bool(image_sources and senders and receivers and channels)

    def build_plan(self, recipe: Recipe, registry: OperationRegistry, options: ExportOptions) -> DeepJsccExportPlan:
        plan = build_deepjscc_export_plan(
            recipe,
            registry,
            optimizable_steps=options.optimizable_steps,
            loss=options.loss,
            framework=options.framework,
        )
        return replace(
            plan,
            source_path=options.source_path,
            project_root=options.project_root,
        )

    def write_export(self, plan: DeepJsccExportPlan, out_dir: Path, *, force: bool = False) -> JsonDict:
        return _write_deepjscc_export(plan, out_dir, source_path=plan.source_path, force=force)


class NeuralReceiverExporter(DifferentiableExporter):
    id = "neural-receiver"
    name = "Neural receiver differentiable exporter"

    def supports(self, recipe: Recipe, registry: OperationRegistry) -> bool:
        try:
            self.build_plan(
                recipe,
                registry,
                ExportOptions(optimizable_steps=[], loss="bit.bce", framework="torch", exporter=self.id),
            )
        except DifferentiableExportError:
            return False
        return True

    def build_plan(self, recipe: Recipe, registry: OperationRegistry, options: ExportOptions) -> NeuralReceiverExportPlan:
        return replace(
            build_neural_receiver_export_plan(recipe, registry, options=options),
            project_root=options.project_root,
        )

    def write_export(self, plan: NeuralReceiverExportPlan, out_dir: Path, *, force: bool = False) -> JsonDict:
        return write_neural_receiver_starter(plan, out_dir, force=force)


class PhaseTrackingReceiverExporter(DifferentiableExporter):
    """Demo scaffold for packet-context carrier tracking.

    This exporter is deliberately stricter than the generic neural-receiver
    exporter.  It only attaches the checked-in phase-tracking example to the
    operation whose portable ABI carries both received symbols and public pilot
    context.  Oracle phase truth is never part of that ABI or its required
    capture boundary.
    """

    id = "phase-tracking-receiver"
    name = "Packet-context phase-tracking receiver exporter"

    def supports(self, recipe: Recipe, registry: OperationRegistry) -> bool:
        try:
            self.build_plan(
                recipe,
                registry,
                ExportOptions(
                    optimizable_steps=[],
                    loss="bit.bce",
                    framework="torch",
                    exporter=self.id,
                ),
            )
        except (DifferentiableExportError, ValueError, KeyError):
            return False
        return True

    def build_plan(
        self,
        recipe: Recipe,
        registry: OperationRegistry,
        options: ExportOptions,
    ) -> PhaseTrackingReceiverExportPlan:
        return replace(
            build_phase_tracking_receiver_export_plan(
                recipe,
                registry,
                options=options,
            ),
            project_root=options.project_root,
        )

    def write_export(
        self,
        plan: PhaseTrackingReceiverExportPlan,
        out_dir: Path,
        *,
        force: bool = False,
    ) -> JsonDict:
        return write_phase_tracking_receiver_starter(plan, out_dir, force=force)


class ModulationRecognitionExporter(DifferentiableExporter):
    id = "modulation-recognition"
    name = "Automatic modulation-recognition exporter"

    def supports(self, recipe: Recipe, registry: OperationRegistry) -> bool:
        try:
            build_modulation_recognition_export_plan(
                recipe,
                registry,
                optimizable_steps=[],
                loss=MODULATION_CLASSIFICATION_LOSS,
                framework="torch",
            )
        except (ModulationRecognitionExportError, ValueError):
            return False
        return True

    def build_plan(
        self,
        recipe: Recipe,
        registry: OperationRegistry,
        options: ExportOptions,
    ) -> ModulationRecognitionExportPlan:
        try:
            plan = build_modulation_recognition_export_plan(
                recipe,
                registry,
                optimizable_steps=options.optimizable_steps,
                loss=options.loss,
                framework=options.framework,
            )
            return replace(plan, project_root=options.project_root)
        except ModulationRecognitionExportError as exc:
            raise DifferentiableExportError(str(exc)) from exc

    def write_export(
        self,
        plan: ModulationRecognitionExportPlan,
        out_dir: Path,
        *,
        force: bool = False,
    ) -> JsonDict:
        try:
            return write_modulation_recognition_starter(plan, out_dir, force=force)
        except ModulationRecognitionExportError as exc:
            raise DifferentiableExportError(str(exc)) from exc


class CsiFeedbackExporter(DifferentiableExporter):
    id = "csi-feedback"
    name = "Paired CSI compression and feedback exporter"

    def supports(self, recipe: Recipe, registry: OperationRegistry) -> bool:
        slots = [
            step
            for step in recipe.steps
            if step.op in {CSI_FEEDBACK_ENCODER_OP, CSI_FEEDBACK_DECODER_OP}
        ]
        if not is_csi_feedback_slot_pair(slots):
            return False
        try:
            build_csi_feedback_export_plan(
                recipe,
                registry,
                optimizable_steps=[step.id for step in slots],
                loss=CSI_FEEDBACK_LOSS,
                framework="torch",
            )
        except (CsiFeedbackExportError, ValueError):
            return False
        return True

    def build_plan(
        self,
        recipe: Recipe,
        registry: OperationRegistry,
        options: ExportOptions,
    ) -> CsiFeedbackExportPlan:
        try:
            plan = build_csi_feedback_export_plan(
                recipe,
                registry,
                optimizable_steps=options.optimizable_steps,
                loss=options.loss,
                framework=options.framework,
            )
            return replace(plan, project_root=options.project_root)
        except CsiFeedbackExportError as exc:
            raise DifferentiableExportError(str(exc)) from exc

    def write_export(
        self,
        plan: CsiFeedbackExportPlan,
        out_dir: Path,
        *,
        force: bool = False,
    ) -> JsonDict:
        try:
            return write_csi_feedback_starter(plan, out_dir, force=force)
        except CsiFeedbackExportError as exc:
            raise DifferentiableExportError(str(exc)) from exc


class MimoOfdmChannelEstimationExporter(DifferentiableExporter):
    id = "mimo-ofdm-channel-estimation"
    name = "MIMO-OFDM channel-estimation demonstration project builder"

    def supports(self, recipe: Recipe, registry: OperationRegistry) -> bool:
        estimators = [
            step for step in recipe.steps if step.op == CHANNEL_ESTIMATOR_OP
        ]
        if len(estimators) != 1:
            return False
        try:
            build_channel_estimation_export_plan(
                recipe,
                registry,
                optimizable_steps=[estimators[0].id],
                loss=CHANNEL_ESTIMATION_LOSS,
                framework="torch",
            )
        except (ChannelEstimationExportError, ValueError):
            return False
        return True

    def build_plan(
        self,
        recipe: Recipe,
        registry: OperationRegistry,
        options: ExportOptions,
    ) -> ChannelEstimationExportPlan:
        try:
            plan = build_channel_estimation_export_plan(
                recipe,
                registry,
                optimizable_steps=options.optimizable_steps,
                loss=options.loss,
                framework=options.framework,
            )
            return replace(plan, project_root=options.project_root)
        except ChannelEstimationExportError as exc:
            raise DifferentiableExportError(str(exc)) from exc

    def write_export(
        self,
        plan: ChannelEstimationExportPlan,
        out_dir: Path,
        *,
        force: bool = False,
    ) -> JsonDict:
        try:
            return write_channel_estimation_starter(plan, out_dir, force=force)
        except ChannelEstimationExportError as exc:
            raise DifferentiableExportError(str(exc)) from exc


class DelayedCsiResourceAllocationExporter(DifferentiableExporter):
    id = "delayed-csi-resource-allocation"
    name = "Finite-blocklength delayed-CSI power-allocation demonstration project builder"

    def supports(self, recipe: Recipe, registry: OperationRegistry) -> bool:
        allocators = [
            step
            for step in recipe.steps
            if step.op == "model.symbol_power_allocator"
        ]
        if len(allocators) != 1:
            return False
        try:
            build_delayed_csi_resource_allocation_export_plan(
                recipe,
                registry,
                optimizable_steps=[allocators[0].id],
                loss=DELAYED_CSI_RESOURCE_ALLOCATION_LOSS,
                framework="torch",
            )
        except (DelayedCsiResourceAllocationExportError, ValueError):
            return False
        return True

    def build_plan(
        self,
        recipe: Recipe,
        registry: OperationRegistry,
        options: ExportOptions,
    ) -> DelayedCsiResourceAllocationExportPlan:
        try:
            plan = build_delayed_csi_resource_allocation_export_plan(
                recipe,
                registry,
                optimizable_steps=options.optimizable_steps,
                loss=options.loss,
                framework=options.framework,
            )
            return replace(plan, project_root=options.project_root)
        except DelayedCsiResourceAllocationExportError as exc:
            raise DifferentiableExportError(str(exc)) from exc

    def write_export(
        self,
        plan: DelayedCsiResourceAllocationExportPlan,
        out_dir: Path,
        *,
        force: bool = False,
    ) -> JsonDict:
        try:
            return write_delayed_csi_resource_allocation_starter(
                plan,
                out_dir,
                force=force,
            )
        except DelayedCsiResourceAllocationExportError as exc:
            raise DifferentiableExportError(str(exc)) from exc


class ResourceAllocationExporter(DifferentiableExporter):
    id = "resource-allocation"
    name = "Label-free CSI power-allocation demonstration project builder"

    def supports(self, recipe: Recipe, registry: OperationRegistry) -> bool:
        allocators = [step for step in recipe.steps if step.op == "model.symbol_power_allocator"]
        if len(allocators) != 1:
            return False
        try:
            build_resource_allocation_export_plan(
                recipe,
                registry,
                optimizable_steps=[allocators[0].id],
                loss=RESOURCE_ALLOCATION_LOSS,
                framework="torch",
            )
        except (ResourceAllocationExportError, ValueError):
            return False
        return True

    def build_plan(
        self,
        recipe: Recipe,
        registry: OperationRegistry,
        options: ExportOptions,
    ) -> ResourceAllocationExportPlan:
        try:
            plan = build_resource_allocation_export_plan(
                recipe,
                registry,
                optimizable_steps=options.optimizable_steps,
                loss=options.loss,
                framework=options.framework,
            )
            return replace(plan, project_root=options.project_root)
        except ResourceAllocationExportError as exc:
            raise DifferentiableExportError(str(exc)) from exc

    def write_export(
        self,
        plan: ResourceAllocationExportPlan,
        out_dir: Path,
        *,
        force: bool = False,
    ) -> JsonDict:
        try:
            return write_resource_allocation_export(plan, out_dir, force=force)
        except ResourceAllocationExportError as exc:
            raise DifferentiableExportError(str(exc)) from exc


def available_exporters() -> List[DifferentiableExporter]:
    return [
        CsiFeedbackExporter(),
        MimoOfdmChannelEstimationExporter(),
        DeepJSCCImageExporter(),
        ModulationRecognitionExporter(),
        PhaseTrackingReceiverExporter(),
        NeuralReceiverExporter(),
        DelayedCsiResourceAllocationExporter(),
        ResourceAllocationExporter(),
        TextSemanticJSCCExporter(),
        TaskHeadExporter(),
        DatasetCaptureOnlyExporter(),
    ]


def exporter_ids() -> List[str]:
    return [exporter.id for exporter in available_exporters()]


def inspect_training_capture(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    optimizable_steps: Optional[Sequence[str] | str] = None,
    route_loss_steps: Optional[Sequence[str] | str] = None,
    project_root: Optional[Path] = None,
) -> JsonDict:
    """Describe editable capture signals and splits for the selected slot boundary."""

    if optimizable_steps is None:
        selected_ids: List[str] = []
    elif isinstance(optimizable_steps, str):
        selected_ids = [
            item.strip() for item in optimizable_steps.split(",") if item.strip()
        ]
    else:
        selected_ids = []
        for value in optimizable_steps:
            selected_ids.extend(
                item.strip() for item in str(value).split(",") if item.strip()
            )
    selected = [_step_by_id(recipe, step_id) for step_id in selected_ids]

    if selected:
        feasibility = inspect_training_feasibility(
            recipe,
            registry,
            optimizable_steps=[step.id for step in selected],
            loss=route_loss_steps,
        )
        if (
            feasibility.get("recommended_mode") == "differentiable_export"
            and not recipe.dataset_capture
        ):
            return _annotate_capture_candidate_lineage({
                "mode": "not_required",
                "ready": True,
                "issue": "",
                "sample_unit": "samples",
                "capture_runner_required": False,
                "candidates": capture_tap_candidates(recipe, registry),
                "required_taps": [],
                "selected_taps": [],
                "explanation": (
                    "The selected boundary has a live downstream gradient route; "
                    "tensor capture is optional rather than required."
                ),
            }, recipe, selected_ids)
        requirements = replacement_boundary_capture_requirements(
            recipe,
            registry,
            [step.id for step in selected],
        )
        required = list(requirements.get("required_taps") or [])
        unsupported = list(requirements.get("unsupported_inputs") or [])
        payload = capture_plan_inspection(
            recipe,
            registry,
            required_taps=required,
            sample_unit="recipe records",
            suggested_total_samples=1000,
        )
        payload["capture_purpose"] = "researcher_defined"
        payload["capture_runner_required"] = True
        payload["unsupported_inputs"] = unsupported
        selected_taps = list(payload.get("selected_taps") or [])
        if unsupported:
            payload["ready"] = False
            details = ", ".join(
                "%s.%s <- %s"
                % (item.get("step_id"), item.get("input"), item.get("from"))
                for item in unsupported
            )
            payload["issue"] = (
                "Generic capture cannot materialize non-array replacement input(s): %s. "
                "Add an operation-owned file-backed data provider or an array boundary."
                % details
            )
        elif not selected_taps:
            payload["ready"] = False
            payload["issue"] = (
                "Select at least one capturable recipe output for the external training dataset."
            )
        return _annotate_capture_candidate_lineage(payload, recipe, selected_ids)

    return _annotate_capture_candidate_lineage({
        "mode": "not_required",
        "ready": True,
        "issue": "",
        "sample_unit": "samples",
        "capture_runner_required": False,
        "candidates": capture_tap_candidates(recipe, registry),
        "required_taps": [],
        "selected_taps": list((recipe.dataset_capture or {}).get("taps") or []),
    }, recipe, selected_ids)


def _annotate_capture_candidate_lineage(
    payload: JsonDict,
    recipe: Recipe,
    selected_step_ids: Sequence[str],
) -> JsonDict:
    """Describe provenance without deciding how a researcher may use a signal."""

    selected = {str(item) for item in selected_step_ids}
    downstream = set(selected)
    changed = True
    while changed:
        changed = False
        for step in recipe.steps:
            if step.id in downstream:
                continue
            producers = {
                str(reference).split(".", 1)[0]
                for reference in step.inputs.values()
                if isinstance(reference, str) and "." in reference
            }
            if producers.intersection(downstream):
                downstream.add(step.id)
                changed = True
    for candidate in list(payload.get("candidates") or []):
        producer = str(candidate.get("step_id") or "")
        if producer in selected:
            candidate["relationship"] = "current_replacement_output"
            candidate["reason"] = (
                "Produced by the operation currently installed in a selected replacement Block. "
                "It may be captured as a teacher target, diagnostic, or auxiliary value, but it "
                "does not represent the future replacement's output."
            )
        elif producer in downstream:
            candidate["relationship"] = "current_pipeline_dependent"
            candidate["reason"] = (
                "Produced downstream of a selected replacement Block under the current pipeline. "
                "It may be useful as a target, diagnostic, or auxiliary value, but an offline "
                "capture will not change when the future replacement changes."
            )
        else:
            candidate["relationship"] = "scenario_signal"
            candidate["reason"] = "Capturable output from the current system scenario."
        candidate["selectable"] = True
    return payload


def _default_options(*, exporter: str = "auto") -> ExportOptions:
    return ExportOptions(
        optimizable_steps=[],
        loss="image.mse",
        framework="torch-sionna",
        exporter=exporter,
    )


def export_differentiable_scenario(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    optimizable_steps: Sequence[str] | str,
    route_loss_steps: Optional[Sequence[str] | str] = None,
    loss: str,
    framework: str,
    out_dir: Path,
    source_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
    force: bool = False,
    exporter: str = "auto",
    include_starter: bool = False,
    training_objective: str = "",
) -> JsonDict:
    trainable_steps = _normalize_optimizable_steps(optimizable_steps)
    selected_route_loss_steps = (
        _normalize_optimizable_steps(route_loss_steps)
        if route_loss_steps
        else []
    )
    options = ExportOptions(
        optimizable_steps=trainable_steps,
        loss=str(loss or ("image.mse" if include_starter else "")),
        framework=str(framework or "torch"),
        exporter=str(exporter or "auto"),
        source_path=source_path,
        project_root=project_root,
        include_starter=bool(include_starter),
    )
    selected: Optional[DifferentiableExporter] = None
    plan = None
    if include_starter:
        selected = select_differentiable_exporter(recipe, registry, options)
        plan = selected.build_plan(recipe, registry, options)
    route_inspection = inspect_training_feasibility(
        recipe,
        registry,
        optimizable_steps=trainable_steps,
        loss=tuple(selected_route_loss_steps) or None,
    )
    analyzed_route_loss_steps = [
        str(item) for item in route_inspection.get("selected_loss_steps") or []
    ]
    destination = Path(out_dir)
    _prepare_contract_export_directory(destination, force=force)
    try:
        compiled = compile_training_contract(
            recipe,
            registry,
            options=_neutral_contract_options(
                recipe,
                registry,
                trainable_steps=trainable_steps,
                framework=options.framework,
                route_loss_steps=selected_route_loss_steps,
            ),
        )
        result = write_training_contract_bundle(compiled, destination, force=force)
    except TrainingContractError as exc:
        raise DifferentiableExportError(str(exc)) from exc
    effective_route_loss_steps = [
        str(item) for item in compiled.contract.get("recipe_loss_steps") or []
    ]

    resolved_project_root = Path(project_root or Path.cwd()).resolve()
    result = _normalize_contract_manifest_paths(
        result,
        destination=destination,
        project_root=resolved_project_root,
    )

    result = _attach_neutral_data_contract(
        result,
        recipe,
        registry,
        trainable_steps=trainable_steps,
        destination=destination,
        project_root=resolved_project_root,
        training_mode=str(route_inspection.get("recommended_mode") or ""),
    )

    training_plan = TrainingPlan(
        selected_steps=trainable_steps,
        # Preserve the researcher's recipe boundary even when feasibility
        # chooses capture-backed training and therefore exports no live
        # downstream autograd route.
        loss_steps=tuple(analyzed_route_loss_steps),
        dataset_capture=dict(recipe.dataset_capture or {}),
        objective=(
            options.loss
            if include_starter
            else str(training_objective or "").strip()
        ),
        starter=options.exporter if include_starter else "",
        framework=options.framework,
    )
    training_plan_payload = training_plan.to_dict()
    training_plan_sha256 = training_plan_fingerprint(training_plan)
    _write_yaml(destination / "training_plan.yaml", training_plan_payload)
    project_manifest = dict(result.get("project_manifest") or {})
    project_manifest["training_plan"] = {
        "path": "training_plan.yaml",
        "sha256": training_plan_sha256,
    }
    result["project_manifest"] = project_manifest
    _write_yaml(destination / "project_manifest.yaml", project_manifest)
    files = list(result.get("files") or [])
    if "training_plan.yaml" not in files:
        files.append("training_plan.yaml")
    result["files"] = files

    result.update(
        {
            "recipe": recipe.name,
            "recipe_sha256": scenario_recipe_fingerprint(recipe),
            "training_plan_sha256": training_plan_sha256,
            "framework": options.framework,
            "replacement_steps": list(trainable_steps),
            "route_loss_steps": list(analyzed_route_loss_steps),
            "live_route_loss_steps": list(effective_route_loss_steps),
            "training_mode": str(route_inspection.get("recommended_mode") or ""),
            "optimizable_steps": list(trainable_steps),
            "include_starter": bool(include_starter),
            "exporter": "training-contract",
            "capture_jobs": list(
                (result.get("project_manifest") or {}).get("capture_jobs") or []
            ),
            "trained_artifacts": list(
                (result.get("project_manifest") or {}).get("trained_artifacts") or []
            ),
        }
    )
    if not include_starter:
        return result

    assert selected is not None and plan is not None
    starter_dir = destination / "reference_training"
    starter_payload = selected.write_export(plan, starter_dir, force=True)
    return _attach_optional_starter(
        result,
        starter_payload,
        destination=destination,
        selected_exporter=selected.id,
    )


def _prepare_contract_export_directory(out_dir: Path, *, force: bool) -> None:
    """Validate an overwrite target, then leave the existing bundle in place.

    The bundle is the researcher's working directory after handoff.  Removing
    the whole tree would also remove their model, loss, trainer, checkpoints,
    and notes.  The writers replace their known contract files in place; any
    researcher-owned additions and already captured datasets remain untouched.
    """

    if not force or not out_dir.is_dir() or not any(out_dir.iterdir()):
        return
    manifest_path = out_dir / "project_manifest.yaml"
    manifest = None
    if manifest_path.is_file():
        try:
            manifest = load_strict_yaml_or_json(manifest_path)
        except (OSError, UnicodeError, ValueError) as exc:
            raise DifferentiableExportError(
                "Cannot verify ownership of existing training bundle %s: %s"
                % (out_dir, exc)
            ) from exc
    generated_kinds = {
        "noema.training_interface_bundle@1",
        "noema.standalone_training_project",
    }
    if not isinstance(manifest, Mapping) or str(manifest.get("kind") or "") not in generated_kinds:
        raise DifferentiableExportError(
            "Refusing to overwrite a nonempty directory that is not a recognized Noema training bundle: %s"
            % out_dir
        )
    _remove_verified_demo_handoff_files(out_dir, manifest)


def _remove_verified_demo_handoff_files(
    out_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Remove only unchanged launchers owned by the optional demo helper.

    A forced neutral re-export replaces the manifest and may invalidate an
    attached demonstration's generated launchers.  The nested training project
    and every researcher-owned file remain untouched.  Modified launchers are
    also preserved because their recorded digest no longer proves ownership.
    """

    external_training = manifest.get("external_training") or {}
    if not isinstance(external_training, Mapping):
        return
    scaffold = external_training.get("optional_demo_scaffold") or {}
    if not isinstance(scaffold, Mapping):
        return
    handoff = scaffold.get("root_handoff") or {}
    if (
        not isinstance(handoff, Mapping)
        or str(handoff.get("ownership") or "") != "noema_demo_helper"
    ):
        return
    allowed = {"RUN_DEMO.md", "train_demo.py", "evaluate_demo.py"}
    marker = b"Generated by demo_trainings/prepare_example.py; non-normative demo handoff."
    for raw_item in list(handoff.get("managed_files") or []):
        if not isinstance(raw_item, Mapping):
            continue
        relative = str(raw_item.get("path") or "")
        expected_sha256 = str(raw_item.get("sha256") or "").lower()
        if relative not in allowed or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            continue
        path = out_dir / relative
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if marker not in payload[:512]:
            continue
        if hashlib.sha256(payload).hexdigest() == expected_sha256:
            path.unlink()


def _normalize_contract_manifest_paths(
    result: JsonDict,
    *,
    destination: Path,
    project_root: Path,
) -> JsonDict:
    def portable(path: Path) -> str:
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(project_root))
        except ValueError:
            return str(resolved)

    manifest = dict(result.get("project_manifest") or {})
    manifest["out_dir"] = portable(destination)
    training = dict(manifest.get("training") or {})
    training["working_directory"] = portable(destination)
    manifest["training"] = training
    trained = []
    for item in list(manifest.get("trained_artifacts") or []):
        row = dict(item)
        row["manifest_path"] = portable(destination / "trained_artifact.yaml")
        trained.append(row)
    manifest["trained_artifacts"] = trained
    _write_yaml(destination / "project_manifest.yaml", manifest)
    merged = dict(result)
    merged["project_manifest"] = manifest
    return merged


def _attach_neutral_data_contract(
    result: JsonDict,
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    trainable_steps: Sequence[str],
    destination: Path,
    project_root: Path,
    training_mode: str,
) -> JsonDict:
    try:
        if (
            str(training_mode) == "differentiable_export"
            and not recipe.dataset_capture
        ):
            return result
        data = write_generic_capture_data_contract(
            recipe,
            registry,
            selected_step_ids=list(trainable_steps),
            out_dir=destination,
            project_root=project_root,
        )
    except (
        GenericCaptureDataContractError,
    ) as exc:
        raise DifferentiableExportError(str(exc)) from exc
    manifest_path = destination / "project_manifest.yaml"
    manifest = dict(result.get("project_manifest") or {})
    manifest["capture_jobs"] = list(data.get("capture_jobs") or [])
    manifest["data_contract"] = {
        "path": "data_contract.yaml",
        "kind": "noema.training_data_contract@1",
        "sha256": str(data.get("data_contract_sha256") or ""),
        "file_sha256": str(data.get("data_contract_file_sha256") or ""),
        "ownership": dict((data.get("data_contract") or {}).get("ownership") or {}),
    }
    _write_yaml(manifest_path, manifest)
    merged = dict(result)
    merged["project_manifest"] = manifest
    merged["capture_jobs"] = list(data.get("capture_jobs") or [])
    merged["data_contract"] = dict(data.get("data_contract") or {})
    merged["files"] = list(result.get("files") or []) + list(data.get("files") or [])
    return merged


def _is_deepjscc_slot_pair(selected: Sequence[RecipeStep]) -> bool:
    return len(selected) == 2 and {
        step.op for step in selected
    } == {
        "model.deepjscc_external_encode",
        "model.deepjscc_external_decode",
    }

def _portable_project_path(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(resolved)


def _neutral_contract_options(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    trainable_steps: Sequence[str],
    framework: str,
    route_loss_steps: Sequence[str] = (),
) -> TrainingContractOptions:
    """Add only domain/interface facts; never prescribe architecture or objective."""

    selected = [_step_by_id(recipe, step_id) for step_id in trainable_steps]
    slot_roles: Dict[str, str] = {}
    groups: List[SlotGroupSpec] = []
    conditioning: List[NamedValueSpec] = []
    signals: List[SignalSpec] = []
    constraints: List[ConstraintSpec] = []
    tensor_overrides: Dict[str, Mapping[str, Any]] = {}

    allocator_steps = [step for step in selected if step.op == "model.symbol_power_allocator"]
    csi_encoders = [step for step in selected if step.op == CSI_FEEDBACK_ENCODER_OP]
    csi_decoders = [step for step in selected if step.op == CSI_FEEDBACK_DECODER_OP]
    deep_encoders = [step for step in selected if step.op == "model.deepjscc_external_encode"]
    deep_decoders = [step for step in selected if step.op == "model.deepjscc_external_decode"]
    neural_receivers = [
        step
        for step in selected
        if step.op == "demodulation.neural_receiver_adapter"
    ]
    phase_tracking_receivers = [
        step
        for step in selected
        if step.op == "demodulation.phase_tracking_receiver_adapter"
    ]
    modulation_classifiers = [
        step for step in selected if step.op == MODULATION_CLASSIFIER_OP
    ]

    if len(csi_encoders) == 1 and len(csi_decoders) == 1 and len(selected) == 2:
        encoder = csi_encoders[0]
        decoder = csi_decoders[0]
        channel = _producer_step(recipe, encoder, "csi")
        feedback_link = _producer_step(recipe, decoder, "received_code")
        feedback_dimension = int(encoder.params.get("feedback_dimension", 64))
        decoder_dimension = int(decoder.params.get("feedback_dimension", 64))
        if decoder_dimension != feedback_dimension:
            raise TrainingContractError(
                "CSI feedback encoder and decoder feedback_dimension values must match"
            )
        tx_antennas = int(channel.params.get("tx_antennas", 8))
        subcarriers = int(channel.params.get("ofdm_fft_size", 32))
        bits_per_latent = int(feedback_link.params.get("bits_per_latent", 8))
        clip_value = float(feedback_link.params.get("clip_value", 1.0))
        feedback_mode = str(
            feedback_link.params.get("mode") or "ideal_noiseless"
        )
        csi_shape = ["batch", 2, tx_antennas, subcarriers]
        code_shape = ["batch", feedback_dimension]
        slot_roles.update({encoder.id: "encoder", decoder.id: "decoder"})
        groups.append(
            SlotGroupSpec(
                id="csi_feedback_codec",
                slots=(encoder.id, decoder.id),
                joint_training=False,
                atomic_artifact_return=True,
                description=(
                    "CSI encoder and decoder form one paired runtime interface and return "
                    "as one atomic two-component artifact. Their training method is external."
                ),
            )
        )
        tensor_overrides.update(
            {
                "%s.inputs.csi" % encoder.id: {
                    "kind": "channel.miso_ofdm_csi.numpy",
                    "dtype": "float32",
                    "shape": csi_shape,
                    "layout": "N_ri_tx_subcarrier",
                    "domain": "normalized_complex_CSI_real_imag",
                },
                "%s.outputs.feedback_code" % encoder.id: {
                    "kind": "channel.csi_feedback_code.numpy",
                    "dtype": "float32",
                    "shape": code_shape,
                    "layout": "N_feedback_latent",
                    "domain": "bounded_real_feedback_latent",
                },
                "%s.inputs.received_code" % decoder.id: {
                    "kind": "channel.csi_feedback_received.numpy",
                    "dtype": "float32",
                    "shape": code_shape,
                    "layout": "N_feedback_latent",
                    "domain": "transported_feedback_latent",
                },
                "%s.outputs.reconstruction" % decoder.id: {
                    "kind": "channel.miso_ofdm_csi_reconstruction.numpy",
                    "dtype": "float32",
                    "shape": csi_shape,
                    "layout": "N_ri_tx_subcarrier",
                    "domain": "reconstructed_complex_CSI_real_imag",
                },
            }
        )
        conditioning.extend(
            [
                NamedValueSpec(
                    id="downlink_snr_db",
                    source="%s.params.downlink_snr_db" % channel.id,
                    units="dB",
                    description="Downlink SNR used for reconstructed-CSI MRT evaluation.",
                ),
                NamedValueSpec(
                    id="feedback_dimension",
                    source="%s.params.feedback_dimension" % encoder.id,
                    dtype="int64",
                    description="Real-valued feedback latent count per CSI realization.",
                ),
                NamedValueSpec(
                    id="bits_per_latent",
                    source="%s.params.bits_per_latent" % feedback_link.id,
                    dtype="int64",
                    units="bits",
                    description="Fixed quantization resolution of each feedback latent.",
                ),
            ]
        )
        signals.extend(
            [
                SignalSpec(
                    id="true_csi",
                    source=str(encoder.inputs.get("csi") or "%s.csi" % channel.id),
                    tensor=TensorSpec(
                        kind="channel.miso_ofdm_csi.numpy",
                        dtype="float32",
                        shape=tuple(csi_shape),
                        layout="N_ri_tx_subcarrier",
                        domain="normalized_complex_CSI_real_imag",
                    ),
                    purpose="replacement_input_or_researcher_selected_target",
                ),
                SignalSpec(
                    id="reconstructed_csi",
                    source="%s.reconstruction" % decoder.id,
                    tensor=TensorSpec(
                        kind="channel.miso_ofdm_csi_reconstruction.numpy",
                        dtype="float32",
                        shape=tuple(csi_shape),
                        layout="N_ri_tx_subcarrier",
                        domain="reconstructed_complex_CSI_real_imag",
                    ),
                    purpose="current_pipeline_output_or_auxiliary_signal",
                ),
            ]
        )
        constraints.extend(
            [
                ConstraintSpec(
                    id="feedback_width",
                    expression="feedback_code.shape[1] = feedback_dimension",
                    enforcement="artifact_compatibility_test",
                    scope="per_batch",
                ),
                ConstraintSpec(
                    id="feedback_bit_budget",
                    expression=(
                        "feedback_bits_per_sample = feedback_dimension * bits_per_latent = %d"
                        % (feedback_dimension * bits_per_latent)
                    ),
                    enforcement="frozen_feedback_link",
                    scope="per_sample",
                ),
                ConstraintSpec(
                    id="feedback_transport",
                    expression=(
                        "mode = %s; clip_value = %.12g"
                        % (feedback_mode, clip_value)
                    ),
                    enforcement="frozen_feedback_link_and_external_training_surrogate",
                    scope="per_feedback_latent",
                    description=(
                        "An external quantization-aware trainer may use a surrogate only for "
                        "the fixed uniform quantizer's backward pass; deployed forward values "
                        "must match the recipe feedback link."
                    ),
                ),
            ]
        )
    elif len(allocator_steps) == 1 and len(selected) == 1:
        allocator = allocator_steps[0]
        slot_roles[allocator.id] = "allocator"
        groups.append(
            SlotGroupSpec(
                id="%s_group" % allocator.id,
                slots=(allocator.id,),
                joint_training=False,
                atomic_artifact_return=False,
            )
        )
        conditioning.extend(
            [
                NamedValueSpec(
                    id="noise_variance",
                    source="channel_state.params.noise_variance",
                    units="normalized_power",
                    description="Noise power used with each realized channel state.",
                ),
                NamedValueSpec(
                    id="average_power_budget",
                    source="%s.params.target_power" % allocator.id,
                    units="normalized_power",
                    description="Average power budget controlled by the recipe.",
                ),
            ]
        )
        constraints.extend(
            [
                ConstraintSpec(
                    id="nonnegative_power",
                    expression="allocated_power >= 0",
                    enforcement="noema_runtime_projection",
                ),
                ConstraintSpec(
                    id="exact_sum_power",
                    expression=(
                        "sum_subcarrier(allocated_power) = "
                        "subcarrier_count * average_power_budget"
                    ),
                    enforcement="noema_runtime_projection",
                ),
            ]
        )
    elif len(deep_encoders) == 1 and len(deep_decoders) == 1 and len(selected) == 2:
        encoder = deep_encoders[0]
        decoder = deep_decoders[0]
        slot_roles.update({encoder.id: "encoder", decoder.id: "decoder"})
        groups.append(
            SlotGroupSpec(
                id="deepjscc_sender_receiver",
                slots=(encoder.id, decoder.id),
                joint_training=False,
                atomic_artifact_return=True,
                description=(
                    "Encoder and decoder return as one atomic runtime artifact; the "
                    "researcher decides whether and how they are trained together."
                ),
            )
        )
        tensor_overrides["%s.outputs.symbols" % encoder.id] = {
            "kind": "channel.symbols.complex_numpy",
            "dtype": "complex64",
            "shape": ["batch", "symbol_channel", "symbol_height", "symbol_width"],
            "layout": "NCHW_complex",
        }
        tensor_overrides["%s.inputs.symbols" % decoder.id] = {
            "kind": "channel.symbols.complex_numpy",
            "dtype": "complex64",
            "shape": ["batch", "symbol_channel", "symbol_height", "symbol_width"],
            "layout": "NCHW_complex",
        }
        conditioning.append(
            NamedValueSpec(
                id="channel_condition",
                source="wireless_channel.params",
                description="Channel and noise settings owned by the frozen scenario.",
            )
        )
        signals.extend(
            [
                SignalSpec(
                    id="source_image",
                    source=str(encoder.inputs.get("images") or "data.images"),
                    tensor=TensorSpec(
                        kind="image.batch.numpy",
                        dtype="uint8",
                        shape=("batch", "height", "width", "channels"),
                        layout="NHWC",
                        domain="integer_[0,255]",
                    ),
                ),
                SignalSpec(
                    id="reconstruction",
                    source="%s.images" % decoder.id,
                    tensor=TensorSpec(
                        kind="image.batch.numpy",
                        dtype="uint8",
                        shape=("batch", "height", "width", "channels"),
                        layout="NHWC",
                        domain="integer_[0,255]",
                    ),
                ),
            ]
        )
        constraints.append(
            ConstraintSpec(
                id="average_transmit_power",
                expression="mean(abs(tx_symbols)**2) = recipe_target_power",
                enforcement="noema_frozen_scenario",
                scope="batch",
            )
        )
    elif len(neural_receivers) == 1 and len(selected) == 1:
        receiver = neural_receivers[0]
        slot_roles[receiver.id] = "receiver"
        groups.append(
            SlotGroupSpec(
                id="%s_group" % receiver.id,
                slots=(receiver.id,),
                joint_training=False,
                atomic_artifact_return=False,
            )
        )
        feature_reference = str(receiver.inputs.get("rx_symbols") or "")
        tensor_overrides["%s.inputs.rx_symbols" % receiver.id] = {
            "kind": "channel.rx_symbols.complex_numpy",
            "dtype": "complex64",
            "shape": ["symbol"],
            "layout": "complex_symbol",
        }
        tensor_overrides["%s.outputs.llr" % receiver.id] = {
            "kind": "channel.llr.numpy",
            "dtype": "float32",
            "shape": ["bit"],
            "layout": "flat_bit_llr",
        }
        conditioning.append(
            NamedValueSpec(
                id="channel_condition",
                source="%s.params" % feature_reference.split(".", 1)[0],
                description="Recipe-owned noise/channel settings accompanying captured received symbols.",
            )
        )
        constraints.append(
            ConstraintSpec(
                id="qpsk_bit_pairing",
                expression="output_bit_logit_count = 2 * received_symbol_count",
                enforcement="artifact_compatibility_test",
                scope="packet",
            )
        )
    elif len(phase_tracking_receivers) == 1 and len(selected) == 1:
        receiver = phase_tracking_receivers[0]
        slot_roles[receiver.id] = "receiver"
        groups.append(
            SlotGroupSpec(
                id="%s_group" % receiver.id,
                slots=(receiver.id,),
                joint_training=False,
                atomic_artifact_return=False,
            )
        )
        feature_reference = str(receiver.inputs.get("rx_symbols") or "")
        pilot_reference = str(receiver.inputs.get("pilot_context") or "")
        tensor_overrides["%s.inputs.rx_symbols" % receiver.id] = {
            "kind": "channel.rx_symbols.complex_numpy",
            "dtype": "complex64",
            "shape": ["packet", "frame_symbol"],
            "layout": "packet_complex_symbol",
        }
        tensor_overrides["%s.inputs.pilot_context" % receiver.id] = {
            "kind": "channel.qpsk_pilot_context.numpy",
            "dtype": "float32",
            "shape": ["packet", "frame_symbol", 3],
            "layout": "packet_symbol_pilot_mask_known_iq",
        }
        tensor_overrides["%s.inputs.phase_truth" % receiver.id] = {
            "kind": "channel.carrier_phase_truth.numpy",
            "dtype": "float32",
            "shape": ["packet", "frame_symbol"],
            "layout": "packet_symbol_phase_rad",
        }
        tensor_overrides["%s.outputs.llr" % receiver.id] = {
            "kind": "channel.llr.numpy",
            "dtype": "float32",
            "shape": ["packet", "data_bit"],
            "layout": "packet_flat_data_bit_llr",
        }
        conditioning.extend(
            [
                NamedValueSpec(
                    id="channel_condition",
                    source="%s.params" % feature_reference.split(".", 1)[0],
                    description=(
                        "Recipe-owned AWGN settings accompanying captured received symbols."
                    ),
                ),
                NamedValueSpec(
                    id="pilot_pattern",
                    source="%s.params" % pilot_reference.split(".", 1)[0],
                    description=(
                        "Public preamble, pilot spacing, and deterministic pilot-sequence settings."
                    ),
                ),
            ]
        )
        constraints.extend(
            [
                ConstraintSpec(
                    id="phase_tracking_context_alignment",
                    expression=(
                        "receiver_features.shape[0:2] = "
                        "rx_symbols.shape[0:2] = pilot_context.shape[0:2]"
                    ),
                    enforcement="artifact_compatibility_test",
                    scope="packet",
                ),
                ConstraintSpec(
                    id="phase_truth_excluded_from_learned_abi",
                    expression=(
                        "learned_runtime_inputs = [rx_symbols, pilot_context]"
                    ),
                    enforcement="operation_owned_artifact_abi",
                    scope="artifact",
                    description=(
                        "The optional oracle phase trace is diagnostic evidence and is never "
                        "an input to the learned receiver."
                    ),
                ),
            ]
        )
    elif len(modulation_classifiers) == 1 and len(selected) == 1:
        classifier = modulation_classifiers[0]
        feature_reference = str(classifier.inputs.get("observation") or "")
        feature_step_id = feature_reference.split(".", 1)[0]
        slot_roles[classifier.id] = "classifier"
        groups.append(
            SlotGroupSpec(
                id="%s_group" % classifier.id,
                slots=(classifier.id,),
                joint_training=False,
                atomic_artifact_return=False,
            )
        )
        tensor_overrides["%s.inputs.observation" % classifier.id] = {
            "kind": "ai_phy.modulation_iq_frames.numpy",
            "dtype": "float32",
            "shape": ["batch", "sample", 2],
            "layout": "batch_symbol_real_imag",
            "domain": "synchronized_complex_baseband_IQ",
        }
        tensor_overrides["%s.outputs.prediction" % classifier.id] = {
            "kind": "ai_phy.modulation_predictions.numpy",
            "dtype": "float32",
            "shape": ["batch", 3],
            "layout": "batch_class_logits",
            "domain": "ordered_classes_bpsk_qpsk_qam16",
        }
        conditioning.extend(
            [
                NamedValueSpec(
                    id="channel_snr_min_db",
                    source="%s.params.snr_db_min" % feature_step_id,
                    units="dB",
                    description="Lower bound of the recipe-owned AWGN capture range.",
                ),
                NamedValueSpec(
                    id="channel_snr_max_db",
                    source="%s.params.snr_db_max" % feature_step_id,
                    units="dB",
                    description="Upper bound of the recipe-owned AWGN capture range.",
                ),
            ]
        )
        constraints.extend(
            [
                ConstraintSpec(
                    id="fixed_class_vocabulary",
                    expression="class_names = [bpsk, qpsk, qam16]",
                    enforcement="artifact_compatibility_test",
                    scope="dataset",
                ),
            ]
        )
    else:
        for step in selected:
            operation = registry.get(step.op).describe()
            role = "trainable_block"
            output_kinds = set((operation.get("output_kinds") or {}).values())
            if "channel.llr.numpy" in output_kinds:
                role = "receiver"
            slot_roles[step.id] = role
            groups.append(
                SlotGroupSpec(
                    id="%s_group" % step.id,
                    slots=(step.id,),
                    joint_training=False,
                    atomic_artifact_return=False,
                )
            )

    inspection = inspect_training_feasibility(
        recipe,
        registry,
        optimizable_steps=trainable_steps,
        loss=tuple(route_loss_steps) or None,
    )
    route_loss_steps = (
        tuple(str(item) for item in inspection.get("selected_loss_steps") or [])
        if inspection.get("recommended_mode") == "differentiable_export"
        else ()
    )

    return TrainingContractOptions(
        trainable_steps=tuple(trainable_steps),
        framework=str(framework or "torch"),
        loss_steps=route_loss_steps,
        slot_roles=slot_roles,
        slot_groups=tuple(groups),
        tensor_overrides=tensor_overrides,
        conditioning=tuple(conditioning),
        signals=tuple(signals),
        constraints=tuple(constraints),
    )


def _attach_optional_starter(
    result: JsonDict,
    starter_payload: Mapping[str, Any],
    *,
    destination: Path,
    selected_exporter: str,
) -> JsonDict:
    """Attach a non-normative demo beneath a normative contract bundle."""

    starter_dir = destination / "reference_training"
    shutil.copy2(destination / "training_contract.yaml", starter_dir / "training_contract.yaml")
    starter_config_path = starter_dir / "train_config.yaml"
    if starter_config_path.is_file():
        starter_config = load_strict_yaml_or_json(starter_config_path)
        if not isinstance(starter_config, Mapping):
            raise DifferentiableExportError(
                "Starter training configuration must contain a mapping: %s"
                % starter_config_path
            )
        starter_config = dict(starter_config)
        starter_config["training_contract"] = {
            "path": "training_contract.yaml",
            "id": result.get("contract_id"),
            "sha256": result.get("contract_sha256"),
            "file_sha256": result.get("contract_file_sha256"),
        }
        # The scaffold executes from reference_training/, but its returned
        # artifact belongs to the neutral interface bundle.  Keep the
        # trainer optional while giving Noema one stable discovery path.
        training = dict(starter_config.get("training") or {})
        if training.get("artifact_manifest_path"):
            training["artifact_manifest_path"] = "../trained_artifact.yaml"
            if training.get("artifact_component_path"):
                training["artifact_component_path"] = (
                    "../artifacts/%s"
                    % Path(str(training["artifact_component_path"])).name
                )
            if training.get("artifact_component_paths"):
                training["artifact_component_paths"] = [
                    "../artifacts/%s" % Path(str(item)).name
                    for item in list(training["artifact_component_paths"])
                ]
            starter_config["training"] = training
        data_contract = dict(result.get("data_contract") or {})
        contract_reference = dict(
            ((result.get("project_manifest") or {}).get("data_contract") or {})
        )
        if str(data_contract.get("mode") or "") == "file_backed_live_differentiable":
            data_config = dict(starter_config.get("data") or {})
            data_config.pop("image_ids", None)
            data_config.update(
                {
                    "contract_path": "../data_contract.yaml",
                    "contract_sha256": str(contract_reference.get("sha256") or ""),
                    "contract_file_sha256": str(
                        contract_reference.get("file_sha256") or ""
                    ),
                    "train_image_ids": split_image_ids(data_contract, "train"),
                    "validation_image_ids": split_image_ids(
                        data_contract, "validation"
                    ),
                    "test_images_exposed_to_trainer": False,
                }
            )
            starter_config["data"] = data_config
        elif str(data_contract.get("mode") or "") == "captured_self_supervised_csi":
            data_config = dict(starter_config.get("data") or {})
            data_config.update(
                {
                    "contract_path": "../data_contract.yaml",
                    "contract_sha256": str(contract_reference.get("sha256") or ""),
                    "contract_file_sha256": str(
                        contract_reference.get("file_sha256") or ""
                    ),
                    "test_split_exposed_to_training": False,
                }
            )
            starter_config["data"] = data_config
        elif str(data_contract.get("mode") or "") == "captured_generic_tensors":
            data_config = dict(starter_config.get("data") or {})
            data_config.update(
                {
                    "contract_path": "../data_contract.yaml",
                    "contract_sha256": str(contract_reference.get("sha256") or ""),
                    "contract_file_sha256": str(
                        contract_reference.get("file_sha256") or ""
                    ),
                    "train_capture_dirs": ["../data/train"],
                    "validation_capture_dirs": ["../data/validation"],
                    "test_capture_dirs": ["../data/test"],
                }
            )
            signals = [
                dict(item)
                for item in list(data_contract.get("signals") or [])
                if isinstance(item, Mapping)
            ]
            tap_by_reference = {
                str(item.get("reference") or ""): str(item.get("tap_id") or "")
                for item in signals
            }
            recipe_config = dict(starter_config.get("recipe") or {})
            feature_reference = str(recipe_config.get("feature_reference") or "")
            target_reference = str(recipe_config.get("target_reference") or "")
            required_taps = [
                str(item.get("tap_id") or "")
                for item in signals
                if bool(item.get("required")) and str(item.get("tap_id") or "")
            ]
            if feature_reference and tap_by_reference.get(feature_reference):
                data_config["feature_tap"] = tap_by_reference[feature_reference]
            elif data_config.get("feature_tap") and required_taps:
                data_config["feature_tap"] = required_taps[0]
            if target_reference and tap_by_reference.get(target_reference):
                data_config["target_tap"] = tap_by_reference[target_reference]
            starter_config["data"] = data_config
        _write_yaml(starter_config_path, starter_config)

    root_manifest_path = destination / "project_manifest.yaml"
    root_manifest = load_strict_yaml_or_json(root_manifest_path)
    if not isinstance(root_manifest, Mapping):
        raise DifferentiableExportError(
            "Training project manifest must contain a mapping: %s"
            % root_manifest_path
        )
    starter_manifest = dict(starter_payload.get("project_manifest") or {})
    root_manifest = dict(root_manifest)
    returned_artifacts = list(root_manifest.get("trained_artifacts") or [])
    returned_manifest_path = str(
        (returned_artifacts[0] if returned_artifacts else {}).get("manifest_path")
        or (destination / "trained_artifact.yaml")
    )
    if starter_manifest:
        starter_training = dict(starter_manifest.get("training") or {})
        if starter_training.get("artifact_manifest_path"):
            starter_training["artifact_manifest_path"] = returned_manifest_path
            component_root = Path(returned_manifest_path).parent / "artifacts"
            if starter_training.get("component_paths"):
                starter_training["component_paths"] = [
                    str(component_root / Path(str(item)).name)
                    for item in list(starter_training["component_paths"])
                ]
            starter_manifest["training"] = starter_training
        starter_trained = []
        for item in list(starter_manifest.get("trained_artifacts") or []):
            row = dict(item)
            row["manifest_path"] = returned_manifest_path
            starter_trained.append(row)
        if starter_trained:
            starter_manifest["trained_artifacts"] = starter_trained
        _write_yaml(starter_dir / "project_manifest.yaml", starter_manifest)
    external_training = dict(root_manifest.get("external_training") or {})
    external_training["optional_demo_scaffold"] = {
        "included": True,
        "normative": False,
        "path": "reference_training",
        "exporter": selected_exporter,
        "architecture": "example_only",
        "loss": "example_only",
        "trainer": "example_only",
        "training": dict(starter_manifest.get("training") or {}),
        "evaluation": dict(starter_manifest.get("evaluation") or {}),
        "post_training": dict(starter_manifest.get("post_training") or {}),
        "capture_jobs": list(starter_manifest.get("capture_jobs") or []),
        "trained_artifacts": list(starter_manifest.get("trained_artifacts") or []),
    }
    root_manifest["external_training"] = external_training
    _write_yaml(root_manifest_path, root_manifest)

    starter_files = [
        "reference_training/%s" % item
        for item in list(starter_payload.get("files") or [])
    ]
    if "reference_training/training_contract.yaml" not in starter_files:
        starter_files.append("reference_training/training_contract.yaml")
    merged = dict(result)
    merged.update(
        {
            "include_starter": True,
            "demo_starter": {
                "path": str(starter_dir),
                "exporter": selected_exporter,
                "training_template": starter_payload.get("training_template"),
                "loss": starter_payload.get("loss"),
            },
            "exporter": "training-contract",
            "starter_exporter": selected_exporter,
            "project_manifest": root_manifest,
            "capture_jobs": list(root_manifest.get("capture_jobs") or []),
            "trained_artifacts": list(root_manifest.get("trained_artifacts") or []),
            "files": list(result.get("files") or []) + starter_files,
        }
    )
    return merged


def select_differentiable_exporter(recipe: Recipe, registry: OperationRegistry, options: ExportOptions) -> DifferentiableExporter:
    requested = str(options.exporter or "auto").strip().lower().replace("_", "-")
    exporters = available_exporters()
    by_id = {item.id: item for item in exporters}
    if requested != "auto":
        if requested not in by_id:
            raise DifferentiableExportError(
                "Unknown differentiable exporter `%s`; expected one of %s or auto."
                % (options.exporter, ", ".join(sorted(by_id)))
            )
        exporter = by_id[requested]
        try:
            exporter.build_plan(recipe, registry, options)
            return exporter
        except DifferentiableExportError as exc:
            reason = str(exc)
        raise DifferentiableExportError(
            "Exporter `%s` does not support recipe `%s`: %s"
            % (requested, recipe.name, reason)
        )
    unsupported: List[str] = []
    for exporter in exporters:
        if exporter.id in {"text-semantic-jscc", "task-head", "dataset-capture-only"}:
            continue
        try:
            exporter.build_plan(recipe, registry, options)
            return exporter
        except DifferentiableExportError as exc:
            unsupported.append("%s: %s" % (exporter.id, str(exc)))
    raise DifferentiableExportError(
        "No differentiable exporter supports recipe `%s`. Missing patterns: %s"
        % (recipe.name, "; ".join(unsupported) if unsupported else "none")
    )


def _write_deepjscc_export(
    plan: DeepJsccExportPlan,
    out_dir: Path,
    *,
    source_path: Optional[Path] = None,
    force: bool = False,
) -> JsonDict:
    out_dir = prepare_demo_starter_directory(
        Path(out_dir),
        force=force,
        error_type=DifferentiableExportError,
    )
    template_dir = _deepjscc_template_dir()
    static_files = [
        "model.py",
        "scenario.py",
        "datamodule.py",
        "losses.py",
        "train.py",
        "evaluate.py",
        "build_benchmark.py",
        "build_slow_fading_benchmark.py",
        "README.md",
        "requirements.txt",
    ]
    for filename in static_files:
        shutil.copy2(template_dir / filename, out_dir / filename)
    write_standalone_structured_input(out_dir)
    training_template = load_strict_yaml_or_json(template_dir / "template.yaml")
    if not isinstance(training_template, Mapping):
        raise DifferentiableExportError(
            "DeepJSCC training template must contain a mapping"
        )
    training_template = dict(training_template)
    training_template["id"] = plan.template_id
    if plan.channel == "flat_rayleigh":
        training_template["name"] = (
            "Nested-bandwidth DeepJSCC over blind slow Rayleigh fading"
        )
        training_template["model"] = {
            **dict(training_template.get("model") or {}),
            "architecture": "nested_bandwidth_blind_csi_residual_cnn_v4",
            "symbol_channel_options": [8, 16, 32],
            "channel_uses_per_source_pixel": [0.125, 0.25, 0.5],
        }
        training_template["frozen_differentiable_path"] = [
            "channel.symbol_power_normalize",
            "wireless.channel.flat_rayleigh.source_item.no_csi",
        ]
        training_template["channel"] = {
            "supported": ["flat_rayleigh"],
            "receiver_processing": "none",
            "fading_scope": "source_item",
            "channel_state_information": "none",
        }
        training_template["post_training"] = {
            **dict(training_template.get("post_training") or {}),
            "helper": "build_slow_fading_benchmark.py",
            "sweep": ["channel.snr_db", "channel.uses_per_pixel"],
        }
    _write_yaml(out_dir / "training_template.yaml", training_template)

    project_root = Path(plan.project_root or Path.cwd()).resolve()
    config = _training_config(plan, project_root=project_root)
    _write_yaml(out_dir / "train_config.yaml", config)
    _write_yaml(out_dir / "noema_recipe.yaml", plan.recipe.to_dict())
    _write_yaml(out_dir / "training_contract.yaml", _deepjscc_training_contract(plan))
    project_manifest = _deepjscc_project_manifest(
        plan,
        out_dir=out_dir,
        project_root=project_root,
    )
    _write_yaml(out_dir / "project_manifest.yaml", project_manifest)

    files = [
        *static_files,
        "structured_input.py",
        "training_template.yaml",
        "train_config.yaml",
        "training_contract.yaml",
        "project_manifest.yaml",
        "noema_recipe.yaml",
    ]
    return {
        "status": "exported",
        "recipe": plan.recipe.name,
        "recipe_sha256": plan.recipe_sha256,
        "exporter": DeepJSCCImageExporter.id,
        "framework": plan.framework,
        "loss": plan.loss,
        "training_template": plan.template_id,
        "out_dir": str(out_dir),
        "optimizable_steps": list(plan.optimizable_steps),
        "channel": {"step_id": plan.channel_step.id, "type": plan.channel, "snr_db": list(plan.snr_db)},
        "project_manifest": project_manifest,
        "capture_jobs": list(project_manifest["capture_jobs"]),
        "trained_artifacts": list(project_manifest["trained_artifacts"]),
        "files": files,
    }


def build_deepjscc_export_plan(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    optimizable_steps: Sequence[str] | str,
    loss: str,
    framework: str,
) -> DeepJsccExportPlan:
    validate_recipe_against_registry(recipe, registry)
    framework = _normalize_framework(framework)
    if framework != "torch":
        raise DifferentiableExportError(
            "DeepJSCC export currently requires framework=torch because its "
            "checked-in reference starter uses Noema's pure-Torch AWGN and "
            "blind slow-Rayleigh blocks. Sionna 2.x/PyTorch channels are "
            "supported by the generic differentiable export graph, but this "
            "specialized starter does not silently substitute them; select "
            "framework=torch."
        )
    if str(loss).strip().lower() != "image.mse":
        raise DifferentiableExportError("differentiable export MVP supports --loss image.mse; got %s" % loss)
    trainable = _normalize_optimizable_steps(optimizable_steps)
    if len(trainable) < 2:
        raise DifferentiableExportError(
            "DeepJSCC export requires sender and receiver replacement step ids."
        )

    sender = _find_sender(recipe, registry, trainable)
    receiver = _find_receiver(recipe, registry, trainable)
    data = _producer_step(recipe, sender, "images")
    if data.op != "source.image_dataset":
        raise DifferentiableExportError(
            "DeepJSCC export MVP expects image batches from source.image_dataset; sender %s consumes %s."
            % (sender.id, data.op)
        )
    if not _path_exists(recipe, sender.id, receiver.id):
        raise DifferentiableExportError("receiver %s is not downstream of sender %s." % (receiver.id, sender.id))
    channel = _find_channel_between(recipe, sender.id, receiver.id)
    channel_name = str(channel.params.get("channel") or "awgn").lower()
    if channel_name not in {"awgn", "flat_rayleigh"}:
        raise DifferentiableExportError(
            "DeepJSCC closed-loop export supports AWGN or blind slow "
            "flat-Rayleigh fading; %s uses %s."
            % (channel.id, channel_name)
        )
    if channel_name == "flat_rayleigh":
        receiver_processing = str(
            channel.params.get("receiver_processing") or "matched"
        )
        fading_scope = str(channel.params.get("fading_scope") or "symbol")
        if receiver_processing != "none" or fading_scope != "source_item":
            raise DifferentiableExportError(
                "DeepJSCC flat-Rayleigh export models blind slow fading and "
                "therefore requires receiver_processing=none and "
                "fading_scope=source_item."
            )
    noise_mode = str(channel.params.get("noise_mode") or "snr_at_unit_power")
    if noise_mode != "snr_at_unit_power":
        raise DifferentiableExportError(
            "DeepJSCC export requires noise_mode=snr_at_unit_power; %s uses %s."
            % (channel.id, noise_mode)
        )
    power_normalization = _find_power_normalization_between(recipe, sender.id, channel.id)

    return DeepJsccExportPlan(
        recipe=recipe,
        recipe_sha256=scenario_recipe_fingerprint(recipe),
        optimizable_steps=trainable,
        sender_step=sender,
        receiver_step=receiver,
        data_step=data,
        channel_step=channel,
        framework=framework,
        loss="image.mse",
        snr_db=_snr_values(recipe, channel),
        channel=channel_name,
        power_normalization_step=power_normalization,
        power_normalization_target=_power_normalization_target(power_normalization),
        template_id=(
            "deepjscc_image_reconstruction.nested_bandwidth_slow_rayleigh"
            if channel_name == "flat_rayleigh"
            else "deepjscc_image_reconstruction.reference_cnn_awgn"
        ),
    )


def _find_power_normalization_between(recipe: Recipe, sender_id: str, channel_id: str) -> Optional[RecipeStep]:
    for step in recipe.steps:
        if step.op == "channel.symbol_power_normalize" and _path_exists(recipe, sender_id, step.id) and _path_exists(recipe, step.id, channel_id):
            return step
    return None


def _power_normalization_target(step: Optional[RecipeStep]) -> float:
    if step is None:
        return 1.0
    try:
        return max(0.0, float(step.params.get("target_power", 1.0)))
    except Exception:
        return 1.0


def build_neural_receiver_export_plan(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    options: ExportOptions,
) -> NeuralReceiverExportPlan:
    validate_recipe_against_registry(recipe, registry)
    framework = _normalize_framework(options.framework)
    loss = str(options.loss or "bit.bce").strip().lower()
    if loss not in {"bit.bce", "bits.bce", "bce"}:
        raise DifferentiableExportError("neural-receiver exporter supports --loss bit.bce; got %s" % options.loss)

    trainable = list(options.optimizable_steps)
    receiver = _find_neural_receiver(recipe, registry, trainable)
    receiver_inputs = dict(receiver.inputs)
    feature_input = next(
        (
            name
            for name in ("rx_symbols", "features", "symbols", "llr")
            if name in receiver_inputs
        ),
        "",
    )
    if not feature_input:
        raise DifferentiableExportError(
            "neural-receiver exporter expects a receiver input named rx_symbols/features/symbols/llr."
        )
    feature_step = _producer_step(recipe, receiver, feature_input)
    feature_reference = str(receiver_inputs[feature_input])
    target_reference = _neural_receiver_target_reference(
        recipe,
        registry,
        receiver,
    )
    target_step = _step_by_id(recipe, target_reference.split(".", 1)[0])
    feature_kind = _output_kind(registry, feature_step, feature_reference)
    target_kind = _output_kind(registry, target_step, target_reference)
    if feature_kind not in {"channel.rx_symbols.complex_numpy", "channel.llr.numpy", "channel.symbols.complex_numpy"}:
        raise DifferentiableExportError(
            "neural-receiver exporter needs channel rx symbols or LLR features; %s produces %s."
            % (receiver_inputs[feature_input], feature_kind)
        )
    if target_kind not in {"channel.coded_bits.numpy", "channel.payload_bits.numpy", "channel.bits.numpy", "channel.demod_bits.numpy"}:
        raise DifferentiableExportError(
            "neural-receiver exporter needs target bits; %s produces %s."
            % (target_reference, target_kind)
        )
    try:
        capture_plan = resolve_training_capture_plan(
            recipe,
            registry,
            required_taps=(
                {
                    "id": "rx_symbols",
                    "from": feature_reference,
                    "role": "receiver_input_features",
                },
                {
                    "id": "target_bits",
                    "from": target_reference,
                    "role": "supervised_target",
                },
            ),
            sample_unit="packets",
            suggested_total_samples=48,
        )
    except TrainingCapturePlanError as exc:
        raise DifferentiableExportError(str(exc)) from exc
    return NeuralReceiverExportPlan(
        recipe=recipe,
        recipe_sha256=scenario_recipe_fingerprint(recipe),
        receiver_step=receiver,
        feature_step=feature_step,
        target_step=target_step,
        feature_input=feature_input,
        feature_reference=feature_reference,
        target_reference=target_reference,
        framework=framework,
        loss="bit.bce",
        feature_kind=feature_kind,
        target_kind=target_kind,
        capture_plan=capture_plan,
    )


def build_phase_tracking_receiver_export_plan(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    options: ExportOptions,
) -> PhaseTrackingReceiverExportPlan:
    """Resolve the graph-owned capture boundary for the phase-tracking demo.

    The deployable model receives only the impaired symbols and public pilot
    context.  Transmitted data bits are an offline supervised target.  The
    channel's phase trace may be selected separately for diagnostics, but is
    intentionally absent from both this required plan and the artifact ABI.
    """

    validate_recipe_against_registry(recipe, registry)
    framework = _normalize_framework(options.framework)
    loss = str(options.loss or "bit.bce").strip().lower()
    if loss not in {"bit.bce", "bits.bce", "bce"}:
        raise DifferentiableExportError(
            "phase-tracking-receiver exporter supports --loss bit.bce; got %s"
            % options.loss
        )

    selected_ids = list(options.optimizable_steps)
    candidates = [
        step
        for step in recipe.steps
        if step.op == "demodulation.phase_tracking_receiver_adapter"
        and (not selected_ids or step.id in selected_ids)
    ]
    if len(candidates) != 1:
        if selected_ids:
            raise DifferentiableExportError(
                "phase-tracking-receiver exporter requires exactly one selected "
                "demodulation.phase_tracking_receiver_adapter block"
            )
        raise DifferentiableExportError(
            "phase-tracking-receiver exporter requires exactly one "
            "demodulation.phase_tracking_receiver_adapter block"
        )
    if selected_ids != [] and selected_ids != [candidates[0].id]:
        raise DifferentiableExportError(
            "phase-tracking-receiver exporter replaces only %s"
            % candidates[0].id
        )

    receiver = candidates[0]
    feature_reference = str(receiver.inputs.get("rx_symbols") or "").strip()
    pilot_context_reference = str(
        receiver.inputs.get("pilot_context") or ""
    ).strip()
    if not feature_reference or not pilot_context_reference:
        raise DifferentiableExportError(
            "phase-tracking receiver requires connected rx_symbols and pilot_context inputs"
        )
    feature_step = _producer_step(recipe, receiver, "rx_symbols")
    pilot_context_step = _producer_step(recipe, receiver, "pilot_context")
    target_reference = _neural_receiver_target_reference(
        recipe,
        registry,
        receiver,
    )
    target_step = _step_by_id(recipe, target_reference.split(".", 1)[0])
    feature_kind = _output_kind(
        registry,
        feature_step,
        feature_reference,
    )
    pilot_context_kind = _output_kind(
        registry,
        pilot_context_step,
        pilot_context_reference,
    )
    target_kind = _output_kind(registry, target_step, target_reference)
    if feature_kind != "channel.rx_symbols.complex_numpy":
        raise DifferentiableExportError(
            "phase-tracking receiver requires channel.rx_symbols.complex_numpy; "
            "%s produces %s" % (feature_reference, feature_kind)
        )
    if pilot_context_kind != "channel.qpsk_pilot_context.numpy":
        raise DifferentiableExportError(
            "phase-tracking receiver requires channel.qpsk_pilot_context.numpy; "
            "%s produces %s" % (pilot_context_reference, pilot_context_kind)
        )
    if target_kind not in {
        "channel.coded_bits.numpy",
        "channel.payload_bits.numpy",
        "channel.bits.numpy",
        "channel.demod_bits.numpy",
    }:
        raise DifferentiableExportError(
            "phase-tracking receiver needs canonical transmitted target bits; "
            "%s produces %s" % (target_reference, target_kind)
        )

    try:
        required_taps = [
            {
                "id": "rx_symbols",
                "from": feature_reference,
                "role": "receiver_input_symbols",
            },
            {
                "id": "pilot_context",
                "from": pilot_context_reference,
                "role": "receiver_input_pilot_context",
            },
            {
                "id": "target_bits",
                "from": target_reference,
                "role": "supervised_target",
            },
        ]
        capture_plan = resolve_training_capture_plan(
            recipe,
            registry,
            required_taps=tuple(required_taps),
            sample_unit="packets",
            suggested_total_samples=1536,
        )
    except TrainingCapturePlanError as exc:
        raise DifferentiableExportError(str(exc)) from exc

    return PhaseTrackingReceiverExportPlan(
        recipe=recipe,
        recipe_sha256=scenario_recipe_fingerprint(recipe),
        receiver_step=receiver,
        feature_step=feature_step,
        pilot_context_step=pilot_context_step,
        target_step=target_step,
        feature_reference=feature_reference,
        pilot_context_reference=pilot_context_reference,
        target_reference=target_reference,
        framework=framework,
        loss="bit.bce",
        feature_kind=feature_kind,
        pilot_context_kind=pilot_context_kind,
        target_kind=target_kind,
        capture_plan=capture_plan,
    )


def _normalize_framework(framework: str) -> str:
    value = str(framework or "torch-sionna").strip().lower().replace("_", "-")
    if value not in {"torch-sionna", "torch"}:
        raise DifferentiableExportError("differentiable export framework must be torch-sionna or torch.")
    return value


def _normalize_optimizable_steps(optimizable_steps: Sequence[str] | str) -> List[str]:
    if isinstance(optimizable_steps, str):
        raw_items = optimizable_steps.split(",")
    else:
        raw_items: List[str] = []
        for item in optimizable_steps:
            raw_items.extend(str(item).split(","))
    values = [item.strip() for item in raw_items if item.strip()]
    if not values:
        raise DifferentiableExportError(
            "--replacement must name one or more recipe step ids "
            "(--optimizable remains a compatibility alias)."
        )
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _find_sender(recipe: Recipe, registry: OperationRegistry, trainable: Sequence[str]) -> RecipeStep:
    for step_id in trainable:
        step = _step_by_id(recipe, step_id)
        kinds = _operation_kinds(registry, step)
        if "image.batch.numpy" in kinds["inputs"] and "channel.symbols.complex_numpy" in kinds["outputs"]:
            return step
    raise DifferentiableExportError(
        "Could not identify a replaceable image-to-symbol sender. Use a step like model.deepjscc_external_encode."
    )


def _find_receiver(recipe: Recipe, registry: OperationRegistry, trainable: Sequence[str]) -> RecipeStep:
    for step_id in trainable:
        step = _step_by_id(recipe, step_id)
        kinds = _operation_kinds(registry, step)
        if (
            {"channel.symbols.complex_numpy", "channel.rx_symbols.complex_numpy"} & kinds["inputs"]
            and "image.batch.numpy" in kinds["outputs"]
        ):
            return step
    raise DifferentiableExportError(
        "Could not identify a replaceable symbol-to-image receiver. Use a step like model.deepjscc_external_decode."
    )


def _find_neural_receiver(recipe: Recipe, registry: OperationRegistry, trainable: Sequence[str]) -> RecipeStep:
    candidate_ids = list(trainable) if trainable else [step.id for step in recipe.steps]
    missing = []
    for step_id in candidate_ids:
        step = _step_by_id(recipe, step_id)
        operation = registry.get(step.op).describe()
        training_capabilities = dict(operation.get("training_capabilities") or {})
        kinds = _operation_kinds(registry, step)
        has_feature = bool(kinds["inputs"].intersection({"channel.rx_symbols.complex_numpy", "channel.llr.numpy", "channel.symbols.complex_numpy"}))
        emits_prediction = bool(kinds["outputs"].intersection({"channel.llr.numpy", "channel.demod_bits.numpy", "channel.payload_bits.numpy", "channel.bits.numpy"}))
        if bool(training_capabilities.get("portable_replacement")) and has_feature and emits_prediction:
            return step
        if step_id in trainable:
            missing.append(
                "%s needs a portable trained-artifact replacement ABI plus a channel rx/LLR "
                "feature input and bit/LLR output" % step_id
            )
    raise DifferentiableExportError(
        "Could not identify a replaceable neural receiver. %s"
        % ("; ".join(missing) if missing else "Use a step that consumes rx_symbols/llr and emits bits or LLRs.")
    )


def _neural_receiver_target_reference(
    recipe: Recipe,
    registry: OperationRegistry,
    receiver: RecipeStep,
) -> str:
    """Discover supervised target bits from graph evidence, never a runtime model input.

    A classical BER/BLER metric already records which transmitted bit boundary is
    compared with the receiver prediction.  Reusing that reference makes the
    capture contract follow recipe topology and keeps target labels out of the
    deployable receiver ABI.  Synthetic training-only operations with an explicit
    target input remain supported for contract tests and third-party operations.
    """

    for name in ("target_bits", "bits"):
        reference = str(receiver.inputs.get(name) or "")
        if reference:
            return reference

    bit_kinds = {
        "channel.coded_bits.numpy",
        "channel.payload_bits.numpy",
        "channel.bits.numpy",
        "channel.demod_bits.numpy",
    }
    candidates: List[str] = []
    for step in recipe.steps:
        reference = str(step.inputs.get("reference") or "")
        prediction = str(step.inputs.get("candidate") or "")
        if not reference or not prediction:
            continue
        prediction_step = prediction.split(".", 1)[0]
        if prediction_step != receiver.id and not _path_exists(
            recipe, receiver.id, prediction_step
        ):
            continue
        try:
            target_step = _step_by_id(recipe, reference.split(".", 1)[0])
            target_kind = _output_kind(registry, target_step, reference)
        except (DifferentiableExportError, ValueError):
            continue
        if target_kind in bit_kinds and reference not in candidates:
            candidates.append(reference)
    if not candidates:
        raise DifferentiableExportError(
            "Could not discover neural-receiver target bits. Connect the receiver "
            "prediction to a metric with `candidate` and a transmitted-bit `reference`."
        )
    if len(candidates) > 1:
        raise DifferentiableExportError(
            "Neural-receiver graph has ambiguous target-bit references: %s"
            % ", ".join(candidates)
        )
    return candidates[0]


def _output_kind(registry: OperationRegistry, step: RecipeStep, reference: str) -> str:
    _producer_id, output_name = reference.split(".", 1)
    return str(registry.get(step.op).output_kinds.get(output_name) or "")


def _operation_kinds(registry: OperationRegistry, step: RecipeStep) -> Dict[str, Set[str]]:
    operation = registry.get(step.op).describe()
    input_kinds: Set[str] = set()
    all_input_kinds = {
        **dict(operation.get("input_kinds") or {}),
        **dict(operation.get("optional_input_kinds") or {}),
    }
    for values in all_input_kinds.values():
        input_kinds.update(str(item) for item in values)
    output_kinds = {str(value) for value in dict(operation.get("output_kinds") or {}).values()}
    return {"inputs": input_kinds, "outputs": output_kinds}


def _step_by_id(recipe: Recipe, step_id: str) -> RecipeStep:
    for step in recipe.steps:
        if step.id == step_id:
            return step
    raise DifferentiableExportError("Optimizable step id is not in the recipe: %s" % step_id)


def _producer_step(recipe: Recipe, step: RecipeStep, input_name: str) -> RecipeStep:
    reference = step.inputs.get(input_name)
    if not reference:
        raise DifferentiableExportError("Step %s does not consume input %s." % (step.id, input_name))
    producer_id, _output_name = reference.split(".", 1)
    return _step_by_id(recipe, producer_id)


def _consumer_graph(recipe: Recipe) -> Dict[str, Set[str]]:
    consumers: Dict[str, Set[str]] = {step.id: set() for step in recipe.steps}
    for step in recipe.steps:
        for reference in step.inputs.values():
            producer_id, _output_name = reference.split(".", 1)
            consumers.setdefault(producer_id, set()).add(step.id)
    return consumers


def _path_exists(recipe: Recipe, start: str, end: str) -> bool:
    consumers = _consumer_graph(recipe)
    queue = [start]
    visited: Set[str] = set()
    while queue:
        step_id = queue.pop(0)
        if step_id == end:
            return True
        if step_id in visited:
            continue
        visited.add(step_id)
        queue.extend(sorted(consumers.get(step_id, set()) - visited))
    return False


def _find_channel_between(recipe: Recipe, sender_id: str, receiver_id: str) -> RecipeStep:
    consumers = _consumer_graph(recipe)
    queue = [sender_id]
    visited: Set[str] = set()
    while queue:
        step_id = queue.pop(0)
        if step_id in visited:
            continue
        visited.add(step_id)
        step = _step_by_id(recipe, step_id)
        if step.op == "wireless.channel":
            if _path_exists(recipe, step.id, receiver_id):
                return step
        queue.extend(sorted(consumers.get(step_id, set()) - visited))
    raise DifferentiableExportError("DeepJSCC differentiable export requires a wireless.channel step between sender and receiver.")


def _snr_values(recipe: Recipe, channel: RecipeStep) -> List[float]:
    try:
        matrix_values = matrix_values_for_step_param(recipe, channel.id, "snr_db")
    except RecipeMatrixError as exc:
        raise DifferentiableExportError("Invalid recipe SNR matrix: %s" % exc) from exc
    if matrix_values is not None:
        return _parse_numeric_values(matrix_values)
    return _parse_numeric_values(channel.params.get("snr_db", 12.0))


def _parse_numeric_values(value: Any) -> List[float]:
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
    else:
        text = str(value).strip()
        if "," in text:
            if ":" in text:
                raise DifferentiableExportError(
                    "Numeric values must use either a:b:c range syntax or a,b,c list syntax; got %s" % text
                )
            parts = [item.strip() for item in text.split(",")]
            if any(not item for item in parts):
                raise DifferentiableExportError("Numeric lists must not contain empty values: %s" % text)
            values = [float(item) for item in parts]
        elif ":" in text:
            parts = [float(item.strip()) for item in text.split(":") if item.strip()]
            if len(parts) == 2:
                start, stop = parts
                step = 1.0
            elif len(parts) == 3:
                start, step, stop = parts
            else:
                raise DifferentiableExportError("Range values must use a:c or a:b:c syntax; got %s" % text)
            if step == 0:
                raise DifferentiableExportError("Range step must be nonzero: %s" % text)
            values = []
            current = start
            epsilon = abs(step) * 1e-9 + 1e-9
            if step > 0:
                while current <= stop + epsilon:
                    values.append(round(current, 12))
                    current += step
            else:
                while current >= stop - epsilon:
                    values.append(round(current, 12))
                    current += step
        else:
            values = [float(text)]
    if not values:
        raise DifferentiableExportError("SNR configuration produced no values.")
    return values


def _deepjscc_template_dir() -> Path:
    relative = Path("demo_trainings") / "deepjscc_image_reconstruction"
    candidate = find_demo_training_dir(relative)
    if candidate is None:
        raise DifferentiableExportError(
            "DeepJSCC demonstration project was not found; expected %s" % relative
        )
    return candidate


def _deepjscc_project_path(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(resolved)


def _deepjscc_data_path(value: Any, project_root: Path) -> str:
    path = Path(str(value or ".noema/datasets/kodak")).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    return str(path.resolve())


def _training_config(plan: DeepJsccExportPlan, *, project_root: Path) -> JsonDict:
    data_params = plan.data_step.params
    safe_recipe = _safe_identifier(plan.recipe.name)
    return {
        "schema_version": 1,
        "training_template": plan.template_id,
        "project_root": str(Path(project_root).resolve()),
        "recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
            "source_step": plan.data_step.id,
            "sender_step": plan.sender_step.id,
            "receiver_step": plan.receiver_step.id,
            "channel_step": plan.channel_step.id,
        },
        "framework": "torch",
        "objective": {
            "loss": plan.loss,
            "direction": "minimize",
            "formula": "mean((reconstruction - source_image)^2)",
        },
        "trainable_slots": [
            {
                "step_id": plan.sender_step.id,
                "operation": plan.sender_step.op,
                "role": "encoder",
            },
            {
                "step_id": plan.receiver_step.id,
                "operation": plan.receiver_step.op,
                "role": "decoder",
            },
        ],
        "channel": {
            "step_id": plan.channel_step.id,
            "type": plan.channel,
            "noise_mode": "snr_at_unit_power",
            "snr_db": list(plan.snr_db),
            "default_snr_db": float(plan.snr_db[0]),
            "receiver_processing": str(
                plan.channel_step.params.get("receiver_processing")
                or "matched"
            ),
            "fading_scope": str(
                plan.channel_step.params.get("fading_scope") or "symbol"
            ),
            "recipe_seed": int(plan.channel_step.params.get("seed") or 0),
            "trainable_parameters": False,
            "autograd": "full_to_transmitted_symbols",
        },
        "symbol_power": {
            "enabled": plan.power_normalization_step is not None,
            "step_id": plan.power_normalization_step.id if plan.power_normalization_step is not None else "",
            "target_power": float(plan.power_normalization_target),
            "eps": 1e-8,
            "trainable_parameters": False,
        },
        "data": {
            "dataset": data_params.get("dataset", "kodak"),
            "dataset_dir": _deepjscc_data_path(
                data_params.get("dataset_dir", ".noema/datasets/kodak"),
                project_root,
            ),
            "image_ids": data_params.get("image_ids", "kodim01"),
            "crop_size": int(data_params.get("crop_size", 64) or 64),
            "repeat_count": int(data_params.get("repeat_count", 1) or 1),
        },
        "model": {
            "class": "ReferenceDeepJSCCModel",
            "architecture": (
                "nested_bandwidth_blind_csi_residual_cnn_v4"
                if plan.channel == "flat_rayleigh"
                else "compact_residual_cnn_v2"
            ),
            "symbol_channels": 32,
            "symbol_channel_options": (
                [8, 16, 32]
                if plan.channel == "flat_rayleigh"
                else [32]
            ),
            "downsampling_factor": 8,
            "channel_uses_per_source_pixel": 0.5,
        },
        "training": {
            "epochs": 80,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "early_stopping_patience": 12,
            "initialization_seeds": [23],
            "validation_noise_seed": 9001,
            "num_workers": 0,
            "device": "cuda_if_available",
            "artifact_manifest_path": "trained_artifact.yaml",
            "artifact_component_paths": [
                "artifacts/encoder.onnx",
                "artifacts/decoder.onnx",
            ],
            "history_path": "training_history.json",
            "artifact_id": "%s.deepjscc.compact_residual_cnn" % safe_recipe,
            "artifact_name": "Learned DeepJSCC model for %s" % plan.recipe.name,
            "artifact_label": "Learned DeepJSCC · %s" % plan.recipe.name,
        },
        "evaluation": {
            "snr_db": list(plan.snr_db),
            "noise_seed": 19001,
            "device": "cpu",
            "metrics_path": "evaluation_metrics.json",
            "split": "validation",
            "held_out_test_owner": "noema_ordinary_recipe_or_benchmark",
        },
        "notes": [
            "Sender and receiver recipe steps are trainable slots, not pre-existing model implementations copied from Noema.",
            "model.py is an optional template-owned example architecture; the exported slot contract remains authoritative.",
            (
                "The frozen power normalizer and blind slow-Rayleigh channel "
                "remain inside the PyTorch autograd graph."
                if plan.channel == "flat_rayleigh"
                else "The frozen power normalizer and AWGN channel remain inside the PyTorch autograd graph."
            ),
            "Training writes a hash-pinned paired artifact for the existing encoder and decoder operations.",
        ],
    }


def _deepjscc_training_contract(plan: DeepJsccExportPlan) -> JsonDict:
    return {
        "schema_version": 1,
        "kind": "noema.deepjscc_training_contract",
        "training_template": plan.template_id,
        "source_recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
        },
        "trainable_slots": [
            {
                "step_id": plan.sender_step.id,
                "operation": plan.sender_step.op,
                "role": "encoder",
                "input": {"kind": "image.batch.numpy", "layout": "NHWC", "runtime_dtype": "uint8"},
                "output": {"kind": "channel.symbols.complex_numpy", "training_layout": "NCHW", "dtype": "complex64"},
            },
            {
                "step_id": plan.receiver_step.id,
                "operation": plan.receiver_step.op,
                "role": "decoder",
                "input": {"kind": "channel.rx_symbols.complex_numpy", "training_layout": "NCHW", "dtype": "complex64"},
                "output": {"kind": "image.batch.numpy", "layout": "NHWC", "runtime_dtype": "uint8"},
            },
        ],
        "gradient_path": [
            {"role": "encoder", "step_id": plan.sender_step.id, "trainable": True},
            {
                "role": "average_power_normalization",
                "step_id": plan.power_normalization_step.id if plan.power_normalization_step else "",
                "enabled": plan.power_normalization_step is not None,
                "target_power": float(plan.power_normalization_target),
                "trainable": False,
                "gradient": "full",
            },
            {
                "role": "wireless_channel",
                "step_id": plan.channel_step.id,
                "channel": plan.channel,
                "noise_mode": "snr_at_unit_power",
                "snr_db": list(plan.snr_db),
                "receiver_processing": str(
                    plan.channel_step.params.get("receiver_processing")
                    or "matched"
                ),
                "fading_scope": str(
                    plan.channel_step.params.get("fading_scope") or "symbol"
                ),
                "trainable": False,
                "gradient": "full_to_transmitted_symbols",
            },
            {"role": "decoder", "step_id": plan.receiver_step.id, "trainable": True},
            {"role": "loss", "id": "image.mse"},
        ],
        "artifact_return": {
            "schema": "noema.trained_block_artifact.v2",
            "format": "onnx",
            "runtime": "onnxruntime",
            "components": ["encoder", "decoder"],
            "application": "all_group_bindings",
            "integrity": "sha256_required",
        },
        "runtime_binding": {
            "application": "all_group_bindings",
            "binding_group": "deepjscc_sender_receiver",
            "operations": [plan.sender_step.op, plan.receiver_step.op],
        },
    }


def _deepjscc_project_manifest(
    plan: DeepJsccExportPlan,
    *,
    out_dir: Path,
    project_root: Path,
) -> JsonDict:
    return {
        "schema_version": 1,
        "kind": "noema.standalone_training_project",
        "exporter": "deepjscc-image",
        "training_template": plan.template_id,
        "source_recipe": plan.recipe.name,
        "source_recipe_sha256": plan.recipe_sha256,
        "out_dir": _deepjscc_project_path(out_dir, project_root),
        "capture_jobs": [],
        "training": {
            "owner": "external",
            "working_directory": _deepjscc_project_path(out_dir, project_root),
            "requires_python": ">=3.11,<3.14",
            "dependency_file": _deepjscc_project_path(out_dir / "requirements.txt", project_root),
            "setup_command": "python -m pip install -r requirements.txt",
            "command": "python train.py",
            "artifact_manifest_path": _deepjscc_project_path(
                out_dir / "trained_artifact.yaml", project_root
            ),
            "component_paths": [
                _deepjscc_project_path(out_dir / "artifacts" / "encoder.onnx", project_root),
                _deepjscc_project_path(out_dir / "artifacts" / "decoder.onnx", project_root),
            ],
            "history_path": _deepjscc_project_path(out_dir / "training_history.json", project_root),
        },
        "evaluation": {
            "owner": "external",
            "command": "python evaluate.py",
            "metrics_path": _deepjscc_project_path(out_dir / "evaluation_metrics.json", project_root),
            "split": "validation",
            "held_out_test": False,
            "held_out_test_owner": "noema_ordinary_recipe_or_benchmark",
        },
        "post_training": {
            "owner": "external",
            "working_directory": _deepjscc_project_path(out_dir, project_root),
            "command": (
                "python build_slow_fading_benchmark.py"
                if plan.channel == "flat_rayleigh"
                else "python build_benchmark.py"
            ),
            "benchmark_pack_path": _deepjscc_project_path(
                out_dir / "benchmark_pack.yaml", project_root
            ),
            "benchmark_command": "noema benchmark run benchmark_pack.yaml",
            "presentation": "metadata.demo@1",
        },
        "trained_artifacts": [
            {
                "role": "trained_model_pair",
                "binding_group": "deepjscc_sender_receiver",
                "operations": [plan.sender_step.op, plan.receiver_step.op],
                "step_ids": [plan.sender_step.id, plan.receiver_step.id],
                "manifest_path": _deepjscc_project_path(out_dir / "trained_artifact.yaml", project_root),
            }
        ],
    }


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _safe_identifier(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return safe or "noema_differentiable_export"
