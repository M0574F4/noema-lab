from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from noema_lab.core.recipes import Recipe, recipe_from_dict
from noema_lab.core.structured_input import (
    StructuredInputError,
    load_strict_yaml_or_json,
)

JsonDict = Dict[str, Any]

TRAINING_PLAN_KIND = "noema.training_plan"
TRAINING_PLAN_SCHEMA_VERSION = 1
LEGACY_TRAINING_METADATA_FIELDS = frozenset({"training_performed"})


class TrainingPlanError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingPlan:
    """Researcher-selected training intent kept outside a runnable recipe."""

    selected_steps: Tuple[str, ...] = ()
    loss_steps: Tuple[str, ...] = ()
    dataset_capture: JsonDict = field(default_factory=dict)
    objective: str = ""
    starter: str = ""
    framework: str = ""
    extra_fields: JsonDict = field(default_factory=dict, repr=False)

    def to_dict(self) -> JsonDict:
        payload = copy.deepcopy(self.extra_fields)
        payload.update(
            {
                "kind": TRAINING_PLAN_KIND,
                "schema_version": TRAINING_PLAN_SCHEMA_VERSION,
            }
        )
        if self.selected_steps:
            payload["selected_steps"] = list(self.selected_steps)
        if self.loss_steps:
            payload["loss_steps"] = list(self.loss_steps)
        if self.dataset_capture:
            payload["dataset_capture"] = copy.deepcopy(self.dataset_capture)
        if self.objective:
            payload["objective"] = self.objective
        if self.starter:
            payload["starter"] = self.starter
        if self.framework:
            payload["framework"] = self.framework
        return payload


def training_plan_from_dict(value: Mapping[str, Any] | None) -> TrainingPlan:
    data = dict(value or {})
    kind = str(data.pop("kind", TRAINING_PLAN_KIND) or TRAINING_PLAN_KIND)
    if kind != TRAINING_PLAN_KIND:
        raise TrainingPlanError("Training plan kind must be %s" % TRAINING_PLAN_KIND)
    schema_version = int(data.pop("schema_version", TRAINING_PLAN_SCHEMA_VERSION))
    if schema_version != TRAINING_PLAN_SCHEMA_VERSION:
        raise TrainingPlanError(
            "Unsupported training-plan schema version: %s" % schema_version
        )
    raw_steps = data.pop("selected_steps", ()) or ()
    if not isinstance(raw_steps, (list, tuple)):
        raise TrainingPlanError("Training plan selected_steps must be a list")
    selected_steps = tuple(dict.fromkeys(str(item).strip() for item in raw_steps if str(item).strip()))
    raw_loss_steps = data.pop("loss_steps", ()) or ()
    if not isinstance(raw_loss_steps, (list, tuple)):
        raise TrainingPlanError("Training plan loss_steps must be a list")
    loss_steps = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in raw_loss_steps
            if str(item).strip()
        )
    )
    raw_capture = data.pop("dataset_capture", {}) or {}
    if not isinstance(raw_capture, Mapping):
        raise TrainingPlanError("Training plan dataset_capture must be a mapping")
    return TrainingPlan(
        selected_steps=selected_steps,
        loss_steps=loss_steps,
        dataset_capture=dict(raw_capture),
        objective=str(data.pop("objective", "") or ""),
        starter=str(data.pop("starter", "") or ""),
        framework=str(data.pop("framework", "") or ""),
        extra_fields=data,
    )


def load_training_plan(path: Path | str) -> TrainingPlan:
    plan_path = Path(path)
    try:
        data = load_strict_yaml_or_json(plan_path)
    except StructuredInputError as exc:
        raise TrainingPlanError(
            "Invalid training-plan YAML/JSON input %s: %s" % (plan_path, exc)
        ) from exc
    if not isinstance(data, Mapping):
        raise TrainingPlanError("Training plan must contain a YAML/JSON object: %s" % plan_path)
    return training_plan_from_dict(data)


def training_plan_fingerprint(plan: TrainingPlan) -> str:
    payload = json.dumps(
        plan.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scenario_recipe_fingerprint(recipe: Recipe) -> str:
    """Hash only the runnable scientific scenario, excluding training intent."""

    # Keep this import local so the migration helpers remain independent of the
    # reproducibility module's recipe serialization internals.
    from noema_lab.core.reproducibility import recipe_fingerprint

    return recipe_fingerprint(neutral_recipe(recipe))


def neutral_recipe(recipe: Recipe) -> Recipe:
    """Return the runnable experiment without legacy training intent."""

    payload = recipe.to_dict()
    payload.pop("dataset_capture", None)
    metadata = dict(payload.get("metadata") or {})
    for field_name in LEGACY_TRAINING_METADATA_FIELDS:
        metadata.pop(field_name, None)
    payload["metadata"] = metadata
    return recipe_from_dict(payload)


def extract_legacy_training_plan(recipe: Recipe) -> tuple[Recipe, TrainingPlan]:
    """Migrate legacy embedded capture fields without losing their values."""

    plan = TrainingPlan(dataset_capture=copy.deepcopy(recipe.dataset_capture or {}))
    return neutral_recipe(recipe), plan


def apply_training_plan(recipe: Recipe, plan: TrainingPlan) -> Recipe:
    """Materialize a transient capture/export recipe from a neutral recipe and plan."""

    payload = neutral_recipe(recipe).to_dict()
    if plan.dataset_capture:
        payload["dataset_capture"] = copy.deepcopy(plan.dataset_capture)
    return recipe_from_dict(payload)
