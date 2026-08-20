from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional

from noema_lab.core.execution_profiles import (
    ExecutionProfileDeclarationError,
    ExecutionProfileRef,
    execution_profile_ref_from_value,
)
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import Recipe, compile_recipe
from noema_lab.core.research import research_specs_from_recipe
from noema_lab.core.research_catalog import ResearchCatalog, load_research_catalog
from noema_lab.core.runtime_readiness import inspect_recipe_run_readiness
from noema_lab.core.structured_input import (
    StructuredInputError,
    decode_strict_yaml_or_json,
)

JsonDict = Dict[str, Any]

RECIPE_TEMPLATE_CATALOG_SCHEMA_VERSION = 1
RECIPE_TEMPLATE_PROVENANCE_SCHEMA_VERSION = 1
RECIPE_TEMPLATE_PROVENANCE_KIND = "noema.recipe_template_provenance"
RECIPE_TEMPLATE_STATUSES = {"supported", "experimental", "planned"}
RECIPE_TEMPLATE_EDITORS = {"graph", "image", "task", "text"}
RECIPE_TEMPLATE_SOURCE_KINDS = {"packaged_builtin", "project_override"}
RECIPE_TEMPLATE_PROVENANCE_METADATA_KEY = "template_provenance"

_ROOT_FIELDS = {"schema_version", "templates"}
_TEMPLATE_FIELDS = {
    "id",
    "label",
    "task_id",
    "editor",
    "recipe_path",
    "order",
    "default",
    "status",
    "execution_profile",
    "starter_resource",
    "editor_bindings",
}
_EDITOR_BINDING_FIELDS = {"step_id", "op", "param"}
_OVERRIDE_FIELDS = {"name", "description", "step_params"}
_UI_TEMPLATE_METADATA_FIELDS = {
    "opened_from",
    "recipe_template_id",
    "ui_editor",
    "ui_preserve_topology",
    "ui_topology_edited",
    "ui_working_copy",
}
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class RecipeTemplateCatalogError(ValueError):
    pass


class RecipeTemplateInstantiationError(RecipeTemplateCatalogError):
    pass


@dataclass(frozen=True)
class RecipeTemplateEditorBinding:
    step_id: str
    op: str
    param: str

    def to_dict(self) -> JsonDict:
        return {
            "step_id": self.step_id,
            "op": self.op,
            "param": self.param,
        }


@dataclass(frozen=True)
class RecipeTemplateDefinition:
    id: str
    label: str
    task_id: str
    editor: str
    recipe_path: str
    order: int
    default: bool
    status: str
    execution_profile: Optional[ExecutionProfileRef] = None
    starter_resource: Optional[str] = None
    editor_bindings: Dict[str, RecipeTemplateEditorBinding] = field(
        default_factory=dict
    )

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "id": self.id,
            "label": self.label,
            "task_id": self.task_id,
            "editor": self.editor,
            "recipe_path": self.recipe_path,
            "order": self.order,
            "default": self.default,
            "status": self.status,
        }
        if self.execution_profile is not None:
            payload["execution_profile"] = self.execution_profile.to_dict()
        if self.starter_resource is not None:
            payload["starter_resource"] = self.starter_resource
        if self.editor_bindings:
            payload["editor_bindings"] = {
                name: binding.to_dict()
                for name, binding in sorted(self.editor_bindings.items())
            }
        return payload


@dataclass(frozen=True)
class RecipeTemplateCatalog:
    templates: Dict[str, RecipeTemplateDefinition]
    schema_version: int = RECIPE_TEMPLATE_CATALOG_SCHEMA_VERSION

    def ordered_templates(self) -> List[RecipeTemplateDefinition]:
        return sorted(self.templates.values(), key=lambda item: (item.order, item.id))

    def template(self, template_id: str) -> Optional[RecipeTemplateDefinition]:
        return self.templates.get(str(template_id))

    def templates_for_task(self, task_id: str) -> List[RecipeTemplateDefinition]:
        return [
            item
            for item in self.ordered_templates()
            if item.task_id == str(task_id)
        ]

    def default_for_task(self, task_id: str) -> Optional[RecipeTemplateDefinition]:
        return next(
            (item for item in self.templates_for_task(task_id) if item.default),
            None,
        )

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "templates": [item.to_dict() for item in self.ordered_templates()],
        }


@dataclass(frozen=True)
class RecipeTemplateProvenance:
    template_id: str
    catalog_schema_version: int
    template_digest: str
    source_kind: str
    source_reference: str
    source_digest: str
    overrides_digest: str
    schema_version: int = RECIPE_TEMPLATE_PROVENANCE_SCHEMA_VERSION
    kind: str = RECIPE_TEMPLATE_PROVENANCE_KIND

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "template_id": self.template_id,
            "catalog_schema_version": self.catalog_schema_version,
            "template_digest": self.template_digest,
            "source_kind": self.source_kind,
            "source_reference": self.source_reference,
            "source_digest": self.source_digest,
            "overrides_digest": self.overrides_digest,
        }


@dataclass(frozen=True)
class RecipeTemplateInstantiation:
    recipe: Recipe
    provenance: RecipeTemplateProvenance

    def to_dict(self) -> JsonDict:
        return {
            "recipe": self.recipe.to_dict(),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class _ResolvedTemplateSource:
    kind: str
    reference: str
    content: bytes
    digest: str


@dataclass(frozen=True)
class _ValidatedTemplate:
    recipe: Recipe
    provenance: RecipeTemplateProvenance
    source_reference: str


class _RecipeTemplateSourceUnavailable(RecipeTemplateInstantiationError):
    pass


@dataclass(frozen=True)
class RecipeTemplateInspection:
    template: RecipeTemplateDefinition
    available: bool
    validation_status: str
    errors: List[str] = field(default_factory=list)
    recipe_name: str = ""
    recipe_description: str = ""
    step_count: int = 0
    resolved_task_id: str = ""
    recipe_execution_profile: Optional[JsonDict] = None
    source_kind: str = ""
    source_reference: str = ""
    source_digest: str = ""
    run_readiness: Optional[JsonDict] = None

    @property
    def valid(self) -> bool:
        return self.validation_status == "valid"

    def to_dict(self) -> JsonDict:
        payload = self.template.to_dict()
        payload.update(
            {
                "available": self.available,
                "validation": {
                    "status": self.validation_status,
                    "errors": list(self.errors),
                },
            }
        )
        if self.available:
            payload["recipe"] = {
                "name": self.recipe_name,
                "description": self.recipe_description,
                "step_count": self.step_count,
                "task_id": self.resolved_task_id,
                "execution_profile": dict(self.recipe_execution_profile or {}),
            }
            payload["source"] = {
                "kind": self.source_kind,
                "reference": self.source_reference,
                "digest": self.source_digest,
            }
            if self.run_readiness is not None:
                payload["run_readiness"] = dict(self.run_readiness)
        return payload


@dataclass(frozen=True)
class RecipeTemplateCatalogInspection:
    rows: List[RecipeTemplateInspection]
    schema_version: int = RECIPE_TEMPLATE_CATALOG_SCHEMA_VERSION

    @property
    def status(self) -> str:
        if any(row.validation_status == "invalid" for row in self.rows):
            return "invalid"
        if any(row.validation_status == "unavailable" for row in self.rows):
            return "degraded"
        return "valid"

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "templates": [row.to_dict() for row in self.rows],
        }


@lru_cache(maxsize=1)
def load_recipe_template_catalog() -> RecipeTemplateCatalog:
    data = decode_strict_yaml_or_json(
        resources.files("noema_lab")
        .joinpath("recipe_templates.yaml")
        .read_text(encoding="utf-8"),
        input_format="yaml",
    )
    return recipe_template_catalog_from_dict(data)


def recipe_template_catalog_from_dict(
    value: Any,
    *,
    research_catalog: Optional[ResearchCatalog] = None,
) -> RecipeTemplateCatalog:
    data = _mapping(value, "recipe template catalog")
    _reject_unknown_fields(data, _ROOT_FIELDS, "recipe template catalog")
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version != RECIPE_TEMPLATE_CATALOG_SCHEMA_VERSION:
        raise RecipeTemplateCatalogError(
            "recipe template catalog schema_version must be integer %d"
            % RECIPE_TEMPLATE_CATALOG_SCHEMA_VERSION
        )
    raw_templates = data.get("templates")
    if not isinstance(raw_templates, list) or not raw_templates:
        raise RecipeTemplateCatalogError(
            "recipe template catalog requires a non-empty `templates` list"
        )

    task_catalog = research_catalog or load_research_catalog()
    templates: Dict[str, RecipeTemplateDefinition] = {}
    for index, raw_template in enumerate(raw_templates):
        label = "templates[%d]" % index
        item = _template_from_dict(raw_template, label, task_catalog)
        if item.id in templates:
            raise RecipeTemplateCatalogError("duplicate recipe template id: %s" % item.id)
        templates[item.id] = item

    task_ids = sorted({item.task_id for item in templates.values()})
    for task_id in task_ids:
        defaults = [
            item
            for item in templates.values()
            if item.task_id == task_id and item.default
        ]
        if len(defaults) != 1:
            raise RecipeTemplateCatalogError(
                "task `%s` must have exactly one default recipe template; found %d"
                % (task_id, len(defaults))
            )
        if defaults[0].status != "supported":
            raise RecipeTemplateCatalogError(
                "default recipe template `%s` must have status `supported`"
                % defaults[0].id
            )

    return RecipeTemplateCatalog(
        schema_version=schema_version,
        templates=templates,
    )


def inspect_recipe_template_catalog(
    project_root: Path,
    registry: Any,
    *,
    catalog: Optional[RecipeTemplateCatalog] = None,
) -> RecipeTemplateCatalogInspection:
    catalog = catalog or load_recipe_template_catalog()
    root = Path(project_root).resolve()
    rows = [
        _inspect_recipe_template(
            item,
            root,
            registry,
            catalog_schema_version=catalog.schema_version,
        )
        for item in catalog.ordered_templates()
    ]
    return RecipeTemplateCatalogInspection(
        schema_version=catalog.schema_version,
        rows=rows,
    )


def instantiate_recipe_template(
    template_id: str,
    project_root: Optional[Path],
    registry: Any,
    *,
    catalog: Optional[RecipeTemplateCatalog] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> RecipeTemplateInstantiation:
    """Instantiate a catalog starter as an ordinary, validated recipe.

    A project file at the catalog entry's ``recipe_path`` is an explicit
    override.  When it is absent, the catalog's packaged ``starter_resource``
    is used.  The chosen source is read once, then overrides and canonical
    provenance are applied before strict compilation and registry planning.
    Invalid project overrides are never hidden by falling back to the built-in
    starter.
    """

    selected_catalog = catalog or load_recipe_template_catalog()
    if not isinstance(template_id, str) or not template_id.strip():
        raise RecipeTemplateInstantiationError(
            "recipe template id must be a non-empty string"
        )
    if template_id != template_id.strip():
        raise RecipeTemplateInstantiationError(
            "recipe template id must be trimmed"
        )
    template = selected_catalog.template(template_id)
    if template is None:
        raise RecipeTemplateInstantiationError(
            "Unknown recipe template id: %s" % template_id
        )

    root = Path(project_root).resolve() if project_root is not None else None
    source = _resolve_recipe_template_source(template, root)
    validated = _validate_recipe_template_source(
        template,
        source,
        registry,
        catalog_schema_version=selected_catalog.schema_version,
        overrides=overrides,
    )
    return RecipeTemplateInstantiation(
        recipe=validated.recipe,
        provenance=validated.provenance,
    )


def _template_from_dict(
    value: Any,
    label: str,
    research_catalog: ResearchCatalog,
) -> RecipeTemplateDefinition:
    data = _mapping(value, label)
    _reject_unknown_fields(data, _TEMPLATE_FIELDS, label)
    template_id = _identifier(data, "id", label)
    task_id = _identifier(data, "task_id", label)
    if research_catalog.task(task_id) is None:
        raise RecipeTemplateCatalogError(
            "%s references unknown research task `%s`" % (label, task_id)
        )
    editor = _identifier(data, "editor", label)
    if editor not in RECIPE_TEMPLATE_EDITORS:
        raise RecipeTemplateCatalogError(
            "%s.editor must be one of %s"
            % (label, sorted(RECIPE_TEMPLATE_EDITORS))
        )
    recipe_path = _safe_recipe_path(data.get("recipe_path"), "%s.recipe_path" % label)
    text_label = _nonempty_string(data, "label", label)
    status = _nonempty_string(data, "status", label)
    if status not in RECIPE_TEMPLATE_STATUSES:
        raise RecipeTemplateCatalogError(
            "%s.status must be one of %s"
            % (label, sorted(RECIPE_TEMPLATE_STATUSES))
        )
    order = data.get("order")
    if type(order) is not int or order < 0:
        raise RecipeTemplateCatalogError("%s.order must be a non-negative integer" % label)
    default = data.get("default")
    if type(default) is not bool:
        raise RecipeTemplateCatalogError("%s.default must be a boolean" % label)

    execution_profile = None
    if "execution_profile" in data:
        raw_profile = data.get("execution_profile")
        if not isinstance(raw_profile, Mapping):
            raise RecipeTemplateCatalogError(
                "%s.execution_profile must be a mapping" % label
            )
        try:
            execution_profile = execution_profile_ref_from_value(raw_profile)
        except ExecutionProfileDeclarationError as exc:
            raise RecipeTemplateCatalogError(
                "%s.execution_profile is invalid: %s" % (label, exc)
            ) from exc

    starter_resource = None
    if "starter_resource" in data:
        starter_resource = _safe_starter_resource(
            data.get("starter_resource"),
            "%s.starter_resource" % label,
        )

    editor_bindings: Dict[str, RecipeTemplateEditorBinding] = {}
    if "editor_bindings" in data:
        raw_bindings = _mapping(
            data.get("editor_bindings"),
            "%s.editor_bindings" % label,
        )
        for raw_name, raw_binding in raw_bindings.items():
            binding_label = "%s.editor_bindings[%r]" % (label, raw_name)
            if (
                not isinstance(raw_name, str)
                or not raw_name.strip()
                or raw_name != raw_name.strip()
                or not _IDENTIFIER_RE.fullmatch(raw_name)
            ):
                raise RecipeTemplateCatalogError(
                    "%s keys must be trimmed identifiers matching %s"
                    % ("%s.editor_bindings" % label, _IDENTIFIER_RE.pattern)
                )
            binding = _mapping(raw_binding, binding_label)
            _reject_unknown_fields(
                binding,
                _EDITOR_BINDING_FIELDS,
                binding_label,
            )
            editor_bindings[raw_name] = RecipeTemplateEditorBinding(
                step_id=_nonempty_string(binding, "step_id", binding_label),
                op=_nonempty_string(binding, "op", binding_label),
                param=_nonempty_string(binding, "param", binding_label),
            )

    return RecipeTemplateDefinition(
        id=template_id,
        label=text_label,
        task_id=task_id,
        editor=editor,
        recipe_path=recipe_path,
        order=order,
        default=default,
        status=status,
        execution_profile=execution_profile,
        starter_resource=starter_resource,
        editor_bindings=editor_bindings,
    )


def _inspect_recipe_template(
    template: RecipeTemplateDefinition,
    project_root: Path,
    registry: Any,
    *,
    catalog_schema_version: int,
) -> RecipeTemplateInspection:
    try:
        source = _resolve_recipe_template_source(template, project_root)
    except _RecipeTemplateSourceUnavailable as exc:
        return RecipeTemplateInspection(
            template=template,
            available=False,
            validation_status="unavailable",
            errors=[str(exc)],
        )
    except Exception as exc:
        return RecipeTemplateInspection(
            template=template,
            available=False,
            validation_status="invalid",
            errors=[str(exc)],
        )

    try:
        validated = _validate_recipe_template_source(
            template,
            source,
            registry,
            catalog_schema_version=catalog_schema_version,
        )
        recipe = validated.recipe
        specs = research_specs_from_recipe(recipe)
        resolved_task_id = str((specs.get("task") or {}).get("id") or "").strip()
        recipe_profile = recipe.execution_profile.to_dict()
        return RecipeTemplateInspection(
            template=template,
            available=True,
            validation_status="valid",
            recipe_name=recipe.name,
            recipe_description=str(recipe.description or ""),
            step_count=len(recipe.steps),
            resolved_task_id=resolved_task_id,
            recipe_execution_profile=recipe_profile,
            source_kind=validated.provenance.source_kind,
            source_reference=validated.source_reference,
            source_digest=validated.provenance.source_digest,
            run_readiness=inspect_recipe_run_readiness(recipe, registry),
        )
    except Exception as exc:
        return RecipeTemplateInspection(
            template=template,
            available=True,
            validation_status="invalid",
            errors=[str(exc)],
            source_kind=source.kind,
            source_reference=source.reference,
            source_digest=source.digest,
        )


def _resolve_recipe_template_source(
    template: RecipeTemplateDefinition,
    project_root: Optional[Path],
) -> _ResolvedTemplateSource:
    if project_root is not None:
        project_source = _read_project_template_override(template, project_root)
        if project_source is not None:
            return project_source

    if template.starter_resource is None:
        raise _RecipeTemplateSourceUnavailable(
            "Recipe template file is unavailable and no packaged starter is declared: %s"
            % template.recipe_path
        )
    try:
        content = (
            resources.files("noema_lab")
            .joinpath(*PurePosixPath(template.starter_resource).parts)
            .read_bytes()
        )
    except (FileNotFoundError, IsADirectoryError) as exc:
        raise _RecipeTemplateSourceUnavailable(
            "Packaged recipe starter is unavailable: %s" % template.starter_resource
        ) from exc
    except OSError as exc:
        raise _RecipeTemplateSourceUnavailable(
            "Packaged recipe starter could not be read: %s (%s)"
            % (template.starter_resource, exc)
        ) from exc
    return _ResolvedTemplateSource(
        kind="packaged_builtin",
        reference="noema_lab:%s" % template.starter_resource,
        content=content,
        digest=_sha256_digest(content),
    )


def _read_project_template_override(
    template: RecipeTemplateDefinition,
    project_root: Path,
) -> Optional[_ResolvedTemplateSource]:
    root = project_root.resolve()
    candidate = root / template.recipe_path
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return None
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RecipeTemplateInstantiationError(
            "Recipe template path escapes project root after resolution: %s"
            % template.recipe_path
        ) from exc
    try:
        content = resolved.read_bytes()
    except FileNotFoundError:
        # The override disappeared after resolution. Resolve the source that is
        # actually available at this call rather than using stale evidence.
        return None
    except (IsADirectoryError, OSError) as exc:
        raise RecipeTemplateInstantiationError(
            "Project recipe template could not be read: %s (%s)"
            % (template.recipe_path, exc)
        ) from exc
    return _ResolvedTemplateSource(
        kind="project_override",
        reference=template.recipe_path,
        content=content,
        digest=_sha256_digest(content),
    )


def _validate_recipe_template_source(
    template: RecipeTemplateDefinition,
    source: _ResolvedTemplateSource,
    registry: Any,
    *,
    catalog_schema_version: int,
    overrides: Optional[Mapping[str, Any]] = None,
) -> _ValidatedTemplate:
    normalized_overrides = _normalize_template_overrides(overrides)
    provenance = RecipeTemplateProvenance(
        template_id=template.id,
        catalog_schema_version=catalog_schema_version,
        template_digest=_canonical_digest(template.to_dict()),
        source_kind=source.kind,
        source_reference=source.reference,
        source_digest=source.digest,
        overrides_digest=_canonical_digest(normalized_overrides),
    )
    raw_recipe = _decode_recipe_template_source(template, source)
    _apply_template_overrides(raw_recipe, normalized_overrides, registry)
    _apply_template_provenance(raw_recipe, provenance)

    try:
        recipe = compile_recipe(
            raw_recipe,
            mode="strict",
            registry=registry,
        ).require_recipe()
        validate_recipe_against_registry(recipe, registry)
        _validate_template_contract(template, recipe, source.reference, registry)
    except RecipeTemplateInstantiationError:
        raise
    except Exception as exc:
        raise RecipeTemplateInstantiationError(
            "Recipe template `%s` from %s is invalid: %s"
            % (template.id, source.reference, exc)
        ) from exc
    return _ValidatedTemplate(
        recipe=recipe,
        provenance=provenance,
        source_reference=source.reference,
    )


def _decode_recipe_template_source(
    template: RecipeTemplateDefinition,
    source: _ResolvedTemplateSource,
) -> JsonDict:
    try:
        text = source.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecipeTemplateInstantiationError(
            "Recipe template `%s` source is not UTF-8: %s"
            % (template.id, source.reference)
        ) from exc
    try:
        value = decode_strict_yaml_or_json(
            text,
            input_format=(
                "json"
                if PurePosixPath(source.reference).suffix.lower() == ".json"
                else "yaml"
            ),
        )
    except StructuredInputError as exc:
        raise RecipeTemplateInstantiationError(
            "Recipe template `%s` source could not be decoded: %s"
            % (template.id, exc)
        ) from exc
    if not isinstance(value, Mapping):
        raise RecipeTemplateInstantiationError(
            "Recipe template `%s` source must contain a mapping" % template.id
        )
    return copy.deepcopy(dict(value))


def _validate_template_contract(
    template: RecipeTemplateDefinition,
    recipe: Recipe,
    source_reference: str,
    registry: Any,
) -> None:
    specs = research_specs_from_recipe(recipe)
    resolved_task_id = str((specs.get("task") or {}).get("id") or "").strip()
    if resolved_task_id != template.task_id:
        raise RecipeTemplateInstantiationError(
            "Recipe template `%s` declares task `%s`, but `%s` resolves to `%s`"
            % (
                template.id,
                template.task_id,
                source_reference,
                resolved_task_id or "<none>",
            )
        )

    steps = {step.id: step for step in recipe.steps}
    for name, binding in sorted(template.editor_bindings.items()):
        step = steps.get(binding.step_id)
        if step is None:
            raise RecipeTemplateInstantiationError(
                "Recipe template `%s` editor binding `%s` references missing step `%s` in `%s`"
                % (template.id, name, binding.step_id, source_reference)
            )
        if step.op != binding.op:
            raise RecipeTemplateInstantiationError(
                "Recipe template `%s` editor binding `%s` requires step `%s` to use `%s`, but `%s` uses `%s`"
                % (
                    template.id,
                    name,
                    binding.step_id,
                    binding.op,
                    source_reference,
                    step.op,
                )
            )
        allowed_params = set(step.params)
        try:
            operation = registry.get(step.op)
            schema = getattr(operation, "params_schema", {})
            properties = schema.get("properties") if isinstance(schema, Mapping) else None
            if isinstance(properties, Mapping):
                allowed_params.update(str(key) for key in properties)
        except Exception:
            # Registry validation has already reported unknown operations. Keep
            # this check focused on the declared editor binding contract.
            pass
        if binding.param not in allowed_params:
            raise RecipeTemplateInstantiationError(
                "Recipe template `%s` editor binding `%s` references unknown parameter `%s` on step `%s`"
                % (template.id, name, binding.param, binding.step_id)
            )
    recipe_profile = recipe.execution_profile.to_dict()
    if (
        template.execution_profile is not None
        and recipe_profile != template.execution_profile.to_dict()
    ):
        raise RecipeTemplateInstantiationError(
            "Recipe template `%s` declares execution profile %s, but `%s` uses %s"
            % (
                template.id,
                template.execution_profile.to_dict(),
                source_reference,
                recipe_profile,
            )
        )


def _normalize_template_overrides(
    value: Optional[Mapping[str, Any]],
) -> JsonDict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RecipeTemplateInstantiationError(
            "recipe template overrides must be a mapping"
        )
    data = dict(value)
    unknown = sorted(set(data).difference(_OVERRIDE_FIELDS))
    if unknown:
        raise RecipeTemplateInstantiationError(
            "recipe template overrides have unknown field(s): %s"
            % ", ".join(unknown)
        )

    normalized: JsonDict = {}
    if "name" in data:
        name = data["name"]
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise RecipeTemplateInstantiationError(
                "recipe template override `name` must be a trimmed non-empty string"
            )
        normalized["name"] = name
    if "description" in data:
        description = data["description"]
        if not isinstance(description, str) or description != description.strip():
            raise RecipeTemplateInstantiationError(
                "recipe template override `description` must be a trimmed string"
            )
        normalized["description"] = description
    if "step_params" in data:
        raw_step_params = data["step_params"]
        if not isinstance(raw_step_params, Mapping):
            raise RecipeTemplateInstantiationError(
                "recipe template override `step_params` must be a mapping"
            )
        step_params: JsonDict = {}
        for raw_step_id, raw_params in raw_step_params.items():
            if (
                not isinstance(raw_step_id, str)
                or not raw_step_id.strip()
                or raw_step_id != raw_step_id.strip()
            ):
                raise RecipeTemplateInstantiationError(
                    "recipe template override step IDs must be trimmed non-empty strings"
                )
            if not isinstance(raw_params, Mapping):
                raise RecipeTemplateInstantiationError(
                    "recipe template override for step `%s` must be a parameter mapping"
                    % raw_step_id
                )
            params: JsonDict = {}
            for raw_name, raw_value in raw_params.items():
                if (
                    not isinstance(raw_name, str)
                    or not raw_name.strip()
                    or raw_name != raw_name.strip()
                ):
                    raise RecipeTemplateInstantiationError(
                        "recipe template override parameter names must be trimmed non-empty strings"
                    )
                params[raw_name] = copy.deepcopy(raw_value)
            step_params[raw_step_id] = params
        normalized["step_params"] = step_params

    # Provenance requires a stable, language-neutral representation.  This
    # also rejects NaN, infinity, non-string mapping keys, and custom objects
    # before they can enter a public recipe.
    _canonical_json(normalized)
    return normalized


def _apply_template_overrides(
    raw_recipe: JsonDict,
    overrides: JsonDict,
    registry: Any,
) -> None:
    if "name" in overrides:
        raw_recipe["name"] = overrides["name"]
    if "description" in overrides:
        raw_recipe["description"] = overrides["description"]

    requested = dict(overrides.get("step_params") or {})
    if not requested:
        return
    raw_steps = raw_recipe.get("steps")
    if not isinstance(raw_steps, list):
        raise RecipeTemplateInstantiationError(
            "recipe template must define steps before step parameter overrides can be applied"
        )
    steps: Dict[str, JsonDict] = {}
    for raw_step in raw_steps:
        if not isinstance(raw_step, Mapping):
            continue
        step_id = raw_step.get("id")
        if isinstance(step_id, str) and step_id:
            steps[step_id] = raw_step  # type: ignore[assignment]

    unknown_steps = sorted(set(requested).difference(steps))
    if unknown_steps:
        raise RecipeTemplateInstantiationError(
            "recipe template overrides reference unknown step(s): %s"
            % ", ".join(unknown_steps)
        )
    for step_id, updates in requested.items():
        raw_step = steps[step_id]
        raw_params = raw_step.get("params")
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, Mapping):
            raise RecipeTemplateInstantiationError(
                "recipe template step `%s` params must be a mapping" % step_id
            )
        params = copy.deepcopy(dict(raw_params))
        allowed_params = set(params)
        operation_id = raw_step.get("op")
        try:
            operation = registry.get(operation_id)
            schema = getattr(operation, "params_schema", {})
            if isinstance(schema, Mapping):
                properties = schema.get("properties") or {}
                if isinstance(properties, Mapping):
                    allowed_params.update(str(key) for key in properties)
        except Exception:
            # Strict compilation below reports an unknown operation with its
            # normal diagnostic. Existing authored parameters remain eligible
            # so the override error does not mask that source error.
            pass
        unknown_params = sorted(set(updates).difference(allowed_params))
        if unknown_params:
            raise RecipeTemplateInstantiationError(
                "recipe template overrides reference unknown parameter(s) on step `%s`: %s"
                % (step_id, ", ".join(unknown_params))
            )
        params.update(copy.deepcopy(updates))
        raw_step["params"] = params


def _apply_template_provenance(
    raw_recipe: JsonDict,
    provenance: RecipeTemplateProvenance,
) -> None:
    raw_metadata = raw_recipe.get("metadata")
    if raw_metadata is None:
        metadata: JsonDict = {}
    elif isinstance(raw_metadata, Mapping):
        metadata = copy.deepcopy(dict(raw_metadata))
    else:
        # Leave shape validation to the strict compiler, where the error uses
        # the same recipe contract as every other entry point.
        return
    for key in list(metadata):
        if key in _UI_TEMPLATE_METADATA_FIELDS or str(key).startswith("ui_"):
            metadata.pop(key, None)
    metadata[RECIPE_TEMPLATE_PROVENANCE_METADATA_KEY] = provenance.to_dict()
    raw_recipe["metadata"] = metadata


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecipeTemplateInstantiationError(
            "recipe template overrides must contain finite JSON values: %s" % exc
        ) from exc


def _canonical_digest(value: Any) -> str:
    return _sha256_digest(_canonical_json(value))


def _sha256_digest(content: bytes) -> str:
    return "sha256:%s" % hashlib.sha256(content).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecipeTemplateCatalogError("%s must be a mapping" % label)
    return value


def _reject_unknown_fields(
    data: Mapping[str, Any],
    allowed: set,
    label: str,
) -> None:
    unknown = sorted(set(data).difference(allowed))
    if unknown:
        raise RecipeTemplateCatalogError(
            "%s has unknown field(s): %s" % (label, ", ".join(unknown))
        )


def _nonempty_string(data: Mapping[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RecipeTemplateCatalogError(
            "%s requires a trimmed non-empty string `%s`" % (label, key)
        )
    return value


def _identifier(data: Mapping[str, Any], key: str, label: str) -> str:
    value = _nonempty_string(data, key, label)
    if not _IDENTIFIER_RE.fullmatch(value):
        raise RecipeTemplateCatalogError(
            "%s.%s must match %s" % (label, key, _IDENTIFIER_RE.pattern)
        )
    return value


def _safe_recipe_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecipeTemplateCatalogError("%s must be a trimmed non-empty string" % label)
    if "\\" in value:
        raise RecipeTemplateCatalogError("%s must use forward slashes" % label)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise RecipeTemplateCatalogError("%s must be a safe project-relative path" % label)
    if path.as_posix() != value or value.startswith("./"):
        raise RecipeTemplateCatalogError("%s must be a normalized project-relative path" % label)
    if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise RecipeTemplateCatalogError(
            "%s must reference a YAML or JSON recipe" % label
        )
    return value


def _safe_starter_resource(value: Any, label: str) -> str:
    resource = _safe_recipe_path(value, label)
    path = PurePosixPath(resource)
    if not path.parts or path.parts[0] != "recipe_starters":
        raise RecipeTemplateCatalogError(
            "%s must be within the `recipe_starters/` package directory" % label
        )
    return resource
