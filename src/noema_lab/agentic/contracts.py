"""Strict, reproducible contracts for slow agentic radio supervision.

The contract deliberately exposes a small action: select an existing allocator
and one declared average-power budget.  It does not allow an agent to generate
per-symbol or per-subcarrier power vectors.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.structured_input import (
    decode_strict_json_object,
    load_strict_yaml_or_json,
)


CONTRACT_KIND = "noema.agentic_supervisory_contract"
CONTRACT_SCHEMA_VERSION = 1
ACTION_TOOL = "configure_allocator"
SUPPORTED_POLICIES = (
    "fixed",
    "observed_csi_water_filling",
    "robust_csi_water_filling",
    "causal_ar_water_filling",
    "causal_ar_box_water_filling",
)
SUPPORTED_PROVIDER_KINDS = (
    "scripted",
    "openai_compatible",
    "ollama",
    "transformers",
)
SUPPORTED_OBSERVATION_FIELDS = (
    "observation_timestamp_utc",
    "episode_index",
    "decision_index",
    "available_after_run_id",
    "public_context.noise_variance",
    "public_context.feedback_delay_ofdm_symbols",
    "public_context.csi_history_length",
    "public_context.csi_estimation_snr_db",
    "public_context.mobility_kmh",
    "public_context.ofdm_fft_size",
    "public_context.allocation_ofdm_symbols",
    "recent.run_id",
    "recent.run_started_at_utc",
    "recent.run_completed_at_utc",
    "recent.action.policy",
    "recent.action.power_budget",
    "recent.predicted_bler",
    "recent.expected_goodput_bps_hz",
    "recent.average_power_budget",
    "recent.action_valid",
    "recent.fallback_used",
    "transmitter_csi.oldest_observation_ofdm_symbol_index",
    "transmitter_csi.newest_observation_ofdm_symbol_index",
    "transmitter_csi.newest_observation_nominal_time_s",
    "transmitter_csi.mean_observed_gain",
    "transmitter_csi.p10_observed_gain",
    "transmitter_csi.p50_observed_gain",
    "transmitter_csi.p90_observed_gain",
    "transmitter_csi.mean_history_delta_magnitude",
)
NUMERIC_RULE_OBSERVATION_FIELDS = tuple(
    field
    for field in SUPPORTED_OBSERVATION_FIELDS
    if field
    not in {
        "observation_timestamp_utc",
        "available_after_run_id",
        "recent.run_id",
        "recent.run_started_at_utc",
        "recent.run_completed_at_utc",
        "recent.action.policy",
        "recent.action_valid",
        "recent.fallback_used",
    }
)
SUPPORTED_FEEDBACK_METRICS = (
    "resource.finite_blocklength.predicted_bler",
    "resource.finite_blocklength.expected_goodput_bps_hz",
    "resource.finite_blocklength.p05_goodput_bps_hz",
    "resource.average_transmit_power_budget",
    "resource.power_constraint.max_abs_error",
    "resource.power_constraint.max_relative_error",
    "resource.power_constraint.max_negative_violation",
)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_MODEL_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_COMPARISON_OPERATORS = {"<", "<=", "==", ">=", ">"}
_FALLBACK_REASONS = {"timeout", "backend_error", "invalid_action"}


class AgenticContractError(ValueError):
    """Raised when an agentic contract or action is ambiguous or unsafe."""


@dataclass(frozen=True)
class BaseRecipeBinding:
    path: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ObjectiveConstraint:
    metric: str
    operator: str
    value: float

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "operator": self.operator, "value": self.value}


@dataclass(frozen=True)
class ObjectiveTieBreaker:
    metric: str
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "direction": self.direction}


@dataclass(frozen=True)
class ObjectiveContract:
    id: str
    primary_metric: str
    direction: str
    aggregation: str
    description: str
    constraints: tuple[ObjectiveConstraint, ...]
    tie_breakers: tuple[ObjectiveTieBreaker, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "aggregation": self.aggregation,
            "description": self.description,
            "constraints": [item.to_dict() for item in self.constraints],
            "tie_breakers": [item.to_dict() for item in self.tie_breakers],
        }


@dataclass(frozen=True)
class EpisodeSeed:
    episode: int
    seeds: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"episode": self.episode, "seeds": dict(self.seeds)}


@dataclass(frozen=True)
class EpisodeContract:
    count: int
    decisions_per_episode: int
    reset_between_episodes: bool
    seed_schedule: tuple[EpisodeSeed, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "decisions_per_episode": self.decisions_per_episode,
            "reset_between_episodes": self.reset_between_episodes,
            "seed_schedule": [item.to_dict() for item in self.seed_schedule],
        }


@dataclass(frozen=True)
class ObservationContract:
    timestamp_field: str
    history_decisions: int
    allowlist: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_field": self.timestamp_field,
            "history_decisions": self.history_decisions,
            "allowlist": list(self.allowlist),
        }


@dataclass(frozen=True)
class ActionEnvelope:
    tool: str
    policy: str
    power_budget: float

    @property
    def arguments(self) -> dict[str, Any]:
        return {"policy": self.policy, "power_budget": self.power_budget}

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "arguments": self.arguments}


@dataclass(frozen=True)
class ActionContract:
    tool: str
    policies: tuple[str, ...]
    allowed_power_budgets: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "policies": list(self.policies),
            "allowed_power_budgets": list(self.allowed_power_budgets),
        }

    def tool_schema(self) -> dict[str, Any]:
        return {
            "name": self.tool,
            "description": "Select one existing allocator and average power budget for the next run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy": {"type": "string", "enum": list(self.policies)},
                    "power_budget": {
                        "type": "number",
                        "enum": list(self.allowed_power_budgets),
                    },
                },
                "required": ["policy", "power_budget"],
                "additionalProperties": False,
            },
        }


@dataclass(frozen=True)
class PromptContract:
    system: str
    user: str

    def to_dict(self) -> dict[str, Any]:
        return {"system": self.system, "user": self.user}


@dataclass(frozen=True)
class RuleBasedComparator:
    observation: str
    operator: str
    threshold: float
    if_true: ActionEnvelope
    if_false: ActionEnvelope

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "operator": self.operator,
            "threshold": self.threshold,
            "if_true": self.if_true.to_dict(),
            "if_false": self.if_false.to_dict(),
        }


@dataclass(frozen=True)
class ComparatorContract:
    policy_actions: tuple[ActionEnvelope, ...]
    rule_based: RuleBasedComparator

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_actions": [item.to_dict() for item in self.policy_actions],
            "rule_based": self.rule_based.to_dict(),
        }


@dataclass(frozen=True)
class LimitContract:
    decision_timeout_seconds: float
    max_tool_calls_per_decision: int
    max_output_tokens: int
    max_invalid_actions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_timeout_seconds": self.decision_timeout_seconds,
            "max_tool_calls_per_decision": self.max_tool_calls_per_decision,
            "max_output_tokens": self.max_output_tokens,
            "max_invalid_actions": self.max_invalid_actions,
        }


@dataclass(frozen=True)
class FallbackContract:
    on: tuple[str, ...]
    action: ActionEnvelope

    def to_dict(self) -> dict[str, Any]:
        return {"on": list(self.on), "action": self.action.to_dict()}


@dataclass(frozen=True)
class ProviderConfig:
    kind: str
    model: str
    temperature: float = 0.0
    seed: Optional[int] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    model_revision: Optional[str] = None
    device: str = "cpu"
    local_files_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        if self.base_url is not None:
            payload["base_url"] = self.base_url
        if self.api_key_env is not None:
            payload["api_key_env"] = self.api_key_env
        if self.model_revision is not None:
            payload["model_revision"] = self.model_revision
        if self.kind == "transformers":
            payload["device"] = self.device
            payload["local_files_only"] = self.local_files_only
        return payload


@dataclass(frozen=True)
class AgenticContract:
    schema_version: int
    kind: str
    id: str
    base_recipe: BaseRecipeBinding
    objective: ObjectiveContract
    episodes: EpisodeContract
    observations: ObservationContract
    actions: ActionContract
    prompts: PromptContract
    comparators: ComparatorContract
    limits: LimitContract
    fallback: FallbackContract
    provider: ProviderConfig
    source_path: Optional[Path] = field(default=None, compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "id": self.id,
            "base_recipe": self.base_recipe.to_dict(),
            "objective": self.objective.to_dict(),
            "episodes": self.episodes.to_dict(),
            "observations": self.observations.to_dict(),
            "actions": self.actions.to_dict(),
            "prompts": self.prompts.to_dict(),
            "comparators": self.comparators.to_dict(),
            "limits": self.limits.to_dict(),
            "fallback": self.fallback.to_dict(),
            "provider": self.provider.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


def load_agentic_contract(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    verify_base_recipe: bool = True,
) -> AgenticContract:
    """Load a duplicate-key-safe YAML/JSON contract and verify its recipe hash."""

    source = Path(path)
    payload = load_strict_yaml_or_json(source)
    return agentic_contract_from_mapping(
        payload,
        source_path=source,
        project_root=project_root,
        verify_base_recipe=verify_base_recipe,
    )


def agentic_contract_from_mapping(
    payload: Any,
    *,
    source_path: str | Path | None = None,
    project_root: str | Path | None = None,
    verify_base_recipe: bool = False,
) -> AgenticContract:
    root = _mapping(payload, "$contract")
    _keys(
        root,
        required={
            "schema_version", "kind", "id", "base_recipe", "objective",
            "episodes", "observations", "actions", "prompts", "comparators",
            "limits", "fallback", "provider",
        },
        path="$contract",
    )
    version = _integer(root["schema_version"], "$contract.schema_version", minimum=1)
    if version != CONTRACT_SCHEMA_VERSION:
        raise AgenticContractError("$contract.schema_version must be 1")
    kind = _text(root["kind"], "$contract.kind")
    if kind != CONTRACT_KIND:
        raise AgenticContractError("$contract.kind must be %r" % CONTRACT_KIND)
    contract_id = _safe_text(root["id"], "$contract.id")

    base_recipe = _parse_base_recipe(root["base_recipe"])
    actions = _parse_actions(root["actions"])
    observations = _parse_observations(root["observations"])
    comparators = _parse_comparators(root["comparators"], actions, observations)
    fallback = _parse_fallback(root["fallback"], actions)
    episodes = _parse_episodes(root["episodes"])
    if observations.history_decisions > episodes.decisions_per_episode:
        raise AgenticContractError(
            "observations.history_decisions cannot exceed decisions_per_episode"
        )
    contract = AgenticContract(
        schema_version=version,
        kind=kind,
        id=contract_id,
        base_recipe=base_recipe,
        objective=_parse_objective(root["objective"]),
        episodes=episodes,
        observations=observations,
        actions=actions,
        prompts=_parse_prompts(root["prompts"]),
        comparators=comparators,
        limits=_parse_limits(root["limits"]),
        fallback=fallback,
        provider=provider_config_from_mapping(root["provider"]),
        source_path=Path(source_path) if source_path is not None else None,
    )
    if verify_base_recipe:
        if source_path is None:
            raise AgenticContractError("source_path is required to verify base_recipe")
        resolved_recipe = _verify_recipe_binding(
            base_recipe,
            Path(source_path).parent,
            project_root=Path(project_root) if project_root is not None else None,
        )
        _verify_base_recipe_capabilities(resolved_recipe)
    return contract


def parse_action_envelope(
    payload: str | Mapping[str, Any],
    action_contract: ActionContract,
) -> ActionEnvelope:
    """Parse and validate exactly one JSON tool/action envelope."""

    if isinstance(payload, str):
        try:
            value: Any = decode_strict_json_object(payload, label="agent action")
        except ValueError as exc:
            raise AgenticContractError("agent action is not strict JSON: %s" % exc) from exc
    else:
        value = payload
    item = _mapping(value, "$action")
    _keys(item, required={"tool", "arguments"}, path="$action")
    tool = _text(item["tool"], "$action.tool")
    if tool != action_contract.tool:
        raise AgenticContractError(
            "$action.tool must be %r; found %r" % (action_contract.tool, tool)
        )
    arguments = _mapping(item["arguments"], "$action.arguments")
    _keys(arguments, required={"policy", "power_budget"}, path="$action.arguments")
    policy = _text(arguments["policy"], "$action.arguments.policy")
    if policy not in action_contract.policies:
        raise AgenticContractError("$action.arguments.policy is not allowed: %r" % policy)
    budget = _number(arguments["power_budget"], "$action.arguments.power_budget", positive=True)
    selected = next(
        (candidate for candidate in action_contract.allowed_power_budgets if budget == candidate),
        None,
    )
    if selected is None:
        raise AgenticContractError(
            "$action.arguments.power_budget is not in allowed_power_budgets"
        )
    return ActionEnvelope(tool=tool, policy=policy, power_budget=selected)


def action_contract_from_mapping(payload: Any) -> ActionContract:
    """Normalize the public action section for backend-only callers."""

    return _parse_actions(payload)


def provider_config_from_mapping(payload: Any) -> ProviderConfig:
    item = _mapping(payload, "$contract.provider")
    _keys(
        item,
        required={"kind", "model"},
        optional={
            "temperature", "seed", "base_url", "api_key_env",
            "model_revision", "device", "local_files_only",
        },
        path="$contract.provider",
    )
    kind = _text(item["kind"], "$contract.provider.kind")
    if kind not in SUPPORTED_PROVIDER_KINDS:
        raise AgenticContractError("unsupported provider kind %r" % kind)
    model = _text(item["model"], "$contract.provider.model")
    temperature = _number(item.get("temperature", 0.0), "$contract.provider.temperature")
    if temperature < 0.0 or temperature > 2.0:
        raise AgenticContractError("$contract.provider.temperature must be between 0 and 2")
    seed_value = item.get("seed")
    seed = None if seed_value is None else _integer(seed_value, "$contract.provider.seed", minimum=0)
    base_url = item.get("base_url")
    if base_url is not None:
        base_url = _validate_base_url(_text(base_url, "$contract.provider.base_url"))
    api_key_env = item.get("api_key_env")
    if api_key_env is not None:
        api_key_env = _text(api_key_env, "$contract.provider.api_key_env")
        if not _ENV_NAME.fullmatch(api_key_env):
            raise AgenticContractError("$contract.provider.api_key_env must be an environment variable name")
        if not api_key_env.startswith("NOEMA_AGENT_"):
            raise AgenticContractError(
                "$contract.provider.api_key_env must use the NOEMA_AGENT_ namespace"
            )
    revision = item.get("model_revision")
    if revision is not None:
        revision = _text(revision, "$contract.provider.model_revision")
    device = _text(item.get("device", "cpu"), "$contract.provider.device")
    if device not in {"cpu", "cuda", "mps", "auto"}:
        raise AgenticContractError("$contract.provider.device is unsupported")
    local_files_only = _boolean(item.get("local_files_only", False), "$contract.provider.local_files_only")

    if kind in {"openai_compatible", "ollama"} and base_url is None:
        raise AgenticContractError("$contract.provider.base_url is required for %s" % kind)
    if kind != "openai_compatible" and api_key_env is not None:
        raise AgenticContractError("api_key_env is only allowed for openai_compatible providers")
    if kind == "transformers" and revision is None:
        raise AgenticContractError("model_revision is required for transformers providers")
    if kind == "transformers" and not _IMMUTABLE_MODEL_REVISION.fullmatch(
        str(revision)
    ):
        raise AgenticContractError(
            "transformers model_revision must be a 40- or 64-character lowercase commit digest"
        )
    if kind != "transformers" and (revision is not None or "device" in item or "local_files_only" in item):
        raise AgenticContractError("model_revision/device/local_files_only are only allowed for transformers")
    if kind not in {"openai_compatible", "ollama"} and base_url is not None:
        raise AgenticContractError("base_url is only allowed for HTTP providers")
    return ProviderConfig(
        kind=kind,
        model=model,
        temperature=temperature,
        seed=seed,
        base_url=base_url,
        api_key_env=api_key_env,
        model_revision=revision,
        device=device,
        local_files_only=local_files_only,
    )


def _parse_base_recipe(payload: Any) -> BaseRecipeBinding:
    item = _mapping(payload, "$contract.base_recipe")
    _keys(item, required={"path", "sha256"}, path="$contract.base_recipe")
    path = _text(item["path"], "$contract.base_recipe.path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or "\\" in path or path != pure.as_posix() or any(part in {"", "."} for part in pure.parts):
        raise AgenticContractError("$contract.base_recipe.path must be a normalized relative POSIX path")
    digest = _text(item["sha256"], "$contract.base_recipe.sha256")
    if not _SHA256.fullmatch(digest):
        raise AgenticContractError("$contract.base_recipe.sha256 must be 64 lowercase hexadecimal characters")
    return BaseRecipeBinding(path=path, sha256=digest)


def _parse_objective(payload: Any) -> ObjectiveContract:
    item = _mapping(payload, "$contract.objective")
    _keys(
        item,
        required={
            "id",
            "primary_metric",
            "direction",
            "aggregation",
            "constraints",
            "tie_breakers",
        },
        optional={"description"},
        path="$contract.objective",
    )
    direction = _text(item["direction"], "$contract.objective.direction")
    if direction not in {"maximize", "minimize"}:
        raise AgenticContractError("$contract.objective.direction must be maximize or minimize")
    aggregation = _text(
        item["aggregation"], "$contract.objective.aggregation"
    )
    if aggregation != "mean":
        raise AgenticContractError(
            "$contract.objective.aggregation must be mean in schema v1"
        )
    constraints_raw = _sequence(item["constraints"], "$contract.objective.constraints")
    if not constraints_raw:
        raise AgenticContractError("$contract.objective.constraints must not be empty")
    constraints: list[ObjectiveConstraint] = []
    for index, raw in enumerate(constraints_raw):
        path = "$contract.objective.constraints[%d]" % index
        constraint = _mapping(raw, path)
        _keys(constraint, required={"metric", "operator", "value"}, path=path)
        operator = _text(constraint["operator"], path + ".operator")
        if operator not in _COMPARISON_OPERATORS:
            raise AgenticContractError(path + ".operator is unsupported")
        constraints.append(
            ObjectiveConstraint(
                metric=_safe_text(constraint["metric"], path + ".metric"),
                operator=operator,
                value=_number(constraint["value"], path + ".value"),
            )
        )
    primary_metric = _safe_text(
        item["primary_metric"], "$contract.objective.primary_metric"
    )
    if primary_metric not in SUPPORTED_FEEDBACK_METRICS:
        raise AgenticContractError(
            "$contract.objective.primary_metric is not produced by the v1 harness"
        )
    unsupported_metrics = sorted(
        {constraint.metric for constraint in constraints}
        - set(SUPPORTED_FEEDBACK_METRICS)
    )
    if unsupported_metrics:
        raise AgenticContractError(
            "$contract.objective.constraints contain unsupported metrics: %s"
            % ", ".join(unsupported_metrics)
        )
    tie_breakers_raw = _sequence(
        item["tie_breakers"], "$contract.objective.tie_breakers"
    )
    tie_breakers: list[ObjectiveTieBreaker] = []
    seen_tie_metrics: set[str] = set()
    for index, raw in enumerate(tie_breakers_raw):
        path = "$contract.objective.tie_breakers[%d]" % index
        tie = _mapping(raw, path)
        _keys(tie, required={"metric", "direction"}, path=path)
        metric = _safe_text(tie["metric"], path + ".metric")
        tie_direction = _text(tie["direction"], path + ".direction")
        if metric not in SUPPORTED_FEEDBACK_METRICS:
            raise AgenticContractError(path + ".metric is not produced by the v1 harness")
        if metric == primary_metric or metric in seen_tie_metrics:
            raise AgenticContractError(
                "$contract.objective.tie_breakers metrics must be unique"
            )
        if tie_direction not in {"maximize", "minimize"}:
            raise AgenticContractError(path + ".direction must be maximize or minimize")
        seen_tie_metrics.add(metric)
        tie_breakers.append(ObjectiveTieBreaker(metric, tie_direction))
    return ObjectiveContract(
        id=_safe_text(item["id"], "$contract.objective.id"),
        primary_metric=primary_metric,
        direction=direction,
        aggregation=aggregation,
        description=_text(item.get("description", ""), "$contract.objective.description", allow_empty=True),
        constraints=tuple(constraints),
        tie_breakers=tuple(tie_breakers),
    )


def _parse_episodes(payload: Any) -> EpisodeContract:
    item = _mapping(payload, "$contract.episodes")
    _keys(item, required={"count", "decisions_per_episode", "reset_between_episodes", "seed_schedule"}, path="$contract.episodes")
    count = _integer(item["count"], "$contract.episodes.count", minimum=1)
    decisions = _integer(item["decisions_per_episode"], "$contract.episodes.decisions_per_episode", minimum=1)
    reset = _boolean(item["reset_between_episodes"], "$contract.episodes.reset_between_episodes")
    if not reset:
        raise AgenticContractError("$contract.episodes.reset_between_episodes must be true")
    raw_schedule = _sequence(item["seed_schedule"], "$contract.episodes.seed_schedule")
    if len(raw_schedule) != count:
        raise AgenticContractError("seed_schedule must have exactly one entry per episode")
    schedule: list[EpisodeSeed] = []
    seen: set[int] = set()
    for index, raw in enumerate(raw_schedule):
        path = "$contract.episodes.seed_schedule[%d]" % index
        entry = _mapping(raw, path)
        _keys(entry, required={"episode", "seeds"}, path=path)
        episode = _integer(entry["episode"], path + ".episode", minimum=0)
        seeds_raw = _mapping(entry["seeds"], path + ".seeds")
        if set(seeds_raw) != {"channel", "traffic"}:
            raise AgenticContractError(
                path + ".seeds must contain exactly channel and traffic"
            )
        seeds: list[tuple[str, int]] = []
        for name, value in seeds_raw.items():
            if not _SAFE_NAME.fullmatch(name):
                raise AgenticContractError(path + ".seeds contains an unsafe seed name")
            seeds.append((name, _integer(value, path + ".seeds." + name, minimum=0)))
        if episode in seen:
            raise AgenticContractError("duplicate episode in seed_schedule")
        seen.add(episode)
        schedule.append(EpisodeSeed(episode=episode, seeds=tuple(sorted(seeds))))
    if seen != set(range(count)):
        raise AgenticContractError("seed_schedule episode values must be 0 through count-1")
    return EpisodeContract(count, decisions, reset, tuple(sorted(schedule, key=lambda row: row.episode)))


def _parse_observations(payload: Any) -> ObservationContract:
    item = _mapping(payload, "$contract.observations")
    _keys(item, required={"timestamp_field", "history_decisions", "allowlist"}, path="$contract.observations")
    timestamp = _safe_text(item["timestamp_field"], "$contract.observations.timestamp_field")
    history = _integer(item["history_decisions"], "$contract.observations.history_decisions", minimum=0)
    allowlist = tuple(_unique_text_sequence(item["allowlist"], "$contract.observations.allowlist"))
    if not allowlist:
        raise AgenticContractError("$contract.observations.allowlist must not be empty")
    if timestamp not in allowlist:
        raise AgenticContractError("timestamp_field must appear in observations.allowlist")
    unsupported = sorted(set(allowlist) - set(SUPPORTED_OBSERVATION_FIELDS))
    if unsupported:
        raise AgenticContractError(
            "observations.allowlist contains unsupported or hidden fields: %s"
            % ", ".join(unsupported)
        )
    return ObservationContract(timestamp, history, allowlist)


def _parse_actions(payload: Any) -> ActionContract:
    item = _mapping(payload, "$contract.actions")
    _keys(item, required={"tool", "policies", "allowed_power_budgets"}, path="$contract.actions")
    tool = _text(item["tool"], "$contract.actions.tool")
    if tool != ACTION_TOOL:
        raise AgenticContractError("$contract.actions.tool must be %r" % ACTION_TOOL)
    policies = tuple(_unique_text_sequence(item["policies"], "$contract.actions.policies"))
    if not policies:
        raise AgenticContractError("$contract.actions.policies must not be empty")
    unsupported = sorted(set(policies) - set(SUPPORTED_POLICIES))
    if unsupported:
        raise AgenticContractError("unsupported allocation policies: %s" % ", ".join(unsupported))
    raw_budgets = _sequence(item["allowed_power_budgets"], "$contract.actions.allowed_power_budgets")
    budgets = tuple(_number(value, "$contract.actions.allowed_power_budgets[%d]" % index, positive=True) for index, value in enumerate(raw_budgets))
    if not budgets or len(set(budgets)) != len(budgets):
        raise AgenticContractError("allowed_power_budgets must be nonempty and unique")
    return ActionContract(tool, policies, budgets)


def _parse_prompts(payload: Any) -> PromptContract:
    item = _mapping(payload, "$contract.prompts")
    _keys(item, required={"system", "user"}, path="$contract.prompts")
    return PromptContract(
        system=_text(item["system"], "$contract.prompts.system"),
        user=_text(item["user"], "$contract.prompts.user"),
    )


def _parse_comparators(payload: Any, actions: ActionContract, observations: ObservationContract) -> ComparatorContract:
    item = _mapping(payload, "$contract.comparators")
    _keys(item, required={"policy_actions", "rule_based"}, path="$contract.comparators")
    raw_actions = _sequence(item["policy_actions"], "$contract.comparators.policy_actions")
    if not raw_actions:
        raise AgenticContractError("comparators.policy_actions must not be empty")
    policy_actions = tuple(parse_action_envelope(value, actions) for value in raw_actions)
    serialized = [canonical_json_sha256(action.to_dict()) for action in policy_actions]
    if len(set(serialized)) != len(serialized):
        raise AgenticContractError("comparators.policy_actions contains duplicates")
    compared_policies = {action.policy for action in policy_actions}
    if compared_policies != set(actions.policies):
        raise AgenticContractError(
            "comparators.policy_actions must cover every allowed policy"
        )
    raw_rule = _mapping(item["rule_based"], "$contract.comparators.rule_based")
    _keys(raw_rule, required={"observation", "operator", "threshold", "if_true", "if_false"}, path="$contract.comparators.rule_based")
    observation = _safe_text(raw_rule["observation"], "$contract.comparators.rule_based.observation")
    if observation not in observations.allowlist:
        raise AgenticContractError("rule_based observation must be in observations.allowlist")
    if observation not in NUMERIC_RULE_OBSERVATION_FIELDS:
        raise AgenticContractError("rule_based observation must be a numeric observation field")
    operator = _text(raw_rule["operator"], "$contract.comparators.rule_based.operator")
    if operator not in _COMPARISON_OPERATORS:
        raise AgenticContractError("rule_based operator is unsupported")
    rule = RuleBasedComparator(
        observation=observation,
        operator=operator,
        threshold=_number(raw_rule["threshold"], "$contract.comparators.rule_based.threshold"),
        if_true=parse_action_envelope(raw_rule["if_true"], actions),
        if_false=parse_action_envelope(raw_rule["if_false"], actions),
    )
    return ComparatorContract(policy_actions, rule)


def _parse_limits(payload: Any) -> LimitContract:
    item = _mapping(payload, "$contract.limits")
    _keys(item, required={"decision_timeout_seconds", "max_tool_calls_per_decision", "max_output_tokens", "max_invalid_actions"}, path="$contract.limits")
    tool_calls = _integer(item["max_tool_calls_per_decision"], "$contract.limits.max_tool_calls_per_decision", minimum=1)
    if tool_calls != 1:
        raise AgenticContractError("max_tool_calls_per_decision must be 1 in schema v1")
    invalid_actions = _integer(item["max_invalid_actions"], "$contract.limits.max_invalid_actions", minimum=0)
    if invalid_actions != 0:
        raise AgenticContractError(
            "max_invalid_actions must be 0 in schema v1; invalid actions use fallback"
        )
    return LimitContract(
        decision_timeout_seconds=_number(item["decision_timeout_seconds"], "$contract.limits.decision_timeout_seconds", positive=True),
        max_tool_calls_per_decision=tool_calls,
        max_output_tokens=_integer(item["max_output_tokens"], "$contract.limits.max_output_tokens", minimum=1),
        max_invalid_actions=invalid_actions,
    )


def _parse_fallback(payload: Any, actions: ActionContract) -> FallbackContract:
    item = _mapping(payload, "$contract.fallback")
    _keys(item, required={"on", "action"}, path="$contract.fallback")
    reasons = tuple(_unique_text_sequence(item["on"], "$contract.fallback.on"))
    if set(reasons) != _FALLBACK_REASONS:
        raise AgenticContractError(
            "fallback.on must cover timeout, backend_error, and invalid_action"
        )
    return FallbackContract(reasons, parse_action_envelope(item["action"], actions))


def _verify_recipe_binding(
    binding: BaseRecipeBinding,
    base_directory: Path,
    *,
    project_root: Path | None,
) -> Path:
    try:
        candidate = (base_directory.resolve(strict=True) / binding.path).resolve(strict=True)
        if project_root is not None:
            candidate.relative_to(project_root.resolve(strict=True))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        detail = " inside project_root" if project_root is not None else ""
        raise AgenticContractError("base_recipe path does not resolve%s" % detail) from exc
    if not candidate.is_file():
        raise AgenticContractError("base_recipe path must identify a regular file")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if digest != binding.sha256:
        raise AgenticContractError(
            "base_recipe SHA-256 mismatch: expected %s, found %s" % (binding.sha256, digest)
        )
    return candidate


def _verify_base_recipe_capabilities(path: Path) -> None:
    from noema_lab.core.recipes import RecipeValidationError, load_recipe

    try:
        recipe = load_recipe(path)
    except RecipeValidationError as exc:
        raise AgenticContractError("base_recipe is not a valid Noema recipe") from exc
    by_id = {step.id: step for step in recipe.steps}
    required = {
        "channel_state": "wireless.ofdm_channel_state",
        "csi_observation": "wireless.ofdm_delayed_csi",
        "tx_power": "model.causal_csi_power_allocator",
        "wireless_channel": "wireless.channel",
        "allocation_evaluation": "metrics.ofdm_finite_blocklength_allocation",
    }
    for step_id, operation in required.items():
        step = by_id.get(step_id)
        if step is None or step.op != operation:
            raise AgenticContractError(
                "base_recipe must contain %s using %s" % (step_id, operation)
            )
    if by_id["tx_power"].inputs.get("channel_state") != "csi_observation.transmitter_csi":
        raise AgenticContractError(
            "base_recipe allocator must consume delayed transmitter_csi"
        )
    if by_id["wireless_channel"].inputs.get("channel_state") != "csi_observation.actual_state":
        raise AgenticContractError(
            "base_recipe channel must consume the separate actual_state"
        )
    evaluation_inputs = by_id["allocation_evaluation"].inputs
    if (
        evaluation_inputs.get("actual_state") != "csi_observation.actual_state"
        or evaluation_inputs.get("transmitter_csi")
        != "csi_observation.transmitter_csi"
    ):
        raise AgenticContractError(
            "base_recipe allocation evaluation must retain actual/transmitter CSI separation"
        )


def _validate_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise AgenticContractError("provider base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise AgenticContractError("provider base_url must not contain credentials, a query, or a fragment")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise AgenticContractError("unencrypted provider base_url is allowed only for loopback hosts")
    return value.rstrip("/")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgenticContractError(path + " must be an object")
    if not all(isinstance(key, str) for key in value):
        raise AgenticContractError(path + " object keys must be strings")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise AgenticContractError(path + " must be an array")
    return value


def _keys(value: Mapping[str, Any], *, required: set[str], path: str, optional: set[str] | None = None) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise AgenticContractError("%s is missing required keys: %s" % (path, ", ".join(missing)))
    if unknown:
        raise AgenticContractError("%s contains unknown keys: %s" % (path, ", ".join(unknown)))


def _text(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise AgenticContractError(path + " must be a string")
    result = value.strip()
    if not allow_empty and not result:
        raise AgenticContractError(path + " must not be empty")
    return result


def _safe_text(value: Any, path: str) -> str:
    result = _text(value, path)
    if not _SAFE_NAME.fullmatch(result):
        raise AgenticContractError(path + " contains unsupported characters")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise AgenticContractError(path + " must be a boolean")
    return value


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgenticContractError(path + " must be an integer")
    if value < minimum:
        raise AgenticContractError("%s must be at least %d" % (path, minimum))
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgenticContractError(path + " must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise AgenticContractError(path + " must be finite")
    if positive and result <= 0.0:
        raise AgenticContractError(path + " must be greater than zero")
    return result


def _unique_text_sequence(value: Any, path: str) -> list[str]:
    result = [_safe_text(item, "%s[%d]" % (path, index)) for index, item in enumerate(_sequence(value, path))]
    if len(set(result)) != len(result):
        raise AgenticContractError(path + " must not contain duplicates")
    return result
