"""Pluggable decision backends for agentic supervisory experiments.

Only a normalized JSON tool envelope crosses the backend boundary.  API keys
are looked up at request time from the declared environment-variable name and
are never included in response/evidence objects.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, runtime_checkable

from noema_lab.agentic.contracts import (
    ActionContract,
    ActionEnvelope,
    AgenticContractError,
    ProviderConfig,
    RuleBasedComparator,
    action_contract_from_mapping,
    parse_action_envelope,
    provider_config_from_mapping,
)
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.structured_input import decode_strict_json_object


_MAX_HTTP_RESPONSE_BYTES = 2 * 1024 * 1024


class DecisionBackendError(RuntimeError):
    """Raised when a provider cannot return one valid bounded action."""

    failure_class = "backend_error"


class InvalidDecisionActionError(DecisionBackendError):
    """Raised when a provider responded but its action violates the contract."""

    failure_class = "invalid_action"


@dataclass(frozen=True)
class DecisionRequest:
    decision_id: str
    observation: Mapping[str, Any]
    action_contract: ActionContract | Mapping[str, Any]
    system_prompt: str
    user_prompt: str = ""
    objective: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    max_output_tokens: int = 160
    max_tool_calls: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.decision_id, str) or not self.decision_id.strip():
            raise ValueError("decision_id must be a nonempty string")
        if not isinstance(self.observation, Mapping):
            raise TypeError("observation must be an object")
        if isinstance(self.action_contract, Mapping):
            object.__setattr__(
                self,
                "action_contract",
                action_contract_from_mapping(self.action_contract),
            )
        if not isinstance(self.action_contract, ActionContract):
            raise TypeError("action_contract must be an ActionContract or mapping")
        if not isinstance(self.objective, Mapping):
            raise TypeError("objective must be an object")
        # Fail before a provider call if the observation cannot be represented
        # unambiguously in the evidence record.
        canonical_json_sha256(dict(self.observation))
        canonical_json_sha256(dict(self.objective))
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("system_prompt must be nonempty")
        if not isinstance(self.user_prompt, str):
            raise TypeError("user_prompt must be a string")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)):
            raise TypeError("timeout_seconds must be numeric")
        if not math.isfinite(float(self.timeout_seconds)) or float(self.timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(self.max_output_tokens, bool) or not isinstance(self.max_output_tokens, int) or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        if self.max_tool_calls != 1:
            raise ValueError("schema v1 requires exactly one tool call")

    def rendered_user_prompt(self) -> str:
        instruction = self.user_prompt.strip() or "Select one valid action for the next run."
        objective_json = json.dumps(
            dict(self.objective),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        actions_json = json.dumps(
            self.action_contract.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        observation_json = json.dumps(
            dict(self.observation),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        replacements = {
            "{{ objective_json }}": objective_json,
            "{{ allowed_actions_json }}": actions_json,
            "{{ allowed_power_budgets_json }}": json.dumps(
                list(self.action_contract.allowed_power_budgets),
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
            "{{ observation_json }}": observation_json,
        }
        used_placeholder = False
        for marker, replacement in replacements.items():
            if marker in instruction:
                used_placeholder = True
                instruction = instruction.replace(marker, replacement)
        if used_placeholder:
            return instruction
        payload = {
            "decision_id": self.decision_id,
            "objective": dict(self.objective),
            "allowed_actions": self.action_contract.to_dict(),
            "observation": dict(self.observation),
        }
        return "%s\n\nRuntime input (strict JSON):\n%s" % (
            instruction,
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
        )


@dataclass(frozen=True)
class DecisionResponse:
    action: ActionEnvelope
    raw_response: str
    provider: Mapping[str, Any]
    usage: Mapping[str, int] = field(default_factory=dict)
    tool_call_count: int = 1
    latency_seconds: float = 0.0
    finish_reason: str = "stop"

    def __post_init__(self) -> None:
        # The raw response retained by the harness is always the normalized
        # action envelope, never an authorization header or provider wrapper.
        parsed = decode_strict_json_object(self.raw_response, label="normalized backend response")
        if parsed != self.action.to_dict():
            raise ValueError("raw_response must equal the normalized action envelope")
        canonical_json_sha256(dict(self.provider))
        canonical_json_sha256(dict(self.usage))
        if self.tool_call_count != 1:
            raise ValueError("a successful response must contain exactly one tool call")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("latency_seconds must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "raw_response": self.raw_response,
            "provider": dict(self.provider),
            "usage": dict(self.usage),
            "tool_call_count": self.tool_call_count,
            "latency_seconds": self.latency_seconds,
            "finish_reason": self.finish_reason,
        }


@runtime_checkable
class DecisionBackend(Protocol):
    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Return one valid action or raise ``DecisionBackendError``."""


class StaticBackend:
    """Comparator that always returns the same declared action."""

    def __init__(self, action: ActionEnvelope | Mapping[str, Any]) -> None:
        self._action = action

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        started = time.monotonic()
        action = _coerce_action(self._action, request.action_contract)
        return _response(action, "static", "static-v1", started)


class RuleBasedBackend:
    """Deterministic comparator over one allowlisted numeric observation.

    Missing history (for example the first decision after reset) takes
    ``if_true`` by default, which is intended to be the conservative branch.
    The behavior is explicit in ``missing_uses_true`` and recorded as the
    backend model identity.
    """

    def __init__(
        self,
        rule: RuleBasedComparator,
        *,
        missing_uses_true: bool = True,
    ) -> None:
        self.rule = rule
        self.missing_uses_true = bool(missing_uses_true)

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        started = time.monotonic()
        found, value = _lookup_dotted(request.observation, self.rule.observation)
        if not found or value is None:
            branch = self.missing_uses_true
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DecisionBackendError("rule observation must be numeric when present")
            number = float(value)
            if not math.isfinite(number):
                raise DecisionBackendError("rule observation must be finite")
            branch = _compare(number, self.rule.operator, self.rule.threshold)
        candidate = self.rule.if_true if branch else self.rule.if_false
        action = _coerce_action(candidate, request.action_contract)
        model = "missing-is-true-v1" if self.missing_uses_true else "missing-is-false-v1"
        return _response(action, "rule_based", model, started)


class ScriptedBackend:
    """Deterministic, network-free backend for smoke tests and tutorials."""

    def __init__(
        self,
        actions: Optional[Iterable[ActionEnvelope | Mapping[str, Any]]] = None,
        *,
        cycle: bool = True,
        model: str = "scripted-v1",
    ) -> None:
        self._actions = tuple(actions or ())
        self._cycle = bool(cycle)
        self._model = model
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        started = time.monotonic()
        if not self._actions:
            policy_count = len(request.action_contract.policies)
            budget_count = len(request.action_contract.allowed_power_budgets)
            span = policy_count * budget_count
            if span < 1:
                raise DecisionBackendError("action contract has no scripted choices")
            offset = self._index % span
            candidate = {
                "tool": request.action_contract.tool,
                "arguments": {
                    "policy": request.action_contract.policies[offset // budget_count],
                    "power_budget": request.action_contract.allowed_power_budgets[offset % budget_count],
                },
            }
            self._index += 1
            action = _coerce_action(candidate, request.action_contract)
            return _response(action, "scripted", self._model, started)
        if self._index >= len(self._actions):
            if not self._cycle:
                raise DecisionBackendError("scripted action sequence is exhausted")
            self._index = 0
        candidate = self._actions[self._index]
        self._index += 1
        action = _coerce_action(candidate, request.action_contract)
        return _response(action, "scripted", self._model, started)


class ReplayBackend:
    """Replay actions keyed by stable decision id, without a model call."""

    def __init__(self, actions: Mapping[str, ActionEnvelope | Mapping[str, Any]]) -> None:
        self._actions = dict(actions)

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        started = time.monotonic()
        if request.decision_id not in self._actions:
            raise DecisionBackendError("replay has no action for decision_id %r" % request.decision_id)
        action = _coerce_action(self._actions[request.decision_id], request.action_contract)
        return _response(action, "replay", "replay-v1", started)


class OpenAICompatibleBackend:
    """Chat-completions backend using only the Python standard library."""

    def __init__(
        self,
        provider: ProviderConfig,
        *,
        opener: Optional[Callable[..., Any]] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        if provider.kind != "openai_compatible":
            raise ValueError("OpenAICompatibleBackend requires kind=openai_compatible")
        if not provider.base_url:
            raise ValueError("OpenAI-compatible provider requires base_url")
        self.provider = provider
        self._opener = opener or urllib.request.urlopen
        # Keep only a lookup function; provider/evidence serialization contains
        # the environment-variable name, never its value.
        source = environ if environ is not None else os.environ
        self._getenv = source.get

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        payload: dict[str, Any] = {
            "model": self.provider.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.rendered_user_prompt()},
            ],
            "tools": [{"type": "function", "function": request.action_contract.tool_schema()}],
            "tool_choice": {
                "type": "function",
                "function": {"name": request.action_contract.tool},
            },
            "temperature": self.provider.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if self.provider.seed is not None:
            payload["seed"] = self.provider.seed
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.provider.api_key_env:
            secret = self._getenv(self.provider.api_key_env)
            if not secret:
                raise DecisionBackendError(
                    "required API key environment variable %s is not set"
                    % self.provider.api_key_env
                )
            headers["Authorization"] = "Bearer " + secret
        url = self.provider.base_url.rstrip("/") + "/chat/completions"
        body, started = _http_json(self._opener, url, payload, headers, request.timeout_seconds)
        action, tool_calls, finish_reason = _action_from_chat_message(
            _openai_message(body), request.action_contract
        )
        usage = _openai_usage(body)
        provider_evidence: dict[str, Any] = {"base_url": self.provider.base_url}
        for source, target in (
            ("model", "response_model"),
            ("system_fingerprint", "system_fingerprint"),
        ):
            value = body.get(source)
            if isinstance(value, str) and 0 < len(value) <= 512:
                provider_evidence[target] = value
        return _response(
            action,
            "openai_compatible",
            self.provider.model,
            started,
            usage=usage,
            tool_call_count=tool_calls,
            finish_reason=finish_reason,
            extra_provider=provider_evidence,
        )


class OllamaBackend:
    """Ollama ``/api/chat`` backend using only the standard library."""

    def __init__(
        self,
        provider: ProviderConfig,
        *,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        if provider.kind != "ollama":
            raise ValueError("OllamaBackend requires kind=ollama")
        if not provider.base_url:
            raise ValueError("Ollama provider requires base_url")
        self.provider = provider
        self._opener = opener or urllib.request.urlopen

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        options: dict[str, Any] = {
            "temperature": self.provider.temperature,
            "num_predict": request.max_output_tokens,
        }
        if self.provider.seed is not None:
            options["seed"] = self.provider.seed
        payload = {
            "model": self.provider.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.rendered_user_prompt()},
            ],
            "tools": [{"type": "function", "function": request.action_contract.tool_schema()}],
            "stream": False,
            "options": options,
        }
        url = self.provider.base_url.rstrip("/") + "/api/chat"
        body, started = _http_json(
            self._opener,
            url,
            payload,
            {"Content-Type": "application/json", "Accept": "application/json"},
            request.timeout_seconds,
        )
        message = body.get("message")
        if not isinstance(message, Mapping):
            raise InvalidDecisionActionError("Ollama response is missing message")
        action, tool_calls, _ = _action_from_chat_message(message, request.action_contract)
        usage = {}
        for source, target in (
            ("prompt_eval_count", "prompt_tokens"),
            ("eval_count", "completion_tokens"),
        ):
            value = body.get(source)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[target] = value
        if "prompt_tokens" in usage and "completion_tokens" in usage:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        provider_evidence: dict[str, Any] = {"base_url": self.provider.base_url}
        for source, target in (
            ("model", "response_model"),
            ("created_at", "response_created_at"),
        ):
            value = body.get(source)
            if isinstance(value, str) and 0 < len(value) <= 512:
                provider_evidence[target] = value
        return _response(
            action,
            "ollama",
            self.provider.model,
            started,
            usage=usage,
            tool_call_count=tool_calls,
            finish_reason=str(body.get("done_reason") or "stop"),
            extra_provider=provider_evidence,
        )


class TransformersBackend:
    """Optional local Hugging Face backend, imported lazily.

    A tiny callable can be injected for tests.  Production loading never enables
    remote model code and uses the contract's pinned revision.
    """

    def __init__(
        self,
        provider: ProviderConfig,
        *,
        generator: Optional[Callable[[str], str | Mapping[str, Any]]] = None,
    ) -> None:
        if provider.kind != "transformers":
            raise ValueError("TransformersBackend requires kind=transformers")
        self.provider = provider
        self._generator = generator
        self._tokenizer: Any = None
        self._model: Any = None

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        started = time.monotonic()
        prompt = _local_prompt(request)
        try:
            raw = self._generator(prompt) if self._generator is not None else self._generate(prompt, request)
            action = _parse_local_action(raw, request.action_contract)
        except AgenticContractError as exc:
            raise InvalidDecisionActionError(
                "local model returned an invalid action: %s" % exc
            ) from exc
        except DecisionBackendError:
            raise
        except Exception as exc:
            raise DecisionBackendError("local Transformers generation failed (%s)" % type(exc).__name__) from exc
        elapsed = time.monotonic() - started
        if elapsed > request.timeout_seconds:
            raise DecisionBackendError("local generation exceeded the decision timeout")
        if self._generator is not None:
            version = "injected"
        else:
            try:
                version = importlib_metadata.version("transformers")
            except importlib_metadata.PackageNotFoundError:
                version = "unavailable"
        return _response(
            action,
            "transformers",
            self.provider.model,
            started,
            extra_provider={
                "model_revision": self.provider.model_revision,
                "device": self.provider.device,
                "transformers_version": version,
            },
        )

    def _generate(self, prompt: str, request: DecisionRequest) -> str:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise DecisionBackendError(
                "Transformers backend requires the optional textgen dependencies"
            ) from exc
        if self._tokenizer is None or self._model is None:
            kwargs = {
                "revision": self.provider.model_revision,
                "local_files_only": self.provider.local_files_only,
                "trust_remote_code": False,
            }
            self._tokenizer = AutoTokenizer.from_pretrained(self.provider.model, **kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(self.provider.model, **kwargs)
            if self.provider.device != "auto":
                self._model.to(self.provider.device)
            self._model.eval()
        if self.provider.seed is not None:
            torch.manual_seed(self.provider.seed)
        rendered_prompt = prompt
        if getattr(self._tokenizer, "chat_template", None):
            try:
                rendered_prompt = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (TypeError, ValueError) as exc:
                raise DecisionBackendError(
                    "local tokenizer could not render its chat template"
                ) from exc
        encoded = self._tokenizer(rendered_prompt, return_tensors="pt")
        if self.provider.device != "auto":
            encoded = {key: value.to(self.provider.device) for key, value in encoded.items()}
        do_sample = self.provider.temperature > 0.0
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_output_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.provider.temperature
        with torch.no_grad():
            output = self._model.generate(**encoded, **generate_kwargs)
        input_length = int(encoded["input_ids"].shape[-1])
        return self._tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip()


def build_backend(
    provider: ProviderConfig | Mapping[str, Any],
    *,
    scripted_actions: Optional[Iterable[ActionEnvelope | Mapping[str, Any]]] = None,
    replay_actions: Optional[Mapping[str, ActionEnvelope | Mapping[str, Any]]] = None,
    opener: Optional[Callable[..., Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
    generator: Optional[Callable[[str], str | Mapping[str, Any]]] = None,
) -> DecisionBackend:
    """Construct a provider backend without importing optional dependencies."""

    if isinstance(provider, Mapping):
        provider = provider_config_from_mapping(provider)
    if not isinstance(provider, ProviderConfig):
        raise TypeError("provider must be a ProviderConfig or mapping")
    if provider.kind == "scripted":
        return ScriptedBackend(scripted_actions, model=provider.model)
    if provider.kind == "replay":
        if replay_actions is None:
            raise ValueError("replay_actions are required for a replay backend")
        return ReplayBackend(replay_actions)
    if provider.kind == "openai_compatible":
        return OpenAICompatibleBackend(provider, opener=opener, environ=environ)
    if provider.kind == "ollama":
        return OllamaBackend(provider, opener=opener)
    if provider.kind == "transformers":
        return TransformersBackend(provider, generator=generator)
    raise ValueError("unsupported provider kind %r" % provider.kind)


def _response(
    action: ActionEnvelope,
    kind: str,
    model: str,
    started: float,
    *,
    usage: Optional[Mapping[str, int]] = None,
    tool_call_count: int = 1,
    finish_reason: str = "stop",
    extra_provider: Optional[Mapping[str, Any]] = None,
) -> DecisionResponse:
    provider: dict[str, Any] = {"kind": kind, "model": model}
    provider.update(dict(extra_provider or {}))
    raw = json.dumps(action.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return DecisionResponse(
        action=action,
        raw_response=raw,
        provider=provider,
        usage=dict(usage or {}),
        tool_call_count=tool_call_count,
        latency_seconds=max(0.0, time.monotonic() - started),
        finish_reason=finish_reason,
    )


def _coerce_action(candidate: ActionEnvelope | Mapping[str, Any], contract: ActionContract) -> ActionEnvelope:
    try:
        return parse_action_envelope(
            candidate.to_dict() if isinstance(candidate, ActionEnvelope) else candidate,
            contract,
        )
    except AgenticContractError as exc:
        raise InvalidDecisionActionError("backend action is invalid: %s" % exc) from exc


def _lookup_dotted(root: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = root
    for part in path.split("."):
        # Observation paths describe the shape of each recent-history item;
        # supervisory rules evaluate the newest completed item.
        if isinstance(current, list):
            if not current:
                return False, None
            current = current[-1]
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _compare(value: float, operator: str, threshold: float) -> bool:
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "==":
        return value == threshold
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    raise DecisionBackendError("unsupported rule operator")


def _http_json(
    opener: Callable[..., Any],
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> tuple[Mapping[str, Any], float]:
    started = time.monotonic()
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
    try:
        response = opener(request, timeout=float(timeout))
        if hasattr(response, "__enter__"):
            with response as opened:
                raw = opened.read(_MAX_HTTP_RESPONSE_BYTES + 1)
        else:
            raw = response.read(_MAX_HTTP_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Provider bodies and headers can contain credentials; report only the
        # exception class, never the provider's raw error string.
        raise DecisionBackendError("provider request failed (%s)" % type(exc).__name__) from exc
    if len(raw) > _MAX_HTTP_RESPONSE_BYTES:
        raise DecisionBackendError("provider response exceeds the 2 MiB limit")
    try:
        text = raw.decode("utf-8", errors="strict")
        return decode_strict_json_object(text, label="provider response"), started
    except (UnicodeDecodeError, ValueError) as exc:
        raise DecisionBackendError("provider returned invalid JSON") from exc


def _openai_message(body: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise InvalidDecisionActionError(
            "OpenAI-compatible response must contain exactly one choice"
        )
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise InvalidDecisionActionError(
            "OpenAI-compatible response is missing choice.message"
        )
    result = dict(message)
    result["_finish_reason"] = str(choices[0].get("finish_reason") or "stop")
    return result


def _action_from_chat_message(
    message: Mapping[str, Any],
    contract: ActionContract,
) -> tuple[ActionEnvelope, int, str]:
    calls = message.get("tool_calls")
    if calls is not None:
        if not isinstance(calls, list) or len(calls) != 1:
            raise InvalidDecisionActionError(
                "provider must return exactly one tool call"
            )
        call = calls[0]
        if not isinstance(call, Mapping):
            raise InvalidDecisionActionError("provider tool call must be an object")
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise InvalidDecisionActionError(
                "provider tool call is missing function"
            )
        name = function.get("name")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = decode_strict_json_object(arguments, label="tool arguments")
            except ValueError as exc:
                raise InvalidDecisionActionError(
                    "provider returned invalid tool arguments"
                ) from exc
        try:
            action = parse_action_envelope({"tool": name, "arguments": arguments}, contract)
        except AgenticContractError as exc:
            raise InvalidDecisionActionError(
                "provider returned an invalid action: %s" % exc
            ) from exc
        return action, 1, str(message.get("_finish_reason") or "tool_call")
    content = message.get("content")
    if not isinstance(content, str):
        raise InvalidDecisionActionError(
            "provider response has neither a tool call nor JSON content"
        )
    try:
        action = parse_action_envelope(content, contract)
    except AgenticContractError as exc:
        raise InvalidDecisionActionError(
            "provider returned invalid JSON action: %s" % exc
        ) from exc
    return action, 1, str(message.get("_finish_reason") or "stop")


def _openai_usage(body: Mapping[str, Any]) -> dict[str, int]:
    raw = body.get("usage")
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[key] = value
    return result


def _local_prompt(request: DecisionRequest) -> str:
    schema = json.dumps(
        {"tool": request.action_contract.tool_schema()},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    example = json.dumps(
        {
            "tool": request.action_contract.tool,
            "arguments": {
                "policy": request.action_contract.policies[0],
                "power_budget": request.action_contract.allowed_power_budgets[0],
            },
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (
        "%s\n\n%s\n\nAllowed action schema (strict JSON):\n%s\n"
        "Return exactly one JSON object shaped like %s, substituting only allowed values. "
        "Use the key arguments, not parameters. Do not use a Markdown code fence or add text."
    ) % (
        request.system_prompt.strip(),
        request.rendered_user_prompt(),
        schema,
        example,
    )


def _parse_local_action(
    raw: str | Mapping[str, Any],
    contract: ActionContract,
) -> ActionEnvelope:
    """Normalize common local-model JSON wrappers, then enforce the contract."""

    value: Any = raw
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```json\n") and text.endswith("\n```"):
            text = text[len("```json\n") : -len("\n```")].strip()
        elif text.startswith("```\n") and text.endswith("\n```"):
            text = text[len("```\n") : -len("\n```")].strip()
        try:
            value = decode_strict_json_object(text, label="local model action")
        except ValueError as exc:
            raise AgenticContractError(
                "agent action is not strict JSON: %s" % exc
            ) from exc
    if not isinstance(value, Mapping):
        raise AgenticContractError("agent action must be a JSON object")
    item = dict(value)
    if set(item) == {"arguments"}:
        item = {"tool": contract.tool, "arguments": item["arguments"]}
    elif set(item) == {"parameters"}:
        item = {"tool": contract.tool, "arguments": item["parameters"]}
    elif set(item) == {"tool", "parameters"}:
        item = {"tool": item["tool"], "arguments": item["parameters"]}
    elif set(item) == {"policy", "power_budget"}:
        item = {"tool": contract.tool, "arguments": item}
    return parse_action_envelope(item, contract)
