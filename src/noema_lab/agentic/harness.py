from __future__ import annotations

"""Slow supervisory-control campaign runner for Noema recipes.

The harness deliberately treats a complete Noema run as the smallest control
interval. A decision request is constructed only from allowlisted public
configuration, completed-run feedback, and the delayed transmitter-visible
CSI artifact. The current-channel artifact is never opened.
"""

import copy
import gzip
import hashlib
import json
import math
import os
import queue
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.executor import LocalExecutor
from noema_lab.core.recipes import Recipe, compile_recipe, load_recipe
from noema_lab.core.reproducibility import canonical_json_sha256, derive_seed, utc_now_iso
from noema_lab.core.storage import LocalStore
from noema_lab.core.structured_input import decode_strict_yaml_or_json
from noema_lab.core.study_io import (
    content_bound_document,
    load_study_mapping,
    resolve_bound_file,
    verify_content_bound_document,
    write_study_json,
)
from noema_lab.ops import build_registry

JsonDict = Dict[str, Any]

_KIND = "noema.agentic_supervisory_campaign"
_REPLAY_KIND = "noema.agentic_supervisory_replay"
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")
_POLICY_ALIASES = {"equal_power": "fixed"}
_BANNED_OBSERVATION_PARTS = (
    "actual_state",
    "current_channel",
    "future_channel",
    "pending",
    "oracle",
    "environment_seed",
)
_FEEDBACK_METRICS = (
    "resource.finite_blocklength.predicted_bler",
    "resource.finite_blocklength.expected_goodput_bps_hz",
    "resource.finite_blocklength.p05_goodput_bps_hz",
    "resource.average_transmit_power_budget",
    "resource.power_constraint.max_abs_error",
    "resource.power_constraint.max_relative_error",
    "resource.power_constraint.max_negative_violation",
)
_MAX_RETAINED_RUN_JSON_BYTES = 64 * 1024 * 1024
_CSI_SUMMARY_FIELDS = (
    "oldest_observation_ofdm_symbol_index",
    "newest_observation_ofdm_symbol_index",
    "newest_observation_nominal_time_s",
    "mean_observed_gain",
    "p10_observed_gain",
    "p50_observed_gain",
    "p90_observed_gain",
    "mean_history_delta_magnitude",
)


class AgenticCampaignError(ValueError):
    """Raised when a supervisory campaign cannot be executed safely."""


class _DecisionDeadlineGuard:
    """Prevent reuse of a backend after an uninterruptible timed-out call.

    Python threads cannot safely cancel a local model generation. Once a call
    exceeds its deadline, the backend is quarantined for the rest of the
    campaign: later decisions take the declared timeout fallback without
    starting another call or resetting mutable backend state.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.worker: Optional[threading.Thread] = None
        self.poisoned = False


def _apply_run_overrides(
    contract: Mapping[str, Any],
    *,
    provider: Optional[str],
    model: Optional[str],
    model_revision: Optional[str],
    endpoint: Optional[str],
    base_url: Optional[str],
    prompt: Optional[Path],
    device: Optional[str],
    api_key_env: Optional[str] = None,
) -> JsonDict:
    """Return a validated effective contract with user-selected model inputs."""

    from noema_lab.agentic.contracts import (
        agentic_contract_from_mapping,
        provider_config_from_mapping,
    )

    result = copy.deepcopy(dict(contract))
    configured = _mapping(result.get("provider"), "provider")
    original_kind = str(configured.get("kind") or "")
    selected_kind = str(provider or original_kind)
    if provider is not None and model is None and selected_kind != original_kind:
        raise AgenticCampaignError(
            "--model is required when --provider changes the configured provider"
        )
    configured["kind"] = selected_kind
    if model is not None:
        configured["model"] = str(model)

    if selected_kind in {"ollama", "openai_compatible"}:
        configured.pop("model_revision", None)
        configured.pop("device", None)
        configured.pop("local_files_only", None)
        selected_url = base_url or endpoint or configured.get("base_url")
        if selected_kind == "ollama" and not selected_url:
            selected_url = "http://127.0.0.1:11434"
        if selected_url:
            configured["base_url"] = str(selected_url)
        if selected_kind == "openai_compatible":
            if api_key_env is not None:
                configured["api_key_env"] = str(api_key_env)
        else:
            configured.pop("api_key_env", None)
            if api_key_env is not None:
                raise AgenticCampaignError(
                    "--api-key-env is only valid for an OpenAI-compatible provider"
                )
    elif selected_kind == "transformers":
        configured.pop("base_url", None)
        configured.pop("api_key_env", None)
        if model_revision is not None:
            configured["model_revision"] = str(model_revision)
        if device is not None:
            configured["device"] = str(device)
        configured.setdefault("device", "cpu")
        configured.setdefault("local_files_only", False)
        if api_key_env is not None:
            raise AgenticCampaignError(
                "--api-key-env is only valid for an OpenAI-compatible provider"
            )
    else:
        for name in (
            "base_url",
            "api_key_env",
            "model_revision",
            "device",
            "local_files_only",
        ):
            configured.pop(name, None)
        if (
            endpoint is not None
            or base_url is not None
            or model_revision is not None
            or device is not None
            or api_key_env is not None
        ):
            raise AgenticCampaignError(
                "endpoint, revision, and device overrides require an HTTP or Transformers provider"
            )

    # Validate the provider independently so malformed overrides fail before
    # any campaign directory or radio run is created.
    result["provider"] = provider_config_from_mapping(configured).to_dict()
    if prompt is not None:
        prompt_path = Path(prompt)
        if prompt_path.is_symlink() or not prompt_path.is_file():
            raise AgenticCampaignError("--prompt must identify a regular, non-symlink file")
        try:
            prompt_text = prompt_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise AgenticCampaignError("cannot read UTF-8 prompt file") from exc
        if not prompt_text.strip():
            raise AgenticCampaignError("--prompt must not be empty")
        prompts = _mapping(result.get("prompts"), "prompts")
        prompts["user"] = prompt_text
        result["prompts"] = prompts

    # Re-run the complete strict schema after applying the effective values.
    return agentic_contract_from_mapping(result, verify_base_recipe=False).to_dict()


def run_agentic_campaign(
    contract_path: Path,
    workspace: Path,
    project_root: Optional[Path] = None,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    model_revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    base_url: Optional[str] = None,
    prompt: Optional[Path] = None,
    device: Optional[str] = None,
    api_key_env: Optional[str] = None,
    out: Optional[Path] = None,
) -> JsonDict:
    """Execute an agent arm and requested paired comparator arms."""

    from noema_lab.agentic.contracts import load_agentic_contract

    contract_path = Path(contract_path)
    root = _project_root(contract_path, project_root)
    contract_object = load_agentic_contract(
        contract_path,
        project_root=root,
        verify_base_recipe=True,
    )
    contract = _json_mapping(contract_object, "normalized agentic contract")
    contract = _apply_run_overrides(
        contract,
        provider=provider,
        model=model,
        model_revision=model_revision,
        endpoint=endpoint,
        base_url=base_url,
        prompt=prompt,
        device=device,
        api_key_env=api_key_env,
    )
    # Overrides are part of the effective campaign contract and therefore its
    # identity.  The authored contract remains recoverable from the CLI input.
    contract_sha256 = canonical_json_sha256(contract)
    base_path = _resolve_base_recipe(contract, root, contract_path)
    base_recipe = load_recipe(base_path, mode="compat")
    provider = _mapping(contract.get("provider"), "provider")
    scripted = provider.get("scripted_actions") or provider.get("actions")
    if provider.get("kind") == "scripted" and not scripted:
        actions = _mapping(contract.get("actions"), "actions")
        policies = list(actions.get("policies") or [])
        budgets = list(actions.get("allowed_power_budgets") or [])
        scripted = [
            {
                "tool": str(actions.get("tool")),
                "arguments": {
                    "policy": str(policy_name),
                    "power_budget": float(budgets[min(index, len(budgets) - 1)]),
                },
            }
            for index, policy_name in enumerate(policies)
        ]
    backend = _build_backend(provider, scripted_actions=scripted)
    return _run_campaign(
        contract=contract,
        contract_sha256=contract_sha256,
        base_recipe=base_recipe,
        base_recipe_path=base_path,
        workspace=Path(workspace),
        backend=backend,
        out=Path(out) if out is not None else None,
    )


def verify_agentic_campaign(campaign_dir: Path) -> JsonDict:
    """Verify internal content bindings and the privacy/action invariants."""

    from noema_lab.agentic.contracts import agentic_contract_from_mapping

    campaign_dir = Path(campaign_dir)
    if campaign_dir.is_symlink() or not campaign_dir.is_dir():
        raise AgenticCampaignError("campaign_dir must be a real directory")
    manifest = verify_content_bound_document(
        load_study_mapping(campaign_dir / "manifest.json"), "agentic manifest"
    )
    if manifest.get("kind") != _KIND or manifest.get("schema_version") != 1:
        raise AgenticCampaignError("unsupported agentic campaign manifest")

    resolved: Dict[str, Path] = {}
    files = _mapping(manifest.get("files"), "manifest.files")
    for name, binding_value in files.items():
        binding = _mapping(binding_value, "manifest.files.%s" % name)
        path = resolve_bound_file(
            campaign_dir,
            binding.get("path"),
            binding.get("sha256"),
            "campaign file %s" % name,
        )
        if int(binding.get("size_bytes", -1)) != path.stat().st_size:
            raise AgenticCampaignError("campaign file %s size mismatch" % name)
        resolved[name] = path

    required = {
        "contract",
        "base_recipe",
        "base_recipe_source",
        "events",
        "summary",
        "prompts",
    }
    if not required.issubset(resolved):
        raise AgenticCampaignError("campaign manifest is missing required files")
    contract_document = verify_content_bound_document(
        load_study_mapping(resolved["contract"]), "normalized contract"
    )
    contract = dict(contract_document)
    contract.pop("sha256", None)
    normalized_contract = agentic_contract_from_mapping(
        contract,
        verify_base_recipe=False,
    ).to_dict()
    if normalized_contract != contract:
        raise AgenticCampaignError("stored contract is not in canonical normalized form")
    summary = verify_content_bound_document(
        load_study_mapping(resolved["summary"]), "campaign summary"
    )
    prompt_document = verify_content_bound_document(
        load_study_mapping(resolved["prompts"]), "campaign prompts"
    )
    prompt_payload = dict(prompt_document)
    prompt_payload.pop("sha256", None)
    if prompt_payload != _mapping(contract.get("prompts"), "contract prompts"):
        raise AgenticCampaignError("stored prompts do not match the effective contract")
    stored_recipe = _mapping(
        load_study_mapping(resolved["base_recipe"]),
        "stored normalized base recipe",
    )
    source_recipe_sha256 = file_sha256(resolved["base_recipe_source"])
    recipe_binding = _mapping(contract.get("base_recipe"), "base_recipe")
    if source_recipe_sha256 != recipe_binding.get("sha256"):
        raise AgenticCampaignError(
            "retained base-recipe source does not match the contract"
        )
    retained_recipe = load_recipe(resolved["base_recipe_source"], mode="compat")
    if retained_recipe.to_dict() != stored_recipe:
        raise AgenticCampaignError(
            "normalized base recipe does not match its retained source"
        )
    events = _read_event_chain(resolved["events"])
    contract_sha256 = canonical_json_sha256(contract)
    if manifest.get("contract_sha256") != contract_sha256:
        raise AgenticCampaignError("manifest contract identity does not match contract")
    if summary.get("contract_sha256") != contract_sha256:
        raise AgenticCampaignError("summary contract identity does not match contract")
    started = next(
        (event for event in events if event.get("event") == "campaign_started"),
        None,
    )
    if started is None:
        raise AgenticCampaignError("campaign trace has no start event")
    if started.get("base_recipe_sha256") != recipe_binding.get("sha256"):
        raise AgenticCampaignError("campaign base-recipe identity does not match contract")
    if started.get("normalized_base_recipe_sha256") != canonical_json_sha256(
        stored_recipe
    ):
        raise AgenticCampaignError(
            "campaign normalized base-recipe identity does not match evidence"
        )
    if int(manifest.get("event_count", -1)) != len(events):
        raise AgenticCampaignError("manifest event count does not match trace")
    expected_tail = events[-1]["sha256"] if events else None
    if manifest.get("last_event_sha256") != expected_tail:
        raise AgenticCampaignError("manifest trace tail does not match trace")
    if int(summary.get("event_count", -1)) != len(events):
        raise AgenticCampaignError("summary event count does not match trace")

    _verify_campaign_topology(
        events,
        contract,
        public_context=_recipe_public_context(retained_recipe),
    )

    allowlist = _observation_allowlist(contract)
    decisions = 0
    runs = 0
    referenced_run_files: set[str] = set()
    verification_registry = build_registry()
    for event in events:
        event_type = event.get("event")
        if event_type == "decision":
            decisions += 1
            observation = _mapping(event.get("observation"), "decision observation")
            _verify_observation(observation, allowlist)
            effective = _mapping(event.get("effective_action"), "effective action")
            _validate_canonical_action(effective, contract)
        elif event_type == "run_completed":
            runs += 1
            metrics = _mapping(event.get("metrics"), "run metrics")
            if set(metrics) != set(_feedback_metric_names()):
                raise AgenticCampaignError("run feedback metric projection is incomplete")
            csi = _mapping(event.get("transmitter_csi"), "transmitter CSI summary")
            _assert_safe_transmitter_csi_summary(csi)
            evidence = _mapping(event.get("run_evidence"), "retained run evidence")
            if set(evidence) != {"summary", "manifest"}:
                raise AgenticCampaignError(
                    "run evidence must reference summary and manifest files"
                )
            summary_key = str(evidence["summary"])
            manifest_key = str(evidence["manifest"])
            for key in (summary_key, manifest_key):
                if key not in resolved or key in referenced_run_files:
                    raise AgenticCampaignError(
                        "run evidence reference is missing or reused: %s" % key
                    )
                referenced_run_files.add(key)
            retained_summary, summary_sha256, summary_size = (
                _load_compressed_json_evidence(
                    resolved[summary_key],
                    "retained run summary",
                )
            )
            retained_manifest, manifest_sha256, _ = (
                _load_compressed_json_evidence(
                    resolved[manifest_key],
                    "retained run manifest",
                )
            )
            if summary_sha256 != event.get("run_summary_sha256"):
                raise AgenticCampaignError("retained run summary hash mismatch")
            if manifest_sha256 != event.get("run_manifest_sha256"):
                raise AgenticCampaignError("retained run manifest hash mismatch")
            run_id = str(event.get("run_id"))
            if (
                retained_summary.get("run_id") != run_id
                or retained_manifest.get("run_id") != run_id
            ):
                raise AgenticCampaignError(
                    "retained run evidence does not match the recorded run id"
                )
            if (
                retained_summary.get("status") != "completed"
                or retained_manifest.get("status") != "completed"
                or retained_summary.get("completed_at_utc")
                != event.get("completed_at_utc")
            ):
                raise AgenticCampaignError(
                    "retained run completion evidence does not match the event"
                )
            summary_binding = _mapping(
                retained_manifest.get("summary"),
                "retained run manifest summary binding",
            )
            if (
                summary_binding.get("sha256") != summary_sha256
                or _integer(
                    summary_binding.get("size_bytes"),
                    "retained summary size",
                )
                != summary_size
            ):
                raise AgenticCampaignError(
                    "retained run manifest does not bind its summary"
                )
            if _extract_feedback_metrics(retained_summary) != metrics:
                raise AgenticCampaignError(
                    "run metrics do not match the retained run summary"
                )
            action = _mapping(event.get("action"), "run action")
            run_seeds = {
                str(name): _integer(value, "recorded %s seed" % name)
                for name, value in _mapping(
                    event.get("run_seeds"), "recorded run seeds"
                ).items()
            }
            expected_recipe = _materialize_recipe(
                retained_recipe,
                action,
                run_seeds,
                arm_id=str(event.get("arm_id")),
                episode=int(event.get("episode", 0)),
                window=int(event.get("window", 0)),
                registry=verification_registry,
            )
            expected_authored_sha256 = canonical_json_sha256(
                expected_recipe.to_dict()
            )
            expected_effective_sha256 = canonical_json_sha256(
                compile_recipe(
                    expected_recipe.to_dict(),
                    mode="strict",
                    registry=verification_registry,
                )
                .require_recipe(effective=True)
                .to_dict()
            )
            manifest_recipe = _mapping(
                retained_manifest.get("recipe"),
                "retained run manifest recipe binding",
            )
            if (
                retained_summary.get("authored_recipe_sha256")
                != expected_authored_sha256
                or manifest_recipe.get("authored_sha256")
                != expected_authored_sha256
                or retained_summary.get("effective_recipe_sha256")
                != expected_effective_sha256
                or retained_summary.get("recipe_sha256")
                != expected_effective_sha256
                or manifest_recipe.get("effective_sha256")
                != expected_effective_sha256
            ):
                raise AgenticCampaignError(
                    "executed recipe identity does not match the recorded action and seeds"
                )

    declared_run_files = {key for key in resolved if key.startswith("run_")}
    if referenced_run_files != declared_run_files:
        raise AgenticCampaignError(
            "campaign manifest has missing or unreferenced retained run evidence"
        )
    if int(summary.get("run_count", -1)) != runs or int(
        summary.get("decision_count", -1)
    ) != decisions:
        raise AgenticCampaignError("summary run/decision counts do not match trace")
    expected_arm_summaries = _arm_summaries(
        events,
        _feedback_metric_names(),
        contract,
    )
    if summary.get("arm_summaries") != expected_arm_summaries:
        raise AgenticCampaignError("campaign arm summaries do not match trace")
    if summary.get("objective_ranking") != _objective_ranking(
        expected_arm_summaries,
        contract,
    ):
        raise AgenticCampaignError("campaign objective ranking does not match trace")
    expected_agent_costs = _cost_summary(
        [
            _mapping(event.get("cost"), "agent decision cost")
            for event in events
            if event.get("event") == "decision"
            and event.get("arm_kind") == "agent"
        ]
    )
    if summary.get("agent_costs") != expected_agent_costs:
        raise AgenticCampaignError("campaign agent cost summary does not match trace")

    return {
        "schema_version": 1,
        "kind": "noema.agentic_campaign_verification",
        "status": "passed",
        "campaign_id": manifest.get("campaign_id"),
        "manifest_sha256": manifest["sha256"],
        "event_count": len(events),
        "decision_count": decisions,
        "run_count": runs,
    }


def replay_agentic_campaign(
    campaign_dir: Path,
    workspace: Path,
    project_root: Optional[Path] = None,
    *,
    out: Optional[Path] = None,
) -> JsonDict:
    """Re-execute the recorded effective actions without calling a model."""

    del project_root  # replay is self-contained by construction
    verification = verify_agentic_campaign(campaign_dir)
    campaign_dir = Path(campaign_dir)
    manifest = verify_content_bound_document(
        load_study_mapping(campaign_dir / "manifest.json"), "agentic manifest"
    )
    files = _mapping(manifest["files"], "manifest.files")
    contract_doc = verify_content_bound_document(
        load_study_mapping(campaign_dir / str(_mapping(files["contract"], "contract binding")["path"])),
        "normalized contract",
    )
    contract = dict(contract_doc)
    contract.pop("sha256", None)
    base_recipe_path = campaign_dir / str(
        _mapping(files["base_recipe"], "base recipe binding")["path"]
    )
    base_recipe = load_recipe(base_recipe_path, mode="compat")
    events_path = campaign_dir / str(_mapping(files["events"], "events binding")["path"])
    source_events = _read_event_chain(events_path)
    source_runs = [event for event in source_events if event.get("event") == "run_completed"]

    replay_dir = _new_evidence_dir(
        Path(workspace),
        "replay_%s" % str(manifest.get("campaign_id") or "campaign"),
        str(manifest["sha256"]),
        out=Path(out) if out is not None else None,
    )
    registry = build_registry()
    store = LocalStore(Path(workspace))
    executor = LocalExecutor(registry, store)
    records: List[JsonDict] = []
    mismatches = 0
    for source in source_runs:
        action = _mapping(source.get("action"), "recorded run action")
        run_seeds = {
            str(name): _integer(value, "recorded %s seed" % name)
            for name, value in _mapping(source.get("run_seeds"), "recorded run seeds").items()
        }
        recipe = _materialize_recipe(
            base_recipe,
            action,
            run_seeds,
            arm_id=str(source.get("arm_id")),
            episode=int(source.get("episode", 0)),
            window=int(source.get("window", 0)),
            registry=registry,
        )
        started = time.perf_counter()
        run_dir = executor.run(recipe)
        elapsed = time.perf_counter() - started
        summary = store.read_json(run_dir / "summary.json")
        metrics = _extract_feedback_metrics(summary)
        expected = _mapping(source.get("metrics"), "recorded run metrics")
        matches = canonical_json_sha256(metrics) == canonical_json_sha256(expected)
        mismatches += int(not matches)
        records.append(
            {
                "source_event_index": source.get("event_index"),
                "arm_id": source.get("arm_id"),
                "episode": source.get("episode"),
                "window": source.get("window"),
                "run_seeds": run_seeds,
                "action": copy.deepcopy(action),
                "expected_metrics_sha256": canonical_json_sha256(expected),
                "actual_metrics_sha256": canonical_json_sha256(metrics),
                "metrics_match": matches,
                "run_id": run_dir.name,
                "run_summary_sha256": file_sha256(run_dir / "summary.json"),
                "wall_time_seconds": elapsed,
            }
        )
    report = content_bound_document(
        {
            "schema_version": 1,
            "kind": _REPLAY_KIND,
            "created_at_utc": utc_now_iso(),
            "source_campaign_id": manifest.get("campaign_id"),
            "source_manifest_sha256": manifest["sha256"],
            "source_verification": verification,
            "status": "passed" if mismatches == 0 else "failed",
            "replayed_run_count": len(records),
            "mismatch_count": mismatches,
            "records": records,
        }
    )
    write_study_json(replay_dir / "replay.json", report)
    result = dict(report)
    result["replay_dir"] = str(replay_dir)
    return result


def _run_campaign(
    *,
    contract: JsonDict,
    contract_sha256: str,
    base_recipe: Recipe,
    base_recipe_path: Path,
    workspace: Path,
    backend: Any,
    out: Optional[Path],
) -> JsonDict:
    allowlist = _observation_allowlist(contract)
    campaign_dir = _new_evidence_dir(
        workspace, str(contract["id"]), contract_sha256, out=out
    )
    registry = build_registry()
    store = LocalStore(workspace)
    executor = LocalExecutor(registry, store)
    events: List[JsonDict] = []
    run_evidence_bindings: Dict[str, JsonDict] = {}
    run_ordinal = 0
    previous_event_sha256: Optional[str] = None
    created_at = utc_now_iso()
    deadline_guard = _DecisionDeadlineGuard()

    def emit(event: Mapping[str, Any]) -> JsonDict:
        nonlocal previous_event_sha256
        payload = dict(event)
        payload["schema_version"] = 1
        payload["event_index"] = len(events)
        payload["previous_event_sha256"] = previous_event_sha256
        bound = content_bound_document(payload)
        events.append(bound)
        previous_event_sha256 = bound["sha256"]
        return bound

    initial_action = _fallback_action(contract)
    episodes = _mapping(contract.get("episodes"), "episodes")
    decision_count = _integer(
        episodes.get("decisions_per_episode"), "episodes.decisions_per_episode"
    )
    schedule = _episode_schedule(episodes)
    arms = _campaign_arms(contract)
    agent_costs: List[JsonDict] = []
    public_context = _recipe_public_context(base_recipe)

    emit(
        {
            "event": "campaign_started",
            "occurred_at_utc": created_at,
            "contract_sha256": contract_sha256,
            "base_recipe_sha256": file_sha256(base_recipe_path),
            "normalized_base_recipe_sha256": canonical_json_sha256(
                base_recipe.to_dict()
            ),
            "arm_ids": [arm["id"] for arm in arms],
        }
    )
    for arm in arms:
        for schedule_row in schedule:
            episode = int(schedule_row["episode"])
            current_action = copy.deepcopy(initial_action)
            history: List[JsonDict] = []
            reset = getattr(backend, "reset", None)
            if (
                arm["kind"] == "agent"
                and callable(reset)
                and not deadline_guard.poisoned
            ):
                try:
                    reset()
                except TypeError:
                    reset(episode=episode)
            emit(
                {
                    "event": "episode_reset",
                    "occurred_at_utc": utc_now_iso(),
                    "arm_id": arm["id"],
                    "arm_kind": arm["kind"],
                    "episode": episode,
                    "initial_action": copy.deepcopy(current_action),
                }
            )
            # decisions_per_episode is the number of complete controlled runs,
            # including decision zero with an explicitly empty history.
            for window in range(decision_count):
                observation = _build_observation(
                    contract,
                    arm["id"],
                    episode,
                    window,
                    history,
                    public_context,
                )
                if arm["kind"] == "agent":
                    decision = _agent_decision(
                        backend,
                        contract,
                        observation,
                        current_action,
                        deadline_guard=deadline_guard,
                    )
                    current_action = decision["effective_action"]
                    agent_costs.append(copy.deepcopy(decision["cost"]))
                elif arm["kind"] == "static":
                    current_action = copy.deepcopy(arm["action"])
                    decision = _comparator_decision(current_action, "static_policy")
                else:
                    current_action = _rule_action(
                        arm["rule"], observation, current_action, contract
                    )
                    decision = _comparator_decision(current_action, "rule_based")
                emit(
                    {
                        "event": "decision",
                        "occurred_at_utc": utc_now_iso(),
                        "arm_id": arm["id"],
                        "arm_kind": arm["kind"],
                        "episode": episode,
                        "decision": window,
                        "observation": observation,
                        **decision,
                    }
                )

                run_seeds = _derive_run_seeds(
                    schedule_row,
                    contract_id=str(contract["id"]),
                    episode=episode,
                    window=window,
                )
                recipe = _materialize_recipe(
                    base_recipe,
                    current_action,
                    run_seeds,
                    arm_id=arm["id"],
                    episode=episode,
                    window=window,
                    registry=registry,
                )
                run_started_at = utc_now_iso()
                run_started = time.perf_counter()
                run_dir = executor.run(recipe)
                run_seconds = time.perf_counter() - run_started
                run_summary = store.read_json(run_dir / "summary.json")
                metrics = _extract_feedback_metrics(run_summary)
                transmitter_csi = _summarize_transmitter_csi(run_dir, run_summary)
                run_evidence, retained_bindings = _retain_run_evidence(
                    campaign_dir,
                    run_dir,
                    run_ordinal,
                )
                run_evidence_bindings.update(retained_bindings)
                run_ordinal += 1
                completed_at = str(run_summary.get("completed_at_utc") or utc_now_iso())
                history.append(
                    {
                        "window": window,
                        "run_id": run_dir.name,
                        "started_at_utc": run_started_at,
                        "completed_at_utc": completed_at,
                        "action": copy.deepcopy(current_action),
                        "metrics": copy.deepcopy(metrics),
                        "action_valid": decision["validation_status"] in {"accepted", "comparator"},
                        "fallback_used": bool(decision["fallback_used"]),
                        "transmitter_csi": transmitter_csi,
                    }
                )
                emit(
                    {
                        "event": "run_completed",
                        "occurred_at_utc": utc_now_iso(),
                        "arm_id": arm["id"],
                        "arm_kind": arm["kind"],
                        "episode": episode,
                        "window": window,
                        "phase": "controlled",
                        "run_seeds": run_seeds,
                        "action": copy.deepcopy(current_action),
                        "started_at_utc": run_started_at,
                        "completed_at_utc": completed_at,
                        "run_wall_time_seconds": run_seconds,
                        "run_id": run_dir.name,
                        "run_summary_sha256": file_sha256(run_dir / "summary.json"),
                        "run_manifest_sha256": file_sha256(run_dir / "manifest.json"),
                        "run_evidence": run_evidence,
                        "metrics": metrics,
                        "transmitter_csi": transmitter_csi,
                    }
                )

    completed_at = utc_now_iso()
    emit(
        {
            "event": "campaign_completed",
            "occurred_at_utc": completed_at,
            "run_count": sum(event.get("event") == "run_completed" for event in events),
            "decision_count": sum(event.get("event") == "decision" for event in events),
        }
    )
    contract_doc = content_bound_document(contract)
    write_study_json(campaign_dir / "contract.normalized.json", contract_doc)
    write_study_json(campaign_dir / "base_recipe.json", base_recipe.to_dict())
    _copy_evidence_file(
        base_recipe_path,
        campaign_dir / "base_recipe.source.yaml",
    )
    _write_events(campaign_dir / "events.jsonl", events)
    arm_summaries = _arm_summaries(
        events,
        _feedback_metric_names(),
        contract,
    )
    summary = content_bound_document(
        {
            "schema_version": 1,
            "kind": "noema.agentic_supervisory_campaign_summary",
            "campaign_id": campaign_dir.name,
            "contract_id": contract["id"],
            "contract_sha256": contract_sha256,
            "status": "completed",
            "created_at_utc": created_at,
            "completed_at_utc": completed_at,
            "event_count": len(events),
            "run_count": sum(event.get("event") == "run_completed" for event in events),
            "decision_count": sum(event.get("event") == "decision" for event in events),
            "arm_summaries": arm_summaries,
            "objective_ranking": _objective_ranking(arm_summaries, contract),
            "agent_costs": _cost_summary(agent_costs),
        }
    )
    write_study_json(campaign_dir / "summary.json", summary)
    write_study_json(
        campaign_dir / "prompts.json",
        content_bound_document(_mapping(contract["prompts"], "prompts")),
    )
    file_names = {
        "contract": "contract.normalized.json",
        "base_recipe": "base_recipe.json",
        "base_recipe_source": "base_recipe.source.yaml",
        "events": "events.jsonl",
        "summary": "summary.json",
        "prompts": "prompts.json",
    }
    bindings = {
        key: _file_binding(campaign_dir / filename, filename)
        for key, filename in file_names.items()
    }
    bindings.update(run_evidence_bindings)
    manifest = content_bound_document(
        {
            "schema_version": 1,
            "kind": _KIND,
            "campaign_id": campaign_dir.name,
            "contract_id": contract["id"],
            "contract_sha256": contract_sha256,
            "created_at_utc": created_at,
            "completed_at_utc": completed_at,
            "status": "completed",
            "event_count": len(events),
            "last_event_sha256": events[-1]["sha256"] if events else None,
            "files": bindings,
        }
    )
    write_study_json(campaign_dir / "manifest.json", manifest)
    result = dict(summary)
    result["campaign_dir"] = str(campaign_dir)
    result["manifest_sha256"] = manifest["sha256"]
    return result


def _agent_decision(
    backend: Any,
    contract: Mapping[str, Any],
    observation: JsonDict,
    current_action: JsonDict,
    *,
    deadline_guard: Optional[_DecisionDeadlineGuard] = None,
) -> JsonDict:
    from noema_lab.agentic.backends import DecisionRequest

    limits = _mapping(contract.get("limits"), "limits")
    prompts = _mapping(contract.get("prompts"), "prompts")
    timeout = float(limits.get("decision_timeout_seconds"))
    decision_id = "episode-%d-decision-%d" % (
        int(observation["episode_index"]),
        int(observation["decision_index"]),
    )
    request = DecisionRequest(
        decision_id=decision_id,
        observation=observation,
        action_contract=copy.deepcopy(_mapping(contract.get("actions"), "actions")),
        system_prompt=str(prompts.get("system") or ""),
        user_prompt=str(prompts.get("user") or ""),
        objective=copy.deepcopy(_mapping(contract.get("objective"), "objective")),
        timeout_seconds=timeout,
        max_output_tokens=int(limits.get("max_output_tokens")),
        max_tool_calls=int(limits.get("max_tool_calls_per_decision", 1)),
    )
    started = time.perf_counter()
    response, failure, error = _decide_with_deadline(
        backend,
        request,
        timeout,
        guard=deadline_guard,
    )
    elapsed = time.perf_counter() - started
    requested_action: Optional[JsonDict] = None
    provider: JsonDict = {}
    usage: JsonDict = {}
    raw_response: Any = None
    tool_calls = 0
    reported_latency: Optional[float] = None
    finish_reason: Optional[str] = None
    if response is not None:
        raw_response = _json_safe(getattr(response, "raw_response", None))
        provider = _json_mapping_or_empty(getattr(response, "provider", {}))
        usage = _json_mapping_or_empty(getattr(response, "usage", {}))
        tool_calls = int(getattr(response, "tool_call_count", 0) or 0)
        latency_value = getattr(response, "latency_seconds", None)
        reported_latency = float(latency_value) if latency_value is not None else None
        finish_reason_value = getattr(response, "finish_reason", None)
        finish_reason = str(finish_reason_value) if finish_reason_value is not None else None
        try:
            requested_action = _normalize_action(getattr(response, "action", None))
            requested_action = _validate_canonical_action(requested_action, contract)
        except (AgenticCampaignError, TypeError, ValueError) as exc:
            failure = "invalid_action"
            error = str(exc)
        maximum_calls = int(limits.get("max_tool_calls_per_decision", 1))
        if tool_calls > maximum_calls:
            failure = "invalid_action"
            error = "tool call budget exceeded"
    if failure is None and requested_action is None:
        failure = "invalid_action"
        error = "backend did not return an action"

    if failure is None:
        effective = (
            copy.deepcopy(current_action)
            if requested_action and requested_action["kind"] == "keep"
            else copy.deepcopy(requested_action)
        )
        fallback_used = False
        validation_status = "accepted"
    else:
        fallback_on = {str(item) for item in _mapping(contract["fallback"], "fallback").get("on", [])}
        if failure not in fallback_on:
            # Contract vocabularies sometimes call these provider_error/malformed_json.
            aliases = {"backend_error": "provider_error", "invalid_action": "malformed_json"}
            if aliases.get(failure) not in fallback_on:
                raise AgenticCampaignError(
                    "decision failed with %s but the contract does not permit that fallback" % failure
                )
        effective = _fallback_action(contract, current_action=current_action)
        fallback_used = True
        validation_status = "fallback"

    raw_hash = canonical_json_sha256(raw_response) if raw_response is not None else None
    provider_kind = str(_mapping(contract.get("provider"), "provider").get("kind"))
    call_skipped = error in {
        "backend quarantined after an earlier timeout",
        "backend has an unfinished earlier decision",
    }
    model_call_count = int(
        not call_skipped and provider_kind not in {"scripted", "replay"}
    )
    cost = {
        "decision_latency_seconds": elapsed,
        "backend_reported_latency_seconds": reported_latency,
        "provider_call_count": int(not call_skipped),
        "model_call_count": model_call_count,
        "tool_call_count": tool_calls,
        "usage": usage,
        "failure_class": failure,
        "fallback_used": fallback_used,
    }
    return {
        "requested_action": requested_action,
        "effective_action": effective,
        "validation_status": validation_status,
        "failure_class": failure,
        "failure_message": error,
        "fallback_used": fallback_used,
        "raw_response": raw_response,
        "raw_response_sha256": raw_hash,
        "provider": provider,
        "usage": usage,
        "tool_call_count": tool_calls,
        "finish_reason": finish_reason,
        "cost": cost,
    }


def _decide_with_deadline(
    backend: Any,
    request: Any,
    timeout_seconds: float,
    *,
    guard: Optional[_DecisionDeadlineGuard] = None,
) -> Tuple[Any, Optional[str], Optional[str]]:
    state = guard or _DecisionDeadlineGuard()
    with state.lock:
        if state.poisoned:
            return None, "timeout", "backend quarantined after an earlier timeout"
        if state.worker is not None and state.worker.is_alive():
            state.poisoned = True
            return None, "timeout", "backend has an unfinished earlier decision"

    results: queue.Queue[Tuple[str, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            results.put(("ok", backend.decide(request)))
        except BaseException as exc:  # preserve provider failure as evidence
            results.put(("error", exc))

    worker = threading.Thread(target=invoke, name="noema-agent-decision", daemon=True)
    with state.lock:
        state.worker = worker
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        with state.lock:
            state.poisoned = True
        return None, "timeout", "decision deadline exceeded"
    with state.lock:
        if state.worker is worker:
            state.worker = None
    try:
        status, value = results.get_nowait()
    except queue.Empty:
        return None, "backend_error", "backend returned no result"
    if status == "error":
        failure_class = str(getattr(value, "failure_class", "backend_error"))
        if failure_class not in {"backend_error", "invalid_action"}:
            failure_class = "backend_error"
        return None, failure_class, "%s: %s" % (type(value).__name__, value)
    return value, None, None


def _build_observation(
    contract: Mapping[str, Any],
    arm_id: str,
    episode: int,
    decision: int,
    history: Sequence[Mapping[str, Any]],
    public_context: Mapping[str, Any],
) -> JsonDict:
    del arm_id  # controller identity is harness evidence, not an observation
    observations = _mapping(contract.get("observations"), "observations")
    history_size = int(observations.get("history_decisions"))
    timestamp_field = str(observations.get("timestamp_field"))
    recent: List[JsonDict] = []
    selected_history = history[-history_size:] if history_size > 0 else []
    for row in selected_history:
        metrics = _mapping(row["metrics"], "history metrics")
        action = _mapping(row["action"], "history action")
        recent.append(
            {
                "run_id": str(row["run_id"]),
                "run_started_at_utc": str(row["started_at_utc"]),
                "run_completed_at_utc": str(row["completed_at_utc"]),
                "action": {
                    "policy": str(action["policy"]),
                    "power_budget": float(action["average_power_budget"]),
                },
                "predicted_bler": metrics["resource.finite_blocklength.predicted_bler"],
                "expected_goodput_bps_hz": metrics[
                    "resource.finite_blocklength.expected_goodput_bps_hz"
                ],
                "average_power_budget": metrics[
                    "resource.average_transmit_power_budget"
                ],
                "action_valid": bool(row["action_valid"]),
                "fallback_used": bool(row["fallback_used"]),
            }
        )
    candidate = {
        timestamp_field: utc_now_iso(),
        "episode_index": episode,
        "decision_index": decision,
        "available_after_run_id": str(history[-1]["run_id"]) if history else None,
        "public_context": copy.deepcopy(dict(public_context)),
        "recent": recent,
        "transmitter_csi": (
            copy.deepcopy(history[-1]["transmitter_csi"])
            if history
            else {field: None for field in _CSI_SUMMARY_FIELDS}
        ),
    }
    observation = _project_observation(candidate, _observation_allowlist(contract))
    _verify_observation(observation, _observation_allowlist(contract))
    return observation


def _verify_observation(observation: Mapping[str, Any], allowlist: Sequence[str]) -> None:
    _assert_no_banned_observation_keys(observation)
    expected_non_recent = {path for path in allowlist if not path.startswith("recent.")}
    actual_non_recent = {
        path for path in _leaf_paths(observation) if not path.startswith("recent.")
    }
    if actual_non_recent != expected_non_recent:
        raise AgenticCampaignError("observation fields differ from contract allowlist")
    expected_recent = {
        path[len("recent.") :] for path in allowlist if path.startswith("recent.")
    }
    recent = observation.get("recent")
    if expected_recent and not isinstance(recent, list):
        raise AgenticCampaignError("allowlisted recent observation must be an array")
    if not expected_recent and "recent" in observation:
        raise AgenticCampaignError("observation contains undeclared recent history")
    for row in recent or []:
        if set(_leaf_paths(_mapping(row, "recent observation"))) != expected_recent:
            raise AgenticCampaignError("recent observation fields differ from allowlist")


def _assert_no_banned_observation_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _BANNED_OBSERVATION_PARTS):
                raise AgenticCampaignError("forbidden observation field at %s.%s" % (path, key))
            _assert_no_banned_observation_keys(item, "%s.%s" % (path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_banned_observation_keys(item, "%s[%d]" % (path, index))


def _observation_allowlist(contract: Mapping[str, Any]) -> List[str]:
    observations = _mapping(contract.get("observations"), "observations")
    raw = observations.get("allowlist")
    if not isinstance(raw, list) or not raw:
        raise AgenticCampaignError("observations.allowlist must be nonempty")
    result: List[str] = []
    for index, item in enumerate(raw):
        metric = str(item).strip()
        if not metric or metric in result:
            raise AgenticCampaignError("invalid or duplicate observation metric at %d" % index)
        lowered = metric.lower()
        if any(part in lowered for part in _BANNED_OBSERVATION_PARTS):
            raise AgenticCampaignError("observation allowlist exposes forbidden state: %s" % metric)
        result.append(metric)
    return result


def _extract_feedback_metrics(summary: Mapping[str, Any]) -> JsonDict:
    evaluation = next(
        (
            step
            for step in summary.get("steps", [])
            if isinstance(step, Mapping) and step.get("id") == "allocation_evaluation"
        ),
        None,
    )
    if evaluation is None:
        raise AgenticCampaignError("run summary is missing allocation_evaluation")
    metrics = _mapping(evaluation.get("metrics"), "allocation_evaluation.metrics")
    missing = [name for name in _FEEDBACK_METRICS if name not in metrics]
    if missing:
        raise AgenticCampaignError(
            "allocation_evaluation is missing feedback metrics: %s" % ", ".join(missing)
        )
    result: JsonDict = {}
    for name in _FEEDBACK_METRICS:
        value = metrics[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise AgenticCampaignError(
                "allocation_evaluation metric %s must be finite and numeric" % name
            )
        result[name] = _json_safe(value)
    return result


def _feedback_metric_names() -> List[str]:
    return list(_FEEDBACK_METRICS)


def _recipe_public_context(recipe: Recipe) -> JsonDict:
    state_params: Optional[Mapping[str, Any]] = None
    csi_params: Optional[Mapping[str, Any]] = None
    for step in recipe.steps:
        if step.op == "wireless.ofdm_channel_state":
            if state_params is not None:
                raise AgenticCampaignError(
                    "base recipe must contain one OFDM channel-state step"
                )
            state_params = step.params
        elif step.op == "wireless.ofdm_delayed_csi":
            if csi_params is not None:
                raise AgenticCampaignError(
                    "base recipe must contain one delayed-CSI step"
                )
            csi_params = step.params
    if state_params is None or csi_params is None:
        raise AgenticCampaignError(
            "base recipe must contain OFDM channel-state and delayed-CSI steps"
        )
    context = {
        "noise_variance": float(state_params["noise_variance"]),
        "feedback_delay_ofdm_symbols": int(
            csi_params["feedback_delay_ofdm_symbols"]
        ),
        "csi_history_length": int(csi_params["csi_history_length"]),
        "csi_estimation_snr_db": float(csi_params["csi_estimation_snr_db"]),
        "mobility_kmh": float(state_params["mobility_kmh"]),
        "ofdm_fft_size": int(state_params["ofdm_fft_size"]),
        "allocation_ofdm_symbols": int(csi_params["allocation_ofdm_symbols"]),
    }
    if not all(
        not isinstance(value, float) or math.isfinite(value)
        for value in context.values()
    ):
        raise AgenticCampaignError("base recipe public context must be finite")
    return context


def _summarize_transmitter_csi(
    run_dir: Path, run_summary: Mapping[str, Any]
) -> JsonDict:
    """Summarize only the delayed/noisy transmitter-visible CSI artifact."""

    step = next(
        (
            item
            for item in run_summary.get("steps", [])
            if isinstance(item, Mapping) and item.get("id") == "csi_observation"
        ),
        None,
    )
    if step is None:
        raise AgenticCampaignError("run summary is missing csi_observation")
    outputs = _mapping(step.get("outputs"), "csi_observation.outputs")
    output = _mapping(
        outputs.get("transmitter_csi"),
        "csi_observation.outputs.transmitter_csi",
    )
    if str(output.get("kind") or "") != "channel.ofdm_channel_state.numpy":
        raise AgenticCampaignError("transmitter CSI artifact has an unexpected kind")
    raw_path = Path(str(output.get("path") or ""))
    candidates = [raw_path] if raw_path.is_absolute() else [Path.cwd() / raw_path, run_dir / raw_path]
    artifact_path: Optional[Path] = None
    resolved_run = Path(run_dir).resolve(strict=True)
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_run)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        if resolved.is_file() and not resolved.is_symlink():
            artifact_path = resolved
            break
    if artifact_path is None:
        raise AgenticCampaignError("transmitter CSI artifact path is unavailable or unsafe")
    expected_sha256 = str(output.get("sha256") or "")
    if len(expected_sha256) != 64 or file_sha256(artifact_path) != expected_sha256:
        raise AgenticCampaignError("transmitter CSI artifact hash does not match run evidence")

    try:
        with np.load(str(artifact_path), allow_pickle=False) as payload:
            gains = np.asarray(payload["capture_gains"], dtype=np.float64)
            history_iq = np.asarray(payload["capture_csi_history"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise AgenticCampaignError("cannot read transmitter-visible CSI artifact") from exc
    if gains.size == 0 or gains.ndim != 2 or not np.all(np.isfinite(gains)):
        raise AgenticCampaignError("transmitter CSI gains must be a finite nonempty matrix")
    if (
        history_iq.ndim != 4
        or history_iq.shape[-1] != 2
        or history_iq.shape[1] < 1
        or not np.all(np.isfinite(history_iq))
    ):
        raise AgenticCampaignError("transmitter CSI history has an invalid shape")
    history = history_iq[..., 0] + 1j * history_iq[..., 1]
    mean_delta = (
        float(np.mean(np.abs(np.diff(history, axis=1))))
        if history.shape[1] > 1
        else 0.0
    )
    metadata = _mapping(output.get("metadata"), "transmitter CSI metadata")
    history_length = int(metadata.get("csi_history_length") or history.shape[1])
    allocation_symbols = int(
        metadata.get("allocation_ofdm_symbols") or history_iq.shape[0]
    )
    spacing_khz = float(metadata.get("subcarrier_spacing_khz") or 15.0)
    newest_index = history_length + allocation_symbols - 2
    summary = {
        "oldest_observation_ofdm_symbol_index": 0,
        "newest_observation_ofdm_symbol_index": newest_index,
        "newest_observation_nominal_time_s": newest_index
        / max(spacing_khz * 1000.0, 1e-30),
        "mean_observed_gain": float(np.mean(gains)),
        "p10_observed_gain": float(np.percentile(gains, 10.0)),
        "p50_observed_gain": float(np.percentile(gains, 50.0)),
        "p90_observed_gain": float(np.percentile(gains, 90.0)),
        "mean_history_delta_magnitude": mean_delta,
    }
    _assert_safe_transmitter_csi_summary(summary)
    return summary


def _assert_safe_transmitter_csi_summary(value: Mapping[str, Any]) -> None:
    if set(value) != set(_CSI_SUMMARY_FIELDS):
        raise AgenticCampaignError("transmitter CSI summary fields are incomplete")
    for name, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AgenticCampaignError("transmitter CSI summary %s must be numeric" % name)
        if not math.isfinite(float(item)):
            raise AgenticCampaignError("transmitter CSI summary %s must be finite" % name)


def _project_observation(candidate: Mapping[str, Any], allowlist: Sequence[str]) -> JsonDict:
    projected: JsonDict = {}
    recent_paths = [path for path in allowlist if path.startswith("recent.")]
    for path in allowlist:
        if path.startswith("recent."):
            continue
        _set_dotted(projected, path, _get_dotted(candidate, path))
    rows = candidate.get("recent", [])
    if recent_paths:
        projected["recent"] = []
        for raw_row in rows if isinstance(rows, list) else []:
            row: JsonDict = {}
            for path in recent_paths:
                relative = path[len("recent.") :]
                _set_dotted(row, relative, _get_dotted(raw_row, relative))
            projected["recent"].append(row)
    return projected


def _get_dotted(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return copy.deepcopy(current)


def _set_dotted(target: JsonDict, path: str, value: Any) -> None:
    components = path.split(".")
    current = target
    for component in components[:-1]:
        child = current.setdefault(component, {})
        if not isinstance(child, dict):
            raise AgenticCampaignError("observation allowlist has colliding paths")
        current = child
    current[components[-1]] = value


def _leaf_paths(value: Mapping[str, Any], prefix: str = "") -> List[str]:
    result: List[str] = []
    for key, item in value.items():
        path = "%s.%s" % (prefix, key) if prefix else str(key)
        if isinstance(item, Mapping):
            result.extend(_leaf_paths(item, path))
        elif isinstance(item, list):
            if key == "recent":
                continue
            result.append(path)
        else:
            result.append(path)
    return result


def _derive_run_seeds(
    schedule_row: Mapping[str, Any],
    *,
    contract_id: str,
    episode: int,
    window: int,
) -> JsonDict:
    declared = _mapping(schedule_row.get("seeds"), "episode seeds")
    if set(declared) != {"channel", "traffic"}:
        raise AgenticCampaignError(
            "each episode must declare exactly channel and traffic seeds"
        )
    channel = _integer(declared["channel"], "channel seed")
    traffic = _integer(declared["traffic"], "traffic seed")
    coordinate = "episode-%d-window-%d" % (episode, window)
    return {
        "master": derive_seed(channel, contract_id, coordinate, "master"),
        "data": derive_seed(traffic, contract_id, coordinate, "data"),
        "channel_state": derive_seed(
            channel, contract_id, coordinate, "channel_state"
        ),
        "csi_observation": derive_seed(
            channel, contract_id, coordinate, "csi_observation"
        ),
        "wireless_channel": derive_seed(
            channel, contract_id, coordinate, "wireless_channel"
        ),
    }


def _comparator_decision(action: Mapping[str, Any], source: str) -> JsonDict:
    effective = copy.deepcopy(dict(action))
    return {
        "requested_action": copy.deepcopy(effective),
        "effective_action": effective,
        "validation_status": "comparator",
        "failure_class": None,
        "failure_message": None,
        "fallback_used": False,
        "raw_response": None,
        "raw_response_sha256": None,
        "provider": {"kind": source, "model": source + "-v1"},
        "usage": {},
        "tool_call_count": 0,
        "finish_reason": "deterministic",
        "cost": {
            "decision_latency_seconds": 0.0,
            "backend_reported_latency_seconds": 0.0,
            "model_call_count": 0,
            "tool_call_count": 0,
            "usage": {},
            "failure_class": None,
            "fallback_used": False,
        },
    }


def _materialize_recipe(
    base_recipe: Recipe,
    action: Mapping[str, Any],
    run_seeds: Mapping[str, int],
    *,
    arm_id: str,
    episode: int,
    window: int,
    registry: Any,
) -> Recipe:
    payload = base_recipe.to_dict()
    safe_arm = _SAFE_ID.sub("_", arm_id)
    payload["name"] = "%s_agentic_%s_e%d_w%d" % (
        base_recipe.name,
        safe_arm,
        episode,
        window,
    )
    metadata = dict(payload.get("metadata") or {})
    for key in ("matrix", "sweeps", "ui_sweeps", "matrix_selection", "matrix_variant_id", "matrix_index"):
        metadata.pop(key, None)
    required_seeds = {
        "master",
        "data",
        "channel_state",
        "csi_observation",
        "wireless_channel",
    }
    if set(run_seeds) != required_seeds:
        raise AgenticCampaignError("run seed plan is incomplete")
    metadata["seed"] = int(run_seeds["master"])
    metadata["seed_namespace"] = "agentic-paired-e%d-w%d" % (episode, window)
    metadata["power_allocator_policy"] = _runtime_policy(str(action["policy"]))
    metadata["average_tx_power_budget"] = float(action["average_power_budget"])
    metadata["agentic_supervision"] = {
        "arm_id": arm_id,
        "episode": episode,
        "window": window,
        "run_seeds": {name: int(value) for name, value in sorted(run_seeds.items())},
    }
    payload["metadata"] = metadata

    allocator_count = 0
    channel_state_count = 0
    for step in payload.get("steps", []):
        params = dict(step.get("params") or {})
        if "seed" in params:
            if step.get("op") == "source.random_bits":
                params["seed"] = int(run_seeds["data"])
            elif step.get("op") == "wireless.ofdm_channel_state":
                params["seed"] = int(run_seeds["channel_state"])
            elif step.get("op") == "wireless.ofdm_delayed_csi":
                params["seed"] = int(run_seeds["csi_observation"])
            elif step.get("op") == "wireless.channel":
                params["seed"] = int(run_seeds["wireless_channel"])
            else:
                params["seed"] = derive_seed(
                    int(run_seeds["master"]), base_recipe.name, str(step["id"])
                )
        if step.get("op") in {"model.causal_csi_power_allocator", "model.symbol_power_allocator"}:
            allocator_count += 1
            params["policy"] = _runtime_policy(str(action["policy"]))
            params["target_power"] = float(action["average_power_budget"])
        if step.get("op") == "wireless.ofdm_channel_state":
            channel_state_count += 1
            params["average_power_budget"] = float(action["average_power_budget"])
        step["params"] = params
    if allocator_count != 1:
        raise AgenticCampaignError("base recipe must contain exactly one supported allocator step")
    if channel_state_count < 1:
        raise AgenticCampaignError("base recipe must contain an OFDM channel-state step")
    return compile_recipe(payload, mode="strict", registry=registry).require_recipe()


def _normalize_action(value: Any) -> JsonDict:
    if value is None:
        raise AgenticCampaignError("action is missing")
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif not isinstance(value, Mapping) and hasattr(value, "tool"):
        value = {
            "tool": getattr(value, "tool"),
            "arguments": {
                "policy": getattr(value, "policy", None),
                "power_budget": getattr(value, "power_budget", None),
            },
        }
    action = _mapping(value, "action")
    if "kind" in action:
        kind = str(action.get("kind"))
        if kind == "keep":
            if set(action) != {"kind"}:
                raise AgenticCampaignError("keep action may not contain extra fields")
            return {"kind": "keep"}
        if set(action) != {"kind", "policy", "average_power_budget"}:
            raise AgenticCampaignError("configure_allocator action has unexpected fields")
        return {
            "kind": kind,
            "policy": str(action.get("policy")),
            "average_power_budget": float(action.get("average_power_budget")),
        }
    if set(action) != {"tool", "arguments"}:
        raise AgenticCampaignError("tool action has unexpected fields")
    tool = str(action.get("tool"))
    arguments = _mapping(action.get("arguments"), "action.arguments")
    if tool == "keep":
        if arguments:
            raise AgenticCampaignError("keep tool arguments must be empty")
        return {"kind": "keep"}
    if set(arguments) != {"policy", "power_budget"}:
        raise AgenticCampaignError("configure_allocator requires policy and power_budget")
    return {
        "kind": "configure_allocator" if tool == "configure_allocator" else tool,
        "policy": str(arguments.get("policy")),
        "average_power_budget": float(arguments.get("power_budget")),
    }


def _validate_canonical_action(action: Mapping[str, Any], contract: Mapping[str, Any]) -> JsonDict:
    normalized = _normalize_action(action)
    if normalized["kind"] == "keep":
        return normalized
    if normalized["kind"] != "configure_allocator":
        raise AgenticCampaignError("unknown action kind %s" % normalized["kind"])
    actions = _mapping(contract.get("actions"), "actions")
    policies = [str(item) for item in actions.get("policies", [])]
    policy = _runtime_policy(str(normalized["policy"]))
    allowed_runtime = [_runtime_policy(item) for item in policies]
    if policy not in allowed_runtime:
        raise AgenticCampaignError("policy %s is outside the action contract" % policy)
    budgets = [float(item) for item in actions.get("allowed_power_budgets", [])]
    budget = float(normalized["average_power_budget"])
    if not math.isfinite(budget) or budget not in budgets:
        raise AgenticCampaignError("power budget is outside the action contract")
    return {
        "kind": "configure_allocator",
        "policy": policy,
        "average_power_budget": budget,
    }


def _fallback_action(
    contract: Mapping[str, Any], current_action: Optional[Mapping[str, Any]] = None
) -> JsonDict:
    fallback = _mapping(contract.get("fallback"), "fallback")
    action_value = fallback.get("action")
    if isinstance(action_value, str) and action_value == "hold_last_valid":
        if current_action is None:
            raise AgenticCampaignError("hold_last_valid needs an initial fallback action")
        return copy.deepcopy(dict(current_action))
    action = _validate_canonical_action(_normalize_action(action_value), contract)
    if action["kind"] == "keep":
        if current_action is None:
            raise AgenticCampaignError("initial fallback cannot be keep")
        return copy.deepcopy(dict(current_action))
    return action


def _campaign_arms(contract: Mapping[str, Any]) -> List[JsonDict]:
    arms: List[JsonDict] = [{"id": "agent", "kind": "agent"}]
    comparators = _mapping(contract.get("comparators"), "comparators")
    for index, value in enumerate(comparators.get("policy_actions", [])):
        action = _validate_canonical_action(_normalize_action(value), contract)
        if action["kind"] == "keep":
            raise AgenticCampaignError("static comparator action cannot be keep")
        arms.append(
            {
                "id": "static_%02d_%s_p%s"
                % (index, action["policy"], format(action["average_power_budget"], "g")),
                "kind": "static",
                "action": action,
            }
        )
    rule = comparators.get("rule_based")
    if rule is not None:
        rule_map = _mapping(rule, "comparators.rule_based")
        # Validate both branches before any expensive run starts.
        _validate_canonical_action(_normalize_action(rule_map.get("if_true")), contract)
        _validate_canonical_action(_normalize_action(rule_map.get("if_false")), contract)
        arms.append({"id": "rule_based", "kind": "rule", "rule": copy.deepcopy(rule_map)})
    return arms


def _rule_action(
    rule: Mapping[str, Any],
    observation: Mapping[str, Any],
    current_action: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> JsonDict:
    path = str(rule.get("observation"))
    current: Any = observation
    found = True
    for component in path.split("."):
        if isinstance(current, list):
            if not current:
                found = False
                break
            current = current[-1]
        if not isinstance(current, Mapping) or component not in current:
            found = False
            break
        current = current[component]
    value = current if found else None
    if value is None:
        return _validate_canonical_action(
            _normalize_action(rule.get("if_true")), contract
        )
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise AgenticCampaignError(
            "rule comparator observation must be finite and numeric when present"
        )
    threshold = float(rule.get("threshold"))
    operator = str(rule.get("operator"))
    comparisons = {
        ">": float(value) > threshold,
        ">=": float(value) >= threshold,
        "<": float(value) < threshold,
        "<=": float(value) <= threshold,
        "==": float(value) == threshold,
        "!=": float(value) != threshold,
    }
    if operator not in comparisons:
        raise AgenticCampaignError("unsupported rule comparator %s" % operator)
    branch = rule.get("if_true" if comparisons[operator] else "if_false")
    action = _validate_canonical_action(_normalize_action(branch), contract)
    if action["kind"] == "keep":
        return copy.deepcopy(dict(current_action))
    return action


def _episode_schedule(episodes: Mapping[str, Any]) -> List[JsonDict]:
    count = _integer(episodes.get("count"), "episodes.count")
    raw = episodes.get("seed_schedule")
    if not isinstance(raw, list) or len(raw) != count:
        raise AgenticCampaignError("episodes.seed_schedule must cover every episode")
    rows = [_mapping(item, "episode seed schedule") for item in raw]
    if sorted(int(row.get("episode", -1)) for row in rows) != list(range(count)):
        raise AgenticCampaignError("episode seed indices must be contiguous from zero")
    return sorted((copy.deepcopy(row) for row in rows), key=lambda row: int(row["episode"]))


def _verify_campaign_topology(
    events: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    public_context: Mapping[str, Any],
) -> None:
    """Bind every executed run to one expected decision and paired seed plan."""

    arms = _campaign_arms(contract)
    episodes = _mapping(contract.get("episodes"), "episodes")
    schedule = _episode_schedule(episodes)
    windows = _integer(
        episodes.get("decisions_per_episode"),
        "episodes.decisions_per_episode",
    )
    expected_count = 2 + len(arms) * len(schedule) * (1 + 2 * windows)
    if len(events) != expected_count:
        raise AgenticCampaignError(
            "campaign event topology has %d events; expected %d"
            % (len(events), expected_count)
        )
    cursor = 0

    def take(expected_type: str) -> Mapping[str, Any]:
        nonlocal cursor
        if cursor >= len(events):
            raise AgenticCampaignError(
                "campaign trace ended before %s" % expected_type
            )
        event = events[cursor]
        cursor += 1
        if event.get("event") != expected_type:
            raise AgenticCampaignError(
                "campaign event %d must be %s" % (cursor - 1, expected_type)
            )
        return event

    started = take("campaign_started")
    expected_arm_ids = [str(arm["id"]) for arm in arms]
    if started.get("arm_ids") != expected_arm_ids:
        raise AgenticCampaignError("campaign start event has an unexpected arm plan")

    run_ids: set[str] = set()
    for arm in arms:
        arm_id = str(arm["id"])
        arm_kind = str(arm["kind"])
        for schedule_row in schedule:
            episode = int(schedule_row["episode"])
            reset = take("episode_reset")
            if (
                reset.get("arm_id") != arm_id
                or reset.get("arm_kind") != arm_kind
                or reset.get("episode") != episode
            ):
                raise AgenticCampaignError("episode reset does not match the declared arm plan")
            current_action = _fallback_action(contract)
            reset_action = _validate_canonical_action(
                _mapping(reset.get("initial_action"), "episode initial action"),
                contract,
            )
            if reset_action != current_action:
                raise AgenticCampaignError("episode reset action differs from the contract fallback")
            episode_history: List[JsonDict] = []

            for window in range(windows):
                decision = take("decision")
                if (
                    decision.get("arm_id") != arm_id
                    or decision.get("arm_kind") != arm_kind
                    or decision.get("episode") != episode
                    or decision.get("decision") != window
                ):
                    raise AgenticCampaignError("decision coordinates do not match the arm plan")
                observation = _mapping(
                    decision.get("observation"), "decision observation"
                )
                if (
                    observation.get("episode_index") != episode
                    or observation.get("decision_index") != window
                ):
                    raise AgenticCampaignError(
                        "decision observation coordinates do not match its event"
                    )
                expected_observation = _build_observation(
                    contract,
                    arm_id,
                    episode,
                    window,
                    episode_history,
                    public_context,
                )
                timestamp_field = str(
                    _mapping(contract.get("observations"), "observations").get(
                        "timestamp_field"
                    )
                )
                expected_observation[timestamp_field] = observation.get(
                    timestamp_field
                )
                if expected_observation != observation:
                    raise AgenticCampaignError(
                        "decision observation does not match completed-run feedback"
                    )
                effective = _validate_canonical_action(
                    _mapping(decision.get("effective_action"), "effective action"),
                    contract,
                )
                if arm_kind == "static":
                    expected_action = _validate_canonical_action(
                        _mapping(arm.get("action"), "static comparator action"),
                        contract,
                    )
                    if effective != expected_action:
                        raise AgenticCampaignError(
                            "static comparator departed from its declared action"
                        )
                elif arm_kind == "rule":
                    expected_action = _rule_action(
                        _mapping(arm.get("rule"), "rule comparator"),
                        observation,
                        current_action,
                        contract,
                    )
                    if effective != expected_action:
                        raise AgenticCampaignError(
                            "rule comparator action does not follow its declared rule"
                        )
                else:
                    fallback_used = bool(decision.get("fallback_used"))
                    status = decision.get("validation_status")
                    if fallback_used:
                        if status != "fallback" or effective != _fallback_action(
                            contract,
                            current_action=current_action,
                        ):
                            raise AgenticCampaignError(
                                "agent fallback decision does not match the contract"
                            )
                    elif status != "accepted" or decision.get("failure_class") is not None:
                        raise AgenticCampaignError(
                            "accepted agent decision has inconsistent validation evidence"
                        )

                run = take("run_completed")
                if (
                    run.get("arm_id") != arm_id
                    or run.get("arm_kind") != arm_kind
                    or run.get("episode") != episode
                    or run.get("window") != window
                    or run.get("phase") != "controlled"
                ):
                    raise AgenticCampaignError("run coordinates do not match its decision")
                run_action = _validate_canonical_action(
                    _mapping(run.get("action"), "run action"),
                    contract,
                )
                if run_action != effective:
                    raise AgenticCampaignError(
                        "executed run action differs from its preceding decision"
                    )
                expected_seeds = _derive_run_seeds(
                    schedule_row,
                    contract_id=str(contract["id"]),
                    episode=episode,
                    window=window,
                )
                if _mapping(run.get("run_seeds"), "run seeds") != expected_seeds:
                    raise AgenticCampaignError(
                        "executed run seeds differ from the paired seed schedule"
                    )
                run_id = str(run.get("run_id") or "")
                if not run_id or run_id in run_ids:
                    raise AgenticCampaignError("run ids must be nonempty and unique")
                run_ids.add(run_id)
                episode_history.append(
                    {
                        "window": window,
                        "run_id": run_id,
                        "started_at_utc": str(run.get("started_at_utc")),
                        "completed_at_utc": str(run.get("completed_at_utc")),
                        "action": copy.deepcopy(effective),
                        "metrics": copy.deepcopy(
                            _mapping(run.get("metrics"), "run metrics")
                        ),
                        "action_valid": decision.get("validation_status")
                        in {"accepted", "comparator"},
                        "fallback_used": bool(decision.get("fallback_used")),
                        "transmitter_csi": copy.deepcopy(
                            _mapping(
                                run.get("transmitter_csi"),
                                "run transmitter CSI",
                            )
                        ),
                    }
                )
                current_action = effective

    completed = take("campaign_completed")
    expected_runs = len(arms) * len(schedule) * windows
    if (
        completed.get("run_count") != expected_runs
        or completed.get("decision_count") != expected_runs
    ):
        raise AgenticCampaignError("campaign completion counts do not match the plan")
    if cursor != len(events):
        raise AgenticCampaignError("campaign trace contains unexpected trailing events")


def _arm_summaries(
    events: Sequence[Mapping[str, Any]],
    allowlist: Sequence[str],
    contract: Mapping[str, Any],
) -> List[JsonDict]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    decisions: Dict[str, List[Mapping[str, Any]]] = {}
    kinds: Dict[str, str] = {}
    for event in events:
        if event.get("event") == "decision":
            arm_id = str(event["arm_id"])
            decisions.setdefault(arm_id, []).append(event)
            kinds[arm_id] = str(event["arm_kind"])
            continue
        if event.get("event") != "run_completed" or event.get("phase") != "controlled":
            continue
        arm_id = str(event["arm_id"])
        grouped.setdefault(arm_id, []).append(event)
        kinds[arm_id] = str(event["arm_kind"])
    result: List[JsonDict] = []
    for arm_id in sorted(grouped):
        rows = grouped[arm_id]
        means: JsonDict = {}
        for metric in allowlist:
            values = [
                float(_mapping(row["metrics"], "run metrics")[metric])
                for row in rows
                if isinstance(_mapping(row["metrics"], "run metrics").get(metric), (int, float))
                and not isinstance(_mapping(row["metrics"], "run metrics").get(metric), bool)
            ]
            means[metric] = sum(values) / len(values) if values else None
        violations: List[JsonDict] = []
        aggregate_constraints: List[JsonDict] = []
        objective = _mapping(contract.get("objective"), "objective")
        for raw_constraint in objective.get("constraints", []):
            constraint = _mapping(raw_constraint, "objective constraint")
            metric = str(constraint.get("metric"))
            operator = str(constraint.get("operator"))
            threshold = float(constraint.get("value"))
            values = [
                float(_mapping(row["metrics"], "run metrics")[metric])
                for row in rows
                if isinstance(
                    _mapping(row["metrics"], "run metrics").get(metric),
                    (int, float),
                )
                and not isinstance(
                    _mapping(row["metrics"], "run metrics").get(metric), bool
                )
            ]
            count = sum(
                not _comparison_holds(value, operator, threshold) for value in values
            )
            violations.append(
                {
                    "metric": metric,
                    "operator": operator,
                    "threshold": threshold,
                    "evaluated_run_count": len(values),
                    "violation_count": count,
                    "violation_rate": count / len(values) if values else None,
                }
            )
            observed = means.get(metric)
            aggregate_constraints.append(
                {
                    "metric": metric,
                    "operator": operator,
                    "threshold": threshold,
                    "aggregation": str(objective.get("aggregation")),
                    "observed": observed,
                    "satisfied": (
                        _comparison_holds(float(observed), operator, threshold)
                        if isinstance(observed, (int, float))
                        and not isinstance(observed, bool)
                        else False
                    ),
                }
            )
        arm_decisions = decisions.get(arm_id, [])
        latencies = [
            float(_mapping(row.get("cost"), "decision cost").get("decision_latency_seconds") or 0.0)
            for row in arm_decisions
        ]
        result.append(
            {
                "arm_id": arm_id,
                "arm_kind": kinds[arm_id],
                "controlled_run_count": len(rows),
                "mean_metrics": means,
                "constraint_violations": violations,
                "objective_evaluation": {
                    "aggregation": str(objective.get("aggregation")),
                    "primary_metric": str(objective.get("primary_metric")),
                    "direction": str(objective.get("direction")),
                    "primary_value": means.get(str(objective.get("primary_metric"))),
                    "feasible": all(
                        bool(item["satisfied"]) for item in aggregate_constraints
                    ),
                    "aggregate_constraints": aggregate_constraints,
                    "tie_breakers": [
                        {
                            "metric": str(
                                _mapping(raw, "objective tie breaker").get("metric")
                            ),
                            "direction": str(
                                _mapping(raw, "objective tie breaker").get("direction")
                            ),
                            "value": means.get(
                                str(
                                    _mapping(
                                        raw,
                                        "objective tie breaker",
                                    ).get("metric")
                                )
                            ),
                        }
                        for raw in objective.get("tie_breakers", [])
                    ],
                },
                "decision_costs": {
                    "decision_count": len(arm_decisions),
                    "mean_latency_seconds": (
                        sum(latencies) / len(latencies) if latencies else None
                    ),
                    "fallback_count": sum(
                        bool(row.get("fallback_used")) for row in arm_decisions
                    ),
                    "timeout_count": sum(
                        row.get("failure_class") == "timeout" for row in arm_decisions
                    ),
                    "invalid_action_count": sum(
                        row.get("failure_class") == "invalid_action"
                        for row in arm_decisions
                    ),
                    "backend_error_count": sum(
                        row.get("failure_class") == "backend_error"
                        for row in arm_decisions
                    ),
                },
            }
        )
    return result


def _comparison_holds(value: float, operator: str, threshold: float) -> bool:
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
    raise AgenticCampaignError("unsupported objective constraint operator")


def _objective_ranking(
    arm_summaries: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> JsonDict:
    objective = _mapping(contract.get("objective"), "objective")
    ordering = [
        {
            "metric": str(objective.get("primary_metric")),
            "direction": str(objective.get("direction")),
        }
    ]
    ordering.extend(
        {
            "metric": str(_mapping(item, "objective tie breaker").get("metric")),
            "direction": str(
                _mapping(item, "objective tie breaker").get("direction")
            ),
        }
        for item in objective.get("tie_breakers", [])
    )

    feasible: List[Mapping[str, Any]] = []
    infeasible: List[str] = []
    for arm in arm_summaries:
        evaluation = _mapping(
            arm.get("objective_evaluation"), "arm objective evaluation"
        )
        if bool(evaluation.get("feasible")):
            feasible.append(arm)
        else:
            infeasible.append(str(arm.get("arm_id")))

    def ranking_key(arm: Mapping[str, Any]) -> Tuple[float, ...]:
        means = _mapping(arm.get("mean_metrics"), "arm mean metrics")
        values: List[float] = []
        for criterion in ordering:
            raw = means.get(criterion["metric"])
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise AgenticCampaignError(
                    "objective ranking metric is missing from an arm summary"
                )
            value = float(raw)
            values.append(value if criterion["direction"] == "minimize" else -value)
        return tuple(values)

    ordered = sorted(feasible, key=lambda arm: (ranking_key(arm), str(arm.get("arm_id"))))
    feasible_order: List[JsonDict] = []
    previous_key: Optional[Tuple[float, ...]] = None
    current_rank = 0
    for index, arm in enumerate(ordered):
        key = ranking_key(arm)
        if previous_key is None or key != previous_key:
            current_rank = index + 1
        feasible_order.append(
            {"rank": current_rank, "arm_id": str(arm.get("arm_id"))}
        )
        previous_key = key
    return {
        "aggregation": str(objective.get("aggregation")),
        "ordering": ordering,
        "tie_policy": "shared_rank; arm_id sorts tied rows for serialization only",
        "feasible_order": feasible_order,
        "infeasible_arms": sorted(infeasible),
    }


def _cost_summary(costs: Sequence[Mapping[str, Any]]) -> JsonDict:
    latencies = [float(row["decision_latency_seconds"]) for row in costs]
    return {
        "decision_count": len(costs),
        "fallback_count": sum(bool(row.get("fallback_used")) for row in costs),
        "timeout_count": sum(row.get("failure_class") == "timeout" for row in costs),
        "invalid_action_count": sum(row.get("failure_class") == "invalid_action" for row in costs),
        "backend_error_count": sum(row.get("failure_class") == "backend_error" for row in costs),
        "tool_call_count": sum(int(row.get("tool_call_count", 0)) for row in costs),
        "provider_call_count": sum(
            int(row.get("provider_call_count", 0)) for row in costs
        ),
        "model_call_count": sum(int(row.get("model_call_count", 0)) for row in costs),
        "mean_decision_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "max_decision_latency_seconds": max(latencies) if latencies else None,
    }


def _read_event_chain(path: Path) -> List[JsonDict]:
    events: List[JsonDict] = []
    previous: Optional[str] = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise AgenticCampaignError("blank line in event trace at %d" % line_number)
        try:
            value = decode_strict_yaml_or_json(line, input_format="json")
        except Exception as exc:
            raise AgenticCampaignError("invalid event JSON at line %d" % line_number) from exc
        event = _mapping(value, "event line %d" % line_number)
        event = verify_content_bound_document(event, "event line %d" % line_number)
        if event.get("event_index") != line_number - 1:
            raise AgenticCampaignError("event index mismatch at line %d" % line_number)
        if event.get("previous_event_sha256") != previous:
            raise AgenticCampaignError("event hash chain mismatch at line %d" % line_number)
        previous = str(event["sha256"])
        events.append(event)
    return events


def _write_events(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s-" % path.name, dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise AgenticCampaignError("refusing to replace symlink event trace")
        os.replace(str(temporary), str(path))
    finally:
        temporary.unlink(missing_ok=True)


def _new_evidence_dir(
    workspace: Path,
    identifier: str,
    digest: str,
    *,
    out: Optional[Path] = None,
) -> Path:
    if out is not None:
        candidate = Path(out)
        if candidate.exists() or candidate.is_symlink():
            raise AgenticCampaignError("campaign output path must not already exist")
        parent = candidate.parent
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise AgenticCampaignError("campaign output parent must be a real directory")
        try:
            candidate.mkdir()
        except OSError as exc:
            raise AgenticCampaignError("cannot create campaign output directory") from exc
        return candidate
    root = Path(workspace) / "agentic"
    if root.is_symlink():
        raise AgenticCampaignError("workspace agentic directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = _SAFE_ID.sub("_", identifier).strip("._-") or "campaign"
    stem = "%s_%s_%s" % (timestamp, safe[:80], digest[:12])
    for index in range(1, 10000):
        name = stem if index == 1 else "%s_%d" % (stem, index)
        candidate = root / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise AgenticCampaignError("could not reserve a unique campaign directory")


def _file_binding(path: Path, relative: str) -> JsonDict:
    return {
        "path": relative,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _copy_evidence_file(source: Path, destination: Path) -> None:
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise AgenticCampaignError("run evidence source must be a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise AgenticCampaignError("refusing to overwrite retained run evidence")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s-" % destination.name,
        dir=str(destination.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(str(temporary), str(destination))
    finally:
        temporary.unlink(missing_ok=True)


def _compress_evidence_file(source: Path, destination: Path) -> None:
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise AgenticCampaignError("run evidence source must be a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise AgenticCampaignError("refusing to overwrite retained run evidence")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s-" % destination.name,
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with (
            source.open("rb") as input_handle,
            os.fdopen(descriptor, "wb") as output_handle,
            gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=output_handle,
                mtime=0,
            ) as compressed,
        ):
            shutil.copyfileobj(input_handle, compressed, length=1024 * 1024)
        os.replace(str(temporary), str(destination))
    finally:
        temporary.unlink(missing_ok=True)


def _load_compressed_json_evidence(
    path: Path,
    label: str,
) -> Tuple[JsonDict, str, int]:
    try:
        with gzip.open(path, "rb") as handle:
            raw = handle.read(_MAX_RETAINED_RUN_JSON_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise AgenticCampaignError("cannot read %s" % label) from exc
    if len(raw) > _MAX_RETAINED_RUN_JSON_BYTES:
        raise AgenticCampaignError("%s exceeds the retained evidence limit" % label)
    try:
        value = decode_strict_yaml_or_json(
            raw.decode("utf-8", errors="strict"),
            input_format="json",
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise AgenticCampaignError("%s is not strict UTF-8 JSON" % label) from exc
    return (
        _mapping(value, label),
        hashlib.sha256(raw).hexdigest(),
        len(raw),
    )


def _retain_run_evidence(
    campaign_dir: Path,
    run_dir: Path,
    ordinal: int,
) -> Tuple[JsonDict, Dict[str, JsonDict]]:
    retained: Dict[str, JsonDict] = {}
    references: JsonDict = {}
    for name in ("summary", "manifest"):
        source = Path(run_dir) / (name + ".json")
        relative = "runs/%04d/%s.json.gz" % (ordinal, name)
        destination = Path(campaign_dir) / relative
        _compress_evidence_file(source, destination)
        key = "run_%04d_%s" % (ordinal, name)
        retained[key] = _file_binding(destination, relative)
        references[name] = key
    return references, retained


def _resolve_base_recipe(contract: Mapping[str, Any], root: Path, contract_path: Path) -> Path:
    base = _mapping(contract.get("base_recipe"), "base_recipe")
    raw = Path(str(base.get("path")))
    candidates = [raw] if raw.is_absolute() else [root / raw, contract_path.parent / raw]
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            expected = str(base.get("sha256") or "")
            if expected and file_sha256(candidate) != expected:
                raise AgenticCampaignError("base recipe SHA-256 does not match contract")
            return candidate.resolve()
    raise AgenticCampaignError("base recipe does not exist: %s" % raw)


def _project_root(contract_path: Path, supplied: Optional[Path]) -> Path:
    if supplied is not None:
        return Path(supplied).resolve()
    start = contract_path.resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def _build_backend(
    provider_config: Mapping[str, Any],
    *,
    scripted_actions: Any = None,
    replay_actions: Any = None,
) -> Any:
    from noema_lab.agentic.backends import build_backend

    return build_backend(
        provider_config,
        scripted_actions=scripted_actions,
        replay_actions=replay_actions,
    )


def _runtime_policy(policy: str) -> str:
    return _POLICY_ALIASES.get(policy, policy)


def _json_mapping(value: Any, label: str) -> JsonDict:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise AgenticCampaignError("%s must be a mapping" % label)
    result = _json_safe(dict(value))
    if not isinstance(result, dict):
        raise AgenticCampaignError("%s must be a JSON mapping" % label)
    return result


def _json_mapping_or_empty(value: Any) -> JsonDict:
    if value is None:
        return {}
    return _json_mapping(value, "backend evidence")


def _mapping(value: Any, label: str) -> JsonDict:
    if not isinstance(value, Mapping):
        raise AgenticCampaignError("%s must be a mapping" % label)
    return dict(value)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AgenticCampaignError("%s must be a nonnegative integer" % label)
    return int(value)


def _json_safe(value: Any) -> Any:
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError):
        return str(value)
