"""Normative minimum traceability profile for canonical benchmark results.

Canonical packs that request the strongest local verification bind this
profile's identifier and canonical digest. Authors may add study-specific
checks, but cannot remove the minimum set while retaining the current
traceability-contract state. The legacy ``publication_ready`` metadata key is
kept for stored-pack compatibility; it is not a venue or publishability claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Mapping


JsonDict = Dict[str, Any]
PUBLICATION_VERIFICATION_PROFILE_ID = "noema.traceability.v2"
LEGACY_PUBLICATION_VERIFICATION_PROFILE_IDS = ("noema.publication.v1",)
TRACEABILITY_PROFILE_REQUEST_FIELD = "traceability_profile_requested"
LEGACY_TRACEABILITY_PROFILE_REQUEST_FIELD = "publication_ready"

_PUBLICATION_VERIFICATION_PROFILE: JsonDict = {
    "schema_version": 2,
    "id": PUBLICATION_VERIFICATION_PROFILE_ID,
    "acceptance": {
        "rule": "all_applicable_required_predicates",
        "author_may_remove_required_predicates": False,
        "failed_predicate": "reject_claimed_identity",
        "legitimate_condition_change": (
            "admit_only_under_new_protocol_or_plan_identity"
        ),
        "profile_change": "requires_new_profile_id_and_digest",
    },
    "applicability": {
        "always_required": ["I1", "I2", "I3", "I5", "I6"],
        "conditionally_required": {
            "I4": {
                "when": (
                    "benchmark declares or result contains an imported or "
                    "returned trained artifact"
                ),
                "otherwise": "not_applicable_with_recorded_reason",
            }
        },
    },
    "predicates": {
        "I1": {
            "name": "protocol_identity",
            "depends_on": [],
            "required_relations": [
                "strict schema and semantic-kind admission succeeds",
                "canonical protocol and dataset digests equal their records",
                "authored recipe, effective recipe, and plan digests equal their records",
            ],
            "validator_modules": [
                "noema_lab.core.benchmarks",
                "noema_lab.core.benchmark_run_evidence",
                "noema_lab.core.verification",
            ],
            "conformance_tests": [
                "tests.test_benchmark_validity",
                "tests.test_execution_plan_verification",
                "tests.test_strict_evidence_inputs",
            ],
        },
        "I2": {
            "name": "accounting_identity",
            "depends_on": ["I1", "I3"],
            "required_relations": [
                "typed payload, framed, coded, and symbol boundaries resolve to bound producers",
                "declared count and resource equalities hold",
                "failure policy and attempted-item denominator equalities hold",
            ],
            "validator_modules": [
                "noema_lab.core.lint",
                "noema_lab.core.benchmarks",
                "noema_lab.core.verification",
            ],
            "conformance_tests": [
                "tests.test_semantic_accounting_kinds",
                "tests.test_verification_task_rate_accounting",
                "tests.test_publication_dataset_conditions",
            ],
        },
        "I3": {
            "name": "materialization_identity",
            "depends_on": ["I1"],
            "required_relations": [
                "exactly one concrete materialization and artifact tuple resolves per step",
                "the pre-side-effect execution-plan digest equals its record",
            ],
            "validator_modules": [
                "noema_lab.core.planner",
                "noema_lab.core.executor",
                "noema_lab.core.verification",
            ],
            "conformance_tests": [
                "tests.test_execution_planner",
                "tests.test_execution_plan_verification",
                "tests.test_core_contract_hardening",
            ],
        },
        "I4": {
            "name": "return_compatibility",
            "depends_on": ["I1", "I3"],
            "required_relations": [
                "manifest, package, return-contract, and component digests equal their records",
                "the named entrypoint and tensor ABI satisfy the unchanged protocol",
                "training/evaluation lineage obligations required by the protocol hold",
            ],
            "validator_modules": [
                "noema_lab.core.trained_artifacts",
                "noema_lab.core.benchmark_evidence",
                "noema_lab.core.verification",
            ],
            "conformance_tests": [
                "tests.test_trained_artifact_v2",
                "tests.test_runtime_artifact_strict_inputs",
                "tests.test_differentiable_export",
            ],
        },
        "I5": {
            "name": "evidence_identity",
            "depends_on": ["I1", "I2", "I3", "I4_if_applicable"],
            "required_relations": [
                "every reported cell resolves to a plan and terminal attempt",
                "every reported cell resolves to seeds, artifacts, and an authoritative metric",
                "per-example and plotted-data projections are complete and content bound",
            ],
            "validator_modules": [
                "noema_lab.core.attempt_ledger",
                "noema_lab.core.benchmark_plots",
                "noema_lab.core.verification",
            ],
            "conformance_tests": [
                "tests.test_attempt_ledger",
                "tests.test_benchmark_metric_provenance",
                "tests.test_benchmark_plot_verification",
            ],
        },
        "I6": {
            "name": "bounded_verification",
            "depends_on": ["I1", "I2", "I3", "I4_if_applicable", "I5"],
            "required_relations": [
                "every applicable predecessor predicate passes under the declared trust assumptions",
                "parse, digest, completion, and cross-object failures reject the claimed identity",
            ],
            "validator_modules": [
                "noema_lab.core.structured_input",
                "noema_lab.core.trained_artifacts",
                "noema_lab.core.verification",
            ],
            "conformance_tests": [
                "tests.test_strict_evidence_inputs",
                "tests.test_reproducibility_fail_closed",
                "tests.test_trained_artifact_v2",
            ],
        },
    },
    "scope": {
        "guarantee": (
            "complete implemented evidence path under declared trust "
            "assumptions and applicable predicates"
        ),
        "does_not_guarantee": [
            "scientific validity or fairness",
            "producer authenticity",
            "sandboxing of trusted in-process adapters",
            "external PHY or implementation correctness",
        ],
    },
}

# Stable verifier-side semantics for turning the normative I1--I6 profile into
# evidence-bearing predicate verdicts.  The v2 profile digest covers this
# mapping, and verification reports also expose its narrower digest so readers
# can identify the exact check-to-predicate interpretation directly.
_PUBLICATION_PREDICATE_CHECK_SEMANTICS: JsonDict = {
    "schema_version": 1,
    "profile_id": PUBLICATION_VERIFICATION_PROFILE_ID,
    "acceptance": {
        "applicable_status_required": "pass",
        "warning_status": "non_failing",
        "missing_required_pass_group": "fail",
        "dependency_failure": "fail",
        "unassigned_verifier_error": "fail_via_I6",
    },
    "predicates": {
        "I1": {
            "name": "protocol_identity",
            "applicability": "always",
            "check_ids": [
                "structure.benchmark_dir",
                "schema.benchmark_result_json",
                "schema.benchmark_pack_json",
                "benchmark.id",
                "benchmark.version",
                "benchmark.status",
            ],
            "check_id_prefixes": [
                "schema.benchmark.",
                "benchmark.protocol.",
            ],
            "required_pass_groups": [
                {
                    "id": "result_document",
                    "any_of": ["schema.benchmark_result_json"],
                },
                {
                    "id": "frozen_protocol_document",
                    "any_of": ["schema.benchmark_pack_json"],
                },
                {
                    "id": "protocol_digest",
                    "any_of": ["benchmark.protocol.sha256"],
                },
                {
                    "id": "certification_state",
                    "any_of": ["benchmark.protocol.certification_state"],
                },
                {
                    "id": "profile_binding",
                    "any_of": ["benchmark.protocol.verification_profile"],
                },
                {
                    "id": "protocol_relations",
                    "any_of": ["benchmark.protocol.identity"],
                },
                {
                    "id": "completed_result",
                    "any_of": ["benchmark.status"],
                },
            ],
            "failure_policy": "any_selected_error",
        },
        "I2": {
            "name": "accounting_identity",
            "applicability": "always",
            "check_ids": [],
            "check_id_prefixes": [
                "benchmark.resource_admission",
                "benchmark.common_conditions",
                "accounting.",
            ],
            "required_pass_groups": [
                {
                    "id": "resource_admission",
                    "any_of": ["benchmark.resource_admission"],
                },
                {
                    "id": "common_condition_and_failure_denominator",
                    "any_of": ["benchmark.common_conditions"],
                },
            ],
            "failure_policy": "any_selected_error",
        },
        "I3": {
            "name": "materialization_identity",
            "applicability": "always",
            "check_ids": [],
            "check_id_prefixes": [
                "benchmark.run_evidence_snapshot",
                "benchmark.backing_run",
            ],
            "required_pass_groups": [
                {
                    "id": "result_local_execution_evidence",
                    "any_of": ["benchmark.run_evidence_snapshot"],
                }
            ],
            "failure_policy": "any_selected_error",
        },
        "I4": {
            "name": "return_compatibility",
            "applicability": "trained_artifact_declared",
            "check_ids": [],
            "check_id_prefixes": ["benchmark.training_"],
            "required_pass_groups": [
                {
                    "id": "training_and_return_snapshot",
                    "any_of": ["benchmark.training_evidence_snapshot"],
                }
            ],
            "failure_policy": "any_selected_error",
        },
        "I5": {
            "name": "evidence_identity",
            "applicability": "always",
            "check_ids": [
                "schema.benchmark_plot_sidecar_json",
                "benchmark.metrics.plausibility",
                "benchmark.required_metrics",
                "benchmark.metric_provenance",
                "benchmark.expected_outputs",
                "benchmark.declared_plot_outputs",
            ],
            "check_id_prefixes": [
                "benchmark.attempt_ledger",
                "benchmark.recipe",
                "benchmark.reports.",
                "benchmark.plot",
            ],
            "required_pass_groups": [
                {
                    "id": "terminal_result_identity",
                    "any_of": ["benchmark.attempt_ledger.result_identity"],
                },
                {
                    "id": "terminal_recipe_outcomes",
                    "any_of": ["benchmark.attempt_ledger.outcomes"],
                },
                {
                    "id": "required_metrics",
                    "any_of": ["benchmark.required_metrics"],
                },
                {
                    "id": "authoritative_metric_producers",
                    "any_of": ["benchmark.metric_provenance"],
                },
                {
                    "id": "metrics_projection",
                    "any_of": ["benchmark.reports.metrics_csv"],
                },
                {
                    "id": "recipe_projection",
                    "any_of": ["benchmark.reports.recipes_csv"],
                },
                {
                    "id": "summary_projection",
                    "any_of": ["benchmark.reports.summary_markdown"],
                },
                {
                    "id": "declared_output_closure",
                    "any_of": ["benchmark.expected_outputs"],
                },
            ],
            "conditional_required_pass_groups": [
                {
                    "id": "declared_plot_closure",
                    "when": "declared_plot_outputs",
                    "any_of": ["benchmark.declared_plot_outputs"],
                }
            ],
            "failure_policy": "any_selected_error",
        },
        "I6": {
            "name": "bounded_verification",
            "applicability": "always",
            "check_ids": [],
            "check_id_prefixes": [],
            "required_pass_groups": [],
            "error_scope": "all_verifier_checks",
            "failure_policy": "any_verifier_error_or_dependency_failure",
        },
    },
}

# The v2 profile digest covers the exact runtime predicate/check mapping.  This
# turns the I1--I6 labels into versioned verifier semantics rather than prose
# labels whose implementation could drift without changing the claimed
# profile identity.
_PUBLICATION_VERIFICATION_PROFILE["predicate_check_semantics"] = copy.deepcopy(
    _PUBLICATION_PREDICATE_CHECK_SEMANTICS
)


def traceability_profile_requested(
    state: Mapping[str, Any],
    *,
    context: str = "metadata",
) -> bool:
    """Resolve the strongest-profile trigger across its canonical and legacy names.

    ``traceability_profile_requested`` says what the bit actually controls:
    application of Noema's strongest local traceability profile.  The former
    ``publication_ready`` spelling is accepted only for stored-pack and stored-
    result compatibility; it does not assert scientific validity, external
    reproduction, legal clearance, or venue acceptance.

    Both fields must be booleans when present.  A document that carries both
    spellings is accepted only when their values agree, so the compatibility
    alias cannot silently weaken or strengthen the requested profile.
    """

    values: Dict[str, bool] = {}
    for field in (
        TRACEABILITY_PROFILE_REQUEST_FIELD,
        LEGACY_TRACEABILITY_PROFILE_REQUEST_FIELD,
    ):
        if field not in state:
            continue
        value = state[field]
        if not isinstance(value, bool):
            raise ValueError("%s.%s must be a boolean" % (context, field))
        values[field] = value

    if (
        TRACEABILITY_PROFILE_REQUEST_FIELD in values
        and LEGACY_TRACEABILITY_PROFILE_REQUEST_FIELD in values
        and values[TRACEABILITY_PROFILE_REQUEST_FIELD]
        != values[LEGACY_TRACEABILITY_PROFILE_REQUEST_FIELD]
    ):
        raise ValueError(
            "%s.%s conflicts with deprecated alias %s.%s"
            % (
                context,
                TRACEABILITY_PROFILE_REQUEST_FIELD,
                context,
                LEGACY_TRACEABILITY_PROFILE_REQUEST_FIELD,
            )
        )
    if TRACEABILITY_PROFILE_REQUEST_FIELD in values:
        return values[TRACEABILITY_PROFILE_REQUEST_FIELD]
    return values.get(LEGACY_TRACEABILITY_PROFILE_REQUEST_FIELD, False)


def publication_verification_profile() -> JsonDict:
    """Return an isolated JSON-compatible copy of the normative profile."""

    return copy.deepcopy(_PUBLICATION_VERIFICATION_PROFILE)


def publication_verification_profile_sha256() -> str:
    """Return the canonical SHA-256 of the normative profile."""

    canonical = json.dumps(
        _PUBLICATION_VERIFICATION_PROFILE,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def publication_verification_profile_binding() -> JsonDict:
    """Return the exact binding required by canonical traceability metadata."""

    return {
        "id": PUBLICATION_VERIFICATION_PROFILE_ID,
        "sha256": publication_verification_profile_sha256(),
    }


def publication_predicate_check_semantics() -> JsonDict:
    """Return the verifier-owned mapping from I1--I6 to concrete checks."""

    return copy.deepcopy(_PUBLICATION_PREDICATE_CHECK_SEMANTICS)


def publication_predicate_check_semantics_sha256() -> str:
    """Return the canonical identity of the verifier predicate semantics."""

    canonical = json.dumps(
        _PUBLICATION_PREDICATE_CHECK_SEMANTICS,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def publication_predicate_check_semantics_binding() -> JsonDict:
    """Return the exact verifier semantics binding emitted in certificates."""

    return {
        "profile_id": PUBLICATION_VERIFICATION_PROFILE_ID,
        "schema_version": _PUBLICATION_PREDICATE_CHECK_SEMANTICS[
            "schema_version"
        ],
        "sha256": publication_predicate_check_semantics_sha256(),
    }
