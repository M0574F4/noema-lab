from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from noema_lab.agentic.backends import (
    DecisionRequest,
    OllamaBackend,
    OpenAICompatibleBackend,
    RuleBasedBackend,
    ScriptedBackend,
    TransformersBackend,
)
from noema_lab.agentic.contracts import (
    AgenticContractError,
    agentic_contract_from_mapping,
    load_agentic_contract,
    parse_action_envelope,
    provider_config_from_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "agentic_allocation" / "contract.yaml"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class AgenticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_agentic_contract(EXAMPLE, project_root=ROOT)

    def test_tutorial_contract_loads_and_recipe_hash_is_verified(self) -> None:
        self.assertEqual(self.contract.kind, "noema.agentic_supervisory_contract")
        self.assertEqual(self.contract.actions.tool, "configure_allocator")
        self.assertEqual(self.contract.objective.aggregation, "mean")
        self.assertEqual(
            self.contract.objective.tie_breakers[0].metric,
            "resource.finite_blocklength.expected_goodput_bps_hz",
        )
        self.assertEqual(len(self.contract.sha256), 64)
        self.assertEqual(self.contract.base_recipe.path, "../../recipes/resource_delayed_csi_finite_blocklength.yaml")

    def test_unknown_contract_field_is_rejected(self) -> None:
        payload = self.contract.to_dict()
        payload["surprise"] = True
        with self.assertRaisesRegex(AgenticContractError, "unknown keys"):
            agentic_contract_from_mapping(payload)

    def test_hidden_observation_field_is_rejected(self) -> None:
        payload = self.contract.to_dict()
        payload["observations"]["allowlist"].append("current_channel.gain")
        with self.assertRaisesRegex(AgenticContractError, "unsupported or hidden"):
            agentic_contract_from_mapping(payload)

    def test_rule_comparator_requires_a_numeric_observation(self) -> None:
        payload = self.contract.to_dict()
        payload["comparators"]["rule_based"]["observation"] = (
            "observation_timestamp_utc"
        )
        with self.assertRaisesRegex(AgenticContractError, "numeric observation"):
            agentic_contract_from_mapping(payload)

    def test_invalid_action_and_extra_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(AgenticContractError, "not in allowed_power_budgets"):
            parse_action_envelope(
                {
                    "tool": "configure_allocator",
                    "arguments": {"policy": "fixed", "power_budget": 999.0},
                },
                self.contract.actions,
            )
        with self.assertRaisesRegex(AgenticContractError, "unknown keys"):
            parse_action_envelope(
                {
                    "tool": "configure_allocator",
                    "arguments": {
                        "policy": "fixed",
                        "power_budget": 0.4,
                        "hidden": "future CSI",
                    },
                },
                self.contract.actions,
            )

    def test_duplicate_json_action_key_is_rejected(self) -> None:
        raw = '{"tool":"configure_allocator","tool":"other","arguments":{"policy":"fixed","power_budget":0.4}}'
        with self.assertRaisesRegex(AgenticContractError, "strict JSON"):
            parse_action_envelope(raw, self.contract.actions)

    def test_recipe_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = root / "recipe.yaml"
            recipe.write_text("name: fixture\n", encoding="utf-8")
            payload = self.contract.to_dict()
            payload["base_recipe"] = {"path": "recipe.yaml", "sha256": "0" * 64}
            with self.assertRaisesRegex(AgenticContractError, "SHA-256 mismatch"):
                agentic_contract_from_mapping(
                    payload,
                    source_path=root / "contract.yaml",
                    project_root=root,
                    verify_base_recipe=True,
                )

    def test_provider_secret_value_is_not_a_contract_field(self) -> None:
        with self.assertRaisesRegex(AgenticContractError, "unknown keys"):
            provider_config_from_mapping(
                {
                    "kind": "openai_compatible",
                    "model": "fixture",
                    "base_url": "https://models.example/v1",
                    "api_key": "must-not-be-stored",
                }
            )

    def test_provider_cannot_name_an_arbitrary_environment_secret(self) -> None:
        with self.assertRaisesRegex(AgenticContractError, "NOEMA_AGENT_ namespace"):
            provider_config_from_mapping(
                {
                    "kind": "openai_compatible",
                    "model": "fixture",
                    "base_url": "https://models.example/v1",
                    "api_key_env": "AWS_SECRET_ACCESS_KEY",
                }
            )

    def test_transformers_revision_must_be_an_immutable_digest(self) -> None:
        with self.assertRaisesRegex(AgenticContractError, "commit digest"):
            provider_config_from_mapping(
                {
                    "kind": "transformers",
                    "model": "fixture",
                    "model_revision": "main",
                }
            )


class AgenticBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_agentic_contract(EXAMPLE, project_root=ROOT)

    def request(self, *, observation=None) -> DecisionRequest:
        return DecisionRequest(
            decision_id="episode-0-decision-0",
            observation=observation
            or {
                "observation_timestamp_utc": "2026-01-01T00:00:00Z",
                "decision_index": 0,
            },
            action_contract=self.contract.actions,
            system_prompt=self.contract.prompts.system,
            user_prompt=self.contract.prompts.user,
            objective=self.contract.objective.to_dict(),
            timeout_seconds=5.0,
            max_output_tokens=80,
        )

    def test_scripted_backend_cycles_declared_actions(self) -> None:
        backend = ScriptedBackend(
            [
                {"tool": "configure_allocator", "arguments": {"policy": "fixed", "power_budget": 0.4}},
                {"tool": "configure_allocator", "arguments": {"policy": "robust_csi_water_filling", "power_budget": 1.0}},
            ]
        )
        policies = [backend.decide(self.request()).action.policy for _ in range(3)]
        self.assertEqual(policies, ["fixed", "robust_csi_water_filling", "fixed"])

    def test_implicit_scripted_cycle_uses_policy_budget_grid(self) -> None:
        backend = ScriptedBackend()
        first = backend.decide(self.request()).action
        second = backend.decide(self.request()).action
        self.assertEqual(first.policy, self.contract.actions.policies[0])
        self.assertEqual(first.power_budget, self.contract.actions.allowed_power_budgets[0])
        self.assertEqual(second.power_budget, self.contract.actions.allowed_power_budgets[1])

    def test_prompt_placeholders_are_filled_with_declared_runtime_values(self) -> None:
        request = DecisionRequest(
            decision_id="placeholder-test",
            observation={"observation_timestamp_utc": "2026-01-01T00:00:00Z"},
            action_contract=self.contract.actions,
            system_prompt="system",
            user_prompt=(
                "objective={{ objective_json }} actions={{ allowed_actions_json }} "
                "observation={{ observation_json }}"
            ),
            objective=self.contract.objective.to_dict(),
        )
        rendered = request.rendered_user_prompt()
        self.assertNotIn("{{", rendered)
        self.assertIn('"value":0.25', rendered)
        self.assertIn('"allowed_power_budgets"', rendered)

    def test_rule_backend_uses_conservative_true_branch_when_history_missing(self) -> None:
        backend = RuleBasedBackend(self.contract.comparators.rule_based)
        response = backend.decide(self.request())
        self.assertEqual(response.action, self.contract.comparators.rule_based.if_true)

    def test_rule_backend_uses_latest_completed_history(self) -> None:
        backend = RuleBasedBackend(self.contract.comparators.rule_based)
        high_bler = self.request(
            observation={
                "observation_timestamp_utc": "2026-01-01T00:00:00Z",
                "recent": [{"predicted_bler": 0.1}, {"predicted_bler": 0.4}],
            }
        )
        low_bler = self.request(
            observation={
                "observation_timestamp_utc": "2026-01-01T00:00:01Z",
                "recent": [{"predicted_bler": 0.1}],
            }
        )
        self.assertEqual(
            backend.decide(high_bler).action,
            self.contract.comparators.rule_based.if_true,
        )
        self.assertEqual(
            backend.decide(low_bler).action,
            self.contract.comparators.rule_based.if_false,
        )

    def test_openai_compatible_tool_call_is_normalized_without_secret(self) -> None:
        captured = {}

        def opener(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            body = {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "configure_allocator",
                                        "arguments": json.dumps(
                                            {"policy": "fixed", "power_budget": 0.4}
                                        ),
                                    }
                                }
                            ]
                        },
                    }
                ],
                "model": "fixture-model-2026-09-01",
                "system_fingerprint": "fixture-fingerprint",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            return _Response(json.dumps(body).encode("utf-8"))

        provider = provider_config_from_mapping(
            {
                "kind": "openai_compatible",
                "model": "fixture-model",
                "base_url": "https://models.example/v1",
                "api_key_env": "NOEMA_AGENT_FIXTURE_KEY",
                "temperature": 0.0,
                "seed": 7,
            }
        )
        response = OpenAICompatibleBackend(
            provider,
            opener=opener,
            environ={"NOEMA_AGENT_FIXTURE_KEY": "top-secret"},
        ).decide(self.request())
        self.assertEqual(response.action.policy, "fixed")
        self.assertEqual(response.usage["total_tokens"], 15)
        self.assertEqual(
            response.provider["response_model"], "fixture-model-2026-09-01"
        )
        self.assertEqual(
            response.provider["system_fingerprint"], "fixture-fingerprint"
        )
        self.assertNotIn("top-secret", json.dumps(response.to_dict()))
        sent = json.loads(captured["request"].data.decode("utf-8"))
        runtime = sent["messages"][1]["content"]
        self.assertIn('"objective"', runtime)
        self.assertIn('"allowed_actions"', runtime)

    def test_ollama_tool_call_is_normalized(self) -> None:
        def opener(request, timeout):
            body = {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "configure_allocator",
                                "arguments": {
                                    "policy": "causal_ar_water_filling",
                                    "power_budget": 0.8,
                                },
                            }
                        }
                    ]
                },
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 4,
            }
            return _Response(json.dumps(body).encode("utf-8"))

        provider = provider_config_from_mapping(
            {
                "kind": "ollama",
                "model": "qwen-fixture",
                "base_url": "http://127.0.0.1:11434",
                "temperature": 0.0,
                "seed": 0,
            }
        )
        response = OllamaBackend(provider, opener=opener).decide(self.request())
        self.assertEqual(response.action.policy, "causal_ar_water_filling")
        self.assertEqual(response.usage["total_tokens"], 16)

    def test_injected_transformers_backend_needs_no_optional_dependency(self) -> None:
        provider = provider_config_from_mapping(
            {
                "kind": "transformers",
                "model": "tiny-local-fixture",
                "model_revision": "0123456789abcdef0123456789abcdef01234567",
                "device": "cpu",
                "local_files_only": True,
                "temperature": 0.0,
                "seed": 0,
            }
        )

        def generator(prompt):
            self.assertIn("Allowed action schema", prompt)
            return {
                "tool": "configure_allocator",
                "arguments": {
                    "policy": "observed_csi_water_filling",
                    "power_budget": 0.6,
                },
            }

        response = TransformersBackend(provider, generator=generator).decide(self.request())
        self.assertEqual(response.action.power_budget, 0.6)
        self.assertEqual(response.provider["transformers_version"], "injected")

    def test_transformers_normalizes_common_single_tool_json_wrapper(self) -> None:
        provider = provider_config_from_mapping(
            {
                "kind": "transformers",
                "model": "tiny-local-fixture",
                "model_revision": "0123456789abcdef0123456789abcdef01234567",
                "device": "cpu",
                "local_files_only": True,
                "temperature": 0.0,
                "seed": 0,
            }
        )
        response = TransformersBackend(
            provider,
            generator=lambda prompt: (
                '```json\n{"parameters":{"policy":"robust_csi_water_filling",'
                '"power_budget":1.0}}\n```'
            ),
        ).decide(self.request())
        self.assertEqual(response.action.policy, "robust_csi_water_filling")
        self.assertEqual(response.action.power_budget, 1.0)


if __name__ == "__main__":
    unittest.main()
