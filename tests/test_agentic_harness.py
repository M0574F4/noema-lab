from __future__ import annotations

import json
import io
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from noema_lab.agentic import harness
from noema_lab.agentic.backends import build_backend
from noema_lab.agentic.contracts import load_agentic_contract
from noema_lab.core.artifacts import file_sha256
from noema_lab.core.recipes import compile_recipe, load_recipe
from noema_lab.core.reproducibility import canonical_json_sha256, utc_now_iso
from noema_lab.cli.main import build_parser, main
from noema_lab.ops import build_registry


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "examples" / "agentic_allocation" / "contract.yaml"
RECIPE_PATH = ROOT / "recipes" / "resource_delayed_csi_finite_blocklength.yaml"


def _contract() -> dict:
    return load_agentic_contract(
        CONTRACT_PATH,
        project_root=ROOT,
        verify_base_recipe=True,
    ).to_dict()


def _history_row() -> dict:
    return {
        "window": 0,
        "run_id": "completed-run",
        "started_at_utc": "2026-01-01T00:00:00.000Z",
        "completed_at_utc": "2026-01-01T00:00:01.000Z",
        "action": {
            "kind": "configure_allocator",
            "policy": "fixed",
            "average_power_budget": 0.8,
        },
        "metrics": {
            "resource.finite_blocklength.predicted_bler": 0.2,
            "resource.finite_blocklength.expected_goodput_bps_hz": 1.6,
            "resource.finite_blocklength.p05_goodput_bps_hz": 0.9,
            "resource.average_transmit_power_budget": 0.8,
            "resource.power_constraint.max_abs_error": 0.0,
            "resource.power_constraint.max_relative_error": 0.0,
            "resource.power_constraint.max_negative_violation": 0.0,
        },
        "action_valid": True,
        "fallback_used": False,
        "transmitter_csi": {
            "oldest_observation_ofdm_symbol_index": 0,
            "newest_observation_ofdm_symbol_index": 26,
            "newest_observation_nominal_time_s": 0.0017,
            "mean_observed_gain": 1.0,
            "p10_observed_gain": 0.1,
            "p50_observed_gain": 0.7,
            "p90_observed_gain": 2.2,
            "mean_history_delta_magnitude": 0.16,
        },
    }


class AgenticHarnessUnitTests(unittest.TestCase):
    def test_feedback_metrics_must_be_finite_numbers(self) -> None:
        metrics = dict(_history_row()["metrics"])
        metrics["resource.finite_blocklength.predicted_bler"] = "0.2"
        summary = {
            "steps": [
                {"id": "allocation_evaluation", "metrics": metrics},
            ]
        }
        with self.assertRaisesRegex(
            harness.AgenticCampaignError,
            "finite and numeric",
        ):
            harness._extract_feedback_metrics(summary)

    def test_agentic_cli_parser_and_validate_dispatch(self) -> None:
        parsed = build_parser().parse_args(
            [
                "agentic",
                "run",
                str(CONTRACT_PATH),
                "--provider",
                "ollama",
                "--model",
                "qwen-test",
                "--base-url",
                "http://127.0.0.1:11434",
            ]
        )
        self.assertEqual(parsed.agentic_command, "run")
        self.assertEqual(parsed.provider, "ollama")

        output = io.StringIO()
        with mock.patch("noema_lab.cli.main.build_registry"), redirect_stdout(output):
            status = main(
                ["agentic", "validate", str(CONTRACT_PATH), "--json"]
            )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "valid")

    def test_agentic_replay_mismatch_is_a_cli_failure(self) -> None:
        output = io.StringIO()
        failed = {
            "status": "failed",
            "replay_dir": "/tmp/fixture-replay",
            "replayed_run_count": 1,
            "mismatch_count": 1,
        }
        with (
            mock.patch("noema_lab.cli.main.build_registry"),
            mock.patch(
                "noema_lab.agentic.harness.replay_agentic_campaign",
                return_value=failed,
            ),
            redirect_stdout(output),
        ):
            status = main(["agentic", "replay", "/tmp/fixture", "--json"])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "failed")

    def test_model_and_prompt_overrides_are_part_of_effective_contract(self) -> None:
        contract = _contract()
        effective = harness._apply_run_overrides(
            contract,
            provider="transformers",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            model_revision="7ae557604adf67be50417f59c2c2f167def9a775",
            endpoint=None,
            base_url=None,
            prompt=ROOT / "examples" / "agentic_allocation" / "prompt.md",
            device="cpu",
        )
        self.assertEqual(effective["provider"]["kind"], "transformers")
        self.assertEqual(
            effective["provider"]["model_revision"],
            "7ae557604adf67be50417f59c2c2f167def9a775",
        )
        self.assertIn("{{ observation_json }}", effective["prompts"]["user"])
        self.assertNotEqual(
            canonical_json_sha256(effective), canonical_json_sha256(contract)
        )

    def test_openai_compatible_override_supports_explicit_or_no_auth(self) -> None:
        contract = _contract()
        common = {
            "provider": "openai_compatible",
            "model": "local-compatible-model",
            "model_revision": None,
            "endpoint": None,
            "base_url": "http://127.0.0.1:8080/v1",
            "prompt": None,
            "device": None,
        }
        no_auth = harness._apply_run_overrides(contract, **common)
        self.assertNotIn("api_key_env", no_auth["provider"])
        authenticated = harness._apply_run_overrides(
            contract,
            api_key_env="NOEMA_AGENT_API_KEY",
            **common,
        )
        self.assertEqual(
            authenticated["provider"]["api_key_env"],
            "NOEMA_AGENT_API_KEY",
        )

    def test_observation_is_exactly_allowlisted_and_contains_no_seed_or_truth(self) -> None:
        contract = _contract()
        recipe = load_recipe(RECIPE_PATH)
        observation = harness._build_observation(
            contract,
            "agent",
            0,
            1,
            [_history_row()],
            harness._recipe_public_context(recipe),
        )

        harness._verify_observation(
            observation,
            contract["observations"]["allowlist"],
        )
        encoded = json.dumps(observation, sort_keys=True)
        self.assertNotIn("seed", encoded.lower())
        self.assertNotIn("actual_state", encoded.lower())
        self.assertNotIn("current_channel", encoded.lower())
        self.assertEqual(observation["available_after_run_id"], "completed-run")
        self.assertEqual(observation["recent"][0]["predicted_bler"], 0.2)

    def test_seed_plan_is_paired_across_arms_and_uses_traffic_seed(self) -> None:
        contract = _contract()
        schedule = contract["episodes"]["seed_schedule"][0]
        seeds = harness._derive_run_seeds(
            schedule,
            contract_id=contract["id"],
            episode=0,
            window=2,
        )
        recipe = load_recipe(RECIPE_PATH)
        action = {
            "kind": "configure_allocator",
            "policy": "robust_csi_water_filling",
            "average_power_budget": 0.6,
        }
        registry = build_registry()
        first = harness._materialize_recipe(
            recipe,
            action,
            seeds,
            arm_id="agent",
            episode=0,
            window=2,
            registry=registry,
        )
        second = harness._materialize_recipe(
            recipe,
            action,
            seeds,
            arm_id="static",
            episode=0,
            window=2,
            registry=registry,
        )

        first_steps = {step.id: step for step in first.steps}
        second_steps = {step.id: step for step in second.steps}
        for step_id in ("data", "channel_state", "csi_observation", "wireless_channel"):
            self.assertEqual(
                first_steps[step_id].params["seed"],
                second_steps[step_id].params["seed"],
            )
        self.assertEqual(first_steps["data"].params["seed"], seeds["data"])
        self.assertEqual(
            first_steps["channel_state"].params["seed"], seeds["channel_state"]
        )
        self.assertEqual(first_steps["tx_power"].params["target_power"], 0.6)
        self.assertEqual(first_steps["tx_power"].params["policy"], "robust_csi_water_filling")
        self.assertNotIn("matrix", first.metadata)
        self.assertEqual(first.metadata["seed_namespace"], second.metadata["seed_namespace"])

    def test_rule_supervisor_reads_only_latest_allowlisted_observation(self) -> None:
        contract = _contract()
        recipe = load_recipe(RECIPE_PATH)
        history = _history_row()
        observation = harness._build_observation(
            contract,
            "rule_based",
            0,
            1,
            [history],
            harness._recipe_public_context(recipe),
        )
        action = harness._rule_action(
            contract["comparators"]["rule_based"],
            observation,
            history["action"],
            contract,
        )
        self.assertEqual(action["policy"], "causal_ar_water_filling")
        self.assertEqual(action["average_power_budget"], 0.8)

        observation["recent"][-1]["predicted_bler"] = 0.4
        action = harness._rule_action(
            contract["comparators"]["rule_based"],
            observation,
            history["action"],
            contract,
        )
        self.assertEqual(action["policy"], "robust_csi_water_filling")
        self.assertEqual(action["average_power_budget"], 1.4)

    def test_objective_ranking_assigns_shared_rank_to_exact_ties(self) -> None:
        contract = _contract()
        metrics = {
            "resource.average_transmit_power_budget": 0.8,
            "resource.finite_blocklength.expected_goodput_bps_hz": 1.2,
        }
        summaries = [
            {
                "arm_id": arm_id,
                "mean_metrics": metrics,
                "objective_evaluation": {"feasible": True},
            }
            for arm_id in ("b", "a")
        ]
        ranking = harness._objective_ranking(summaries, contract)
        self.assertEqual(
            ranking["feasible_order"],
            [{"rank": 1, "arm_id": "a"}, {"rank": 1, "arm_id": "b"}],
        )

    def test_timeout_and_invalid_action_use_declared_fallback(self) -> None:
        contract = _contract()
        contract["limits"]["decision_timeout_seconds"] = 0.005
        observation = harness._build_observation(
            contract,
            "agent",
            0,
            0,
            [],
            harness._recipe_public_context(load_recipe(RECIPE_PATH)),
        )
        current = harness._fallback_action(contract)

        class SlowBackend:
            calls = 0

            def decide(self, request):
                del request
                self.calls += 1
                time.sleep(0.05)
                raise AssertionError("late result must not be accepted")

        slow = SlowBackend()
        deadline_guard = harness._DecisionDeadlineGuard()
        timed_out = harness._agent_decision(
            slow,
            contract,
            observation,
            current,
            deadline_guard=deadline_guard,
        )
        self.assertEqual(timed_out["failure_class"], "timeout")
        self.assertTrue(timed_out["fallback_used"])
        self.assertEqual(timed_out["effective_action"]["average_power_budget"], 1.4)
        quarantined = harness._agent_decision(
            slow,
            contract,
            observation,
            current,
            deadline_guard=deadline_guard,
        )
        self.assertEqual(quarantined["failure_class"], "timeout")
        self.assertEqual(quarantined["cost"]["provider_call_count"], 0)
        self.assertEqual(slow.calls, 1)

        class InvalidBackend:
            def decide(self, request):
                del request
                return SimpleNamespace(
                    action={
                        "tool": "configure_allocator",
                        "arguments": {"policy": "oracle", "power_budget": 99},
                    },
                    raw_response="{}",
                    provider={"kind": "test", "model": "invalid"},
                    usage={},
                    tool_call_count=1,
                    latency_seconds=0.0,
                    finish_reason="stop",
                )

        invalid = harness._agent_decision(
            InvalidBackend(), contract, observation, current
        )
        self.assertEqual(invalid["failure_class"], "invalid_action")
        self.assertTrue(invalid["fallback_used"])


class _FakeExecutor:
    def __init__(self, registry, store) -> None:
        self.registry = registry
        self.store = store

    def run(self, recipe):
        run_dir = self.store.create_run_dir(recipe.name)
        csi_dir = run_dir / "artifacts" / "csi_observation"
        csi_dir.mkdir()
        csi_path = csi_dir / "transmitter_csi.npz"
        gains = np.linspace(0.1, 2.0, 32, dtype=np.float32).reshape(4, 8)
        history = np.zeros((4, 4, 8, 2), dtype=np.float32)
        history[..., 0] = np.arange(4, dtype=np.float32)[None, :, None]
        np.savez_compressed(
            csi_path,
            capture_gains=gains,
            capture_csi_history=history,
        )
        tx_step = next(step for step in recipe.steps if step.id == "tx_power")
        budget = float(tx_step.params["target_power"])
        now = utc_now_iso()
        authored_recipe_sha256 = canonical_json_sha256(recipe.to_dict())
        effective_recipe_sha256 = canonical_json_sha256(
            compile_recipe(
                recipe.to_dict(),
                mode="strict",
                registry=self.registry,
            )
            .require_recipe(effective=True)
            .to_dict()
        )
        metrics = {
            "resource.finite_blocklength.predicted_bler": max(0.0, 0.5 - 0.2 * budget),
            "resource.finite_blocklength.expected_goodput_bps_hz": 1.0 + budget,
            "resource.finite_blocklength.p05_goodput_bps_hz": 0.5 + budget,
            "resource.average_transmit_power_budget": budget,
            "resource.power_constraint.max_abs_error": 0.0,
            "resource.power_constraint.max_relative_error": 0.0,
            "resource.power_constraint.max_negative_violation": 0.0,
        }
        summary = {
            "status": "completed",
            "run_id": run_dir.name,
            "created_at_utc": now,
            "completed_at_utc": now,
            "authored_recipe_sha256": authored_recipe_sha256,
            "effective_recipe_sha256": effective_recipe_sha256,
            "recipe_sha256": effective_recipe_sha256,
            "steps": [
                {
                    "id": "csi_observation",
                    "outputs": {
                        "transmitter_csi": {
                            "kind": "channel.ofdm_channel_state.numpy",
                            "path": str(csi_path),
                            "sha256": file_sha256(csi_path),
                            "metadata": {
                                "csi_history_length": 4,
                                "allocation_ofdm_symbols": 24,
                                "subcarrier_spacing_khz": 15.0,
                            },
                        }
                    },
                    "metrics": {},
                },
                {
                    "id": "allocation_evaluation",
                    "outputs": {},
                    "metrics": metrics,
                },
            ],
        }
        self.store.write_json(run_dir / "summary.json", summary)
        self.store.write_json(
            run_dir / "manifest.json",
            {
                "status": "completed",
                "run_id": run_dir.name,
                "summary": {
                    "sha256": file_sha256(run_dir / "summary.json"),
                    "size_bytes": (run_dir / "summary.json").stat().st_size,
                },
                "recipe": {
                    "authored_sha256": authored_recipe_sha256,
                    "effective_sha256": effective_recipe_sha256,
                },
            },
        )
        return run_dir


class AgenticHarnessCampaignTests(unittest.TestCase):
    def test_mock_campaign_is_content_bound_and_verifies(self) -> None:
        contract = _contract()
        contract["id"] = "agentic-harness-test"
        contract["episodes"]["count"] = 1
        contract["episodes"]["seed_schedule"] = contract["episodes"]["seed_schedule"][:1]
        recipe = load_recipe(RECIPE_PATH)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            harness, "LocalExecutor", _FakeExecutor
        ):
            result = harness._run_campaign(
                contract=contract,
                contract_sha256=canonical_json_sha256(contract),
                base_recipe=recipe,
                base_recipe_path=RECIPE_PATH,
                workspace=Path(tmp),
                backend=build_backend(contract["provider"]),
                out=None,
            )
            report = harness.verify_agentic_campaign(Path(result["campaign_dir"]))
            replay = harness.replay_agentic_campaign(
                Path(result["campaign_dir"]),
                Path(tmp) / "replay-workspace",
            )

            self.assertEqual(result["run_count"], 12)
            self.assertEqual(result["decision_count"], 12)
            self.assertEqual(result["objective_ranking"]["aggregation"], "mean")
            self.assertTrue(
                all(
                    "objective_evaluation" in arm
                    for arm in result["arm_summaries"]
                )
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["run_count"], 12)
            self.assertEqual(replay["status"], "passed")
            self.assertEqual(replay["replayed_run_count"], 12)
            self.assertEqual(replay["mismatch_count"], 0)

            events = harness._read_event_chain(
                Path(result["campaign_dir"]) / "events.jsonl"
            )
            run_event = next(
                event for event in events if event.get("event") == "run_completed"
            )
            run_event["action"] = {
                "kind": "configure_allocator",
                "policy": "fixed",
                "average_power_budget": 0.6,
            }
            with self.assertRaisesRegex(
                harness.AgenticCampaignError,
                "run action differs",
            ):
                harness._verify_campaign_topology(
                    events,
                    contract,
                    public_context=harness._recipe_public_context(recipe),
                )

            events = harness._read_event_chain(
                Path(result["campaign_dir"]) / "events.jsonl"
            )
            feedback_event = next(
                event
                for event in events
                if event.get("event") == "decision"
                and event.get("arm_id") == "agent"
                and event.get("decision") == 1
            )
            feedback_event["observation"]["recent"][-1][
                "predicted_bler"
            ] = 0.999
            with self.assertRaisesRegex(
                harness.AgenticCampaignError,
                "observation does not match",
            ):
                harness._verify_campaign_topology(
                    events,
                    contract,
                    public_context=harness._recipe_public_context(recipe),
                )


if __name__ == "__main__":
    unittest.main()
