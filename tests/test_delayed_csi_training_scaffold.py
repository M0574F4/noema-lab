from __future__ import annotations

import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from noema_lab.core.benchmarks import (
    _collect_summary_metrics_with_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD = (
    ROOT
    / "demo_trainings"
    / "resource_allocation_delayed_csi_finite_blocklength"
)


class DelayedCsiTrainingScaffoldTests(unittest.TestCase):
    def test_paired_cluster_evidence_requires_practical_statistical_gain(self):
        module = runpy.run_path(str(SCAFFOLD / "datamodule.py"))
        evidence = module["paired_cluster_evidence"]
        cluster_ids = [
            "trajectory-%02d" % (index % 30) for index in range(60)
        ]
        points = [
            {"average_power_budget": 0.4, "noise_variance": 0.2},
            {"average_power_budget": 0.8, "noise_variance": 0.2},
        ]
        equal = np.ones((2, 60), dtype=np.float64)
        robust = np.full((2, 60), 0.98, dtype=np.float64)
        observed = np.full((2, 60), 0.9, dtype=np.float64)
        common = {
            "baseline_goodput_by_method": {
                "equal_power": equal,
                "robust_csi_water_filling": robust,
                "observed_csi_water_filling": observed,
            },
            "cluster_ids": cluster_ids,
            "cluster_method": (
                "time_major_ofdm_symbol_rows_clustered_by_independent_tdl_block"
            ),
            "operating_points": points,
        }

        passed = evidence(np.full((2, 60), 1.02), **common)
        self.assertEqual(passed["status"], "passed")
        self.assertEqual(
            passed["strongest_deployable_baseline"]["method_id"],
            "equal_power",
        )
        self.assertEqual(
            passed["paired_cluster_confidence_interval"]["cluster_count"],
            30,
        )
        self.assertGreater(
            passed["paired_cluster_confidence_interval"]["lower_bps_hz"],
            0.0,
        )

        too_small = evidence(np.full((2, 60), 1.001), **common)
        self.assertEqual(too_small["status"], "insufficient_evidence")
        self.assertFalse(
            too_small["checks"]["practical_relative_improvement"]
        )

        regressed_point = evidence(
            np.stack(
                (
                    np.full(60, 1.02),
                    np.full(60, 0.995),
                )
            ),
            **common,
        )
        self.assertEqual(regressed_point["status"], "insufficient_evidence")
        self.assertFalse(
            regressed_point["checks"]["bounded_operating_point_regression"]
        )

    def test_paired_cluster_evidence_rejects_uncertain_and_unclustered_gain(self):
        module = runpy.run_path(str(SCAFFOLD / "datamodule.py"))
        evidence = module["paired_cluster_evidence"]
        cluster_ids = [
            "trajectory-%02d" % (index % 30) for index in range(60)
        ]
        baseline = np.ones((1, 60), dtype=np.float64)
        cluster_effects = np.asarray(
            [0.05 if index % 2 else -0.038 for index in range(30)],
            dtype=np.float64,
        )
        candidate = np.asarray(
            [
                1.0
                + np.asarray(
                    [cluster_effects[index % 30] for index in range(60)]
                )
            ]
        )
        uncertain = evidence(
            candidate,
            {"equal_power": baseline},
            cluster_ids=cluster_ids,
            cluster_method=(
                "time_major_ofdm_symbol_rows_clustered_by_independent_tdl_block"
            ),
            operating_points=[
                {"average_power_budget": 0.8, "noise_variance": 0.2}
            ],
            maximum_relative_point_regression=0.05,
        )
        self.assertGreater(uncertain["relative_improvement"], 0.005)
        self.assertFalse(
            uncertain["checks"]["paired_cluster_ci_lower_bound_positive"]
        )

        unclustered = evidence(
            np.full((1, 60), 1.02),
            {"equal_power": baseline},
            cluster_ids=["row-%d" % index for index in range(60)],
            cluster_method="independent_row_fallback_missing_run_metadata",
            operating_points=[
                {"average_power_budget": 0.8, "noise_variance": 0.2}
            ],
        )
        self.assertEqual(unclustered["status"], "insufficient_evidence")
        self.assertFalse(
            unclustered["checks"]["trajectory_cluster_metadata_available"]
        )

    def test_capture_provenance_recovers_independent_tdl_clusters(self):
        module = runpy.run_path(str(SCAFFOLD / "datamodule.py"))
        cluster_ids, method = module["_capture_trajectory_clusters"](
            {
                "runs": [
                    {
                        "captured_samples": 8,
                        "channel_distribution": {
                            "steps": [
                                {
                                    "id": "channel_state",
                                    "metadata": {"ofdm_block_count": 4},
                                }
                            ]
                        },
                    }
                ]
            },
            capture_index=0,
            expected_count=8,
        )
        self.assertIn("clustered_by_independent_tdl_block", method)
        self.assertEqual(cluster_ids[:4], cluster_ids[4:])

    def test_model_enforces_exact_nonnegative_power_budget(self):
        module = runpy.run_path(str(SCAFFOLD / "model.py"))
        model = module["CausalCsiHistoryPowerAllocator"](
            history_length=4,
            hidden_dim=8,
            dilations=(1, 2),
            kernel_size=3,
        )
        history = torch.randn((5, 4, 16, 2), dtype=torch.float32)
        noise = torch.full((5,), 0.2, dtype=torch.float32)
        budget = torch.tensor(
            [0.4, 0.6, 0.8, 1.0, 1.4],
            dtype=torch.float32,
        )
        power = model(history, noise, budget)
        self.assertEqual(tuple(power.shape), (5, 16))
        self.assertGreaterEqual(float(torch.min(power)), 0.0)
        torch.testing.assert_close(
            torch.sum(power, dim=1),
            budget * 16.0,
            atol=3e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            power,
            budget[:, None].expand_as(power),
            atol=3e-6,
            rtol=1e-6,
        )

    def test_model_is_invariant_to_arbitrary_common_carrier_phase(self):
        module = runpy.run_path(str(SCAFFOLD / "model.py"))
        model = module["FrequencyResidualPowerAllocator"](
            history_length=4,
            hidden_dim=8,
            dilations=(1, 2),
            kernel_size=3,
        )
        torch.manual_seed(31)
        with torch.no_grad():
            model.output_projection.weight.normal_(0.0, 0.05)
            model.output_projection.bias.normal_(0.0, 0.05)
        history = torch.randn((3, 4, 16, 2), dtype=torch.float32)
        phase = torch.tensor([0.7, -1.1, 2.2], dtype=torch.float32)
        cosine = torch.cos(phase)[:, None, None]
        sine = torch.sin(phase)[:, None, None]
        rotated = torch.stack(
            (
                history[..., 0] * cosine - history[..., 1] * sine,
                history[..., 0] * sine + history[..., 1] * cosine,
            ),
            dim=-1,
        )
        noise = torch.full((3, 1), 0.2, dtype=torch.float32)
        budget = torch.full((3, 1), 0.8, dtype=torch.float32)
        torch.testing.assert_close(
            model(history, noise, budget),
            model(rotated, noise, budget),
            atol=2e-5,
            rtol=2e-5,
        )

    def test_finite_blocklength_loss_uses_current_outcome_and_backpropagates(self):
        model_module = runpy.run_path(str(SCAFFOLD / "model.py"))
        loss_module = runpy.run_path(str(SCAFFOLD / "losses.py"))
        model = model_module["CausalCsiHistoryPowerAllocator"](
            history_length=4,
            hidden_dim=8,
            dilations=(1,),
            kernel_size=3,
        )
        history = torch.randn((6, 4, 12, 2), dtype=torch.float32)
        current = torch.rand((6, 12), dtype=torch.float32) + 0.1
        noise = torch.full((6,), 0.2, dtype=torch.float32)
        budget = torch.full((6,), 0.8, dtype=torch.float32)
        power = model(history, noise, budget)
        loss = loss_module[
            "negative_expected_finite_blocklength_goodput"
        ](
            power,
            current,
            noise,
            blocklength_channel_uses=128,
            target_rate_bps_hz=2.0,
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.all(torch.isfinite(parameter.grad)))
                for parameter in model.parameters()
            )
        )

    def test_current_csi_numerical_reference_is_feasible_and_improves_seed(self):
        saved = sys.modules.pop("datamodule", None)
        sys.path.insert(0, str(SCAFFOLD))
        try:
            module = runpy.run_path(str(SCAFFOLD / "evaluate.py"))
        finally:
            sys.path.pop(0)
            sys.modules.pop("datamodule", None)
            if saved is not None:
                sys.modules["datamodule"] = saved
        gains = np.asarray(
            [
                [0.2, 0.6, 1.1, 2.0],
                [2.0, 0.3, 0.8, 0.5],
            ],
            dtype=np.float64,
        )
        noise = 0.2
        budget = 0.8
        total_power = budget * gains.shape[1]
        initial = np.stack(
            [
                module["_water_filling"](row, noise, total_power)
                for row in gains
            ],
            axis=0,
        )
        optimized = module[
            "_finite_blocklength_current_csi_numerical_reference"
        ](
            gains,
            noise=noise,
            budget=budget,
            blocklength=128,
            target_rate=2.0,
            third_order=True,
            initial_power=initial,
        )
        seed_score = module["_finite_blocklength_z_score_numpy"](
            gains,
            initial,
            noise=noise,
            blocklength=128,
            target_rate=2.0,
            third_order=True,
        )
        optimized_score = module["_finite_blocklength_z_score_numpy"](
            gains,
            optimized,
            noise=noise,
            blocklength=128,
            target_rate=2.0,
            third_order=True,
        )
        self.assertTrue(bool(np.all(optimized >= 0.0)))
        np.testing.assert_allclose(
            np.sum(optimized, axis=1),
            total_power,
            rtol=1e-9,
            atol=1e-9,
        )
        self.assertTrue(bool(np.all(optimized_score >= seed_score - 1e-12)))

    def test_aligned_pair_hashes_fail_closed_on_cross_split_overlap(self):
        module = runpy.run_path(str(SCAFFOLD / "datamodule.py"))
        load_capture_dataset = module["load_capture_dataset"]
        split_integrity_report = module["split_integrity_report"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_dir = root / "train"
            validation_dir = root / "validation"
            latest_gains = np.asarray(
                [[0.4, 0.7, 1.0], [0.8, 0.3, 1.2]],
                dtype=np.float32,
            )
            history = _complex_csi_history(latest_gains, history_length=4)
            current = np.asarray(
                [[0.5, 0.6, 1.1], [0.7, 0.4, 1.0]],
                dtype=np.float32,
            )
            _write_capture(train_dir, "train", history, current)
            _write_capture(
                validation_dir,
                "validation",
                history[:1],
                current[:1],
            )
            train = load_capture_dataset(
                train_dir,
                delayed_csi_tap="csi_history",
                expected_split="train",
            )
            validation = load_capture_dataset(
                validation_dir,
                delayed_csi_tap="csi_history",
                expected_split="validation",
            )
            with self.assertRaisesRegex(
                ValueError,
                "train and validation share 1 state pair",
            ):
                split_integrity_report(train, validation)

    def test_training_plan_captures_observation_and_aligned_outcome(self):
        plan = yaml.safe_load(
            (SCAFFOLD / "training_plan.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            plan["objective"],
            "resource.negative_expected_finite_blocklength_goodput",
        )
        self.assertEqual(plan["selected_steps"], ["tx_power"])
        self.assertEqual(
            plan["dataset_capture"]["taps"],
            [
                {
                    "id": "csi_history",
                    "from": "csi_observation.transmitter_csi",
                },
                {
                    "id": "current_csi",
                    "from": "csi_observation.actual_state",
                },
            ],
        )

    def test_benchmark_uses_five_deployable_policies_with_paired_seeds(self):
        module = runpy.run_path(str(SCAFFOLD / "build_benchmark.py"))
        pack = module["_build_pack"](
            recipe_reference="noema_recipe.yaml",
            artifact_reference=".noema/trained_artifacts/policy/trained_artifact.yaml",
            artifact_runtime_identity="a" * 64,
            artifact_evidence={"path": "trained_artifact.yaml", "sha256": "b" * 64},
            training_history={"path": "training_history.json", "sha256": "c" * 64},
            evaluation_metrics={"path": "evaluation_metrics.json", "sha256": "d" * 64},
            budgets=[0.4, 0.8],
            seeds=[95101, 95201],
        )
        self.assertEqual(len(pack["recipes"]), 2 * 2 * 5)
        method_ids = {
            recipe["params"]["method_id"] for recipe in pack["recipes"]
        }
        self.assertEqual(
            method_ids,
            {
                "equal_power",
                "observed_csi_water_filling",
                "robust_csi_water_filling",
                "causal_ar_water_filling",
                "learned_allocator",
            },
        )
        self.assertNotIn(
            "current_csi_shannon_diagnostic",
            method_ids,
        )
        self.assertEqual(
            pack["dataset"],
            {
                "id": "synthetic_random_bits_sionna_tdl_delayed_csi",
                "modality": "wireless",
                "version": (
                    "synthetic-random-bits-sionna-tdl-delayed-csi-v2"
                ),
                "split": "held_out_seeded_trajectory",
            },
        )
        self.assertEqual(
            pack["task"],
            {
                "id": "resource_allocation",
                "kind": "policy_optimization",
                "modality": "wireless",
            },
        )
        groups = {}
        for recipe in pack["recipes"]:
            selection = recipe["params"]["matrix_selection"]
            key = (
                selection["resource.average_transmit_power_budget"],
                selection["benchmark.paired_seed"],
            )
            groups.setdefault(key, []).append(recipe)
        for group in groups.values():
            self.assertEqual(len(group), 5)
            metadata = [
                recipe["params"]["metadata"] for recipe in group
            ]
            self.assertEqual(
                {item["pairing_id"] for item in metadata},
                {
                    str(item["benchmark_paired_seed"])
                    for item in metadata
                },
            )
            self.assertEqual(
                len(
                    {
                        item["aggregation_cell_id"]
                        for item in metadata
                    }
                ),
                1,
            )
            self.assertEqual(
                {
                    item["statistical_unit"]
                    for item in metadata
                },
                {"paired held-out TDL trajectory seed"},
            )
            seed_bindings = [
                {
                    step: dict(params)
                    for step, params in recipe["params"]["step_params"].items()
                    if step != "tx_power"
                }
                for recipe in group
            ]
            self.assertTrue(
                all(binding == seed_bindings[0] for binding in seed_bindings[1:])
            )

    def test_benchmark_refuses_insufficient_held_out_evidence(self):
        module = runpy.run_path(str(SCAFFOLD / "build_benchmark.py"))
        with tempfile.TemporaryDirectory() as temporary:
            artifact_path = Path(temporary) / "trained_artifact.yaml"
            artifact_path.write_text("schema_version: 2\n", encoding="utf-8")
            payload = {
                "trained_artifact": {
                    "manifest_sha256": module["_sha256"](artifact_path),
                    "components": [
                        {"id": "policy", "sha256": "a" * 64}
                    ],
                },
                "split_integrity": {"status": "passed"},
                "demo_evidence_status": {
                    "status": "insufficient_evidence"
                },
            }
            with self.assertRaisesRegex(
                ValueError,
                "lacks sufficient evidence",
            ):
                module["_validate_evaluation_binding"](
                    payload,
                    artifact_path=artifact_path,
                    component_sha256="a" * 64,
                )

    def test_benchmark_metrics_bind_authoritative_producers(self):
        module = runpy.run_path(str(SCAFFOLD / "build_benchmark.py"))
        pack = module["_build_pack"](
            recipe_reference="noema_recipe.yaml",
            artifact_reference="trained_artifact.yaml",
            artifact_runtime_identity="a" * 64,
            artifact_evidence={"path": "trained_artifact.yaml", "sha256": "b" * 64},
            training_history={"path": "training_history.json", "sha256": "c" * 64},
            evaluation_metrics={"path": "evaluation_metrics.json", "sha256": "d" * 64},
            budgets=[0.8],
            seeds=[95101],
        )
        metric_sources = {
            metric["id"]: (
                metric["definition_version"],
                metric["source_step"],
                metric["source_operation"],
            )
            for metric in pack["metrics"]
        }
        evaluation_source = (
            1,
            "allocation_evaluation",
            "metrics.ofdm_finite_blocklength_allocation",
        )
        self.assertEqual(
            metric_sources[
                "resource.csi.observed_actual_gain_correlation"
            ],
            (1, "csi_observation", "wireless.ofdm_delayed_csi"),
        )
        for metric_id, source in metric_sources.items():
            if metric_id != "resource.csi.observed_actual_gain_correlation":
                self.assertEqual(source, evaluation_source, metric_id)

        correlation_id = "resource.csi.observed_actual_gain_correlation"
        evaluator_metrics = {
            metric["id"]: 0.5
            for metric in pack["metrics"]
        }
        summary = {
            "metrics": {},
            "steps": [
                {
                    "id": "csi_observation",
                    "op": "wireless.ofdm_delayed_csi",
                    "metrics": {correlation_id: 0.37},
                    "outputs": {},
                    "execution_binding": {
                        "implementation": "test-observer",
                        "implementation_metadata": {
                            "source_sha256": "e" * 64,
                        },
                    },
                },
                {
                    "id": "allocation_evaluation",
                    "op": "metrics.ofdm_finite_blocklength_allocation",
                    "metrics": evaluator_metrics,
                    "outputs": {},
                    "execution_binding": {
                        "implementation": "test-evaluator",
                        "implementation_metadata": {
                            "source_sha256": "f" * 64,
                        },
                    },
                },
            ],
        }
        metrics, provenance = _collect_summary_metrics_with_provenance(
            summary,
            pack["metrics"],
        )
        self.assertEqual(metrics[correlation_id], 0.37)
        self.assertEqual(
            provenance[correlation_id]["source_step"],
            "csi_observation",
        )


def _write_capture(
    directory: Path,
    split: str,
    history: np.ndarray,
    current: np.ndarray,
) -> None:
    directory.mkdir(parents=True)
    np.savez_compressed(
        directory / "shard_00000.npz",
        csi_history=history,
        current_csi=current,
    )
    schema = {
        "kind": "noema.capture_dataset",
        "schema_version": 1,
        "split": split,
        "tap_schemas": {
            "csi_history": {
                "dtype": "float32",
                "record_shape": [
                    int(history.shape[1]),
                    int(history.shape[2]),
                    int(history.shape[3]),
                ],
            },
            "current_csi": {
                "dtype": "float32",
                "record_shape": [int(current.shape[1])],
            },
        },
        "shards": [{"path": "shard_00000.npz"}],
    }
    (directory / "schema.json").write_text(
        json.dumps(schema),
        encoding="utf-8",
    )


def _complex_csi_history(
    latest_gains: np.ndarray,
    *,
    history_length: int,
) -> np.ndarray:
    gains = np.asarray(latest_gains, dtype=np.float32)
    history = np.zeros(
        (gains.shape[0], history_length, gains.shape[1], 2),
        dtype=np.float32,
    )
    amplitudes = np.sqrt(gains)
    for index in range(history_length):
        phase = 0.07 * float(index)
        scale = 0.85 + 0.05 * float(index)
        history[:, index, :, 0] = scale * amplitudes * np.cos(phase)
        history[:, index, :, 1] = scale * amplitudes * np.sin(phase)
    return history


if __name__ == "__main__":
    unittest.main()
