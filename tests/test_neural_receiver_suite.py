from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from noema_lab.cli.main import main
from noema_lab.core.lint import lint_recipe_invariants
from noema_lab.core.matrix import materialize_recipe_matrix_selection
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import load_recipe
from noema_lab.ops import build_registry
from noema_lab.training.differentiability import sionna_available

ROOT = Path(__file__).resolve().parents[1]


class NeuralReceiverSuiteTests(unittest.TestCase):
    def test_neural_receiver_recipe_lints_and_runs(self):
        registry = build_registry()
        recipe_path = ROOT / "recipes" / "neural_receiver_qpsk_iq_calibration.yaml"
        recipe = load_recipe(recipe_path)
        validate_recipe_against_registry(recipe, registry)
        lint_report = lint_recipe_invariants(recipe, registry)
        self.assertEqual(lint_report["status"], "passed")
        self.assertIn("bit_transport", lint_report["profiles"])

        with tempfile.TemporaryDirectory() as tmpdir:
            concrete_recipe_path = Path(tmpdir) / "qpsk_awgn_snr6.json"
            concrete_recipe_path.write_text(
                json.dumps(
                    materialize_recipe_matrix_selection(
                        recipe, {"channel.snr_db": 6.0}
                    )
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "--workspace",
                        tmpdir,
                        "recipe",
                        "run",
                        str(concrete_recipe_path),
                    ]
                )
            self.assertEqual(code, 0, stdout.getvalue())
            summary_path = _line_value(stdout.getvalue(), "summary:")
            summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "completed")
        metrics = _step_metrics(summary)
        self.assertEqual(metrics["channel.transmitted_bit_count"], 4096)
        self.assertEqual(metrics["channel.channel_use_count"], 2048)
        self.assertEqual(metrics["channel.bits_per_symbol"], 2)
        self.assertIn("channel.coded.ber", metrics)
        self.assertIn("channel.coded.bler", metrics)
        self.assertGreaterEqual(float(metrics["channel.coded.ber"]), 0.0)
        self.assertLessEqual(float(metrics["channel.coded.ber"]), 1.0)
        self.assertGreaterEqual(float(metrics["channel.coded.bler"]), 0.0)
        self.assertLessEqual(float(metrics["channel.coded.bler"]), 1.0)
        demodulator = next(step for step in summary["steps"] if step["id"] == "demodulator")
        frontend = next(
            step for step in summary["steps"] if step["id"] == "receiver_frontend"
        )
        self.assertEqual(
            frontend["outputs"]["rx_symbols"]["metadata"]["receiver_frontend"],
            "fixed_iq_imbalance",
        )
        preview = demodulator["outputs"]["bits"]["metadata"]["receiver_decision_preview"]
        self.assertEqual(preview["kind"], "memoryless_qpsk_iq_decision_regions")
        self.assertEqual(preview["receiver_mode"], "reference_qpsk")
        self.assertEqual(preview["grid"]["width"], 64)
        self.assertEqual(preview["coordinate_space"], "impaired_received_iq")
        self.assertEqual({point["bits"] for point in preview["constellation"]}, {"00", "01", "10", "11"})

    def test_classical_qpsk_demodulator_records_analytical_decision_regions(self):
        recipe_path = ROOT / "recipes" / "neural_receiver_qpsk_awgn_classical.yaml"
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["--workspace", tmpdir, "recipe", "run", str(recipe_path)])
            self.assertEqual(code, 0, stdout.getvalue())
            summary_path = _line_value(stdout.getvalue(), "summary:")
            summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))

        demodulator = next(step for step in summary["steps"] if step["id"] == "demodulator")
        preview = demodulator["outputs"]["bits"]["metadata"]["receiver_decision_preview"]
        self.assertEqual(preview["receiver_mode"], "analytical_qpsk")
        rows = preview["grid"]["class_rows"]
        self.assertEqual(set("".join(rows)), {"0", "1", "2", "3"})
        midpoint = len(rows) // 2
        self.assertTrue(all(value == "2" for value in rows[-1][:midpoint]))
        self.assertTrue(all(value == "0" for value in rows[-1][midpoint:]))

    def test_calibrated_oracle_removes_the_fixed_frontend_error_floor(self):
        recipe = load_recipe(
            ROOT / "recipes" / "neural_receiver_qpsk_iq_calibration.yaml"
        ).to_dict()
        recipe["metadata"].pop("matrix", None)
        for step in recipe["steps"]:
            if step["id"] == "data":
                step["params"]["bit_count"] = 32768
            elif step["id"] == "wireless_channel":
                step["params"]["snr_db"] = 10

        observed = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            for mode in ("reference_qpsk", "oracle_frontend_calibrated"):
                payload = json.loads(json.dumps(recipe))
                payload["name"] = "receiver_iq_%s" % mode
                demodulator = next(
                    step
                    for step in payload["steps"]
                    if step["id"] == "demodulator"
                )
                demodulator["params"] = {
                    "mode": mode,
                    "modulation": "qpsk",
                }
                path = Path(tmpdir) / ("%s.json" % mode)
                path.write_text(json.dumps(payload), encoding="utf-8")
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = main(
                        [
                            "--workspace",
                            str(Path(tmpdir) / "workspace"),
                            "recipe",
                            "run",
                            str(path),
                        ]
                    )
                self.assertEqual(code, 0, stdout.getvalue())
                summary = json.loads(
                    Path(_line_value(stdout.getvalue(), "summary:")).read_text(
                        encoding="utf-8"
                    )
                )
                observed[mode] = float(
                    _step_metrics(summary)["channel.coded.ber"]
                )

        self.assertGreater(observed["reference_qpsk"], 0.04)
        self.assertLess(observed["oracle_frontend_calibrated"], 0.005)
        self.assertLess(
            observed["oracle_frontend_calibrated"],
            observed["reference_qpsk"] / 10.0,
        )

    def test_neural_receiver_benchmark_pack_runs(self):
        benchmark_path = ROOT / "benchmarks" / "neural_receiver_ai_phy" / "qpsk_awgn_receiver_v1.yaml"
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["--workspace", tmpdir, "benchmark", "run", str(benchmark_path)])
            self.assertEqual(code, 0, stdout.getvalue())
            result_path = Path(_line_value(stdout.getvalue(), "result:"))
            metrics_path = Path(_line_value(stdout.getvalue(), "metrics:"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            with metrics_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["benchmark"]["suite"]["id"], "neural_receiver")
        recipe_ids = {row["recipe_id"] for row in rows}
        self.assertIn("classical_qpsk_snr6", recipe_ids)
        self.assertIn("adapter_qpsk_snr14", recipe_ids)
        metrics = {(row["recipe_id"], row["metric"]): row["value"] for row in rows}
        self.assertIn(("classical_qpsk_snr6", "channel.coded.ber"), metrics)
        self.assertIn(("adapter_qpsk_snr14", "channel.coded.bler"), metrics)
        self.assertIn(("adapter_qpsk_snr14", "receiver.neural_receiver_adapter"), metrics)


class SionnaBackedExpansionSuiteTests(unittest.TestCase):
    PACKS = {
        "channel_estimation": {
            "path": ROOT / "benchmarks" / "channel_estimation" / "pilot_awgn_v1.yaml",
            "metric": "channel_estimation.nmse",
            "recipe": "ls_snr0",
            "snr_db": 0.0,
        },
        "mimo_ofdm": {
            "path": ROOT / "benchmarks" / "mimo_ofdm" / "channel_estimation_v1.yaml",
            "metric": "mimo.channel_estimation.nmse",
            "recipe": "ls_snr0",
            "snr_db": 0.0,
            "requires_sionna": True,
            "required_extra": "wireless",
        },
        "beamforming_precoding": {
            "path": ROOT / "benchmarks" / "beamforming_precoding" / "beam_selection_v1.yaml",
            "metric": "beamforming.spectral_efficiency_bps_hz",
            "recipe": "mrt_snr0",
            "snr_db": 0.0,
        },
        "localization_sensing": {
            "path": ROOT / "benchmarks" / "localization_sensing" / "range_localization_v1.yaml",
            "metric": "localization.rmse_m",
            "recipe": "trilateration_snr10",
            "snr_db": 10.0,
        },
        "resource_allocation": {
            "path": ROOT / "benchmarks" / "resource_allocation" / "power_allocation_v1.yaml",
            "metric": "channel.achieved_payload_goodput_bits_per_resource_element",
            "recipe": "equal_power_p10",
            "snr_db": float(10.0 * math.log10(1.0 / 0.2)),
            "recipe_count": 6,
            "requires_sionna": True,
            "required_extra": "wireless",
        },
    }

    def test_experimental_suite_benchmark_packs_validate_and_run(self):
        for suite_id, spec in self.PACKS.items():
            with self.subTest(suite=suite_id):
                # A legacy Sionna distribution still exposes the `sionna`
                # namespace, so module discovery alone is not a runnable
                # Sionna 2.x contract.
                if spec.get("requires_sionna") and not sionna_available():
                    self.skipTest(
                        "%s runtime requires the optional `%s` extra"
                        % (suite_id, spec["required_extra"])
                    )

                validate_stdout = io.StringIO()
                with contextlib.redirect_stdout(validate_stdout):
                    code = main(["benchmark", "validate", str(spec["path"])])
                self.assertEqual(code, 0, validate_stdout.getvalue())
                validation = json.loads(validate_stdout.getvalue())
                self.assertEqual(validation["suite"]["id"], suite_id)
                self.assertEqual(validation["benchmark_tier"], "experimental")
                self.assertEqual(validation["recipe_count"], spec.get("recipe_count", 4))

                with tempfile.TemporaryDirectory() as tmpdir:
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        code = main(["--workspace", tmpdir, "benchmark", "run", str(spec["path"])])
                    self.assertEqual(code, 0, stdout.getvalue())
                    result_path = Path(_line_value(stdout.getvalue(), "result:"))
                    metrics_path = Path(_line_value(stdout.getvalue(), "metrics:"))
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    with metrics_path.open("r", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))

                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["benchmark"]["suite"]["id"], suite_id)
                metrics = {(row["recipe_id"], row["metric"]): row["value"] for row in rows}
                self.assertIn((spec["recipe"], spec["metric"]), metrics)
                self.assertIn((spec["recipe"], "task.score"), metrics)
                if suite_id == "resource_allocation":
                    for suffix in ("p05", "p10", "p20"):
                        equal_id = f"equal_power_{suffix}"
                        water_id = f"water_filling_{suffix}"
                        equal_rate = float(metrics[(equal_id, "resource.theoretical_shannon_spectral_efficiency_bps_hz")])
                        water_rate = float(metrics[(water_id, "resource.theoretical_shannon_spectral_efficiency_bps_hz")])
                        self.assertGreaterEqual(water_rate, equal_rate)
                        self.assertGreater(float(metrics[(equal_id, "resource.water_filling_relative_optimality_gap")]), 0.0)
                        self.assertLess(float(metrics[(water_id, "resource.water_filling_relative_optimality_gap")]), 1e-8)
                        self.assertLess(float(metrics[(water_id, "resource.water_filling_power_normalized_rmse")]), 1e-6)
                        self.assertLess(float(metrics[(water_id, "resource.power_constraint.max_relative_error")]), 1e-6)
                        for recipe_id in (equal_id, water_id):
                            bler = float(metrics[(recipe_id, "channel.payload_delivery_block_error_rate")])
                            self.assertGreaterEqual(bler, 0.0)
                            self.assertLessEqual(bler, 1.0)
                            self.assertGreaterEqual(
                                float(metrics[(recipe_id, "channel.payload_energy_efficiency_bits_per_normalized_energy")]),
                                0.0,
                            )
                self.assertIn((spec["recipe"], "channel.snr_db"), metrics)
                self.assertAlmostEqual(float(metrics[(spec["recipe"], "channel.snr_db")]), spec["snr_db"])

    def test_experimental_suite_plot_export_uses_task_score(self):
        spec = self.PACKS["beamforming_precoding"]
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["--workspace", tmpdir, "benchmark", "run", str(spec["path"])])
            self.assertEqual(code, 0, stdout.getvalue())
            result_id = Path(_line_value(stdout.getvalue(), "result:")).parent.name

            plot_stdout = io.StringIO()
            with contextlib.redirect_stdout(plot_stdout):
                code = main([
                    "--workspace",
                    tmpdir,
                    "benchmark",
                    "plot",
                    result_id,
                    "--plot",
                    "graceful-degradation",
                    "--x",
                    "channel.snr_db",
                    "--y",
                    "task.score",
                    "--group",
                    "method",
                    "--out",
                    "figures/beamforming_score.png",
                    "--json",
                ])
            self.assertEqual(code, 0, plot_stdout.getvalue())
            payload = json.loads(plot_stdout.getvalue())
            plot_path = Path(payload["plot"]["path"])
            self.assertTrue(plot_path.is_file())
            self.assertTrue(plot_path.with_suffix(".csv").is_file())


def _line_value(output: str, prefix: str) -> str:
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.split(prefix, 1)[1].strip()
    raise AssertionError("missing %s in output: %s" % (prefix, output))


def _step_metrics(summary: dict) -> dict:
    metrics = {}
    for step in summary.get("steps", []):
        metrics.update(step.get("metrics") or {})
    return metrics


if __name__ == "__main__":
    unittest.main()
