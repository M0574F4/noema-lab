from __future__ import annotations

import contextlib
import io
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from demo_trainings import prepare_example
from noema_lab.core.recipes import load_recipe
from noema_lab.core.training_plans import (
    TrainingPlan,
    scenario_recipe_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]


class PrepareExampleOutputTests(unittest.TestCase):
    def test_planned_recipe_preserves_custom_capture_ids_without_changing_scenario_identity(self):
        neutral = load_recipe(ROOT / "recipes" / "resource_equal_power_baseline.yaml")
        plan = TrainingPlan(
            selected_steps=("tx_power",),
            dataset_capture={
                "taps": [
                    {
                        "id": "researcher_named_channel_state",
                        "from": "channel_state.state",
                    }
                ],
                "split_plan": {
                    "total_samples": 12,
                    "percentages": {
                        "train": 50,
                        "validation": 25,
                        "test": 25,
                    },
                },
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "noema_recipe.yaml").write_text(
                yaml.safe_dump(neutral.to_dict(), sort_keys=False),
                encoding="utf-8",
            )
            (bundle / "training_plan.yaml").write_text(
                yaml.safe_dump(plan.to_dict(), sort_keys=False),
                encoding="utf-8",
            )

            planned, loaded_plan = prepare_example._load_planned_recipe(bundle)

        self.assertEqual(loaded_plan.selected_steps, ("tx_power",))
        self.assertEqual(
            planned.dataset_capture["taps"],
            [
                {
                    "id": "researcher_named_channel_state",
                    "from": "channel_state.state",
                }
            ],
        )
        self.assertEqual(
            scenario_recipe_fingerprint(planned),
            scenario_recipe_fingerprint(neutral),
        )

    def test_main_reports_missing_demo_signal_without_traceback(self):
        stderr = io.StringIO()
        plan = mock.Mock(
            feature_reference="wireless_channel.rx_symbols",
            target_reference="tx_bit_boundary.bits",
        )

        def fail_with_missing_target(_bundle, demo, *, project_root):
            self.assertEqual(Path(project_root).resolve(), ROOT)
            prepare_example._require_demo_signals(
                {
                    "data_contract": {
                        "signals": [
                            {"reference": "wireless_channel.rx_symbols"},
                        ]
                    }
                },
                plan,
                demo,
            )

        with mock.patch.object(
            sys,
            "argv",
            ["prepare_example.py", "neural-receiver", "/tmp/qpsk-bundle"],
        ), mock.patch.object(
            prepare_example,
            "prepare_example",
            side_effect=fail_with_missing_target,
        ), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                prepare_example.main()

        self.assertEqual(raised.exception.code, 2)
        rendered = stderr.getvalue()
        self.assertIn(
            "error: The neural-receiver demonstration project needs captured signal(s) "
            "tx_bit_boundary.bits",
            rendered,
        )
        self.assertIn("Dataset definition > Captured signals", rendered)
        self.assertIn("recapture the datasets before training", rendered)
        self.assertNotIn("Traceback", rendered)

    def test_neural_receiver_matrix_rejects_redundant_capture_sweeps(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            recipe_payload = load_recipe(
                ROOT / "recipes" / "neural_receiver_qpsk_iq_calibration.yaml"
            ).to_dict()
            (bundle / "noema_recipe.yaml").write_text(
                yaml.safe_dump(recipe_payload, sort_keys=False),
                encoding="utf-8",
            )
            (bundle / "training_plan.yaml").write_text(
                yaml.safe_dump({"dataset_capture": {"split_plan": {}}}),
                encoding="utf-8",
            )
            for split in ("train", "validation", "test"):
                capture_payload = yaml.safe_load(
                    yaml.safe_dump(recipe_payload, sort_keys=False)
                )
                capture_payload["dataset_capture"] = {
                    "sweep": {"wireless_channel.snr_db": [6.0]}
                }
                (bundle / ("capture_%s_recipe.yaml" % split)).write_text(
                    yaml.safe_dump(capture_payload, sort_keys=False),
                    encoding="utf-8",
                )

            with self.assertRaises(ValueError) as raised:
                prepare_example._require_neural_receiver_capture_sweep(bundle)

            message = str(raised.exception)
            self.assertIn("values [-2, 2, 6, 10]", message)
            self.assertIn(
                "capture_train_recipe.yaml also declares dataset_capture.sweep",
                message,
            )
            self.assertIn("recapture all datasets before training", message)

    def test_neural_receiver_matrix_accepts_published_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            recipe_payload = load_recipe(
                ROOT / "recipes" / "neural_receiver_qpsk_iq_calibration.yaml"
            ).to_dict()
            (bundle / "noema_recipe.yaml").write_text(
                yaml.safe_dump(recipe_payload, sort_keys=False),
                encoding="utf-8",
            )
            (bundle / "training_plan.yaml").write_text(
                yaml.safe_dump({"dataset_capture": {"split_plan": {}}}),
                encoding="utf-8",
            )
            capture_payload = yaml.safe_load(
                yaml.safe_dump(recipe_payload, sort_keys=False)
            )
            capture_payload["dataset_capture"] = {"split": "train"}
            for relative in (
                "capture_train_recipe.yaml",
                "capture_validation_recipe.yaml",
                "capture_test_recipe.yaml",
            ):
                (bundle / relative).write_text(
                    yaml.safe_dump(capture_payload, sort_keys=False),
                    encoding="utf-8",
                )

            prepare_example._require_neural_receiver_capture_sweep(bundle)

    def test_checkout_next_steps_use_uv_and_bundle_root_launchers(self):
        output = Path("/tmp/noema project/reference_training")
        stream = io.StringIO()
        project_root = Path(__file__).resolve().parents[1]

        with contextlib.redirect_stdout(stream):
            prepare_example._print_next_steps(output, project_root=project_root)

        rendered = stream.getvalue()
        quoted_project = str(project_root)
        self.assertIn(
            "cd '/tmp/noema project'",
            rendered,
        )
        self.assertIn(
            "uv run --project %s --extra onnx python --version" % quoted_project,
            rendered,
        )
        self.assertIn(
            "uv run --project %s --extra onnx python train_demo.py" % quoted_project,
            rendered,
        )
        self.assertIn(
            "uv run --project %s --extra onnx python evaluate_demo.py" % quoted_project,
            rendered,
        )
        self.assertNotIn("-m pip", rendered)
        self.assertNotIn("uv sync", rendered)
        self.assertNotIn("&&", rendered)
        self.assertIn("Validate returned model", rendered)
        self.assertIn("detects the returned artifact automatically", rendered)
        self.assertNotIn("Refresh bundle", rendered)
        self.assertIn("No dataset capture is required", rendered)
        self.assertNotIn("Capture all datasets", rendered)
        self.assertNotIn("Recapture all datasets", rendered)

    def test_capture_backed_next_steps_keep_capture_readiness_instructions(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            output = bundle / "reference_training"
            output.mkdir(parents=True)
            (bundle / "project_manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "capture_jobs": [
                            {
                                "split": "train",
                                "label": "Train",
                                "requested_samples": 1024,
                            },
                            {
                                "split": "validation",
                                "label": "Validation",
                                "requested_samples": 256,
                            },
                            {
                                "split": "test",
                                "label": "Held-out test",
                                "requested_samples": 256,
                            },
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                prepare_example._print_next_steps(output, project_root=ROOT)

            rendered = stream.getvalue()
            self.assertIn("Capture all datasets", rendered)
            self.assertIn("Recapture all datasets", rendered)
            self.assertIn(
                "Train 1,024 · Validation 256 · Held-out test 256",
                rendered,
            )
            self.assertIn("Only then run:", rendered)
            self.assertNotIn("No dataset capture is required", rendered)

    def test_refresh_reports_existing_complete_captures_as_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            output = bundle / "reference_training"
            output.mkdir(parents=True)
            jobs = []
            for split, label, requested in (
                ("train", "Train", 32),
                ("validation", "Validation", 8),
                ("test", "Held-out test", 8),
            ):
                jobs.append(
                    {
                        "split": split,
                        "label": label,
                        "requested_samples": requested,
                        "expected_taps": [
                            {
                                "id": "rx_symbols",
                                "from": "receiver_frontend.rx_symbols",
                            },
                            {
                                "id": "target_bits",
                                "from": "tx_bit_boundary.bits",
                            },
                        ],
                    }
                )
                capture = bundle / "data" / split
                capture.mkdir(parents=True)
                (capture / "schema.json").write_text(
                    json.dumps(
                        {
                            "captured_samples": requested,
                            "tap_schemas": {
                                "rx_symbols": {
                                    "from": "receiver_frontend.rx_symbols"
                                },
                                "target_bits": {
                                    "from": "tx_bit_boundary.bits"
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            (bundle / "project_manifest.yaml").write_text(
                yaml.safe_dump({"capture_jobs": jobs}, sort_keys=False),
                encoding="utf-8",
            )
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                prepare_example._print_next_steps(output, project_root=ROOT)

            rendered = stream.getvalue()
            self.assertIn("datasets are already ready", rendered)
            self.assertIn("trainer refresh preserved them", rendered)
            self.assertIn("Run:", rendered)
            self.assertNotIn("Capture all datasets", rendered)
            self.assertNotIn("Only then run:", rendered)

    def test_generic_next_steps_use_invoking_interpreter_without_assuming_pip(self):
        output = Path("/tmp/noema project/reference_training")
        stream = io.StringIO()

        with mock.patch.object(
            prepare_example.sys,
            "executable",
            "/tmp/training env/bin/python",
        ), contextlib.redirect_stdout(stream):
            prepare_example._print_next_steps(
                output,
                project_root=Path("/tmp/not-a-noema-checkout"),
            )

        rendered = stream.getvalue()
        self.assertIn("'/tmp/training env/bin/python' train_demo.py", rendered)
        self.assertIn("'/tmp/training env/bin/python' evaluate_demo.py", rendered)
        self.assertIn(
            "install every package listed in reference_training/requirements.txt",
            rendered,
        )
        self.assertNotIn("-m pip", rendered)
        self.assertNotIn("uv sync", rendered)

    def test_root_handoff_launchers_run_nested_scripts_from_their_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            nested = bundle / "reference_training"
            nested.mkdir(parents=True)
            (nested / "train.py").write_text(
                "from pathlib import Path\n"
                "Path('../train-ran.txt').write_text(Path.cwd().name, encoding='utf-8')\n",
                encoding="utf-8",
            )
            (nested / "evaluate.py").write_text(
                "from pathlib import Path\n"
                "Path('../evaluate-ran.txt').write_text(Path.cwd().name, encoding='utf-8')\n",
                encoding="utf-8",
            )
            (bundle / "project_manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "external_training": {
                            "optional_demo_scaffold": {
                                "included": True,
                                "normative": False,
                                "path": "reference_training",
                            }
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            prepare_example._write_root_demo_handoff(bundle, "resource-allocation")
            (bundle / "trained_artifact.yaml").write_text(
                "schema_version: 2\n",
                encoding="utf-8",
            )

            for command in ("train_demo.py", "evaluate_demo.py"):
                completed = subprocess.run(
                    [sys.executable, command],
                    cwd=bundle,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                (bundle / "train-ran.txt").read_text(encoding="utf-8"),
                "reference_training",
            )
            self.assertEqual(
                (bundle / "evaluate-ran.txt").read_text(encoding="utf-8"),
                "reference_training",
            )
            quickstart = (bundle / "RUN_DEMO.md").read_text(encoding="utf-8")
            self.assertIn("train_demo.py", quickstart)
            self.assertIn(
                "uv run --project %s --extra onnx python train_demo.py"
                % shlex.quote(str(ROOT.resolve())),
                quickstart,
            )
            self.assertIn("No dataset capture is required", quickstart)
            self.assertNotIn("Capture all datasets", quickstart)
            self.assertNotIn("Recapture all datasets", quickstart)
            self.assertIn("run the example evaluator", quickstart)
            self.assertNotIn("run the held-out evaluator", quickstart)
            manifest = yaml.safe_load(
                (bundle / "project_manifest.yaml").read_text(encoding="utf-8")
            )
            handoff = manifest["external_training"]["optional_demo_scaffold"]["root_handoff"]
            self.assertEqual(handoff["ownership"], "noema_demo_helper")
            self.assertFalse(handoff["normative"])
            self.assertEqual(handoff["training"]["target"], "reference_training/train.py")
            self.assertEqual(handoff["evaluation"]["target"], "reference_training/evaluate.py")
            self.assertEqual(
                {item["path"] for item in handoff["managed_files"]},
                set(prepare_example.DEMO_HANDOFF_FILES),
            )

            # Reattaching the same published demo refreshes only helper-owned files.
            prepare_example._write_root_demo_handoff(bundle, "resource-allocation")
            refreshed = yaml.safe_load(
                (bundle / "project_manifest.yaml").read_text(encoding="utf-8")
            )["external_training"]["optional_demo_scaffold"]["root_handoff"]
            self.assertEqual(refreshed["managed_files"], handoff["managed_files"])

    def test_train_launcher_reports_missing_captures_without_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            nested = bundle / "reference_training"
            nested.mkdir(parents=True)
            (nested / "train.py").write_text(
                "raise AssertionError('preflight should stop before training')\n",
                encoding="utf-8",
            )
            (nested / "evaluate.py").write_text("pass\n", encoding="utf-8")
            (bundle / "project_manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "out_dir": "bundle",
                        "capture_jobs": [
                            {
                                "split": "train",
                                "label": "Train",
                                "output_dir": "bundle/data/train",
                                "requested_samples": 1024,
                                "sample_unit": "recipe records",
                                "expected_taps": [
                                    {
                                        "id": "phase_truth",
                                        "from": "carrier_impairment.phase_truth",
                                    }
                                ],
                            },
                            {
                                "split": "validation",
                                "label": "Validation",
                                "output_dir": "bundle/data/validation",
                                "requested_samples": 256,
                                "sample_unit": "recipe records",
                                "expected_taps": [
                                    {
                                        "id": "phase_truth",
                                        "from": "carrier_impairment.phase_truth",
                                    }
                                ],
                            },
                        ],
                        "external_training": {
                            "optional_demo_scaffold": {
                                "included": True,
                                "normative": False,
                                "path": "reference_training",
                            }
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            prepare_example._write_root_demo_handoff(bundle, "phase-tracking-receiver")
            quickstart = (bundle / "RUN_DEMO.md").read_text(encoding="utf-8")
            self.assertIn("Capture all datasets", quickstart)
            self.assertIn("Recapture all datasets", quickstart)
            self.assertIn("run the held-out evaluator", quickstart)

            completed = subprocess.run(
                [sys.executable, "train_demo.py"],
                cwd=bundle,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertIn("Capture all datasets", completed.stderr)
            self.assertIn("Recapture all datasets", completed.stderr)
            self.assertIn("Train: 1,024 recipe records", completed.stderr)
            self.assertIn("Validation: 256 recipe records", completed.stderr)
            self.assertIn("carrier_impairment.phase_truth", completed.stderr)
            self.assertIn("Then rerun train_demo.py", completed.stderr)

    def test_evaluate_launcher_reports_train_first_without_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            nested = bundle / "reference_training"
            nested.mkdir(parents=True)
            (nested / "train.py").write_text("pass\n", encoding="utf-8")
            (nested / "evaluate.py").write_text(
                "raise AssertionError('preflight should stop before evaluation')\n",
                encoding="utf-8",
            )
            (bundle / "project_manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "external_training": {
                            "optional_demo_scaffold": {
                                "included": True,
                                "normative": False,
                                "path": "reference_training",
                            }
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            prepare_example._write_root_demo_handoff(bundle, "phase-tracking-receiver")

            completed = subprocess.run(
                [sys.executable, "evaluate_demo.py"],
                cwd=bundle,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertIn("No trained model is available", completed.stderr)
            self.assertIn("Run train_demo.py successfully first", completed.stderr)
            self.assertIn("trained_artifact.yaml", completed.stderr)

    def test_evaluate_launcher_preflights_test_capture_after_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            nested = bundle / "reference_training"
            nested.mkdir(parents=True)
            (nested / "train.py").write_text("pass\n", encoding="utf-8")
            (nested / "evaluate.py").write_text(
                "raise AssertionError('preflight should stop before evaluation')\n",
                encoding="utf-8",
            )
            (bundle / "trained_artifact.yaml").write_text(
                "schema_version: 2\n",
                encoding="utf-8",
            )
            (bundle / "project_manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "out_dir": "bundle",
                        "capture_jobs": [
                            {
                                "split": "test",
                                "label": "Held-out test",
                                "output_dir": "bundle/data/test",
                                "requested_samples": 256,
                                "sample_unit": "recipe records",
                                "expected_taps": [
                                    {
                                        "id": "rx_symbols",
                                        "from": "carrier_impairment.rx_symbols",
                                    }
                                ],
                            }
                        ],
                        "external_training": {
                            "optional_demo_scaffold": {
                                "included": True,
                                "normative": False,
                                "path": "reference_training",
                            }
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            prepare_example._write_root_demo_handoff(bundle, "phase-tracking-receiver")

            completed = subprocess.run(
                [sys.executable, "evaluate_demo.py"],
                cwd=bundle,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertIn("Held-out test: 256 recipe records", completed.stderr)
            self.assertIn("Then rerun evaluate_demo.py", completed.stderr)

    def test_root_handoff_refuses_to_overwrite_researcher_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            researcher_file = bundle / "train_demo.py"
            researcher_file.write_text("print('mine')\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "researcher-owned"):
                prepare_example._require_safe_demo_handoff_paths(bundle)

            self.assertEqual(
                researcher_file.read_text(encoding="utf-8"),
                "print('mine')\n",
            )


if __name__ == "__main__":
    unittest.main()
