from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import yaml

from noema_lab.core.benchmarks import load_benchmark_pack, validate_benchmark_pack
from noema_lab.core.recipes import load_recipe
from noema_lab.core.reproducibility import derive_seed
from noema_lab.core.training_plans import apply_training_plan, load_training_plan
from noema_lab.ops import build_registry
from noema_lab.training.exporter import export_differentiable_scenario


ROOT = Path(__file__).resolve().parents[1]


class PostTrainingBenchmarkBuilderTests(unittest.TestCase):
    def test_resource_builder_generates_one_recipe_paired_three_method_pack(self):
        recipe = load_recipe(ROOT / "recipes" / "resource_equal_power_baseline.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "resource_bundle"
            exported = export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["tx_power"],
                loss="resource.negative_shannon_spectral_efficiency",
                framework="torch",
                out_dir=bundle,
                project_root=root,
                exporter="resource-allocation",
                include_starter=True,
            )
            starter = bundle / "reference_training"
            self.assertTrue((starter / "build_benchmark.py").is_file())
            config = yaml.safe_load(
                (starter / "train_config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                config["training"]["artifact_manifest_path"],
                "../trained_artifact.yaml",
            )
            self.assertEqual(
                config["training"]["artifact_component_path"],
                "../artifacts/power_policy.onnx",
            )
            self.assertIn(
                "post_training",
                exported["project_manifest"]["external_training"][
                    "optional_demo_scaffold"
                ],
            )
            _write_stub_artifact(
                bundle / "trained_artifact.yaml",
                operation="model.symbol_power_allocator",
                entrypoint="power_policy",
            )
            _write_training_evidence(starter)

            completed = subprocess.run(
                [
                    sys.executable,
                    "build_benchmark.py",
                    "--budgets",
                    "0.5,1",
                    "--seeds",
                    "81001,82001",
                ],
                cwd=starter,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("noema benchmark validate ", completed.stdout)
            self.assertIn(str(starter / "benchmark_pack.yaml"), completed.stdout)
            pack = load_benchmark_pack(starter / "benchmark_pack.yaml")
            validation = validate_benchmark_pack(pack, build_registry(), starter)
            self.assertEqual(validation["recipe_count"], 12)
            self.assertEqual({str(item.path) for item in pack.recipes}, {"noema_recipe.yaml"})
            demo = pack.metadata["demo"]
            self.assertEqual(demo["schema_version"], 1)
            self.assertEqual(
                demo["primary_metric"],
                "resource.theoretical_shannon_spectral_efficiency_bps_hz",
            )
            artifact_spec = demo["training_evidence"][0][
                "trained_artifact_manifest"
            ]
            self.assertEqual(artifact_spec["path"], "../trained_artifact.yaml")
            self.assertEqual(
                artifact_spec["sha256"],
                _file_sha256(bundle / "trained_artifact.yaml"),
            )
            for field in (
                "trained_artifact_manifest",
                "training_history",
                "evaluation_metrics",
            ):
                self.assertRegex(
                    demo["training_evidence"][0][field]["sha256"],
                    r"^[0-9a-f]{64}$",
                )
            self.assertTrue(all(plot["kind"] == "line" for plot in demo["plots"]))
            self.assertEqual(
                [plot["id"] for plot in demo["plots"]],
                ["spectral-efficiency-vs-power", "shannon-gap-vs-power"],
            )
            self.assertEqual(
                [plot["y"] for plot in demo["plots"]],
                [
                    "resource.theoretical_shannon_spectral_efficiency_bps_hz",
                    "resource.water_filling_relative_optimality_gap",
                ],
            )
            self.assertNotIn(
                "channel.payload.bler",
                demo["table_metrics"],
            )

            grouped = defaultdict(dict)
            for entry in pack.recipes:
                params = entry.params
                metadata = params["metadata"]
                self.assertTrue(
                    entry.id.startswith(metadata["benchmark_method"] + "_")
                )
                selection = params["matrix_selection"]
                key = (
                    selection["resource.average_transmit_power_budget"],
                    metadata["benchmark_paired_seed"],
                )
                grouped[key][metadata["benchmark_method"]] = params["step_params"]
            self.assertEqual(len(grouped), 4)
            for (selected_budget, _paired_seed), methods in grouped.items():
                self.assertEqual(
                    set(methods),
                    {"equal_power", "learned_allocator", "water_filling"},
                )
                self.assertTrue(
                    all(
                        row["channel_state"]["average_power_budget"]
                        == selected_budget
                        == row["tx_power"]["target_power"]
                        for row in methods.values()
                    )
                )
                paired = {
                    (
                        row["data"]["seed"],
                        row["channel_state"]["seed"],
                        row["wireless_channel"]["seed"],
                    )
                    for row in methods.values()
                }
                self.assertEqual(len(paired), 1)
                for row in methods.values():
                    self.assertEqual(
                        row["wireless_channel"]["channel"],
                        "ofdm_tdl",
                    )
                    self.assertEqual(
                        row["wireless_channel"]["wireless_backend"],
                        "sionna",
                    )
                    self.assertEqual(
                        row["wireless_channel"]["channel_state_mode"],
                        "explicit",
                    )
                    self.assertEqual(
                        row["wireless_channel"]["receiver_processing"],
                        "matched",
                    )
                self.assertEqual(
                    methods["learned_allocator"]["tx_power"]["artifact_manifest_path"],
                    "resource_bundle/trained_artifact.yaml",
                )
                self.assertRegex(
                    methods["learned_allocator"]["tx_power"][
                        "artifact_package_sha256"
                    ],
                    r"^[0-9a-f]{64}$",
                )
                self.assertTrue(
                    (
                        root
                        / methods["learned_allocator"]["tx_power"][
                            "artifact_manifest_path"
                        ]
                    ).is_file()
                )

            seed_overlap = subprocess.run(
                [
                    sys.executable,
                    "build_benchmark.py",
                    "--seeds",
                    "23",
                    "--output",
                    "seed_overlap_pack.yaml",
                ],
                cwd=starter,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(seed_overlap.returncode, 0)
            self.assertIn("not held out from dataset capture", seed_overlap.stderr)
            self.assertFalse((starter / "seed_overlap_pack.yaml").exists())

    def test_receiver_matrix_and_generated_pairing_are_explicit(self):
        recipe = apply_training_plan(
            load_recipe(ROOT / "recipes" / "neural_receiver_qpsk_iq_calibration.yaml"),
            load_training_plan(
                ROOT
                / "demo_trainings"
                / "neural_receiver_supervised_qpsk"
                / "training_plan.yaml"
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "receiver_bundle"
            exported = export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["demodulator"],
                loss="bit.bce",
                framework="torch",
                out_dir=bundle,
                project_root=root,
                exporter="neural-receiver",
                include_starter=True,
            )
            starter = bundle / "reference_training"
            self.assertTrue((starter / "build_benchmark.py").is_file())
            source_recipe_path = starter / "noema_recipe.yaml"
            source_recipe_bytes = source_recipe_path.read_bytes()
            source_recipe_file_sha256 = _file_sha256(source_recipe_path)
            source_recipe = yaml.safe_load(source_recipe_bytes.decode("utf-8"))
            source_recipe_sha256 = _canonical_sha256(source_recipe)
            self.assertIn("dataset_capture", source_recipe)
            config = yaml.safe_load(
                (starter / "train_config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                config["training"]["artifact_manifest_path"],
                "../trained_artifact.yaml",
            )
            self.assertEqual(
                config["training"]["artifact_component_path"],
                "../artifacts/neural_receiver.onnx",
            )
            contract = yaml.safe_load(
                (bundle / "data_contract.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(contract["mode"], "captured_generic_tensors")
            self.assertEqual(
                {
                    item["reference"]
                    for item in contract["signals"]
                },
                {"receiver_frontend.rx_symbols", "tx_bit_boundary.bits"},
            )
            split_seeds = []
            train_capture = None
            for split in ("train", "validation", "test"):
                capture = yaml.safe_load(
                    (bundle / ("capture_%s_recipe.yaml" % split)).read_text(
                        encoding="utf-8"
                    )
                )
                if split == "train":
                    train_capture = capture
                self.assertNotIn("sweep", capture["dataset_capture"])
                self.assertEqual(
                    capture["metadata"]["matrix"]["dimensions"]["channel.snr_db"],
                    [-2.0, 2.0, 6.0, 10.0],
                )
                split_seeds.append(capture["metadata"]["seed"])
            self.assertEqual(len(set(split_seeds)), 3)
            post_training = exported["project_manifest"]["external_training"][
                "optional_demo_scaffold"
            ]["post_training"]
            self.assertEqual(post_training["command"], "python build_benchmark.py")
            self.assertEqual(
                post_training["benchmark_recipe_path"],
                "receiver_bundle/reference_training/benchmark_recipe.yaml",
            )
            training_template = yaml.safe_load(
                (starter / "training_template.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                training_template["post_training"]["recipe_output"],
                "benchmark_recipe.yaml",
            )

            _write_stub_artifact(
                bundle / "trained_artifact.yaml",
                operation="demodulation.neural_receiver_adapter",
                entrypoint="neural_receiver",
            )
            _write_training_evidence(starter)
            completed = subprocess.run(
                [
                    sys.executable,
                    "build_benchmark.py",
                    "--snr-db=0,4,8",
                    "--seeds",
                    "81001,82001",
                ],
                cwd=starter,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(source_recipe_path.read_bytes(), source_recipe_bytes)
            benchmark_recipe = yaml.safe_load(
                (starter / "benchmark_recipe.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                benchmark_recipe["name"],
                "%s_post_training_benchmark" % source_recipe["name"],
            )
            self.assertNotIn("dataset_capture", benchmark_recipe)
            self.assertEqual(
                benchmark_recipe["metadata"]["matrix"],
                {
                    "dimensions": {
                        "channel.snr_db": [0.0, 4.0, 8.0],
                    },
                    "step_params": {
                        "wireless_channel": {
                            "snr_db": {"matrix": "channel.snr_db"},
                        },
                    },
                },
            )
            provenance = benchmark_recipe["metadata"]["benchmark_recipe"]
            self.assertEqual(provenance["purpose"], "post_training_evaluation")
            self.assertEqual(
                provenance["source_recipe_name"], source_recipe["name"]
            )
            self.assertEqual(
                provenance["source_recipe_sha256"], source_recipe_sha256
            )
            self.assertEqual(
                provenance["source_recipe_file_sha256"],
                source_recipe_file_sha256,
            )
            self.assertEqual(
                provenance["evaluation_snr_db_grid"], [0.0, 4.0, 8.0]
            )
            pack = load_benchmark_pack(starter / "benchmark_pack.yaml")
            validation = validate_benchmark_pack(pack, build_registry(), starter)
            self.assertEqual(validation["recipe_count"], 18)
            self.assertEqual(
                {str(item.path) for item in pack.recipes},
                {"benchmark_recipe.yaml"},
            )
            self.assertEqual(pack.metadata["demo"]["comparison_axis"], "channel.snr_db")
            self.assertTrue(
                all(
                    plot["kind"] == "line"
                    for plot in pack.metadata["demo"]["plots"]
                )
            )
            self.assertEqual(
                pack.metadata["demo"]["training_evidence"][0]["series"],
                "learned_receiver",
            )
            self.assertEqual(
                pack.metadata["demo"]["training_evidence"][0][
                    "trained_artifact_manifest"
                ]["path"],
                "../trained_artifact.yaml",
            )
            self.assertEqual(
                pack.metadata["demo"]["plots"][0]["title"],
                "Pre-decoder BER vs SNR (identity code)",
            )
            self.assertEqual(
                pack.metadata["evaluation_bit_count_per_run"], 1_048_576
            )
            self.assertEqual(pack.metadata["evaluation_block_size_bits"], 1024)
            runtime_identity = pack.metadata[
                "trained_artifact_runtime_identity_sha256"
            ]
            self.assertRegex(runtime_identity, r"^[0-9a-f]{64}$")
            self.assertIn(
                "channel.coded.error_count",
                [metric["id"] for metric in pack.metrics],
            )
            metric_owners = {
                metric["id"]: (
                    metric.get("source_step"),
                    metric.get("source_operation"),
                )
                for metric in pack.metrics
            }
            self.assertEqual(
                metric_owners["channel.channel_use_count"],
                ("wireless_channel", "wireless.channel"),
            )
            self.assertEqual(
                metric_owners["channel.transmitted_bit_count"],
                ("modulator", "modulation.digital_modulate"),
            )
            self.assertEqual(
                metric_owners["channel.coded.ber"],
                ("coded_ber", "metrics.bit_error_rate"),
            )
            self.assertTrue(
                all(metric.get("definition_version") == 1 for metric in pack.metrics)
            )

            grouped = defaultdict(dict)
            for entry in pack.recipes:
                params = entry.params
                metadata = params["metadata"]
                self.assertTrue(
                    entry.id.startswith(metadata["benchmark_method"] + "_")
                )
                selection = params["matrix_selection"]
                self.assertEqual(params["method_id"], metadata["benchmark_method"])
                self.assertEqual(
                    selection["benchmark.paired_seed"],
                    metadata["benchmark_paired_seed"],
                )
                self.assertEqual(metadata["statistical_unit"], "paired_seed")
                self.assertEqual(
                    metadata["aggregation_cell_id"],
                    "snr_db=%s"
                    % (
                        ("%.9g" % float(selection["channel.snr_db"]))
                        .replace("-", "m")
                        .replace(".", "p")
                    ),
                )
                key = (
                    selection["channel.snr_db"],
                    metadata["benchmark_paired_seed"],
                )
                grouped[key][metadata["benchmark_method"]] = params["step_params"]
            self.assertEqual(len(grouped), 6)
            self.assertEqual(
                {snr_db for snr_db, _seed in grouped},
                {0.0, 4.0, 8.0},
            )
            for methods in grouped.values():
                self.assertEqual(
                    set(methods),
                    {
                        "uncompensated_qpsk",
                        "calibrated_iq_oracle",
                        "learned_receiver",
                    },
                )
                paired = {
                    (
                        row["data"]["seed"],
                        row["wireless_channel"]["seed"],
                        row["wireless_channel"]["snr_db"],
                    )
                    for row in methods.values()
                }
                self.assertEqual(len(paired), 1)
                self.assertEqual(
                    {row["data"]["bit_count"] for row in methods.values()},
                    {1_048_576},
                )
                self.assertEqual(
                    {row["coded_bler"]["block_size"] for row in methods.values()},
                    {1024},
                )
                self.assertEqual(
                    methods["learned_receiver"]["demodulator"][
                        "artifact_manifest_path"
                    ],
                    "receiver_bundle/trained_artifact.yaml",
                )
                self.assertEqual(
                    methods["learned_receiver"]["demodulator"][
                        "artifact_package_sha256"
                    ],
                    runtime_identity,
                )
                self.assertTrue(
                    (
                        root
                        / methods["learned_receiver"]["demodulator"][
                            "artifact_manifest_path"
                        ]
                    ).is_file()
                )

            self.assertIsNotNone(train_capture)
            capture_split = str(
                (train_capture.get("dataset_capture") or {}).get("split")
                or "train"
            )
            capture_namespace = "%s|dataset_capture_split=%s" % (
                str(
                    (train_capture.get("metadata") or {}).get("seed_namespace")
                    or train_capture["name"]
                ),
                capture_split,
            )
            overlapping_data_seed = derive_seed(
                int(train_capture["metadata"]["seed"]),
                capture_namespace,
                "data",
                "random_bits",
            )
            seed_overlap = subprocess.run(
                [
                    sys.executable,
                    "build_benchmark.py",
                    "--seeds",
                    str(overlapping_data_seed),
                    "--output",
                    "seed_overlap_pack.yaml",
                ],
                cwd=starter,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(seed_overlap.returncode, 0)
            self.assertIn(
                "not held out from dataset capture", seed_overlap.stderr
            )
            self.assertFalse((starter / "seed_overlap_pack.yaml").exists())

    def test_resource_builder_rejects_tampered_artifact_component(self):
        recipe = load_recipe(ROOT / "recipes" / "resource_equal_power_baseline.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "resource_bundle"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["tx_power"],
                loss="resource.negative_shannon_spectral_efficiency",
                framework="torch",
                out_dir=bundle,
                project_root=root,
                exporter="resource-allocation",
                include_starter=True,
            )
            starter = bundle / "reference_training"
            component_path = _write_stub_artifact(
                bundle / "trained_artifact.yaml",
                operation="model.symbol_power_allocator",
                entrypoint="power_policy",
            )
            _write_training_evidence(starter)
            component_path.write_bytes(component_path.read_bytes() + b"tampered")

            completed = subprocess.run(
                [sys.executable, "build_benchmark.py"],
                cwd=starter,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("SHA-256 does not match", completed.stderr)
            self.assertFalse((starter / "benchmark_pack.yaml").exists())

    def test_receiver_builder_rejects_unconfined_artifact_component(self):
        recipe = load_recipe(
            ROOT / "recipes" / "neural_receiver_qpsk_iq_calibration.yaml"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "receiver_bundle"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["demodulator"],
                loss="bit.bce",
                framework="torch",
                out_dir=bundle,
                project_root=root,
                exporter="neural-receiver",
                include_starter=True,
            )
            starter = bundle / "reference_training"
            artifact_path = bundle / "trained_artifact.yaml"
            component_path = _write_stub_artifact(
                artifact_path,
                operation="demodulation.neural_receiver_adapter",
                entrypoint="neural_receiver",
            )
            _write_training_evidence(starter)
            escaped_path = root / "escaped.onnx"
            escaped_path.write_bytes(component_path.read_bytes())
            manifest = yaml.safe_load(artifact_path.read_text(encoding="utf-8"))
            manifest["components"][0]["path"] = "../escaped.onnx"
            artifact_path.write_text(
                yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
            )

            completed = subprocess.run(
                [sys.executable, "build_benchmark.py"],
                cwd=starter,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("path escapes the artifact package", completed.stderr)
            self.assertFalse((starter / "benchmark_recipe.yaml").exists())
            self.assertFalse((starter / "benchmark_pack.yaml").exists())

    def test_receiver_builder_explains_missing_training_prerequisites(self):
        recipe = load_recipe(
            ROOT / "recipes" / "neural_receiver_qpsk_iq_calibration.yaml"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "receiver_bundle"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["demodulator"],
                loss="bit.bce",
                framework="torch",
                out_dir=bundle,
                project_root=root,
                exporter="neural-receiver",
                include_starter=True,
            )

            completed = subprocess.run(
                [sys.executable, "build_benchmark.py"],
                cwd=bundle / "reference_training",
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "requires a trained and held-out-evaluated receiver",
                completed.stderr,
            )
            self.assertIn("python train.py", completed.stderr)
            self.assertIn("python evaluate.py", completed.stderr)
            self.assertFalse(
                (bundle / "reference_training" / "benchmark_pack.yaml").exists()
            )

    def test_builders_require_parseable_nonempty_training_evidence(self):
        cases = (
            (
                "resource",
                ROOT / "recipes" / "resource_equal_power_baseline.yaml",
                ["tx_power"],
                "resource.negative_shannon_spectral_efficiency",
                "resource-allocation",
                "model.symbol_power_allocator",
                "power_policy",
            ),
            (
                "receiver",
                ROOT / "recipes" / "neural_receiver_qpsk_iq_calibration.yaml",
                ["demodulator"],
                "bit.bce",
                "neural-receiver",
                "demodulation.neural_receiver_adapter",
                "neural_receiver",
            ),
        )
        for (
            label,
            recipe_path,
            optimizable_steps,
            loss,
            exporter,
            operation,
            entrypoint,
        ) in cases:
            with self.subTest(builder=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bundle = root / (label + "_bundle")
                export_differentiable_scenario(
                    load_recipe(recipe_path),
                    build_registry(),
                    optimizable_steps=optimizable_steps,
                    loss=loss,
                    framework="torch",
                    out_dir=bundle,
                    project_root=root,
                    exporter=exporter,
                    include_starter=True,
                )
                starter = bundle / "reference_training"
                _write_stub_artifact(
                    bundle / "trained_artifact.yaml",
                    operation=operation,
                    entrypoint=entrypoint,
                )
                _write_training_evidence(starter)
                (starter / "training_history.json").write_text(
                    "[]", encoding="utf-8"
                )

                completed = subprocess.run(
                    [sys.executable, "build_benchmark.py"],
                    cwd=starter,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(
                    "training history must contain a non-empty JSON",
                    completed.stderr,
                )
                self.assertFalse((starter / "benchmark_pack.yaml").exists())

                (starter / "training_history.json").write_text(
                    json.dumps([{"epoch": 1, "loss": 0.1}]), encoding="utf-8"
                )
                (starter / "evaluation_metrics.json").write_text(
                    "{not-json", encoding="utf-8"
                )
                completed = subprocess.run(
                    [sys.executable, "build_benchmark.py"],
                    cwd=starter,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(
                    "evaluation metrics must contain valid JSON", completed.stderr
                )
                self.assertFalse((starter / "benchmark_pack.yaml").exists())

                (starter / "evaluation_metrics.json").write_text(
                    json.dumps(
                        {
                            "primary_metric": 0.1,
                            "test_capture_schema_sha256": ["a" * 64],
                        }
                    ),
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    [sys.executable, "build_benchmark.py"],
                    cwd=starter,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                expected = (
                    "trained_artifact binding evidence"
                    if label == "resource"
                    else "component_sha256 binding evidence"
                )
                self.assertIn(expected, completed.stderr)
                self.assertFalse((starter / "benchmark_pack.yaml").exists())

    def test_builders_reject_evaluation_from_a_different_component(self):
        cases = (
            (
                "resource",
                ROOT / "recipes" / "resource_equal_power_baseline.yaml",
                ["tx_power"],
                "resource.negative_shannon_spectral_efficiency",
                "resource-allocation",
                "model.symbol_power_allocator",
                "power_policy",
            ),
            (
                "receiver",
                ROOT / "recipes" / "neural_receiver_qpsk_iq_calibration.yaml",
                ["demodulator"],
                "bit.bce",
                "neural-receiver",
                "demodulation.neural_receiver_adapter",
                "neural_receiver",
            ),
        )
        for (
            label,
            recipe_path,
            optimizable_steps,
            loss,
            exporter,
            operation,
            entrypoint,
        ) in cases:
            with self.subTest(builder=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bundle = root / (label + "_bundle")
                export_differentiable_scenario(
                    load_recipe(recipe_path),
                    build_registry(),
                    optimizable_steps=optimizable_steps,
                    loss=loss,
                    framework="torch",
                    out_dir=bundle,
                    project_root=root,
                    exporter=exporter,
                    include_starter=True,
                )
                starter = bundle / "reference_training"
                _write_stub_artifact(
                    bundle / "trained_artifact.yaml",
                    operation=operation,
                    entrypoint=entrypoint,
                )
                _write_training_evidence(starter)
                evaluation_path = starter / "evaluation_metrics.json"
                evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                if label == "resource":
                    evaluation["trained_artifact"]["components"][0]["sha256"] = "f" * 64
                else:
                    evaluation["component_sha256"] = "f" * 64
                evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

                completed = subprocess.run(
                    [sys.executable, "build_benchmark.py"],
                    cwd=starter,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(
                    "component SHA-256 does not match the selected artifact component",
                    completed.stderr,
                )
                self.assertFalse((starter / "benchmark_pack.yaml").exists())

    def test_receiver_builder_rejects_stale_held_out_capture_schema(self):
        recipe = load_recipe(
            ROOT / "recipes" / "neural_receiver_qpsk_iq_calibration.yaml"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "receiver_bundle"
            export_differentiable_scenario(
                recipe,
                build_registry(),
                optimizable_steps=["demodulator"],
                loss="bit.bce",
                framework="torch",
                out_dir=bundle,
                project_root=root,
                exporter="neural-receiver",
                include_starter=True,
            )
            starter = bundle / "reference_training"
            _write_stub_artifact(
                bundle / "trained_artifact.yaml",
                operation="demodulation.neural_receiver_adapter",
                entrypoint="neural_receiver",
            )
            _write_training_evidence(starter)
            config = yaml.safe_load(
                (starter / "train_config.yaml").read_text(encoding="utf-8")
            )
            test_capture = (
                starter / config["data"]["test_capture_dirs"][0]
            ).resolve()
            test_capture.mkdir(parents=True)
            (test_capture / "schema.json").write_text(
                json.dumps({"kind": "noema.capture_dataset", "split": "test"}),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, "build_benchmark.py"],
                cwd=starter,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "test capture schema SHA-256 does not match the configured held-out capture",
                completed.stderr,
            )
            self.assertFalse((starter / "benchmark_pack.yaml").exists())

    def test_campaign_builders_reject_unknown_artifact_abi_versions(self):
        cases = (
            (
                ROOT
                / "demo_trainings"
                / "resource_allocation_unsupervised_shannon"
                / "build_benchmark.py",
                "model.symbol_power_allocator",
                "power_policy",
            ),
            (
                ROOT
                / "demo_trainings"
                / "neural_receiver_supervised_qpsk"
                / "build_benchmark.py",
                "demodulation.neural_receiver_adapter",
                "neural_receiver",
            ),
            (
                ROOT
                / "demo_trainings"
                / "modulation_recognition_supervised_cnn"
                / "build_benchmark.py",
                "model.modulation_classifier_adapter",
                "modulation_classifier",
            ),
        )
        for builder_path, operation, entrypoint in cases:
            with self.subTest(builder=builder_path.parent.name), tempfile.TemporaryDirectory() as tmp:
                manifest_path = Path(tmp) / "trained_artifact.yaml"
                _write_stub_artifact(
                    manifest_path,
                    operation=operation,
                    entrypoint=entrypoint,
                )
                payload = yaml.safe_load(
                    manifest_path.read_text(encoding="utf-8")
                )
                payload["runtime"]["abi_version"] = 999
                manifest_path.write_text(
                    yaml.safe_dump(payload, sort_keys=False),
                    encoding="utf-8",
                )
                module = runpy.run_path(str(builder_path))
                with self.assertRaisesRegex(
                    ValueError,
                    "abi_version 999 is unsupported by this campaign",
                ):
                    module["_validate_artifact"](
                        manifest_path,
                        operation,
                        entrypoint,
                        project_root=manifest_path.parent,
                    )


def _write_stub_artifact(
    path: Path, *, operation: str, entrypoint: str
) -> Path:
    artifact_files = path.parent / "artifact_files"
    artifact_files.mkdir(parents=True, exist_ok=True)
    contract_path = artifact_files / "training_contract.yaml"
    contract = {
        "schema_version": 1,
        "kind": "noema.trainable_slot_contract@1",
        "id": "test.post_training_contract",
        "version": 1,
        "artifact_return": {
            "bindings": [{"operation": operation}],
        },
    }
    contract_path.write_text(
        yaml.safe_dump(contract, sort_keys=False), encoding="utf-8"
    )
    component_path = artifact_files / (entrypoint + ".onnx")
    component_path.write_bytes(b"test-onnx-component")
    if operation == "model.symbol_power_allocator":
        component_id = "policy"
        component_role = "power_policy"
        runtime_inputs = [
            {
                "name": "channel_gain",
                "dtype": "float32",
                "shape": ["batch", "subcarrier"],
                "semantic": "per_subcarrier_channel_power_gain",
            },
            {
                "name": "noise_variance",
                "dtype": "float32",
                "shape": ["batch", 1],
                "semantic": "complex_noise_variance",
            },
            {
                "name": "average_power_budget",
                "dtype": "float32",
                "shape": ["batch", 1],
                "semantic": "average_power_per_subcarrier",
            },
        ]
        runtime_outputs = [
            {
                "name": "allocation_scores",
                "dtype": "float32",
                "shape": ["batch", "subcarrier"],
                "semantic": "unconstrained_power_allocation_scores",
            }
        ]
        required_inputs = ["channel_state"]
        binding_params = {
            "policy": "learned_artifact",
            "granularity": "per_subcarrier",
            "budget_mode": "fixed_average",
            "artifact_manifest_path": "trained_artifact.yaml",
            "artifact_entrypoint": "power_policy",
        }
    else:
        component_id = entrypoint + "_component"
        component_role = entrypoint
        runtime_inputs = [
            {
                "name": "input",
                "dtype": "float32",
                "shape": ["batch", 1],
                "semantic": "test_input",
            }
        ]
        runtime_outputs = [
            {
                "name": "output",
                "dtype": "float32",
                "shape": ["batch", 1],
                "semantic": "test_output",
            }
        ]
        required_inputs = []
        binding_params = {}
    payload = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": "test.post_training_artifact",
        "name": "Test post-training artifact",
        "contract": {
            "id": contract["id"],
            "version": contract["version"],
            "path": "artifact_files/training_contract.yaml",
            "sha256": _canonical_sha256(contract),
            "file_sha256": _file_sha256(contract_path),
        },
        "components": [
            {
                "id": component_id,
                "role": component_role,
                "path": "artifact_files/%s.onnx" % entrypoint,
                "sha256": _file_sha256(component_path),
                "format": "onnx",
            }
        ],
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 1,
            "entrypoints": [
                {
                    "id": entrypoint,
                    "component": component_id,
                    "inputs": runtime_inputs,
                    "outputs": runtime_outputs,
                }
            ],
        },
        "compatible_operations": [
            {
                "operation": operation,
                "runtime_entrypoint": entrypoint,
                "required_inputs": required_inputs,
                "params": binding_params,
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return component_path


def _write_training_evidence(starter: Path) -> None:
    manifest_path = starter.parent / "trained_artifact.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    component = dict(manifest["components"][0])
    (starter / "training_history.json").write_text(
        json.dumps([{"epoch": 1, "training_loss": 0.25}]), encoding="utf-8"
    )
    operation = str(manifest["compatible_operations"][0]["operation"])
    if operation == "model.symbol_power_allocator":
        evaluation = {
            "trained_artifact": {
                "manifest_sha256": _file_sha256(manifest_path),
                "components": [
                    {
                        "id": component["id"],
                        "sha256": component["sha256"],
                    }
                ],
            },
            "test_capture_schema_sha256": ["a" * 64],
            "primary_metric": 0.1,
        }
    else:
        evaluation = {
            "component_sha256": component["sha256"],
            "test_capture_schema_sha256": ["a" * 64],
            "primary_metric": 0.1,
        }
    (starter / "evaluation_metrics.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    unittest.main()
