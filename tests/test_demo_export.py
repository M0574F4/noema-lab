import contextlib
import csv
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.cli.main import main
from noema_lab.core.benchmark_evidence import (
    BenchmarkEvidenceError,
    snapshot_benchmark_training_evidence,
)
from noema_lab.core.benchmark_run_evidence import snapshot_benchmark_run_evidence
from noema_lab.core.benchmarks import (
    BenchmarkPack,
    BenchmarkRecipe,
    benchmark_protocol_sha256,
    run_benchmark_pack,
    semantic_benchmark_recipe_sha256,
)
from noema_lab.core.demo_export import (
    DemoExportError,
    _public_benchmark,
    _render_index_html,
    publish_benchmark_demo,
)
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.core.verification import verify_benchmark_result


class DemoExportTests(unittest.TestCase):
    def test_experimental_claim_never_receives_strongest_profile_badge(self):
        benchmark = {
            "id": "experimental",
            "benchmark_tier": "experimental",
            "publication_ready": True,
        }

        projected = _public_benchmark(benchmark, benchmark)
        rendered = _render_index_html(
            {
                "demo": {"title": "Experimental result"},
                "benchmark": benchmark,
                "verification": {"status": "valid"},
            }
        )

        self.assertFalse(projected["publication_ready"])
        self.assertFalse(projected["traceability_profile_requested"])
        self.assertIn("strongest traceability profile requested: no", rendered)
        self.assertIn('class="verify internal-only"', rendered)
        self.assertNotIn('class="verify profile-requested"', rendered)

    def test_training_evidence_snapshot_rejects_source_changed_during_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_copy = shutil.copyfile

            def unstable_copy(source, destination, *args, **kwargs):
                copied = original_copy(source, destination, *args, **kwargs)
                source_path = Path(source)
                if source_path.name == "training_history.json":
                    source_path.write_text(
                        source_path.read_text(encoding="utf-8") + " ",
                        encoding="utf-8",
                    )
                return copied

            with patch(
                "noema_lab.core.benchmark_evidence.shutil.copyfile",
                side_effect=unstable_copy,
            ):
                with self.assertRaisesRegex(
                    BenchmarkEvidenceError,
                    "changed while it was being copied",
                ):
                    self._stored_fixture(root, with_training_evidence=True)

    def test_benchmark_run_binds_effective_recipe_to_result_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_store, fixture_result_id = self._stored_fixture(
                root, with_training_evidence=True
            )
            fixture_dir = fixture_store.get_benchmark_result_dir(fixture_result_id)
            source_benchmark = json.loads(
                (fixture_dir / "benchmark.json").read_text(encoding="utf-8")
            )
            artifact_source = root / "training_export" / "trained_artifact.yaml"
            pack = BenchmarkPack(
                id="snapshot.binding.test",
                version="1",
                path=Path(source_benchmark["path"]),
                metrics=[
                    {
                        "id": "quality",
                        "definition_version": 1,
                        "source_step": "evaluation",
                        "source_operation": "metrics.synthetic_fixture",
                    },
                    {
                        "id": "ber",
                        "definition_version": 1,
                        "source_step": "evaluation",
                        "source_operation": "metrics.synthetic_fixture",
                    },
                ],
                metadata={"demo": source_benchmark["metadata"]["demo"]},
                recipes=[
                    BenchmarkRecipe(
                        id="learned",
                        path=ROOT / "recipes" / "neural_receiver_qpsk_awgn_adapter.yaml",
                        params={
                            "metadata": {"benchmark_method": "Learned policy"},
                            "step_params": {
                                "demodulator": {
                                    "mode": "learned_artifact",
                                    "artifact_manifest_path": str(
                                        artifact_source.relative_to(root)
                                    ),
                                    "artifact_entrypoint": "neural_receiver",
                                }
                            },
                        },
                    )
                ],
            )
            store = LocalStore(root / "execution-workspace")
            run_dir = store.runs_dir / "synthetic-frozen-run"
            captured = []

            def fake_run(_executor, recipe, **_execution):
                recipe_payload = recipe.to_dict()
                recipe_sha = canonical_json_sha256(recipe_payload)
                authored_sha = canonical_json_sha256(recipe_payload)
                demodulator_source_sha = hashlib.sha256(
                    b"tests.test_demo_export.synthetic_demodulator"
                ).hexdigest()
                evaluator_source_sha = hashlib.sha256(
                    b"tests.test_demo_export.synthetic_evaluator"
                ).hexdigest()
                execution_plan = {
                    "schema_version": 1,
                    "kind": "noema.execution_plan",
                    "runner": "local",
                    "recipe": {"sha256": recipe_sha},
                }
                execution_plan["sha256"] = canonical_json_sha256(execution_plan)
                artifact_sha = hashlib.sha256(artifact_source.read_bytes()).hexdigest()
                recipe_metadata = recipe_payload.get("metadata") or {}
                research = recipe_metadata.get("research")
                summary = {
                    "schema_version": 1,
                    "run_id": run_dir.name,
                    "recipe_name": recipe.name,
                    "status": "completed",
                    "created_at_utc": "2026-07-18T10:00:00Z",
                    "completed_at_utc": "2026-07-18T10:00:01Z",
                    "recipe_sha256": recipe_sha,
                    "metrics": {"power": 1.0, "quality": 2.1, "ber": 0.1},
                    "steps": [
                        {
                            "id": "demodulator",
                            "op": "demodulation.neural_receiver_adapter",
                            "metadata": {
                                "artifact_manifest_sha256": artifact_sha,
                            },
                            "outputs": {},
                            "execution_binding": {
                                "implementation_metadata": {
                                    "source_sha256": demodulator_source_sha,
                                }
                            },
                            "metrics": {},
                        },
                        {
                            "id": "evaluation",
                            "op": "metrics.synthetic_fixture",
                            "outputs": {},
                            "execution_binding": {
                                "implementation_metadata": {
                                    "source_sha256": evaluator_source_sha,
                                }
                            },
                            "metrics": {"quality": 2.1, "ber": 0.1},
                        },
                    ],
                }
                manifest = {
                    "schema_version": 1,
                    "run_id": run_dir.name,
                    "recipe_name": recipe.name,
                    "status": "completed",
                    "created_at_utc": "2026-07-18T10:00:00Z",
                    "completed_at_utc": "2026-07-18T10:00:01Z",
                    "recipe": {
                        "sha256": recipe_sha,
                        "authored_sha256": authored_sha,
                        "research": research,
                    },
                    "execution_plan": {
                        "schema_version": 1,
                        "sha256": execution_plan["sha256"],
                        "runner": "local",
                    },
                    "steps": [],
                    "artifacts": [],
                }
                run_dir.mkdir(parents=True, exist_ok=True)
                runtime_artifact = run_dir / "artifacts" / "demodulator" / "prediction.json"
                runtime_artifact.parent.mkdir(parents=True, exist_ok=True)
                runtime_artifact.write_text("{}", encoding="utf-8")
                manifest["artifacts"] = [
                    {
                        "step_id": "demodulator",
                        "output_name": "prediction",
                        "path": str(runtime_artifact),
                        "relative_path": runtime_artifact.relative_to(run_dir).as_posix(),
                        "sha256": hashlib.sha256(runtime_artifact.read_bytes()).hexdigest(),
                        "metadata": {
                            "artifact_manifest_sha256": artifact_sha,
                        },
                    }
                ]
                for name, payload in (
                    ("recipe.json", recipe_payload),
                    ("recipe.authored.json", recipe_payload),
                    ("summary.json", summary),
                    ("manifest.json", manifest),
                    ("execution-plan.json", execution_plan),
                ):
                    (run_dir / name).write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
                captured.append(recipe_payload)
                return run_dir

            validation = {
                "sha256": benchmark_protocol_sha256(pack),
                "catalog_validation": {"status": "valid", "errors": []},
            }
            with (
                patch(
                    "noema_lab.core.benchmarks.validate_benchmark_pack",
                    return_value=validation,
                ),
                patch("noema_lab.core.benchmarks.LocalExecutor.run", new=fake_run),
            ):
                benchmark_dir = run_benchmark_pack(
                    pack,
                    object(),
                    store,
                    root,
                )

            self.assertEqual(len(captured), 1)
            demodulator = next(
                step for step in captured[0]["steps"] if step["id"] == "demodulator"
            )
            bound_path = Path(demodulator["params"]["artifact_manifest_path"])
            self.assertTrue(bound_path.is_file())
            self.assertTrue(bound_path.is_relative_to(benchmark_dir / "training_evidence"))
            self.assertNotEqual(bound_path.resolve(), artifact_source.resolve())
            result = json.loads(
                (benchmark_dir / "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "completed")
            self.assertIn("training_evidence_snapshot", result)
            completed = result["recipes"][0]
            snapshot = completed["run_evidence_snapshot"]
            self.assertEqual(
                snapshot["semantic_recipe_sha256"],
                completed["semantic_recipe_sha256"],
            )
            snapshot_manifest = json.loads(
                (benchmark_dir / snapshot["manifest"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                snapshot_manifest["semantic_recipe_sha256"],
                completed["semantic_recipe_sha256"],
            )

            shutil.rmtree(store.runs_dir)
            verification = verify_benchmark_result(store, benchmark_dir.name)
            self.assertNotEqual(verification["status"], "invalid")
            with patch(
                "noema_lab.core.verification.verify_run_bundle"
            ) as external_verifier:
                publication = publish_benchmark_demo(
                    store,
                    benchmark_dir.name,
                    root / "portable-publication",
                    "deterministic-demo",
                    project_root=root,
                    allow_warnings=True,
                )
            external_verifier.assert_not_called()
            self.assertTrue(Path(publication["index"]).is_file())

    def test_semantic_recipe_hash_ignores_result_local_snapshot_path(self):
        digest = "b" * 64
        first_path = Path("/tmp/result-one/training_evidence/artifact.yaml")
        second_path = Path("/tmp/result-two/training_evidence/artifact.yaml")

        def recipe_at(path):
            return recipe_from_dict(
                {
                    "schema_version": 1,
                    "name": "semantic-artifact-binding",
                    "steps": [
                        {
                            "id": "model",
                            "op": "model.neural_receiver_adapter",
                            "params": {"artifact_manifest_path": str(path)},
                        }
                    ],
                }
            )

        first = recipe_at(first_path)
        second = recipe_at(second_path)
        self.assertNotEqual(
            canonical_json_sha256(first.to_dict()),
            canonical_json_sha256(second.to_dict()),
        )
        self.assertEqual(
            semantic_benchmark_recipe_sha256(
                first,
                [{"snapshot_path": str(first_path), "sha256": digest}],
            ),
            semantic_benchmark_recipe_sha256(
                second,
                [{"snapshot_path": str(second_path), "sha256": digest}],
            ),
        )

    def test_publish_is_read_only_deterministic_redacted_and_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, result_id = self._stored_fixture(root)
            out_one = root / "publication-one"
            out_two = root / "publication-two"
            benchmark_reports = [
                self._verification("benchmark_result", result_id, root, stamp="first"),
                self._verification("benchmark_result", result_id, root, stamp="second"),
            ]

            with (
                patch("noema_lab.core.demo_export.verify_benchmark_result", side_effect=benchmark_reports),
                patch("noema_lab.core.executor.LocalExecutor.run") as executor_run,
                patch("noema_lab.training.exporter.export_differentiable_scenario") as trainer_export,
            ):
                first = publish_benchmark_demo(
                    store,
                    result_id,
                    out_one,
                    "deterministic-demo",
                    project_root=root,
                )
                second = publish_benchmark_demo(
                    store,
                    result_id,
                    out_two,
                    "deterministic-demo",
                    project_root=root,
                )

            executor_run.assert_not_called()
            trainer_export.assert_not_called()
            self.assertEqual(first["publication_sha256"], second["publication_sha256"])
            self.assertEqual(self._tree_bytes(out_one), self._tree_bytes(out_two))

            all_bytes = b"\n".join(self._tree_bytes(out_one).values())
            self.assertNotIn(str(root).encode("utf-8"), all_bytes)
            demo = json.loads((out_one / "data" / "demo.json").read_text(encoding="utf-8"))
            self.assertEqual(len(demo["runs"]), 3)
            self.assertEqual(len(demo["series"]), 2)
            self.assertTrue(
                all(
                    plot["selection_status"] == "benchmark_protocol"
                    and len(plot["selection_protocol_sha256"]) == 64
                    for plot in demo["plots"]
                )
            )
            learned = next(row for row in demo["series"] if row["label"] == "Learned neural policy")
            self.assertEqual(learned["run_count"], 2)
            self.assertEqual(len(learned["recipe_sha256"]), 2)
            self.assertNotIn("generated_at_utc", (out_one / "evidence" / "verification.json").read_text(encoding="utf-8"))

            rows = list(csv.DictReader((out_one / "data" / "plots" / "quality.csv").read_text(encoding="utf-8").splitlines()))
            learned_point = next(row for row in rows if row["series_id"] == "Learned policy")
            self.assertEqual(learned_point["series"], "Learned neural policy")
            self.assertEqual(learned_point["sample_count"], "2")
            self.assertAlmostEqual(float(learned_point["y_value"]), 2.1)
            self.assertGreater(float(learned_point["y_ci95"]), 0.0)

            ber_rows = list(csv.DictReader((out_one / "data" / "plots" / "ber.csv").read_text(encoding="utf-8").splitlines()))
            zero_row = next(row for row in ber_rows if row["series_id"] == "Learned policy")
            self.assertEqual(float(zero_row["y_value"]), 0.0)
            self.assertGreater(float(zero_row["zero_floor"]), 0.0)
            self.assertIn("Zero values are displayed", (out_one / "figures" / "ber.svg").read_text(encoding="utf-8"))
            index_html = (out_one / "index.html").read_text(encoding="utf-8")
            self.assertIn('href="README.md"', index_html)
            self.assertIn("Open the experiment tutorial", index_html)
            definition_sha = demo["benchmark"]["definition_sha256"]
            self.assertEqual(len(definition_sha), 64)
            self.assertIn(definition_sha, index_html)
            self.assertIn("Portable benchmark definition SHA-256", index_html)
            self.assertIn(
                "verification: valid · tier: unspecified · "
                "strongest traceability profile requested: no",
                index_html,
            )
            self.assertFalse(demo["benchmark"]["publication_ready"])
            self.assertFalse(
                demo["benchmark"]["traceability_profile_requested"]
            )

            manifest = json.loads((out_one / "publication-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["scope"], "all publication files except this manifest")
            listed = {row["path"] for row in manifest["files"]}
            self.assertIn("index.html", listed)
            self.assertIn("data/demo.json", listed)
            self.assertIn("figures/quality.svg", listed)

    def test_publish_preserves_resource_rejections_without_plotting_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, result_id = self._stored_fixture(root)
            result_dir = store.get_benchmark_result_dir(result_id)
            result = store.get_benchmark_result(result_id)
            rejected = self._mark_resource_rejected(
                store, result_dir, result, 1
            )
            store.write_json(result_dir / "result.json", result)
            output = root / "publication"
            with patch(
                "noema_lab.core.demo_export.verify_benchmark_result",
                return_value=self._verification(
                    "benchmark_result", result_id, root
                ),
            ):
                publish_benchmark_demo(
                    store,
                    result_id,
                    output,
                    "deterministic-demo",
                    project_root=root,
                )

            demo = json.loads(
                (output / "data" / "demo.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(demo["runs"]), 2)
            self.assertEqual(len(demo["excluded_methods"]), 1)
            self.assertEqual(
                demo["excluded_methods"][0]["status"],
                "rejected_resource_budget",
            )
            excluded_csv = (
                output / "data" / "excluded_methods.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("rejected_resource_budget", excluded_csv)
            index = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("Excluded by frozen resource admission", index)
            plot_rows = list(
                csv.DictReader(
                    (output / "data" / "plots" / "quality.csv")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            )
            self.assertNotIn(
                rejected["run_id"], {row["run_id"] for row in plot_rows}
            )

    def test_publish_rejects_result_with_no_resource_admitted_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, result_id = self._stored_fixture(root)
            result_dir = store.get_benchmark_result_dir(result_id)
            result = store.get_benchmark_result(result_id)
            for index in range(len(result["recipes"])):
                self._mark_resource_rejected(store, result_dir, result, index)
            store.write_json(result_dir / "result.json", result)
            with patch(
                "noema_lab.core.demo_export.verify_benchmark_result",
                return_value=self._verification(
                    "benchmark_result", result_id, root
                ),
            ):
                with self.assertRaisesRegex(
                    DemoExportError, "no completed, resource-admitted methods"
                ):
                    publish_benchmark_demo(
                        store,
                        result_id,
                        root / "publication",
                        "deterministic-demo",
                        project_root=root,
                    )

    def test_publish_resolves_pack_relative_training_evidence_and_updates_docs_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, result_id = self._stored_fixture(root, with_training_evidence=True)
            result_dir = store.get_benchmark_result_dir(result_id)
            snapshot_manifest = json.loads(
                (result_dir / "training_evidence" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            snapshotted_paths = {row["path"] for row in snapshot_manifest["files"]}
            self.assertTrue(any(path.endswith("artifacts/policy.onnx") for path in snapshotted_paths))
            self.assertTrue(any(path.endswith("artifact_files/training_contract.yaml") for path in snapshotted_paths))
            self.assertTrue(any(path.endswith("MODEL_CARD.md") for path in snapshotted_paths))
            shutil.rmtree(root / "training_export")
            output = root / "docs" / "demo" / "experiments" / "deterministic-demo"
            report = self._verification("benchmark_result", result_id, root)
            with patch(
                "noema_lab.core.demo_export.verify_benchmark_result",
                return_value=report,
            ):
                publish_benchmark_demo(
                    store,
                    result_id,
                    output,
                    "deterministic-demo",
                    project_root=root,
                )

            training_root = output / "evidence" / "training" / "learned-policy"
            self.assertTrue((training_root / "trained_artifact_manifest.json").is_file())
            self.assertTrue((training_root / "training_history.json").is_file())
            self.assertTrue((training_root / "evaluation_metrics.json").is_file())
            history = json.loads(
                (training_root / "training_history.json").read_text(encoding="utf-8")
            )
            self.assertIsInstance(history, list)
            self.assertNotIn(str(root), (training_root / "trained_artifact_manifest.json").read_text(encoding="utf-8"))
            demo_payload = json.loads((output / "data" / "demo.json").read_text(encoding="utf-8"))
            provenance = demo_payload["training_evidence"][0]["runtime_provenance"]
            self.assertEqual(provenance["run_count"], 2)
            self.assertEqual(
                {row["field"] for row in provenance["runs"]},
                {"artifact_manifest_sha256", "checkpoint_sha256"},
            )
            registry = json.loads((output.parent / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["kind"], "noema.demo_registry")
            self.assertEqual(registry["demos"][0]["slug"], "deterministic-demo")
            self.assertEqual(registry["demos"][0]["path"], "deterministic-demo/index.html")
            registry_bytes = (output.parent / "index.json").read_bytes()
            with (
                patch(
                    "noema_lab.core.demo_export.verify_benchmark_result",
                    return_value={**report, "generated_at_utc": "later"},
                ),
            ):
                publish_benchmark_demo(
                    store,
                    result_id,
                    output,
                    "deterministic-demo",
                    project_root=root,
                    force=True,
                )
            self.assertEqual(registry_bytes, (output.parent / "index.json").read_bytes())

    def test_publish_rejects_invalid_and_incomplete_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, result_id = self._stored_fixture(root)
            output = root / "publication"
            invalid = {
                "status": "invalid",
                "errors": ["artifact hash mismatch"],
                "warnings": [],
            }
            with patch("noema_lab.core.demo_export.verify_benchmark_result", return_value=invalid):
                with self.assertRaisesRegex(DemoExportError, "failed verification"):
                    publish_benchmark_demo(store, result_id, output, "deterministic-demo", project_root=root)
            self.assertFalse(output.exists())

            result_path = store.get_benchmark_result_dir(result_id) / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["status"] = "running"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            valid = self._verification("benchmark_result", result_id, root)
            with patch("noema_lab.core.demo_export.verify_benchmark_result", return_value=valid):
                with self.assertRaisesRegex(DemoExportError, "incomplete"):
                    publish_benchmark_demo(store, result_id, output, "deterministic-demo", project_root=root)
            self.assertFalse(output.exists())

    def test_verify_and_publish_reject_tampered_training_evidence_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, result_id = self._stored_fixture(root, with_training_evidence=True)
            result_dir = store.get_benchmark_result_dir(result_id)
            result = store.get_benchmark_result(result_id)
            manifest_spec = result["benchmark"]["metadata"]["demo"][
                "training_evidence"
            ][0]["trained_artifact_manifest"]
            trained_path = result_dir / manifest_spec["path"]
            trained_path.write_text(
                trained_path.read_text(encoding="utf-8") + "\n# tampered\n",
                encoding="utf-8",
            )

            verification = verify_benchmark_result(store, result_id)
            self.assertEqual(verification["status"], "invalid")
            self.assertTrue(
                any("training-evidence snapshot" in error for error in verification["errors"])
            )
            report = self._verification("benchmark_result", result_id, root)
            with patch(
                "noema_lab.core.demo_export.verify_benchmark_result",
                return_value=report,
            ):
                with self.assertRaisesRegex(DemoExportError, "snapshot.*invalid"):
                    publish_benchmark_demo(
                        store,
                        result_id,
                        root / "publication",
                        "deterministic-demo",
                        project_root=root,
                    )
            self.assertFalse((root / "publication").exists())

    def test_verify_and_publish_reject_tampered_run_evidence_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, result_id = self._stored_fixture(root)
            result_dir = store.get_benchmark_result_dir(result_id)
            result = store.get_benchmark_result(result_id)
            descriptor = result["recipes"][0]["run_evidence_snapshot"]
            snapshot_manifest = json.loads(
                (result_dir / descriptor["manifest"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            summary_record = next(
                row for row in snapshot_manifest["files"] if row["role"] == "summary"
            )
            summary_path = result_dir / summary_record["path"]
            summary_path.write_bytes(summary_path.read_bytes() + b" ")

            verification = verify_benchmark_result(store, result_id)
            self.assertEqual(verification["status"], "invalid")
            self.assertTrue(
                any("run-evidence snapshot" in error for error in verification["errors"])
            )
            report = self._verification("benchmark_result", result_id, root)
            with patch(
                "noema_lab.core.demo_export.verify_benchmark_result",
                return_value=report,
            ):
                with self.assertRaisesRegex(DemoExportError, "run-evidence snapshot"):
                    publish_benchmark_demo(
                        store,
                        result_id,
                        root / "publication",
                        "deterministic-demo",
                        project_root=root,
                    )
            self.assertFalse((root / "publication").exists())

    def test_semantic_recipe_hash_is_bound_to_run_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, result_id = self._stored_fixture(root)
            result_path = store.get_benchmark_result_dir(result_id) / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["recipes"][0]["semantic_recipe_sha256"] = "c" * 64
            result_path.write_text(json.dumps(result), encoding="utf-8")

            verification = verify_benchmark_result(store, result_id)
            self.assertEqual(verification["status"], "invalid")
            self.assertTrue(
                any("semantic recipe SHA" in error for error in verification["errors"])
            )

    def test_publish_tutorial_link_allows_paths_and_http_and_rejects_dangerous_schemes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, result_id = self._stored_fixture(root)
            result_path = store.get_benchmark_result_dir(result_id) / "result.json"
            base_result = json.loads(result_path.read_text(encoding="utf-8"))
            report = self._verification("benchmark_result", result_id, root)

            for index, (tutorial, external) in enumerate(
                (
                    ("guides/tutorial.html", False),
                    ("/docs/tutorials/demo.html", False),
                    ("http://example.org/noema/tutorial", True),
                    ("https://example.org/noema/tutorial", True),
                )
            ):
                with self.subTest(tutorial=tutorial):
                    allowed = json.loads(json.dumps(base_result))
                    allowed["benchmark"]["metadata"]["demo"]["tutorial"] = tutorial
                    result_path.write_text(json.dumps(allowed), encoding="utf-8")
                    output = root / ("safe-publication-%d" % index)
                    with (
                        patch(
                            "noema_lab.core.demo_export.verify_benchmark_result",
                            return_value=report,
                        ),
                    ):
                        publish_benchmark_demo(
                            store,
                            result_id,
                            output,
                            "deterministic-demo",
                            project_root=root,
                        )
                    index_html = (output / "index.html").read_text(encoding="utf-8")
                    self.assertIn('href="%s"' % tutorial, index_html)
                    if external:
                        self.assertIn('rel="noopener noreferrer"', index_html)

            for index, dangerous in enumerate(
                (
                    "javascript:alert(1)",
                    "data:text/html,unsafe",
                    "file:///etc/passwd",
                    "//evil.example/tutorial",
                    "https://user:password@example.org/tutorial",
                )
            ):
                with self.subTest(tutorial=dangerous):
                    invalid = json.loads(json.dumps(base_result))
                    invalid["benchmark"]["metadata"]["demo"]["tutorial"] = dangerous
                    result_path.write_text(json.dumps(invalid), encoding="utf-8")
                    with patch(
                        "noema_lab.core.demo_export.verify_benchmark_result",
                        return_value=report,
                    ):
                        with self.assertRaisesRegex(DemoExportError, "tutorial"):
                            publish_benchmark_demo(
                                store,
                                result_id,
                                root / ("unsafe-publication-%d" % index),
                                "deterministic-demo",
                                project_root=root,
                            )

    def test_publish_cli_dispatches_without_running_a_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {
                "index": str(root / "demo" / "index.html"),
                "manifest": str(root / "demo" / "publication-manifest.json"),
                "publication_sha256": "a" * 64,
            }
            output = io.StringIO()
            with (
                patch("noema_lab.cli.main.publish_benchmark_demo", return_value=payload) as publish,
                contextlib.redirect_stdout(output),
            ):
                code = main(
                    [
                        "--workspace",
                        str(root / "workspace"),
                        "benchmark",
                        "publish",
                        "stored-result",
                        "--slug",
                        "demo-slug",
                        "--out",
                        str(root / "demo"),
                        "--allow-warnings",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["publication_sha256"], "a" * 64)
            self.assertEqual(publish.call_args.args[1], "stored-result")
            self.assertTrue(publish.call_args.kwargs["allow_warnings"])

    def _stored_fixture(self, root: Path, with_training_evidence: bool = False):
        workspace = root / "workspace"
        store = LocalStore(workspace)
        result_id = "stored-benchmark-result"
        result_dir = store.get_benchmark_result_dir(result_id)
        result_dir.mkdir(parents=True)

        entries = [
            ("learned_seed_1", "Learned P=1 seed=1", "Learned policy", 1, 2.0, 0.0),
            ("learned_seed_2", "Learned P=1 seed=2", "Learned policy", 2, 2.2, 0.0),
            ("oracle_seed_1", "Oracle P=1 seed=1", "Theoretical oracle", 1, 2.3, 0.1),
        ]
        benchmark_recipes = []
        result_recipes = []
        for entry_id, label, method, seed, quality, ber in entries:
            run_id = "run-" + entry_id
            run_dir = store.runs_dir / run_id
            run_dir.mkdir(parents=True)
            recipe_name = method.lower().replace(" ", "_")
            recipe = {
                "schema_version": 1,
                "name": recipe_name,
                "metadata": {
                    "checkpoint_path": str(root / "private" / "model.onnx"),
                    "pairing_id": str(seed),
                    "aggregation_cell_id": "power=1",
                    "statistical_unit": "paired channel realization",
                },
                "steps": [],
            }
            recipe_sha = canonical_json_sha256(recipe)
            authored_sha = canonical_json_sha256(recipe)
            execution_plan = {
                "schema_version": 1,
                "kind": "noema.execution_plan",
                "runner": "local",
                "recipe": {"sha256": recipe_sha},
            }
            execution_plan["sha256"] = canonical_json_sha256(execution_plan)
            summary = {
                "schema_version": 1,
                "run_id": run_id,
                "recipe_name": recipe_name,
                "status": "completed",
                "created_at_utc": "2026-07-18T10:00:00Z",
                "completed_at_utc": "2026-07-18T10:00:01Z",
                "recipe_sha256": recipe_sha,
                "metrics": {"power": 1.0, "quality": quality, "ber": ber},
                "steps": [],
            }
            manifest = {
                "schema_version": 1,
                "run_id": run_id,
                "recipe_name": recipe_name,
                "status": "completed",
                "created_at_utc": "2026-07-18T10:00:00Z",
                "completed_at_utc": "2026-07-18T10:00:01Z",
                "recipe": {
                    "sha256": recipe_sha,
                    "authored_sha256": authored_sha,
                },
                "execution_plan": {
                    "schema_version": 1,
                    "sha256": execution_plan["sha256"],
                    "runner": "local",
                },
                "seed_policy": {"master_seed": seed},
                "environment": {"process": {"cwd": str(root), "pid": 123}},
                "steps": [],
                "artifacts": [],
            }
            for name, payload in (
                ("recipe.json", recipe),
                ("recipe.authored.json", recipe),
                ("summary.json", summary),
                ("manifest.json", manifest),
                ("execution-plan.json", execution_plan),
            ):
                (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")
            benchmark_recipes.append(
                {
                    "id": entry_id,
                    "label": label,
                    "path": str(root / "recipes" / (entry_id + ".yaml")),
                    "params": {
                        "metadata": {
                            "benchmark_method": method,
                            "paired_seed": seed,
                            "aggregation_cell_id": "power=1",
                            "statistical_unit": "paired channel realization",
                        }
                    },
                }
            )
            result_recipes.append(
                {
                    "id": entry_id,
                    "label": label,
                    "role": "oracle" if "oracle" in entry_id else "candidate",
                    "recipe_name": recipe_name,
                    "recipe_path": str(root / "recipes" / (entry_id + ".yaml")),
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "manifest": str(run_dir / "manifest.json"),
                    "recipe_sha256": recipe_sha,
                    "semantic_recipe_sha256": recipe_sha,
                    "status": "completed",
                    "pairing_id": str(seed),
                    "aggregation_cell_id": "power=1",
                    "statistical_unit": "paired channel realization",
                    "metrics": {"power": 1.0, "quality": quality, "ber": ber},
                }
            )

        demo = {
            "schema_version": 1,
            "slug": "deterministic-demo",
            "title": "Deterministic comparison",
            "summary": "Stored evidence only.",
            "question": "Can a learned policy approach the oracle?",
            "tutorial": "README.md",
            "held_constant": ["channel distribution"],
            "changed": ["allocation method"],
            "primary_metric": "quality",
            "comparison_axis": "power",
            "series": [
                {"id": "Learned policy", "label": "Learned neural policy", "role": "learned"},
                {"id": "Theoretical oracle", "label": "Theoretical water-filling oracle", "role": "oracle"},
            ],
            "table_metrics": ["quality", "ber"],
            "plots": [
                {
                    "id": "quality",
                    "title": "Quality",
                    "kind": "line",
                    "x": "power",
                    "y": "quality",
                    "group": "benchmark_method",
                    "method_order": ["Learned policy", "Theoretical oracle"],
                    "style": {"aggregation": "mean_ci", "y_scale": "linear"},
                },
                {
                    "id": "ber",
                    "title": "BER",
                    "kind": "scatter",
                    "x": "power",
                    "y": "ber",
                    "group": "benchmark_method",
                    "method_order": ["Learned policy", "Theoretical oracle"],
                    "style": {"aggregation": "mean_ci", "y_scale": "log"},
                },
            ],
        }
        pack_path = root / "training_export" / "reference_training" / "benchmark_pack.yaml"
        pack_path.parent.mkdir(parents=True)
        pack_path.write_text("schema_version: 1\n", encoding="utf-8")
        if with_training_evidence:
            model_path = pack_path.parent.parent / "artifacts" / "policy.onnx"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"deterministic-model")
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            contract_path = (
                pack_path.parent.parent / "artifact_files" / "training_contract.yaml"
            )
            contract_path.parent.mkdir(parents=True)
            contract_path.write_text(
                "schema_version: 1\nkind: noema.trainable_slot_contract@1\n",
                encoding="utf-8",
            )
            contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            model_card_path = pack_path.parent.parent / "MODEL_CARD.md"
            model_card_path.write_text("# Deterministic policy\n", encoding="utf-8")
            model_card_sha = hashlib.sha256(model_card_path.read_bytes()).hexdigest()
            trained_manifest = {
                "schema_version": 2,
                "kind": "noema.trained_block_artifact",
                "id": "learned-policy",
                "components": [{"id": "policy", "path": "artifacts/policy.onnx", "sha256": model_sha}],
                "contract": {
                    "id": "test.contract",
                    "path": "artifact_files/training_contract.yaml",
                    "file_sha256": contract_sha,
                },
                "support_files": [
                    {"role": "model_card", "path": "MODEL_CARD.md", "sha256": model_card_sha}
                ],
                "source": {"workspace_path": str(root / "private")},
            }
            trained_path = pack_path.parent.parent / "trained_artifact.yaml"
            trained_path.write_text(yaml.safe_dump(trained_manifest, sort_keys=True), encoding="utf-8")
            trained_sha = hashlib.sha256(trained_path.read_bytes()).hexdigest()
            history_path = pack_path.parent / "training_history.json"
            history_path.write_text(
                json.dumps([{"epoch": 1, "loss": 2.0}, {"epoch": 2, "loss": 1.0}]),
                encoding="utf-8",
            )
            evaluation_path = pack_path.parent / "evaluation_metrics.json"
            evaluation_path.write_text(json.dumps({"quality": 2.1}), encoding="utf-8")
            demo["training_evidence"] = [
                {
                    "series": "Learned policy",
                    "trained_artifact_manifest": {
                        "path": "../trained_artifact.yaml",
                        "sha256": trained_sha,
                    },
                    "training_history": "training_history.json",
                    "evaluation_metrics": "evaluation_metrics.json",
                }
            ]
            for index, run_id in enumerate(("run-learned_seed_1", "run-learned_seed_2")):
                run_dir = store.runs_dir / run_id
                summary_path = run_dir / "summary.json"
                manifest_path = run_dir / "manifest.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if index == 0:
                    metadata = {"artifact_manifest_sha256": trained_sha}
                    summary["steps"] = [
                        {"id": "learned_runtime", "metadata": metadata, "outputs": {}}
                    ]
                else:
                    metadata = {"checkpoint_sha256": trained_sha}
                    summary["steps"] = [
                        {
                            "id": "learned_runtime",
                            "metadata": {},
                            "outputs": {"prediction": {"metadata": metadata}},
                        }
                    ]
                manifest["artifacts"] = [
                    {
                        "step_id": "learned_runtime",
                        "output_name": "prediction",
                        "path": str(run_dir / "artifacts" / "learned_runtime" / "prediction.json"),
                        "relative_path": "artifacts/learned_runtime/prediction.json",
                        "sha256": "",
                        "metadata": metadata,
                    }
                ]
                runtime_artifact = run_dir / "artifacts" / "learned_runtime" / "prediction.json"
                runtime_artifact.parent.mkdir(parents=True, exist_ok=True)
                runtime_artifact.write_text("{}", encoding="utf-8")
                manifest["artifacts"][0]["sha256"] = hashlib.sha256(
                    runtime_artifact.read_bytes()
                ).hexdigest()
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        benchmark = {
            "schema_version": 1,
            "id": "stored.demo",
            "version": "1",
            "name": "Stored demo",
            "path": str(pack_path),
            "metadata": {"demo": demo},
            "recipes": benchmark_recipes,
        }
        for index, row in enumerate(result_recipes):
            row["run_evidence_snapshot"] = snapshot_benchmark_run_evidence(
                result_dir,
                entry_id=str(row["id"]),
                entry_index=index,
                run_dir=store.runs_dir / str(row["run_id"]),
                run_id=str(row["run_id"]),
                semantic_recipe_sha256=str(row["semantic_recipe_sha256"]),
            )
        result = {
            "schema_version": 1,
            "kind": "noema.benchmark_result",
            "benchmark": {
                "id": "stored.demo",
                "version": "1",
                "name": "Stored demo",
                "dataset": {"id": "synthetic"},
                "task": {"id": "allocation"},
                "metrics": [{"id": "quality"}, {"id": "ber"}],
                "metadata": {"demo": demo},
            },
            "status": "completed",
            "created_at_utc": "2026-07-18T10:00:00Z",
            "completed_at_utc": "2026-07-18T10:00:02Z",
            "recipes": result_recipes,
        }
        if with_training_evidence:
            snapshot_benchmark_training_evidence(
                result_dir,
                result,
                benchmark,
                root,
            )
        (result_dir / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
        (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return store, result_id

    @staticmethod
    def _mark_resource_rejected(store, result_dir, result, index):
        row = result["recipes"][index]
        run_dir = store.runs_dir / row["run_id"]
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["steps"] = list(summary.get("steps") or []) + [
            {
                "id": "budget",
                "metrics": {"resource.average_power": 2.0},
                "outputs": {},
            }
        ]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        metric = "steps.budget.resource.average_power"
        row["status"] = "rejected_resource_budget"
        row["metrics"].update(
            {
                "resource.average_power": 2.0,
                metric: 2.0,
                "benchmark.resource_budget.admitted": 0,
                "benchmark.resource_budget.observed": 2.0,
                "benchmark.resource_budget.maximum": 1.0,
                "benchmark.resource_budget.tolerance": 0.0,
                "benchmark.resource_budget.excess": 1.0,
            }
        )
        row["resource_admission"] = {
            "admitted": False,
            "decision": "rejected_resource_budget",
            "metric": metric,
            "observed": 2.0,
            "maximum": 1.0,
            "tolerance": 0.0,
            "excess": 1.0,
            "policy": "maximum",
            "unit": "normalized",
            "protocol_sha256": "a" * 64,
        }
        snapshot_root = result_dir / row["run_evidence_snapshot"]["root"]
        shutil.rmtree(snapshot_root)
        row["run_evidence_snapshot"] = snapshot_benchmark_run_evidence(
            result_dir,
            entry_id=str(row["id"]),
            entry_index=index,
            run_dir=run_dir,
            run_id=str(row["run_id"]),
            semantic_recipe_sha256=str(row["semantic_recipe_sha256"]),
        )
        return row

    @staticmethod
    def _verification(target_type, target_id, root, stamp="fixed"):
        return {
            "status": "valid",
            "target_type": target_type,
            "target_id": target_id,
            "path": str(root / "workspace" / target_type / target_id),
            "generated_at_utc": stamp,
            "errors": [],
            "warnings": [],
            "checks": [{"id": "stored", "status": "pass", "message": "verified at %s" % str(root)}],
        }

    @staticmethod
    def _tree_bytes(root):
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        }


if __name__ == "__main__":
    unittest.main()
