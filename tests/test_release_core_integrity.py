import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from noema_lab import __version__
from noema_lab.cli.main import main
from noema_lab.core.artifacts import Artifact, artifact, file_sha256
from noema_lab.core.executor import (
    LocalExecutor,
    _validate_operation_input_metadata,
)
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationRegistry,
    OperationResult,
)
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.reproducibility import environment_snapshot
from noema_lab.core.storage import LocalStore
from noema_lab.core.verification import verify_run_bundle


class _EvidenceOperation(Operation):
    id = "test.release_integrity_evidence"
    name = "Release integrity evidence"
    output_kinds = {"value": "test.value"}
    output_metadata_guarantees = {
        "value": ["shape", "nested.token"],
    }

    def run(self, ctx: OperationContext) -> OperationResult:
        path = ctx.output_path("value", ".bin")
        path.write_bytes(b"release-integrity")
        return OperationResult(
            outputs={
                "value": artifact(
                    "test.value",
                    path,
                    {"shape": [1], "nested": {"token": "present"}},
                )
            },
            metrics={"quality.release_score": 1.0},
        )


class _BrokenGuaranteeOperation(_EvidenceOperation):
    id = "test.release_integrity_broken_guarantee"
    name = "Broken release integrity guarantee"

    def run(self, ctx: OperationContext) -> OperationResult:
        path = ctx.output_path("value", ".bin")
        path.write_bytes(b"missing-metadata")
        return OperationResult(
            outputs={"value": artifact("test.value", path, {"shape": [1]})}
        )


class _MetadataConsumerOperation(Operation):
    id = "test.release_integrity_consumer"
    name = "Release integrity consumer"
    input_kinds = {"value": ["test.value"]}
    input_metadata_requirements = {
        "value": {
            "all_of": ["nested.token"],
            "any_of": ["shape", "original_shape"],
        }
    }

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


class ReleaseCoreIntegrityTests(unittest.TestCase):
    def _completed_run(self, root: Path):
        registry = OperationRegistry()
        registry.register(_EvidenceOperation())
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "release_integrity_fixture",
                "steps": [
                    {
                        "id": "evidence",
                        "op": _EvidenceOperation.id,
                        "inputs": {},
                        "params": {},
                    }
                ],
            }
        )
        store = LocalStore(root / ".noema")
        run_dir = LocalExecutor(registry, store).run(recipe)
        return store, run_dir

    @staticmethod
    def _refresh_summary_descriptor(run_dir: Path) -> None:
        summary_path = run_dir / "summary.json"
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["summary"] = {
            "kind": "noema.run_summary",
            "relative_path": "summary.json",
            "sha256": file_sha256(summary_path),
            "size_bytes": summary_path.stat().st_size,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_metric_tampering_stays_invalid_after_summary_descriptor_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_dir = self._completed_run(Path(tmp))
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["steps"][0]["metrics"]["quality.release_score"] = 0.0
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._refresh_summary_descriptor(run_dir)

            report = verify_run_bundle(store, run_dir.name)

            self.assertEqual(report["status"], "invalid")
            self.assertTrue(
                any("metric values differ" in message for message in report["errors"]),
                report,
            )

    def test_output_metadata_tampering_stays_invalid_after_descriptor_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_dir = self._completed_run(Path(tmp))
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["steps"][0]["outputs"]["value"]["metadata"]["shape"] = [999]
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._refresh_summary_descriptor(run_dir)

            report = verify_run_bundle(store, run_dir.name)

            self.assertEqual(report["status"], "invalid")
            self.assertTrue(
                any("summary metadata differs" in message for message in report["errors"]),
                report,
            )

    def test_runtime_rejects_missing_producer_metadata_guarantee(self):
        registry = OperationRegistry()
        registry.register(_BrokenGuaranteeOperation())
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "broken_metadata_guarantee",
                "steps": [
                    {
                        "id": "producer",
                        "op": _BrokenGuaranteeOperation.id,
                        "inputs": {},
                        "params": {},
                    }
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(OperationError, "metadata guarantees"):
                LocalExecutor(registry, LocalStore(Path(tmp) / ".noema")).run(recipe)

    def test_runtime_rejects_missing_consumer_metadata_requirement(self):
        operation = _MetadataConsumerOperation()
        missing = Artifact(
            kind="test.value",
            path=Path("unused.bin"),
            metadata={"shape": [1]},
            sha256="0" * 64,
        )
        with self.assertRaisesRegex(OperationError, "required runtime metadata"):
            _validate_operation_input_metadata(
                {"value": missing},
                operation=operation,
                step_id="consumer",
            )

    def test_run_summary_and_manifest_versions_and_kinds_are_fail_closed(self):
        mutations = (
            ("summary", "schema_version", 999, "schema version"),
            ("summary", "kind", "not.noema", "summary kind"),
            ("manifest", "schema_version", 999, "schema version"),
            ("manifest", "kind", "not.noema", "manifest kind"),
        )
        for filename, field, value, expected_error in mutations:
            with self.subTest(filename=filename, field=field), tempfile.TemporaryDirectory() as tmp:
                store, run_dir = self._completed_run(Path(tmp))
                path = run_dir / (filename + ".json")
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload[field] = value
                path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if filename == "summary":
                    self._refresh_summary_descriptor(run_dir)

                report = verify_run_bundle(store, run_dir.name)

                self.assertEqual(report["status"], "invalid")
                self.assertTrue(
                    any(expected_error in message for message in report["errors"]),
                    report,
                )

    def test_unknown_run_status_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, run_dir = self._completed_run(Path(tmp))
            for filename in ("summary.json", "manifest.json"):
                path = run_dir / filename
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["status"] = "mysterious"
                path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            self._refresh_summary_descriptor(run_dir)

            report = verify_run_bundle(store, run_dir.name)

            self.assertEqual(report["status"], "invalid")
            self.assertTrue(any("not recognized" in item for item in report["errors"]))

    def test_run_inspection_rejects_traversal_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalStore(root / ".noema")
            store.ensure()
            outside = root / "outside"
            outside.mkdir()
            (outside / "summary.json").write_text('{"secret": true}', encoding="utf-8")
            (outside / "manifest.json").write_text('{"secret": true}', encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                store.get_run("../../outside")
            with self.assertRaises(FileNotFoundError):
                store.get_manifest("../../outside")

            linked_run = store.runs_dir / "linked"
            linked_run.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(FileNotFoundError):
                store.get_run("linked")
            with self.assertRaises(FileNotFoundError):
                verify_run_bundle(store, "linked")

            real_run = store.runs_dir / "real"
            real_run.mkdir()
            (real_run / "summary.json").symlink_to(outside / "summary.json")
            with self.assertRaises(FileNotFoundError):
                store.get_run("real")

    def test_cli_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".noema"
            (workspace / "runs").mkdir(parents=True)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch(
                "noema_lab.cli.main.build_registry",
                return_value=OperationRegistry(),
            ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "--workspace",
                        str(workspace),
                        "runs",
                        "show",
                        "../../outside",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("must be a directory name", stderr.getvalue())

    def test_strict_research_validation_returns_nonzero_when_invalid(self):
        invalid = {"status": "invalid", "errors": ["unknown"], "warnings": []}
        stdout = io.StringIO()
        with mock.patch(
            "noema_lab.cli.main.build_registry",
            return_value=OperationRegistry(),
        ), mock.patch(
            "noema_lab.cli.main.load_research_catalog",
            return_value=object(),
        ), mock.patch(
            "noema_lab.cli.main.load_recipe",
            return_value=SimpleNamespace(name="invalid_research"),
        ), mock.patch(
            "noema_lab.cli.main.validate_recipe_against_registry"
        ), mock.patch(
            "noema_lab.cli.main.research_specs_from_recipe",
            return_value={},
        ), mock.patch(
            "noema_lab.cli.main.validate_research_specs_against_catalog",
            return_value=invalid,
        ), contextlib.redirect_stdout(stdout):
            code = main(["research", "validate-recipe", "recipe.yaml", "--strict"])
        self.assertEqual(code, 1)
        self.assertEqual(
            json.loads(stdout.getvalue())["catalog_validation"]["status"],
            "invalid",
        )

    def test_json_suite_errors_are_emitted_as_json(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["suite", "show", "definitely_missing", "--json"])
        self.assertEqual(code, 1)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "error")
        self.assertIn("unknown suite", payload["error"]["message"])

    def test_kodak_limit_is_bounded_before_download(self):
        for limit in ("0", "25"):
            with self.subTest(limit=limit), mock.patch(
                "noema_lab.cli.main.download_kodak_dataset"
            ) as download, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["data", "fetch", "kodak", "--limit", limit])
                self.assertEqual(raised.exception.code, 2)
                download.assert_not_called()

    def test_verify_cli_returns_zero_only_for_valid(self):
        report = {
            "status": "warning",
            "target_type": "run",
            "target_id": "warning-run",
            "path": ".noema/runs/warning-run",
            "errors": [],
            "warnings": ["warning"],
            "checks": [],
        }
        with mock.patch(
            "noema_lab.cli.main.build_registry",
            return_value=OperationRegistry(),
        ), mock.patch(
            "noema_lab.cli.main.verify_run_bundle",
            return_value=report,
        ), contextlib.redirect_stdout(io.StringIO()):
            code = main(["runs", "verify", "warning-run", "--json"])
        self.assertEqual(code, 1)

    def test_version_action_uses_package_version(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as raised:
                main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(stdout.getvalue().strip(), "noema %s" % __version__)

    def test_environment_snapshot_covers_backends_hardware_and_safe_runtime_state(self):
        runtime_environment = {
            "OMP_NUM_THREADS": "4",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_VISIBLE_DEVICES": "GPU-private-identifier",
            "AWS_SECRET_ACCESS_KEY": "must-not-be-captured",
        }
        with mock.patch.dict(os.environ, runtime_environment, clear=True):
            snapshot = environment_snapshot()

        for dependency in (
            "jsonschema",
            "matplotlib",
            "numpy",
            "Pillow",
            "PyYAML",
            "torch",
            "sionna",
            "onnxruntime",
            "openvino",
            "transformers",
            "ultralytics",
            "pyarrow",
        ):
            self.assertIn(dependency, snapshot["dependencies"])
        self.assertIn("neural", snapshot["dependency_groups"]["optional"])
        self.assertIn("cpu", snapshot["hardware"])
        self.assertIn("accelerators", snapshot["hardware"])
        self.assertIn("linear_algebra", snapshot["runtime"])
        recorded_environment = snapshot["runtime"][
            "thread_and_determinism_environment"
        ]
        self.assertEqual(recorded_environment["OMP_NUM_THREADS"], "4")
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", recorded_environment)
        self.assertIsInstance(recorded_environment["CUDA_VISIBLE_DEVICES"], dict)
        self.assertNotIn(
            "GPU-private-identifier",
            json.dumps(recorded_environment),
        )


if __name__ == "__main__":
    unittest.main()
