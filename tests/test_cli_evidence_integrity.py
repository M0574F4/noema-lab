from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from noema_lab.cli.main import main
from noema_lab.core.artifacts import file_sha256
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationRegistry,
    OperationResult,
    object_schema,
)


class _CliEvidenceOperation(Operation):
    id = "test.cli_evidence"
    name = "CLI evidence fixture"
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": [],
        "differentiable_export": [],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "cli_evidence_fixture",
            "status": "implemented",
        }
    ]
    params_schema = object_schema(
        {
            "threshold": {
                "type": "number",
                "default": 0.75,
            },
            "fail": {
                "type": "boolean",
                "default": False,
            },
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        if bool(ctx.params.get("fail")):
            raise OperationError("deliberate CLI failure")
        return OperationResult(
            metrics={"fixture.threshold": float(ctx.params["threshold"])}
        )


class CliEvidenceIntegrityTests(unittest.TestCase):
    @staticmethod
    def _registry() -> OperationRegistry:
        registry = OperationRegistry()
        registry.register(_CliEvidenceOperation())
        return registry

    @staticmethod
    def _write_recipe(path: Path, *, fail: bool = False) -> None:
        params = {"fail": True} if fail else {}
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "cli_evidence_fixture",
                    "steps": [
                        {
                            "id": "work",
                            "op": _CliEvidenceOperation.id,
                            "params": params,
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def test_cli_preserves_authored_params_before_default_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".noema"
            recipe_path = root / "recipe.yaml"
            self._write_recipe(recipe_path)
            with patch(
                "noema_lab.cli.main.build_registry",
                return_value=self._registry(),
            ):
                code = main(
                    [
                        "--workspace",
                        str(workspace),
                        "recipe",
                        "run",
                        str(recipe_path),
                    ]
                )
            run_dir = next((workspace / "runs").iterdir())
            authored = json.loads(
                (run_dir / "recipe.authored.json").read_text(encoding="utf-8")
            )
            effective = json.loads(
                (run_dir / "recipe.json").read_text(encoding="utf-8")
            )

        self.assertEqual(code, 0)
        self.assertNotIn("threshold", authored["steps"][0]["params"])
        self.assertEqual(effective["steps"][0]["params"]["threshold"], 0.75)

    def test_cli_failure_prints_persisted_run_identity_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".noema"
            recipe_path = root / "recipe.yaml"
            self._write_recipe(recipe_path, fail=True)
            stderr = io.StringIO()
            with patch(
                "noema_lab.cli.main.build_registry",
                return_value=self._registry(),
            ), contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "--workspace",
                        str(workspace),
                        "recipe",
                        "run",
                        str(recipe_path),
                    ]
                )
            run_dir = next((workspace / "runs").iterdir())
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertIn("failed run: %s" % run_dir.name, stderr.getvalue())
        self.assertIn(str(run_dir / "summary.json"), stderr.getvalue())

    def test_cli_benchmark_supervisor_evidence_does_not_rewrite_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".noema"
            recipe_path = root / "recipe.yaml"
            self._write_recipe(recipe_path)
            pack_path = root / "benchmark.yaml"
            pack_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "id": "cli_benchmark_fixture",
                        "version": "1",
                        "dataset": {},
                        "task": {},
                        "metrics": [],
                        "recipes": [
                            {
                                "id": "candidate",
                                "path": str(recipe_path),
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            result_id = "sealed-cli-result"
            result_dir = workspace / "benchmarks" / result_id
            result_dir.mkdir(parents=True)
            result_path = result_dir / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "noema.benchmark_result",
                        "status": "completed",
                        "benchmark": {
                            "id": "cli_benchmark_fixture",
                            "version": "1",
                        },
                        "recipes": [],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            sealed_sha256 = file_sha256(result_path)
            supervisor = SimpleNamespace(
                run=lambda: SimpleNamespace(
                    payload={"result_id": result_id},
                    evidence={"kind": "fixture-resource-guard"},
                )
            )
            with patch(
                "noema_lab.cli.main.build_registry",
                return_value=self._registry(),
            ), patch(
                "noema_lab.cli.main.IsolatedJobSupervisor",
                return_value=supervisor,
            ):
                code = main(
                    [
                        "--workspace",
                        str(workspace),
                        "benchmark",
                        "run",
                        str(pack_path),
                    ]
                )

            sidecar = json.loads(
                (result_dir / "resource-guard.json").read_text(
                    encoding="utf-8"
                )
            )
            final_sha256 = file_sha256(result_path)

        self.assertEqual(code, 0)
        self.assertEqual(final_sha256, sealed_sha256)
        self.assertEqual(
            sidecar["result"]["result_json_sha256"],
            sealed_sha256,
        )


if __name__ == "__main__":
    unittest.main()
