from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from noema_lab.core.external_adapters import (
    ExternalAdapterManifest,
    ExternalAdapterOperationSpec,
    ManifestWrappedOperation,
)
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationResult,
)


class _NoopOperation(Operation):
    id = "test.noop"
    name = "No-op"

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


class ExternalAdapterIdentityTests(unittest.TestCase):
    def test_manifest_and_callable_hashes_are_contract_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "noema_adapter.yaml"
            manifest_path.write_text(
                "schema_version: 1\nname: identity-test\n",
                encoding="utf-8",
            )
            source_path = root / "adapter.py"
            source_path.write_text(
                "def invoke(value, params):\n    return value\n",
                encoding="utf-8",
            )
            operation = ManifestWrappedOperation(
                ExternalAdapterManifest(
                    path=manifest_path,
                    schema_version=1,
                    name="identity-test",
                ),
                ExternalAdapterOperationSpec(
                    id="test.identity_adapter",
                    name="Identity adapter",
                    wraps=_NoopOperation.id,
                    adapter_params={
                        "path": str(source_path),
                        "callable": "invoke",
                    },
                ),
                _NoopOperation(),
            )

            evidence = operation.describe()["external_adapter"]
            self.assertEqual(len(evidence["manifest_sha256"]), 64)
            self.assertEqual(len(evidence["callable_sha256"]), 64)
            self.assertEqual(evidence["callable_source"], str(source_path))

    def test_callable_substitution_after_registration_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "noema_adapter.yaml"
            manifest_path.write_text(
                "schema_version: 1\nname: identity-test\n",
                encoding="utf-8",
            )
            source_path = root / "adapter.py"
            source_path.write_text(
                "def invoke(value, params):\n    return value\n",
                encoding="utf-8",
            )
            operation = ManifestWrappedOperation(
                ExternalAdapterManifest(
                    path=manifest_path,
                    schema_version=1,
                    name="identity-test",
                ),
                ExternalAdapterOperationSpec(
                    id="test.identity_adapter",
                    name="Identity adapter",
                    wraps=_NoopOperation.id,
                    adapter_params={
                        "path": str(source_path),
                        "callable": "invoke",
                    },
                ),
                _NoopOperation(),
            )
            source_path.write_text(
                "def invoke(value, params):\n    return 'substituted'\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                OperationError,
                "adapter bytes changed after registration",
            ):
                operation.run(
                    OperationContext(
                        recipe_name="adapter-identity",
                        step_id="adapter",
                        params={},
                        inputs={},
                        run_dir=root,
                        step_dir=root / "adapter-step",
                    )
                )

    def test_sibling_python_dependency_substitution_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "noema_adapter.yaml"
            manifest_path.write_text(
                "schema_version: 1\nname: transitive-identity-test\n",
                encoding="utf-8",
            )
            source_path = root / "adapter.py"
            source_path.write_text(
                "def invoke(value, params):\n    return value\n",
                encoding="utf-8",
            )
            helper_path = root / "helper.py"
            helper_path.write_text("VALUE = 1\n", encoding="utf-8")
            operation = ManifestWrappedOperation(
                ExternalAdapterManifest(
                    path=manifest_path,
                    schema_version=1,
                    name="transitive-identity-test",
                ),
                ExternalAdapterOperationSpec(
                    id="test.transitive_identity_adapter",
                    name="Transitive identity adapter",
                    wraps=_NoopOperation.id,
                    adapter_params={
                        "path": str(source_path),
                        "callable": "invoke",
                    },
                ),
                _NoopOperation(),
            )
            helper_path.write_text("VALUE = 2\n", encoding="utf-8")

            with self.assertRaisesRegex(
                OperationError,
                "adapter bytes changed after registration",
            ):
                operation.run(
                    OperationContext(
                        recipe_name="adapter-transitive-identity",
                        step_id="adapter",
                        params={},
                        inputs={},
                        run_dir=root,
                        step_dir=root / "adapter-step",
                    )
                )


if __name__ == "__main__":
    unittest.main()
