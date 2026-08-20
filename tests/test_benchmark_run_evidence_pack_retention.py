from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from noema_lab.core.benchmarks import (
    BenchmarkError,
    BenchmarkRecipe,
    _benchmark_run_evidence_retained_artifact_paths,
    benchmark_protocol_sha256,
    load_benchmark_pack,
    run_benchmark_pack,
)
from noema_lab.core.operations import OperationRegistry
from noema_lab.core.storage import LocalStore
from noema_lab.ops.source.random_bits import RandomBitsOperation


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkRunEvidencePackRetentionTests(unittest.TestCase):
    @staticmethod
    def _pack_payload(recipe_path: Path):
        return {
            "schema_version": 1,
            "id": "run_evidence_retention_fixture",
            "version": "1",
            "dataset": {
                "id": "synthetic_random_bits",
                "selection_role": "development_regression",
            },
            "task": {
                "id": "bit_transport",
                "kind": "transport_integrity",
                "modality": "bits",
            },
            "metrics": [
                {
                    "id": "source.bit_count",
                    "definition_version": 1,
                    "source_step": "data",
                    "source_operation": "source.random_bits",
                }
            ],
            "recipes": [
                {
                    "id": "retained",
                    "path": str(recipe_path),
                    "params": {
                        "run_evidence": {
                            "retained_artifact_paths": [
                                "artifacts/data/bits.npz"
                            ]
                        }
                    },
                },
                {
                    "id": "default_metric_only",
                    "path": str(recipe_path),
                },
            ],
        }

    def test_pack_schema_accepts_exact_entry_allowlist_and_rejects_unsafe_shape(self):
        schema = json.loads(
            (ROOT / "schemas/benchmark_pack.schema.json").read_text(
                encoding="utf-8"
            )
        )
        payload = self._pack_payload(Path("recipe.yaml"))
        Draft202012Validator(schema).validate(payload)

        invalid = self._pack_payload(Path("recipe.yaml"))
        invalid["recipes"][0]["params"]["run_evidence"][
            "retained_artifact_paths"
        ] = ["../bits.npz"]
        with self.assertRaises(ValidationError):
            Draft202012Validator(schema).validate(invalid)

    def test_runtime_validation_requires_unique_canonical_artifact_paths(self):
        valid = BenchmarkRecipe(
            id="valid",
            path=Path("recipe.yaml"),
            params={
                "run_evidence": {
                    "retained_artifact_paths": [
                        "artifacts/z/state.npz",
                        "artifacts/a/coded_bits.npz",
                    ]
                }
            },
        )
        self.assertEqual(
            _benchmark_run_evidence_retained_artifact_paths(valid),
            [
                "artifacts/a/coded_bits.npz",
                "artifacts/z/state.npz",
            ],
        )
        self.assertEqual(
            _benchmark_run_evidence_retained_artifact_paths(
                BenchmarkRecipe(id="default", path=Path("recipe.yaml"))
            ),
            [],
        )

        invalid_values = (
            {"run_evidence": "artifacts/data/bits.npz"},
            {"run_evidence": {}},
            {
                "run_evidence": {
                    "retained_artifact_paths": ["artifacts/data/../bits.npz"]
                }
            },
            {
                "run_evidence": {
                    "retained_artifact_paths": ["artifacts//data/bits.npz"]
                }
            },
            {
                "run_evidence": {
                    "retained_artifact_paths": [
                        "artifacts/data/bits.npz",
                        "artifacts/data/bits.npz",
                    ]
                }
            },
        )
        for params in invalid_values:
            with self.subTest(params=params), self.assertRaises(BenchmarkError):
                _benchmark_run_evidence_retained_artifact_paths(
                    BenchmarkRecipe(
                        id="invalid",
                        path=Path("recipe.yaml"),
                        params=params,
                    )
                )

    def test_ordinary_execution_retains_only_the_opted_in_entry_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = root / "recipe.yaml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "name": "run_evidence_retention_recipe",
                        "steps": [
                            {
                                "id": "data",
                                "op": "source.random_bits",
                                "params": {"bit_count": 16, "seed": 7},
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            pack_path = root / "benchmark.yaml"
            pack_path.write_text(
                yaml.safe_dump(
                    self._pack_payload(recipe_path),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            pack = load_benchmark_pack(pack_path)
            registry = OperationRegistry()
            registry.register(RandomBitsOperation())
            result_dir = run_benchmark_pack(
                pack,
                registry,
                LocalStore(root / "workspace"),
                ROOT,
            )

            result = json.loads(
                (result_dir / "result.json").read_text(encoding="utf-8")
            )
            rows = {row["id"]: row for row in result["recipes"]}
            retained_root = result_dir / rows["retained"][
                "run_evidence_snapshot"
            ]["root"]
            default_root = result_dir / rows["default_metric_only"][
                "run_evidence_snapshot"
            ]["root"]
            self.assertTrue(
                (retained_root / "artifacts/data/bits.npz").is_file()
            )
            self.assertFalse(
                (default_root / "artifacts/data/bits.npz").exists()
            )

            retained_snapshot = json.loads(
                (retained_root / "snapshot.json").read_text(encoding="utf-8")
            )
            default_snapshot = json.loads(
                (default_root / "snapshot.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                retained_snapshot["artifact_projection"][
                    "explicit_artifact_paths"
                ],
                ["artifacts/data/bits.npz"],
            )
            self.assertEqual(
                default_snapshot["artifact_projection"][
                    "explicit_artifact_paths"
                ],
                [],
            )

            frozen_pack = json.loads(
                (result_dir / "benchmark.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                result["benchmark"]["sha256"],
                benchmark_protocol_sha256(frozen_pack),
            )
            self.assertEqual(
                frozen_pack["recipes"][0]["params"]["run_evidence"][
                    "retained_artifact_paths"
                ],
                ["artifacts/data/bits.npz"],
            )


if __name__ == "__main__":
    unittest.main()
