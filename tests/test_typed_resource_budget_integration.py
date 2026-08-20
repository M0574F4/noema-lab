from __future__ import annotations

import json
import copy
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from noema_lab.core.benchmarks import (
    BenchmarkError,
    BenchmarkPack,
    BenchmarkRecipe,
    _evaluate_resource_admission,
    _resource_budget_declaration,
    _resource_admission_metrics,
)
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.verification import (
    _CheckRecorder,
    _check_retained_public_benchmark_schema,
    _check_benchmark_resource_admission,
)


ROOT = Path(__file__).resolve().parents[1]
METRIC = "steps.codec.rate.native_codec_bpp"


def _pack(resource_budget: dict) -> tuple[BenchmarkPack, BenchmarkRecipe]:
    entry = BenchmarkRecipe(id="candidate", path=Path("unused.yaml"))
    pack = BenchmarkPack(
        id="typed-resource-test",
        version="1",
        recipes=[entry],
        metadata={"resource_budget": resource_budget},
    )
    return pack, entry


class TypedResourceBudgetIntegrationTests(unittest.TestCase):
    @staticmethod
    def _converted_budget() -> dict:
        return {
            "metric": METRIC,
            "maximum": 0.25,
            "tolerance": 0.0,
            "policy": "reject",
            "metric_unit": "bit/source_pixel",
            "unit": "complex_channel_use/source_pixel",
            "aggregation_policy": "mean_over_declared_source_items",
            "conversion": {
                "kind": (
                    "noema.resource_conversion."
                    "idealized_native_payload_use_proxy"
                ),
                "source_unit": "bit/source_pixel",
                "output_unit": "complex_channel_use/source_pixel",
                "modulation_order": 16,
                "nominal_code_rate": 0.75,
                "executed_bindings": {
                    "modulation_order": 16,
                    "code_rate": 0.75,
                    "binding_identity": "plan:coder-modulator",
                },
            },
        }

    def test_runtime_rejects_cross_unit_budget_without_transform(self) -> None:
        pack, entry = _pack(
            {
                "metric": METRIC,
                "maximum": 0.5,
                "tolerance": 0.0,
                "policy": "reject",
                "metric_unit": "bit/source_pixel",
                "unit": "complex_channel_use/source_pixel",
                "aggregation_policy": "mean_over_declared_source_items",
            }
        )
        with self.assertRaisesRegex(
            BenchmarkError,
            "differs from budget unit.*without an explicit conversion",
        ):
            _resource_budget_declaration(pack, entry)

    def test_runtime_applies_nonidentity_typed_transform_and_retains_it(
        self,
    ) -> None:
        pack, entry = _pack(self._converted_budget())
        admission = _evaluate_resource_admission(
            pack,
            entry,
            {METRIC: 0.9},
            protocol_sha256="a" * 64,
        )
        assert admission is not None
        self.assertAlmostEqual(admission["observed"], 0.3)
        self.assertFalse(admission["admitted"])
        self.assertEqual(
            admission["decision"],
            "rejected_resource_budget",
        )
        contract = admission["unit_contract"]
        self.assertEqual(contract["observed"]["value"], 0.9)
        self.assertEqual(contract["conversion"]["coefficient"], 1.0 / 3.0)
        self.assertEqual(
            contract["conversion"]["executed_bindings"][
                "binding_identity"
            ],
            "plan:coder-modulator",
        )

    def test_production_verifier_recomputes_typed_conversion(self) -> None:
        pack, entry = _pack(self._converted_budget())
        benchmark_json = pack.to_dict()
        protocol_sha256 = canonical_json_sha256(benchmark_json)
        admission = _evaluate_resource_admission(
            pack,
            entry,
            {METRIC: 0.9},
            protocol_sha256=protocol_sha256,
        )
        assert admission is not None
        metrics = {METRIC: 0.9, **_resource_admission_metrics(admission)}
        result = {
            "benchmark": {"sha256": protocol_sha256},
            "recipes": [
                {
                    "id": entry.id,
                    "status": "rejected_resource_budget",
                    "metrics": metrics,
                    "resource_admission": admission,
                }
            ],
        }
        recorder = _CheckRecorder()
        _check_benchmark_resource_admission(
            result,
            benchmark_json,
            recorder,
        )
        self.assertFalse(
            [check for check in recorder.checks if check.status == "error"],
            recorder.checks,
        )

        tampered = copy.deepcopy(result)
        tampered["recipes"][0]["resource_admission"]["unit_contract"][
            "conversion"
        ]["coefficient"] = 1.0
        recorder = _CheckRecorder()
        _check_benchmark_resource_admission(
            tampered,
            benchmark_json,
            recorder,
        )
        self.assertIn(
            "benchmark.resource_admission.unit_contract",
            [
                check.id
                for check in recorder.checks
                if check.status == "error"
            ],
        )

    def test_public_schema_requires_explicit_policy_for_conversion(
        self,
    ) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "benchmark_pack.schema.json").read_text(
                encoding="utf-8"
            )
        )
        validator = Draft202012Validator(schema)
        resource_budget = {
            "metric": METRIC,
            "maximum": 0.5,
            "tolerance": 0.0,
            "policy": "reject",
            "metric_unit": "bit/source_pixel",
            "unit": "complex_channel_use/source_pixel",
            "conversion": {
                "kind": (
                    "noema.resource_conversion."
                    "idealized_native_payload_use_proxy"
                ),
                "source_unit": "bit/source_pixel",
                "output_unit": "complex_channel_use/source_pixel",
                "modulation_order": 4,
                "nominal_code_rate": 0.5,
            },
        }
        payload = {
            "schema_version": 1,
            "id": "typed-resource-schema-test",
            "version": "1",
            "dataset": {"id": "synthetic"},
            "task": {},
            "metrics": [{"id": "rate.native_codec_bpp"}],
            "recipes": [{"id": "candidate", "path": "unused.yaml"}],
            "metadata": {"resource_budget": resource_budget},
        }
        errors = list(validator.iter_errors(payload))
        self.assertTrue(
            any("aggregation_policy" in error.message for error in errors),
            errors,
        )
        resource_budget["aggregation_policy"] = (
            "mean_over_declared_source_items"
        )
        validator.validate(payload)

    def test_production_verifier_rejects_off_schema_retained_pack(
        self,
    ) -> None:
        benchmark_paths = sorted(
            (
                ROOT
                / "paper"
                / "evidence"
                / "results"
                / "current_profile_v1"
                / "benchmarks"
            ).glob("*/benchmark.json")
        )
        self.assertEqual(len(benchmark_paths), 1, benchmark_paths)
        benchmark = json.loads(
            benchmark_paths[0].read_text(encoding="utf-8")
        )
        benchmark["dataset"]["access_policy"]["undeclared_control"] = True
        recorder = _CheckRecorder()
        self.assertFalse(
            _check_retained_public_benchmark_schema(benchmark, recorder)
        )
        self.assertIn(
            "benchmark.protocol.public_schema",
            [
                check.id
                for check in recorder.checks
                if check.status == "error"
            ],
        )


if __name__ == "__main__":
    unittest.main()
