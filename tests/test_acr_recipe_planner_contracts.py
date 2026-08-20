from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.executor import LocalExecutor
from noema_lab.core.graph import recipe_graph
from noema_lab.core.lint import lint_recipe_invariants
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationRegistry,
    OperationResult,
)
from noema_lab.core.params import validate_params
from noema_lab.core.plan_cache import ExecutionPlanCache
from noema_lab.core.planner import (
    RecipePlanningError,
    plan_recipe,
    validate_recipe_against_registry,
)
from noema_lab.core.recipes import (
    Recipe,
    RecipeStep,
    RecipeValidationError,
    compile_recipe,
    load_recipe,
    recipe_from_dict,
)
from noema_lab.core.research import research_specs_from_recipe
from noema_lab.core.runner_contracts import recipe_runner_support
from noema_lab.core.runtime_readiness import inspect_recipe_run_readiness
from noema_lab.core.storage import LocalStore
from noema_lab.ops import build_registry
from noema_lab.ops.foundation import (
    ClipImageEmbeddingOperation,
    ClipTextEmbeddingOperation,
    SamImageSegmentsOperation,
    TextMaskRepairOperation,
    VlmImageSemanticStateOperation,
)


ROOT = Path(__file__).resolve().parents[1]


class _NoDeclarationOperation(Operation):
    id = "test.no_declaration"
    name = "Python operation without backend declarations"

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


class _MissingShapeProducer(Operation):
    id = "test.image_without_shape_metadata"
    name = "Image producer without a shape guarantee"
    output_kinds = {"images": "image.batch.numpy"}

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


class _MutableAvailabilityOperation(Operation):
    id = "test.mutable_availability"
    name = "Operation with mutable dependency availability"

    def __init__(self, state: dict[str, bool]) -> None:
        self.state = state

    def runtime_availability(self, params):
        if self.state["available"]:
            return {"available": True, "missing": []}
        return {
            "available": False,
            "missing": ["test_dependency"],
            "reason": "test dependency is unavailable",
        }

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


def _single_step_recipe(operation_id: str, params=None):
    step = {"id": "work", "op": operation_id}
    if params is not None:
        step["params"] = dict(params)
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "acr_contract_test",
            "steps": [step],
        }
    )


class RecipeDecodeAndValidationHardeningTests(unittest.TestCase):
    def test_yaml_and_json_duplicate_keys_are_rejected_before_normalization(self):
        yaml_payload = (
            "schema_version: 1\n"
            "name: duplicate_yaml\n"
            "steps:\n"
            "  - id: work\n"
            "    op: test.no_declaration\n"
            "    params:\n"
            "      value: 1\n"
            "      value: 2\n"
        )
        json_payload = (
            '{"schema_version":1,"name":"duplicate_json","steps":['
            '{"id":"work","op":"test.no_declaration","params":{"value":1,"value":2}}]}'
        )
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "duplicate.yaml"
            json_path = Path(tmp) / "duplicate.json"
            yaml_path.write_text(yaml_payload, encoding="utf-8")
            json_path.write_text(json_payload, encoding="utf-8")
            with self.assertRaisesRegex(RecipeValidationError, "Duplicate YAML"):
                load_recipe(yaml_path, mode="strict")
            with self.assertRaisesRegex(RecipeValidationError, "Duplicate JSON"):
                load_recipe(json_path, mode="strict")

    def test_non_finite_and_non_json_native_recipe_values_are_rejected(self):
        payload = {
            "schema_version": 1,
            "name": "invalid_json_value",
            "metadata": {"score": float("nan")},
            "steps": [{"id": "work", "op": "test.no_declaration"}],
        }
        compilation = compile_recipe(payload, mode="strict")
        self.assertFalse(compilation.is_valid)
        self.assertIn("NaN or infinity", compilation.errors[0].message)

        payload["metadata"] = {"items": ("tuple",)}
        compilation = compile_recipe(payload, mode="strict")
        self.assertFalse(compilation.is_valid)
        self.assertIn("unsupported tuple", compilation.errors[0].message)

        schema = {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "additionalProperties": False,
        }
        with self.assertRaisesRegex(OperationError, "NaN or infinity"):
            validate_params("test.number", {"score": float("inf")}, schema)

    def test_yaml_and_json_non_finite_constants_are_rejected(self):
        yaml_payload = (
            "schema_version: 1\nname: nonfinite\nmetadata: {score: .nan}\n"
            "steps: [{id: work, op: test.no_declaration}]\n"
        )
        json_payload = (
            '{"schema_version":1,"name":"nonfinite","metadata":{"score":NaN},'
            '"steps":[{"id":"work","op":"test.no_declaration"}]}'
        )
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "nonfinite.yaml"
            json_path = Path(tmp) / "nonfinite.json"
            yaml_path.write_text(yaml_payload, encoding="utf-8")
            json_path.write_text(json_payload, encoding="utf-8")
            with self.assertRaisesRegex(RecipeValidationError, "NaN or infinity"):
                load_recipe(yaml_path, mode="strict")
            with self.assertRaisesRegex(RecipeValidationError, "non-finite"):
                load_recipe(json_path, mode="strict")


class PlannerAndContractHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()

    def test_cross_field_failures_are_caught_by_validation_preflight(self):
        cases = (
            (
                "wireless.modulation_awgn_observation",
                {"snr_db_min": 10.0, "snr_db_max": 5.0},
                "snr_db_max",
            ),
            (
                "wireless.channel",
                {"noise_mode": "fixed_variance"},
                "requires noise_variance",
            ),
            (
                "wireless.digital_link",
                {"noise_mode": "fixed_variance"},
                "requires noise_variance",
            ),
            (
                "channel.nr_ldpc_encoder",
                {"channel_type": "PUSCH", "codeword_index": 1},
                "codeword_index=0",
            ),
        )
        for operation_id, params, message in cases:
            with self.subTest(operation_id=operation_id):
                with self.assertRaisesRegex(RecipePlanningError, message):
                    validate_recipe_against_registry(
                        _single_step_recipe(operation_id, params),
                        self.registry,
                    )

    def test_unsafe_step_ids_are_rejected_during_planning(self):
        for step_id in ("", "nested/work", r"nested\work", ".", "..", " leading"):
            with self.subTest(step_id=step_id):
                recipe = Recipe(
                    name="unsafe_step_id",
                    steps=[
                        RecipeStep(
                            id=step_id,
                            op="source.text_dataset",
                        )
                    ],
                )
                with self.assertRaisesRegex(
                    RecipePlanningError,
                    "unsafe for artifact storage",
                ):
                    validate_recipe_against_registry(recipe, self.registry)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            executor = LocalExecutor(self.registry, LocalStore(workspace))
            recipe = Recipe(
                name="unsafe_step_id_executor",
                steps=[
                    RecipeStep(
                        id="nested/work",
                        op="source.text_dataset",
                    )
                ],
            )
            with self.assertRaisesRegex(
                RecipeValidationError,
                "unsafe for artifact storage",
            ):
                executor.run(recipe)
            self.assertFalse((workspace / "runs").exists())

    def test_compressai_shape_metadata_requirement_is_planner_checked(self):
        registry = build_registry()
        registry.register(_MissingShapeProducer())
        invalid = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "missing_shape_contract",
                "steps": [
                    {
                        "id": "source",
                        "op": "test.image_without_shape_metadata",
                    },
                    {
                        "id": "compress",
                        "op": "model.compressai_encode",
                        "inputs": {"images": "source.images"},
                    },
                ],
            }
        )
        with self.assertRaisesRegex(
            RecipePlanningError,
            r"requires one of producer metadata.*metadata\.original_shapes",
        ):
            validate_recipe_against_registry(invalid, registry)

        valid = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "declared_shape_contract",
                "steps": [
                    {"id": "source", "op": "source.image_dataset"},
                    {
                        "id": "compress",
                        "op": "model.compressai_encode",
                        "inputs": {"images": "source.images"},
                    },
                ],
            }
        )
        validate_recipe_against_registry(valid, registry)

    def test_absent_backend_declarations_do_not_claim_numpy(self):
        description = _NoDeclarationOperation().describe()
        self.assertEqual(description["backends"]["benchmark_run"], ["python"])
        self.assertNotIn(
            "numpy",
            {
                item["backend"]
                for item in description["materializations"]
                if item["runner"] in {"benchmark_run", "dataset_capture"}
            },
        )

    def test_dead_enum_and_phantom_external_backend_claims_are_removed(self):
        enums = (
            ("foundation.text_semantic_state_encode", "extractor"),
            ("foundation.semantic_state_to_text", "generator"),
            ("foundation.text_mask_repair", "generator"),
            ("foundation.sam_segment", "backend"),
            ("foundation.vlm_image_to_state", "backend"),
        )
        dead_values = {"external_llm", "segment_anything", "external_vlm"}
        for operation_id, parameter in enums:
            choices = set(
                self.registry.get(operation_id)
                .params_schema["properties"][parameter]["enum"]
            )
            self.assertFalse(choices.intersection(dead_values))
        self.assertNotIn(
            "checkpoint_path",
            self.registry.get("foundation.sam_segment")
            .params_schema["properties"],
        )

        adapter_ids = (
            "model.channel_estimator_adapter",
            "model.beamforming_adapter",
            "model.localization_adapter",
            "model.aoa_estimator_adapter",
            "model.resource_allocation_adapter",
            "demodulation.neural_receiver_adapter",
        )
        for operation_id in adapter_ids:
            description = self.registry.get(operation_id).describe()
            advertised = {
                (runner, backend)
                for runner, backends in description["backends"].items()
                for backend in backends
            }
            materialized = {
                (item["runner"], item["backend"])
                for item in description["materializations"]
            }
            self.assertNotIn("external", {backend for _runner, backend in advertised})
            self.assertTrue(advertised.issubset(materialized))

    def test_removed_foundation_modes_fail_loudly_on_direct_operation_calls(self):
        cases = (
            (
                TextMaskRepairOperation(),
                {"generator": "removed_generator"},
                "unsupported",
            ),
            (
                SamImageSegmentsOperation(),
                {"backend": "removed_segmenter"},
                "unsupported",
            ),
            (
                VlmImageSemanticStateOperation(),
                {"backend": "removed_vlm"},
                "unsupported",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for operation, params, expected in cases:
                with self.subTest(operation=operation.id):
                    step_dir = root / operation.id.replace(".", "_")
                    step_dir.mkdir()
                    context = OperationContext(
                        recipe_name="direct_invalid_mode",
                        step_id="work",
                        params=params,
                        inputs={},
                        run_dir=root,
                        step_dir=step_dir,
                    )
                    with self.assertRaisesRegex(OperationError, expected):
                        operation.run(context)

    def test_runtime_availability_blocks_planning_and_is_rechecked_on_cache_hit(self):
        state = {"available": False}
        operation = _MutableAvailabilityOperation(state)
        registry = OperationRegistry()
        registry.register(operation)
        recipe = _single_step_recipe(operation.id)
        with self.assertRaisesRegex(RecipePlanningError, "test dependency"):
            plan_recipe(recipe, registry)

        state["available"] = True
        cache = ExecutionPlanCache()
        first = cache.plan(recipe, registry)
        self.assertEqual(first.evidence.outcome, "miss")
        state["available"] = False
        with self.assertRaisesRegex(RecipePlanningError, "test dependency"):
            cache.plan(recipe, registry)

    def test_runner_support_uses_not_applicable_and_relevant_counts(self):
        registry = OperationRegistry()
        registry.register(_NoDeclarationOperation())
        support = recipe_runner_support(
            _single_step_recipe("test.no_declaration"),
            registry,
        )
        differentiable = support["summary"]["differentiable_export"]
        self.assertFalse(differentiable["applicable"])
        self.assertEqual(differentiable["status"], "not_applicable")
        self.assertFalse(differentiable["supported"])
        self.assertEqual(differentiable["relevant_count"], 0)
        self.assertEqual(differentiable["not_applicable_count"], 1)
        step = support["steps"][0]["runner_support"]["differentiable_export"]
        self.assertEqual(step["status"], "not_applicable")
        self.assertIsNone(step["supported"])

    def test_runner_support_respects_export_only_runtime_selection(self):
        support = recipe_runner_support(
            load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml"),
            build_registry(),
        )

        self.assertFalse(support["summary"]["benchmark_run"]["supported"])
        self.assertFalse(support["summary"]["dataset_capture"]["supported"])
        self.assertTrue(support["summary"]["differentiable_export"]["supported"])
        benchmark_failures = {
            item["step_id"]: item["reason"]
            for item in support["summary"]["benchmark_run"]["unsupported_steps"]
        }
        self.assertEqual(set(benchmark_failures), {"sender", "receiver"})
        self.assertTrue(
            all("training_interface" in reason for reason in benchmark_failures.values())
        )
        self.assertTrue(
            all(
                "runner=differentiable_export" in reason
                for reason in benchmark_failures.values()
            )
        )

    def test_compat_typo_is_not_reported_runnable_by_inspection_surfaces(self):
        registry = OperationRegistry()
        registry.register(_NoDeclarationOperation())
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "compat_typo",
                "metdata": {"seed": 7},
                "steps": [{"id": "work", "op": "test.no_declaration"}],
            }
        )
        readiness = inspect_recipe_run_readiness(recipe, registry)
        self.assertFalse(readiness["runnable"])
        self.assertTrue(
            any("Unknown recipe field `metdata`" in issue for issue in readiness["issues"])
        )
        with self.assertRaisesRegex(RecipeValidationError, "strict-compatible"):
            recipe_graph(recipe, registry)
        with self.assertRaisesRegex(RecipeValidationError, "strict-compatible"):
            research_specs_from_recipe(recipe)
        with self.assertRaisesRegex(RecipeValidationError, "strict-compatible"):
            lint_recipe_invariants(recipe, registry, strict=True)


class DirectOperationDefaultTests(unittest.TestCase):
    def test_clip_direct_run_defaults_match_declared_local_semantic_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_path = root / "texts.json"
            text_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "text.batch",
                        "examples": [{"id": "red-square", "text": "red square"}],
                    }
                ),
                encoding="utf-8",
            )
            image_path = root / "images.npz"
            images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
            images[:, 2:6, 2:6, 0] = 255
            np.savez_compressed(
                image_path,
                images=images,
                metadata_json=json.dumps({"sample_ids": ["red-square"]}),
            )
            text_result = ClipTextEmbeddingOperation().run(
                OperationContext(
                    recipe_name="direct_defaults",
                    step_id="text",
                    params={},
                    inputs={
                        "texts": artifact("text.batch.json", text_path, {})
                    },
                    run_dir=root,
                    step_dir=root / "text",
                )
            )
            image_result = ClipImageEmbeddingOperation().run(
                OperationContext(
                    recipe_name="direct_defaults",
                    step_id="image",
                    params={},
                    inputs={
                        "images": artifact("image.batch.numpy", image_path, {})
                    },
                    run_dir=root,
                    step_dir=root / "image",
                )
            )
        self.assertEqual(text_result.metadata["backend"], "local_semantic")
        self.assertEqual(image_result.metadata["backend"], "local_semantic")
        self.assertEqual(text_result.metadata["dimensions"], 8)
        self.assertEqual(image_result.metadata["dimensions"], 8)


if __name__ == "__main__":
    unittest.main()
