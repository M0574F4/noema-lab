from __future__ import annotations

import json
import hashlib
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from noema_lab.core.operations import Operation, OperationRegistry, object_schema
from noema_lab.core.recipes import Recipe, RecipeStep, load_recipe
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.training import inspect_training_feasibility
from noema_lab.ops import build_registry
from noema_lab.training.contracts import (
    ConstraintSpec,
    NamedValueSpec,
    SignalSpec,
    SlotGroupSpec,
    TensorSpec,
    TrainingContractError,
    TrainingContractOptions,
    compile_training_contract,
    validate_compiled_training_contract,
    write_training_contract_bundle,
    _validator_py,
)
from noema_lab.training.exporter import (
    export_differentiable_scenario,
    inspect_training_capture,
)


ROOT = Path(__file__).resolve().parents[1]


class _ContractGraphOperation(Operation):
    def __init__(
        self,
        operation_id: str,
        *,
        inputs=(),
        outputs=("out",),
        replaceable: bool = False,
        differentiable: bool = True,
        kind: str = "test.tensor",
    ) -> None:
        self.id = operation_id
        self.name = operation_id
        self.input_kinds = {name: [kind] for name in inputs}
        self.output_kinds = {name: kind for name in outputs}
        self.differentiability = {
            "framework": "torch" if differentiable else "numpy",
            "gradient": "full" if differentiable else "none",
            "trainable_params": False,
            "exportable": differentiable,
        }
        self.backends = {
            "benchmark_run": ["numpy"],
            "dataset_capture": ["numpy"],
            "differentiable_export": ["torch"] if differentiable else [],
        }
        self.params_schema = object_schema(
            {
                "artifact_manifest_path": {"type": "string", "default": ""},
                "artifact_entrypoint": {"type": "string", "default": "forward"},
            }
        )
        self.trained_artifact_abi = (
            {
                "component_id": operation_id.replace(".", "_"),
                "component_role": "contract_test_component",
                "entrypoint_id": "forward",
                "required_operation_inputs": list(inputs),
                "inputs": {
                    name: {"dtype": "float32", "shape": ["batch", "feature"]}
                    for name in inputs
                },
                "outputs": {
                    name: {"dtype": "float32", "shape": ["batch", "feature"]}
                    for name in outputs
                },
                "binding_params": {
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": "forward",
                },
            }
            if replaceable
            else {}
        )

    def run(self, ctx):
        raise NotImplementedError("contract-only synthetic operation")


class NeutralTrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.deepjscc = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml")
        cls.resource = load_recipe(ROOT / "recipes" / "resource_water_filling_baseline.yaml")

    def test_generated_validator_uses_canonical_utf8_recipe_hashing(self) -> None:
        namespace: dict[str, object] = {}
        exec(_validator_py(), namespace)
        payload = {"description": "Learned 2×2 MIMO-OFDM estimator"}
        self.assertEqual(
            namespace["_sha"](payload),
            canonical_json_sha256(payload),
        )

    def deepjscc_contract(self, *, scenario_steps=()):
        return compile_training_contract(
            self.deepjscc,
            self.registry,
            options=TrainingContractOptions(
                trainable_steps=("sender", "receiver"),
                framework="torch",
                slot_roles={"sender": "encoder", "receiver": "decoder"},
                slot_groups=(
                    SlotGroupSpec(
                        id="deepjscc_sender_receiver",
                        slots=("sender", "receiver"),
                        joint_training=True,
                        atomic_artifact_return=True,
                    ),
                ),
                tensor_overrides={
                    "sender.outputs.symbols": {
                        "kind": "channel.symbols.complex_numpy",
                        "dtype": "complex64",
                        "shape": ["batch", "symbol_channels", "symbol_height", "symbol_width"],
                        "layout": "NCHW_complex",
                    },
                    "receiver.inputs.symbols": {
                        "kind": "channel.symbols.complex_numpy",
                        "dtype": "complex64",
                        "shape": ["batch", "symbol_channels", "symbol_height", "symbol_width"],
                        "layout": "NCHW_complex",
                    },
                },
                conditioning=(
                    NamedValueSpec(
                        id="snr_db",
                        source="wireless_channel.params.snr_db",
                        dtype="float32",
                        units="dB",
                    ),
                ),
                signals=(
                    SignalSpec(
                        id="reconstruction_for_loss",
                        source="receiver.images",
                        tensor=TensorSpec(
                            kind="image.batch.numpy",
                            dtype="float32",
                            shape=("batch", "channels", "height", "width"),
                            layout="NCHW",
                            domain="continuous_[0,1]",
                        ),
                    ),
                ),
                constraints=(
                    ConstraintSpec(
                        id="average_transmit_power",
                        expression="mean(abs(tx_symbols)**2) = target_power",
                        enforcement="frozen_scenario_projection",
                        scope="batch",
                    ),
                ),
                scenario_steps=tuple(scenario_steps),
            ),
        )

    def diamond_contract_fixture(self, *, branch2_differentiable=True):
        registry = OperationRegistry()
        registry.register(
            _ContractGraphOperation(
                "test.source",
                differentiable=False,
            )
        )
        registry.register(
            _ContractGraphOperation(
                "test.replace_b",
                inputs=("x",),
                replaceable=True,
                differentiable=False,
            )
        )
        registry.register(_ContractGraphOperation("test.branch_c1", inputs=("x",)))
        registry.register(
            _ContractGraphOperation(
                "test.branch_c2",
                inputs=("x",),
                differentiable=branch2_differentiable,
            )
        )
        registry.register(
            _ContractGraphOperation(
                "test.merge_d",
                inputs=("left", "right"),
                replaceable=True,
            )
        )
        registry.register(
            _ContractGraphOperation(
                "test.metric",
                inputs=("prediction", "reference"),
                outputs=("report",),
                differentiable=False,
            )
        )
        recipe = Recipe(
            name="replacement_diamond",
            steps=[
                RecipeStep("a", "test.source"),
                RecipeStep("b", "test.replace_b", inputs={"x": "a.out"}),
                RecipeStep("c1", "test.branch_c1", inputs={"x": "b.out"}),
                RecipeStep("c2", "test.branch_c2", inputs={"x": "b.out"}),
                RecipeStep(
                    "d",
                    "test.merge_d",
                    inputs={"left": "c1.out", "right": "c2.out"},
                ),
                RecipeStep(
                    "evaluation",
                    "test.metric",
                    inputs={"prediction": "d.out", "reference": "a.out"},
                ),
            ],
        )
        return recipe, registry

    def test_contract_is_versioned_and_records_recipe_and_execution_profile(self) -> None:
        compiled = self.deepjscc_contract()
        contract = compiled.contract
        self.assertEqual(contract["kind"], "noema.trainable_slot_contract@1")
        self.assertEqual(contract["schema_version"], 1)
        self.assertEqual(contract["source_recipe"]["name"], self.deepjscc.name)
        self.assertEqual(len(contract["source_recipe"]["sha256"]), 64)
        self.assertEqual(
            contract["source_recipe"]["execution_profile"],
            {"id": "joint_source_channel_symbols", "version": 1},
        )
        self.assertEqual(contract["source_recipe"]["execution_profile_status"], "conformant")
        self.assertEqual(contract["training_policy"]["architecture"], "external")
        self.assertEqual(contract["training_policy"]["loss"], "external")
        self.assertEqual(contract["training_policy"]["trainer"], "external")
        self.assertEqual(len(contract["identity_sha256"]), 64)

        sender_only = compile_training_contract(
            self.deepjscc,
            self.registry,
            options=TrainingContractOptions(trainable_steps=("sender",), framework="torch"),
        )
        self.assertNotEqual(contract["id"], sender_only.contract["id"])
        self.assertNotEqual(contract["identity_sha256"], sender_only.contract["identity_sha256"])

    def test_slots_are_typed_and_grouped_without_model_architecture(self) -> None:
        contract = self.deepjscc_contract().contract
        slots = {slot["step_id"]: slot for slot in contract["trainable_slots"]}
        self.assertEqual(set(slots), {"sender", "receiver"})
        self.assertEqual(slots["sender"]["inputs"]["images"]["dtype"], "uint8")
        self.assertEqual(slots["sender"]["inputs"]["images"]["layout"], "NHWC")
        self.assertEqual(slots["sender"]["outputs"]["symbols"]["layout"], "NCHW_complex")
        self.assertEqual(slots["receiver"]["inputs"]["symbols"]["layout"], "NCHW_complex")
        self.assertEqual(slots["sender"]["architecture"], "unspecified")
        self.assertEqual(
            contract["slot_groups"],
            [
                {
                    "id": "deepjscc_sender_receiver",
                    "slots": ["sender", "receiver"],
                    "joint_training": True,
                    "artifact_application": "all_group_bindings",
                }
            ],
        )
        self.assertEqual(contract["conditioning"][0]["id"], "snr_db")
        self.assertEqual(contract["constraints"][0]["enforcement"], "frozen_scenario_projection")
        self.assertIn("reconstruction_for_loss", {signal["id"] for signal in contract["signals"]})
        self.assertNotIn("model", contract)
        self.assertNotIn("loss", contract)
        self.assertNotIn("optimizer", contract)

    def test_scenario_is_a_port_wired_typed_dag_with_placeholders(self) -> None:
        graph = self.deepjscc_contract().scenario_graph
        self.assertEqual(graph["kind"], "noema.typed_training_scenario_graph@1")
        self.assertNotIn("blocks", graph)
        self.assertEqual(
            graph["topological_order"],
            [
                "sender",
                "tx_power",
                "tx_symbol_boundary",
                "wireless_channel",
                "rx_symbol_boundary",
                "receiver",
            ],
        )
        nodes = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(nodes["sender"]["role"], "trainable_placeholder")
        self.assertEqual(nodes["receiver"]["role"], "trainable_placeholder")
        self.assertEqual(nodes["tx_power"]["role"], "frozen_differentiable")
        self.assertEqual(nodes["wireless_channel"]["role"], "frozen_differentiable")
        self.assertEqual(nodes["sender"]["implementation_owner"], "external")
        self.assertEqual(nodes["sender"]["boundary_kind"], "portable_replacement")
        self.assertEqual(nodes["sender"]["params"], {})
        self.assertFalse(nodes["sender"]["differentiability"]["applicable"])
        self.assertEqual(
            nodes["sender"]["materialization"]["implementation"],
            "researcher_supplied_slot",
        )
        self.assertNotIn("data", nodes)
        self.assertNotIn("evaluation", nodes)
        self.assertNotIn("channel_symbol_count_match", nodes)
        edge = next(
            item
            for item in graph["edges"]
            if item["source"] == {"step_id": "sender", "port": "symbols"}
        )
        self.assertEqual(edge["target"], {"step_id": "tx_power", "port": "symbols"})
        self.assertEqual(edge["tensor"]["kind"], "channel.symbols.complex_numpy")

    def test_minimal_scenario_keeps_every_reconverging_downstream_branch(self) -> None:
        recipe, registry = self.diamond_contract_fixture()
        compiled = compile_training_contract(
            recipe,
            registry,
            options=TrainingContractOptions(
                trainable_steps=("b",),
                signals=(
                    SignalSpec(
                        id="task_loss_signal",
                        source="d.out",
                        tensor=TensorSpec("test.tensor", "float32", ("batch", "feature")),
                    ),
                ),
            ),
        )
        graph = compiled.scenario_graph
        self.assertEqual(graph["topological_order"], ["b", "c1", "c2", "d"])
        boundary_sources = {
            str(item.get("recipe_reference") or "")
            for item in graph["external_inputs"]
        }
        self.assertIn("a.out", boundary_sources)
        self.assertNotIn("c1.out", boundary_sources)
        self.assertNotIn("c2.out", boundary_sources)

    def test_feasibility_checks_every_reconverging_downstream_branch(self) -> None:
        recipe, registry = self.diamond_contract_fixture()
        inspection = inspect_training_feasibility(
            recipe,
            registry,
            optimizable_steps=("b",),
        )
        self.assertEqual(inspection["recommended_mode"], "differentiable_export")
        self.assertEqual(
            inspection["paths"][0]["downstream_route_steps"],
            ["b", "c1", "c2", "d", "evaluation"],
        )

        broken_recipe, broken_registry = self.diamond_contract_fixture(
            branch2_differentiable=False
        )
        broken = inspect_training_feasibility(
            broken_recipe,
            broken_registry,
            optimizable_steps=("b",),
        )
        self.assertEqual(broken["recommended_mode"], "dataset_capture")
        self.assertEqual(
            [item["step_id"] for item in broken["gradient_breaks"]],
            ["c2"],
        )

    def test_explicit_loss_boundary_can_exclude_an_unrelated_hard_metric_sink(self) -> None:
        registry = OperationRegistry()
        registry.register(_ContractGraphOperation("test.source", differentiable=False))
        registry.register(
            _ContractGraphOperation(
                "test.replace", inputs=("x",), replaceable=True, differentiable=False
            )
        )
        registry.register(_ContractGraphOperation("test.clean", inputs=("x",)))
        registry.register(
            _ContractGraphOperation("test.hard", inputs=("x",), differentiable=False)
        )
        registry.register(
            _ContractGraphOperation(
                "test.metric", inputs=("prediction",), differentiable=False
            )
        )
        recipe = Recipe(
            name="multi_sink_route_selection",
            steps=[
                RecipeStep("a", "test.source"),
                RecipeStep("b", "test.replace", inputs={"x": "a.out"}),
                RecipeStep("clean", "test.clean", inputs={"x": "b.out"}),
                RecipeStep("hard", "test.hard", inputs={"x": "b.out"}),
                RecipeStep(
                    "evaluation",
                    "test.metric",
                    inputs={"prediction": "clean.out"},
                ),
                RecipeStep(
                    "faithfulness",
                    "test.metric",
                    inputs={"prediction": "hard.out"},
                ),
            ],
        )
        all_sinks = inspect_training_feasibility(
            recipe, registry, optimizable_steps=("b",)
        )
        clean_sink = inspect_training_feasibility(
            recipe,
            registry,
            optimizable_steps=("b",),
            loss=("evaluation",),
        )
        self.assertEqual(
            all_sinks["loss_step_candidates"], ["evaluation", "faithfulness"]
        )
        self.assertEqual(all_sinks["recommended_mode"], "dataset_capture")
        self.assertEqual(clean_sink["selected_loss_steps"], ["evaluation"])
        self.assertEqual(clean_sink["recommended_mode"], "differentiable_export")
        self.assertEqual(clean_sink["selected_downstream_support_blocks"], ["clean"])

    def test_generic_export_derives_complete_recipe_loss_boundary(self) -> None:
        recipe, registry = self.diamond_contract_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            result = export_differentiable_scenario(
                recipe,
                registry,
                optimizable_steps=("b",),
                loss="researcher.defined",
                framework="torch",
                out_dir=Path(temporary) / "contract",
            )
            graph = result["project_manifest"]["contracts"]["scenario_graph"]
            self.assertEqual(graph["kind"], "noema.typed_training_scenario_graph@1")
            graph_payload = json.loads(
                (Path(temporary) / "contract" / "scenario_graph.json").read_text(
                    encoding="utf-8"
                )
            )
            contract_payload = yaml.safe_load(
                (Path(temporary) / "contract" / "training_contract.yaml").read_text(
                    encoding="utf-8"
                )
            )
            plan_payload = yaml.safe_load(
                (Path(temporary) / "contract" / "training_plan.yaml").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(graph_payload["topological_order"], ["b", "c1", "c2", "d"])
        self.assertNotIn("evaluation", {node["id"] for node in graph_payload["nodes"]})
        self.assertIn(
            "a.out",
            {
                item.get("recipe_reference")
                for item in graph_payload["external_inputs"]
            },
        )
        self.assertEqual(contract_payload["recipe_loss_steps"], ["evaluation"])
        self.assertNotIn("objective", plan_payload)
        self.assertNotIn("starter", plan_payload)
        loss_operands = {
            item["source"]: item["tensor"]["kind"]
            for item in contract_payload["signals"]
            if item.get("purpose") == "external_loss_input"
        }
        self.assertEqual(
            loss_operands,
            {"a.out": "test.tensor", "d.out": "test.tensor"},
        )

    def test_capture_backed_custom_slot_gets_generic_ui_plan_and_data_contract(self) -> None:
        registry = OperationRegistry()
        registry.register(
            _ContractGraphOperation(
                "test.array_source", differentiable=False, kind="test.tensor.numpy"
            )
        )
        registry.register(
            _ContractGraphOperation(
                "test.array_replacement",
                inputs=("x",),
                replaceable=True,
                differentiable=False,
                kind="test.tensor.numpy",
            )
        )
        registry.register(
            _ContractGraphOperation(
                "test.hard_postprocess",
                inputs=("x",),
                differentiable=False,
                kind="test.tensor.numpy",
            )
        )
        registry.register(
            _ContractGraphOperation(
                "test.array_metric",
                inputs=("prediction", "reference"),
                differentiable=False,
                kind="test.tensor.numpy",
            )
        )
        recipe = Recipe(
            name="generic_capture_custom_slot",
            steps=[
                RecipeStep("a", "test.array_source"),
                RecipeStep("b", "test.array_replacement", inputs={"x": "a.out"}),
                RecipeStep("hard", "test.hard_postprocess", inputs={"x": "b.out"}),
                RecipeStep(
                    "evaluation",
                    "test.array_metric",
                    inputs={"prediction": "hard.out", "reference": "a.out"},
                ),
            ],
        )

        inspection = inspect_training_capture(
            recipe,
            registry,
            optimizable_steps=("b",),
            route_loss_steps=("evaluation",),
        )
        self.assertEqual(inspection["mode"], "captured_tensors")
        self.assertTrue(inspection["ready"], inspection["issue"])
        self.assertEqual(
            inspection["required_taps"],
            [
                {
                    "id": "a_out",
                    "from": "a.out",
                    "role": "replacement_input:b.x",
                }
            ],
        )

        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "generic_contract"
            result = export_differentiable_scenario(
                recipe,
                registry,
                optimizable_steps=("b",),
                route_loss_steps=("evaluation",),
                loss="",
                framework="torch",
                out_dir=out_dir,
            )
            data_contract = yaml.safe_load(
                (out_dir / "data_contract.yaml").read_text(encoding="utf-8")
            )
            plan = yaml.safe_load(
                (out_dir / "training_plan.yaml").read_text(encoding="utf-8")
            )
            validation = subprocess.run(
                [sys.executable, "validate_contract.py"],
                cwd=out_dir,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result["training_mode"], "dataset_capture")
        self.assertEqual(result["route_loss_steps"], ["evaluation"])
        self.assertEqual(result["live_route_loss_steps"], [])
        self.assertEqual(plan["loss_steps"], ["evaluation"])
        self.assertEqual(data_contract["mode"], "captured_generic_tensors")
        self.assertEqual(
            [(item["tap_id"], item["reference"]) for item in data_contract["signals"]],
            [("a_out", "a.out")],
        )
        self.assertEqual(len(result["capture_jobs"]), 3)
        self.assertEqual(validation.returncode, 0, validation.stderr)

    def test_standalone_captured_signal_keeps_its_declared_tensor(self) -> None:
        declared = TensorSpec(
            "image.batch.numpy",
            "uint8",
            ("batch", "height", "width", "channels"),
            layout="NHWC",
            domain="integer_[0,255]",
        )
        compiled = compile_training_contract(
            self.deepjscc,
            self.registry,
            options=TrainingContractOptions(
                trainable_steps=("receiver",),
                signals=(
                    SignalSpec(
                        id="supervised_target",
                        source="data.images",
                        tensor=declared,
                        purpose="supervised_target",
                    ),
                ),
            ),
        )
        self.assertEqual(compiled.scenario_graph["topological_order"], ["receiver"])
        boundary = next(
            item
            for item in compiled.scenario_graph["external_inputs"]
            if item.get("recipe_reference") == "data.images"
        )
        self.assertEqual(boundary["purpose"], "supervised_target")
        self.assertEqual(boundary["tensor"], declared.to_dict())

    def test_only_joint_slot_groups_pull_paths_between_replacements(self) -> None:
        recipe, registry = self.diamond_contract_fixture()
        independent = compile_training_contract(
            recipe,
            registry,
            options=TrainingContractOptions(
                trainable_steps=("b", "d"),
                slot_groups=(
                    SlotGroupSpec("b_group", ("b",), joint_training=False),
                    SlotGroupSpec("d_group", ("d",), joint_training=False),
                ),
            ),
        )
        self.assertEqual(independent.scenario_graph["topological_order"], ["b", "d"])

        joint = compile_training_contract(
            recipe,
            registry,
            options=TrainingContractOptions(
                trainable_steps=("b", "d"),
                slot_groups=(SlotGroupSpec("joint", ("b", "d")),),
            ),
        )
        self.assertEqual(joint.scenario_graph["topological_order"], ["b", "c1", "c2", "d"])

    def test_explicit_scenario_steps_must_be_topologically_ordered(self) -> None:
        recipe, registry = self.diamond_contract_fixture()
        with self.assertRaisesRegex(TrainingContractError, "not topologically ordered"):
            compile_training_contract(
                recipe,
                registry,
                options=TrainingContractOptions(
                    trainable_steps=("b",),
                    scenario_steps=("c1", "b"),
                ),
            )

    def test_scenario_subset_declares_external_inputs_and_outputs(self) -> None:
        compiled = self.deepjscc_contract(
            scenario_steps=(
                "sender",
                "tx_power",
                "tx_symbol_boundary",
                "wireless_channel",
                "rx_symbol_boundary",
                "channel_symbol_count_match",
                "receiver",
            )
        )
        graph = compiled.scenario_graph
        self.assertEqual(graph["external_inputs"][0]["recipe_reference"], "data.images")
        outputs = {item["id"]: item for item in graph["external_outputs"]}
        self.assertIn("receiver.images", outputs)
        self.assertEqual(outputs["receiver.images"]["recipe_consumers"], ["evaluation.reconstruction"])

    def test_resource_allocator_contract_is_neutral_but_preserves_constraints(self) -> None:
        compiled = compile_training_contract(
            self.resource,
            self.registry,
            options=TrainingContractOptions(
                trainable_steps=("tx_power",),
                slot_roles={"tx_power": "allocator"},
                conditioning=(
                    NamedValueSpec("noise_variance", "channel_state.params.noise_variance", units="normalized_power"),
                    NamedValueSpec("average_power_budget", "tx_power.params.target_power", units="normalized_power"),
                ),
                constraints=(
                    ConstraintSpec("nonnegative_power", "power >= 0", "runtime_projection"),
                    ConstraintSpec(
                        "sum_power",
                        "sum_subcarrier(power) = subcarrier_count * average_power_budget",
                        "runtime_projection",
                    ),
                ),
            ),
        )
        slot = compiled.contract["trainable_slots"][0]
        self.assertEqual(slot["role"], "allocator")
        self.assertIn("channel_state", slot["inputs"])
        self.assertEqual(slot["inputs"]["channel_state"]["dtype"], "float32")
        self.assertEqual(slot["outputs"]["allocation"]["domain"], "nonnegative")
        self.assertEqual(len(compiled.contract["constraints"]), 2)
        self.assertEqual(compiled.contract["training_policy"]["loss"], "external")
        self.assertEqual(
            compiled.contract["artifact_return"]["bindings"][0]["required_inputs"],
            ["channel_state"],
        )

    def test_writer_produces_self_validating_neutral_bundle_and_package_request(self) -> None:
        compiled = self.deepjscc_contract()
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "bundle"
            result = write_training_contract_bundle(compiled, out_dir)
            expected = {
                "training_contract.yaml",
                "scenario_graph.json",
                "noema_recipe.yaml",
                "project_manifest.yaml",
                "interfaces.py",
                "structured_input.py",
                "validate_contract.py",
                "package_artifact.py",
                "trained_artifact.template.yaml",
                "test_vectors/contract_vectors.json",
                "README.md",
            }
            self.assertEqual(set(result["files"]), expected)
            self.assertTrue(all((out_dir / filename).is_file() for filename in expected))
            manifest = yaml.safe_load((out_dir / "project_manifest.yaml").read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "noema.training_interface_bundle@1")
            self.assertEqual(manifest["external_training"]["architecture"], "not_supplied")
            self.assertEqual(manifest["external_training"]["loss"], "not_supplied")
            self.assertEqual(manifest["external_training"]["trainer"], "not_supplied")
            self.assertFalse(manifest["training"]["trainer_included"])
            self.assertEqual(manifest["training"]["command_status"], "not_supplied")
            self.assertNotIn("command", manifest["training"])
            self.assertEqual(manifest["training"]["instructions"], "README.md")

            interfaces = runpy.run_path(str(out_dir / "interfaces.py"))
            sender_boundary = interfaces["OPERATION_BOUNDARIES"]["sender"]
            sender_runtime = interfaces["RUNTIME_ARTIFACT_INTERFACES"]["sender"]
            self.assertEqual(sender_boundary["inputs"]["images"]["dtype"], "uint8")
            self.assertEqual(
                sender_boundary["inputs"]["images"]["shape"],
                ["batch", "height", "width", "channels"],
            )
            self.assertEqual(sender_runtime["entrypoint_id"], "encoder")
            self.assertEqual(sender_runtime["inputs"]["images"]["dtype"], "float32")
            self.assertEqual(
                sender_runtime["inputs"]["images"]["shape"],
                ["batch", 3, "height", "width"],
            )
            self.assertEqual(set(sender_runtime["outputs"]), {"symbols_ri"})
            self.assertEqual(
                interfaces["RUNTIME_INPUT_BINDINGS"]["sender"]["images"]["resolution"],
                "operation_adapter_required",
            )
            with self.assertRaisesRegex(ValueError, "explicit operation preprocessing"):
                interfaces["prepare_runtime_inputs"]("sender", {"images": object()})
            prepared_images = object()
            self.assertIs(
                interfaces["prepare_runtime_inputs"](
                    "sender",
                    {"images": object()},
                    prepared_values={"images": prepared_images},
                )["images"],
                prepared_images,
            )

            vectors = json.loads(
                (out_dir / "test_vectors" / "contract_vectors.json").read_text(
                    encoding="utf-8"
                )
            )
            sender_vector = next(
                item for item in vectors["slots"] if item["step_id"] == "sender"
            )
            self.assertEqual(
                sender_vector["operation_boundary"]["inputs"]["images"]["dtype"],
                "uint8",
            )
            self.assertEqual(
                sender_vector["runtime_artifact"]["inputs"]["images"]["dtype"],
                "float32",
            )
            readme = (out_dir / "README.md").read_text(encoding="utf-8")
            self.assertIn("Operation boundary versus returned-model runtime ABI", readme)
            self.assertIn("Returned-artifact entrypoint: `encoder`", readme)
            self.assertIn("operation preprocessing required: `images`", readme)
            self.assertIn("## Commands included in this bundle", readme)
            self.assertIn("**No training program is included.**", readme)
            self.assertIn("`python train.py` is not a valid command", readme)
            self.assertIn("python validate_contract.py", readme)
            self.assertIn(
                "python package_artifact.py path/to/model.onnx --format onnx --runtime onnx",
                readme,
            )

            validation = subprocess.run(
                [sys.executable, "validate_contract.py"],
                cwd=out_dir,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validation.returncode, 0, validation.stderr)
            self.assertIn("contract valid", validation.stdout)

            artifact = out_dir / "candidate.onnx"
            artifact.write_bytes(b"not-a-real-model; packaging-helper-contract-test")
            package = subprocess.run(
                [
                    sys.executable,
                    "package_artifact.py",
                    str(artifact),
                    "--format",
                    "onnx",
                    "--runtime",
                    "onnx",
                ],
                cwd=out_dir,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(package.returncode, 0, package.stderr)
            request = yaml.safe_load((out_dir / "artifact_package_request.yaml").read_text(encoding="utf-8"))
            self.assertEqual(request["kind"], "noema.trained_artifact_package_request@1")
            self.assertEqual(request["status"], "requires_runtime_adapter_validation")
            self.assertEqual(
                request["training_contract"]["sha256"],
                compiled.contract_sha256,
            )
            self.assertEqual(len(request["bindings"]), 2)

    def test_generated_validator_rejects_duplicate_yaml_and_json_keys(self) -> None:
        compiled = self.deepjscc_contract()
        cases = (
            (
                "training_contract.yaml",
                "\nkind: overwritten-by-duplicate\n",
                "Duplicate YAML mapping key `kind`",
            ),
            (
                "scenario_graph.json",
                '{"kind":"first","kind":"second"}\n',
                "Duplicate JSON object key `kind`",
            ),
        )
        for filename, replacement, expected_error in cases:
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as temporary:
                    out_dir = Path(temporary) / "bundle"
                    write_training_contract_bundle(compiled, out_dir)
                    path = out_dir / filename
                    if filename.endswith(".yaml"):
                        path.write_text(
                            path.read_text(encoding="utf-8") + replacement,
                            encoding="utf-8",
                        )
                    else:
                        path.write_text(replacement, encoding="utf-8")
                    validation = subprocess.run(
                        [sys.executable, "validate_contract.py"],
                        cwd=out_dir,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(validation.returncode, 0)
                    self.assertIn(expected_error, validation.stderr)

    def test_generated_interface_helper_copies_only_proven_identity_bindings(self) -> None:
        recipe, registry = self.diamond_contract_fixture()
        compiled = compile_training_contract(
            recipe,
            registry,
            options=TrainingContractOptions(
                trainable_steps=("b",),
                tensor_overrides={
                    "a.outputs.out": {
                        "kind": "test.tensor",
                        "dtype": "float32",
                        "shape": ["batch", "feature"],
                    },
                },
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "bundle"
            write_training_contract_bundle(compiled, out_dir)
            interfaces = runpy.run_path(str(out_dir / "interfaces.py"))

        binding = interfaces["RUNTIME_INPUT_BINDINGS"]["b"]["x"]
        self.assertEqual(binding["resolution"], "identity")
        self.assertEqual(binding["source"]["recipe_reference"], "a.out")
        operation_value = object()
        prepared = interfaces["prepare_runtime_inputs"](
            "b",
            {"x": operation_value},
        )
        self.assertIs(prepared["x"], operation_value)
        interfaces["assert_operation_input_ports"]("b", {"x": operation_value})
        interfaces["assert_runtime_output_ports"]("b", {"out": operation_value})
        with self.assertRaisesRegex(ValueError, "Operation boundary b omitted"):
            interfaces["assert_operation_input_ports"]("b", {})

    def test_generated_allocator_interface_separates_capture_dependencies_from_runtime_tensors(self) -> None:
        compiled = compile_training_contract(
            self.resource,
            self.registry,
            options=TrainingContractOptions(trainable_steps=("tx_power",)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "bundle"
            write_training_contract_bundle(compiled, out_dir)
            interfaces = runpy.run_path(str(out_dir / "interfaces.py"))
            readme = (out_dir / "README.md").read_text(encoding="utf-8")

        boundary = interfaces["OPERATION_BOUNDARIES"]["tx_power"]
        runtime = interfaces["RUNTIME_ARTIFACT_INTERFACES"]["tx_power"]
        self.assertEqual(boundary["required_inputs"], ["symbols"])
        self.assertEqual(boundary["artifact_adapter_inputs"], ["channel_state"])
        self.assertEqual(runtime["entrypoint_id"], "power_policy")
        self.assertEqual(
            set(runtime["inputs"]),
            {"channel_gain", "noise_variance", "average_power_budget"},
        )
        self.assertEqual(
            set(interfaces["INTERFACE_CONTRACT"]["slots"]["tx_power"]["recipe_params"]),
            set(next(step for step in self.resource.steps if step.id == "tx_power").params),
        )
        interfaces["assert_artifact_adapter_inputs"](
            "tx_power",
            {"channel_state": object()},
        )
        with self.assertRaisesRegex(ValueError, "captured operation input.*channel_state"):
            interfaces["assert_artifact_adapter_inputs"]("tx_power", {})
        self.assertIn(
            "Captured artifact-adapter input(s): `channel_state` from `channel_state.state`",
            readme,
        )

    def test_writer_refuses_nonempty_directory_without_force(self) -> None:
        compiled = self.deepjscc_contract()
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            (out_dir / "researcher_file.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(TrainingContractError, "not empty"):
                write_training_contract_bundle(compiled, out_dir)
            write_training_contract_bundle(compiled, out_dir, force=True)
            self.assertEqual((out_dir / "researcher_file.txt").read_text(encoding="utf-8"), "keep")

    def test_validator_detects_scenario_graph_mutation(self) -> None:
        compiled = self.deepjscc_contract()
        graph = json.loads(json.dumps(compiled.scenario_graph))
        graph["nodes"] = [node for node in graph["nodes"] if node["id"] != "sender"]
        with self.assertRaisesRegex(TrainingContractError, "SHA-256"):
            validate_compiled_training_contract(compiled.contract, graph)

    def test_generated_validator_and_packager_enforce_contract_to_graph_reference(self) -> None:
        compiled = self.deepjscc_contract()
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "bundle"
            write_training_contract_bundle(compiled, out_dir)
            graph_path = out_dir / "scenario_graph.json"
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            graph["semantics"]["tampered"] = "must be rejected"
            graph_path.write_text(
                json.dumps(graph, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest_path = out_dir / "project_manifest.yaml"
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            manifest["contracts"]["scenario_graph"]["sha256"] = hashlib.sha256(
                json.dumps(graph, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest()
            manifest["contracts"]["scenario_graph"]["file_sha256"] = hashlib.sha256(
                graph_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

            validation = subprocess.run(
                [sys.executable, "validate_contract.py"],
                cwd=out_dir,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(validation.returncode, 0)
            self.assertIn("scenario graph reference mismatch", validation.stderr)

            candidate = out_dir / "candidate.onnx"
            candidate.write_bytes(b"not-a-real-model")
            package = subprocess.run(
                [
                    sys.executable,
                    "package_artifact.py",
                    str(candidate),
                    "--format",
                    "onnx",
                    "--runtime",
                    "onnx",
                ],
                cwd=out_dir,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(package.returncode, 0)
            self.assertFalse((out_dir / "artifact_package_request.yaml").exists())

    def test_signal_and_conditioning_sources_must_resolve(self) -> None:
        with self.assertRaisesRegex(TrainingContractError, "does not resolve"):
            compile_training_contract(
                self.deepjscc,
                self.registry,
                options=TrainingContractOptions(
                    trainable_steps=("sender", "receiver"),
                    signals=(
                        SignalSpec(
                            id="invalid_source",
                            source="sender.images",
                            tensor=TensorSpec("image.batch.numpy", "uint8"),
                        ),
                    ),
                ),
            )

    def test_block_without_portable_replacement_abi_cannot_be_declared_as_slot(self) -> None:
        with self.assertRaisesRegex(TrainingContractError, "not a replaceable block with a trained-artifact ABI"):
            compile_training_contract(
                self.deepjscc,
                self.registry,
                options=TrainingContractOptions(trainable_steps=("wireless_channel",)),
            )

    def test_connected_tensor_overrides_cannot_disagree(self) -> None:
        with self.assertRaisesRegex(TrainingContractError, "Connected tensor overrides"):
            compile_training_contract(
                self.deepjscc,
                self.registry,
                options=TrainingContractOptions(
                    trainable_steps=("sender", "receiver"),
                    tensor_overrides={
                        "sender.outputs.symbols": {
                            "kind": "channel.symbols.complex_numpy",
                            "dtype": "complex64",
                            "shape": ["batch", "channel_use"],
                        },
                        "tx_power.inputs.symbols": {
                            "kind": "channel.symbols.complex_numpy",
                            "dtype": "complex64",
                            "shape": ["channel_use"],
                        },
                    },
                ),
            )


if __name__ == "__main__":
    unittest.main()
