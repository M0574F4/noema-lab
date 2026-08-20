from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Union

from noema_lab.core.operations import OperationRegistry
from noema_lab.core.execution_profiles import (
    CUSTOM_EXECUTION_PROFILE_ID,
    JOINT_SOURCE_CHANNEL_SYMBOLS_PROFILE_ID,
    LAYERED_DIGITAL_PROFILE_ID,
    inspect_execution_profile,
)
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import Recipe, RecipeStep
from noema_lab.core.research import research_specs_from_recipe
from noema_lab.core.research_catalog import load_research_catalog
from noema_lab.core.runner_contracts import recipe_runner_support, unsupported_steps_for_runner

JsonDict = Dict[str, Any]


BIT_TRANSPORT_KINDS = {
    "channel.payload_bits.numpy",
    "channel.framed_bits.numpy",
    "channel.coded_bits.numpy",
    "channel.bits.numpy",
    "channel.demod_bits.numpy",
}
SYMBOL_TRANSPORT_KINDS = {
    "channel.symbols.complex_numpy",
    "channel.rx_symbols.complex_numpy",
}
BIT_CHANNEL_OPS = {
    "channel.identity_link",
    "channel.capacity_oracle_digital_link",
    "wireless.digital_link",
}
SYMBOL_CHANNEL_OPS = {
    "channel.identity_symbol_link",
    "wireless.channel",
}
BIT_TO_SYMBOL_MODULATOR_OPS = {
    "modulation.digital_modulate",
    "modulation.identity_modulate",
    "modulation.qpsk_pilot_modulate",
}
SYMBOL_TO_BIT_DEMODULATOR_OPS = {
    "demodulation.digital_demodulate",
    "demodulation.identity_demodulate",
    "demodulation.neural_receiver_adapter",
    "demodulation.phase_tracking_receiver_adapter",
}
TX_POWER_OPS = {
    "channel.symbol_power_identity",
    "channel.symbol_power_normalize",
    "model.symbol_power_allocator",
    "model.causal_csi_power_allocator",
}
CSI_POWER_ALLOCATOR_OPS = {
    "model.symbol_power_allocator",
    "model.causal_csi_power_allocator",
}
NON_NATIVE_RUNNERS = {
    "onnxruntime",
    "onnxruntime_cpp",
    "openvino",
    "aot_inductor",
    "torchscript",
    "libtorch",
}
NATIVE_RUNNERS = {
    "",
    "auto",
    "local_python",
    "pillow",
    "pil",
    "pytorch",
    "python",
}
RUNNER_MODEL_KINDS = {
    "onnxruntime": {"model.onnx.bundle"},
    "onnxruntime_cpp": {"model.onnx.bundle"},
    "openvino": {"model.onnx.bundle"},
    "aot_inductor": {"model.aot_inductor.bundle"},
    "torchscript": {"model.torchscript.bundle"},
    "libtorch": {"model.libtorch.bundle"},
}


@dataclass(frozen=True)
class RecipeLintIssue:
    severity: str
    code: str
    message: str
    step_id: Optional[str] = None

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.step_id:
            payload["step_id"] = self.step_id
        return payload


class _LintContext:
    def __init__(self, recipe: Recipe, registry: OperationRegistry) -> None:
        self.recipe = recipe
        self.registry = registry
        self.steps_by_id = {step.id: step for step in recipe.steps}
        self.output_kinds_by_step: Dict[str, Dict[str, str]] = {}
        self.input_kinds_by_step: Dict[str, Dict[str, List[str]]] = {}
        for step in recipe.steps:
            operation = registry.get(step.op)
            self.output_kinds_by_step[step.id] = dict(operation.output_kinds)
            self.input_kinds_by_step[step.id] = {
                name: list(kinds)
                for name, kinds in {
                    **dict(operation.input_kinds),
                    **dict(getattr(operation, "optional_input_kinds", {}) or {}),
                }.items()
            }

    def step(self, step_id: str) -> Optional[RecipeStep]:
        return self.steps_by_id.get(step_id)

    def produced_kinds(self) -> Set[str]:
        kinds: Set[str] = set()
        for outputs in self.output_kinds_by_step.values():
            kinds.update(outputs.values())
        return kinds

    def model_producer_kind(self, reference: str) -> str:
        step_id, output_name = reference.split(".", 1)
        return self.output_kinds_by_step.get(step_id, {}).get(output_name, "")

    def refs_to(self, reference: str) -> List[RecipeStep]:
        users = []
        for step in self.recipe.steps:
            if reference in dict(step.inputs).values():
                users.append(step)
        return users


def lint_recipe_invariants(
    recipe: Recipe,
    registry: OperationRegistry,
    strict: bool = True,
) -> JsonDict:
    """Validate platform-level recipe invariants beyond basic DAG legality.

    `validate_recipe_against_registry` answers whether a recipe can be planned. This linter answers
    whether a recipe uses Noema's shared benchmarking contracts: fixed bit/symbol boundaries,
    count-match checks, and explicit conversion artifacts for non-native runners.
    """

    # Profile issues are collected below so lint can return a complete diagnostic
    # report instead of raising at the shared planning gate.
    validate_recipe_against_registry(recipe, registry, enforce_execution_profile=False)
    ctx = _LintContext(recipe, registry)
    profiles = _classify_profiles(ctx)
    issues: List[RecipeLintIssue] = []
    profile_inspection = inspect_execution_profile(recipe)
    profile_declaration_issues = [
        RecipeLintIssue("error", issue.code, issue.message, issue.step_id)
        for issue in profile_inspection.issues
    ]
    issues.extend(profile_declaration_issues)
    issues.extend(_lint_research_catalog(recipe, strict=strict))
    profile_spine_issues: List[RecipeLintIssue] = []
    declared_profile_id = profile_inspection.reference.id
    if declared_profile_id == LAYERED_DIGITAL_PROFILE_ID:
        profile_spine_issues.extend(_lint_bit_transport_spine(ctx, structure_checked=True))
    elif declared_profile_id == JOINT_SOURCE_CHANNEL_SYMBOLS_PROFILE_ID:
        profile_spine_issues.extend(
            _lint_symbol_transport_spine(ctx, structure_checked=True, enforce_joint=True)
        )
    else:
        if "bit_transport" in profiles:
            profile_spine_issues.extend(_lint_bit_transport_spine(ctx))
        if "symbol_transport" in profiles:
            profile_spine_issues.extend(_lint_symbol_transport_spine(ctx))
    issues.extend(profile_spine_issues)
    issues.extend(_lint_runtime_artifacts(ctx))
    issues.extend(_lint_power_noise_protocol(ctx))
    runner_support = recipe_runner_support(recipe, registry)
    issues.extend(_lint_benchmark_materialization_support(runner_support))

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    execution_profile_report = profile_inspection.to_dict()
    if declared_profile_id != CUSTOM_EXECUTION_PROFILE_ID:
        execution_profile_report["issues"] = [
            issue.to_dict() for issue in [*profile_declaration_issues, *profile_spine_issues]
        ]
        if any(issue.severity == "error" for issue in [*profile_declaration_issues, *profile_spine_issues]):
            execution_profile_report["status"] = "invalid"
    return {
        "schema_version": 1,
        "recipe": recipe.name,
        "status": "failed" if error_count else "passed",
        "strict": bool(strict),
        "profiles": profiles,
        "execution_profile": execution_profile_report,
        "issue_count": len(issues),
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": [issue.to_dict() for issue in issues],
        "runner_support": runner_support,
    }


def _lint_power_noise_protocol(ctx: _LintContext) -> List[RecipeLintIssue]:
    issues: List[RecipeLintIssue] = []
    wireless_steps = [step for step in ctx.recipe.steps if step.op in {"wireless.channel", "wireless.digital_link"}]
    power_steps = [step for step in ctx.recipe.steps if step.op in TX_POWER_OPS]
    variable_power = any(
        step.op in CSI_POWER_ALLOCATOR_OPS
        and str(step.params.get("budget_mode") or "fixed_average") == "variable_average"
        for step in power_steps
    )
    for step in wireless_steps:
        noise_mode = str(step.params.get("noise_mode") or "snr_at_unit_power")
        if noise_mode == "fixed_variance":
            raw_variance = step.params.get("noise_variance")
            try:
                variance = float(raw_variance)
            except (TypeError, ValueError):
                variance = 0.0
            if variance <= 0.0:
                issues.append(
                    RecipeLintIssue(
                        "error",
                        "fixed_noise_variance_missing",
                        "Fixed-noise mode requires a positive `noise_variance`.",
                        step.id,
                    )
                )
        if variable_power and noise_mode == "snr_at_unit_power":
            issues.append(
                RecipeLintIssue(
                    "warning",
                    "reference_snr_with_variable_power",
                    "Variable transmit power uses `snr_db` only to derive a unit-power noise floor; compare methods using `channel.effective_snr_db`.",
                    step.id,
                )
            )
    return issues


def _lint_benchmark_materialization_support(runner_support: Mapping[str, Any]) -> List[RecipeLintIssue]:
    issues: List[RecipeLintIssue] = []
    for item in unsupported_steps_for_runner(runner_support, "benchmark_run"):
        issues.append(
            RecipeLintIssue(
                "error",
                "benchmark_runner_unsupported",
                "Benchmark runner cannot materialize step `%s` (%s): %s"
                % (item.get("step_id", ""), item.get("op", ""), item.get("reason", "unsupported")),
                str(item.get("step_id") or ""),
            )
        )
    return issues


def _classify_profiles(ctx: _LintContext) -> List[str]:
    op_ids = {step.op for step in ctx.recipe.steps}
    produced = ctx.produced_kinds()
    profiles: List[str] = []
    if op_ids.intersection(
        {
            "channel.bit_boundary",
            "channel.bit_count_match",
            "channel.identity_encoder",
            "channel.repetition_encoder",
            "channel.nr_ldpc_encoder",
            "channel.identity_decoder",
            "channel.repetition_decoder",
            "channel.nr_ldpc_decoder",
            "channel.packetize_crc32",
            "channel.crc32_check",
            "channel.capacity_oracle_digital_link",
            "modulation.identity_modulate",
            "demodulation.identity_demodulate",
            "modulation.digital_modulate",
            "demodulation.digital_demodulate",
            "wireless.digital_link",
        }
    ) or produced.intersection(BIT_TRANSPORT_KINDS):
        profiles.append("bit_transport")
    if "channel.symbol_boundary" in op_ids or str(ctx.recipe.metadata.get("codec_output_form") or "") == "symbols":
        profiles.append("symbol_transport")
    if not profiles:
        profiles.append("task_direct")
    return profiles


def _lint_research_catalog(recipe: Recipe, strict: bool) -> List[RecipeLintIssue]:
    issues: List[RecipeLintIssue] = []
    specs = research_specs_from_recipe(recipe)
    validation = dict(specs.get("catalog_validation") or {})
    for message in validation.get("errors") or []:
        issues.append(RecipeLintIssue("error", "research_catalog_error", str(message)))
    for message in validation.get("warnings") or []:
        severity = "error" if strict and "cataloged as `" in str(message) else "warning"
        issues.append(RecipeLintIssue(severity, "research_catalog_warning", str(message)))

    catalog = load_research_catalog()
    task = dict(specs.get("task") or {})
    task_id = str(task.get("id") or "")
    task_def = catalog.task(task_id) if task_id else None
    if task_def and strict and task_def.status != "supported":
        issues.append(
            RecipeLintIssue(
                "error",
                "task_not_supported",
                "Task `%s` is cataloged as `%s`; strict recipes must use supported tasks."
                % (task_id, task_def.status),
            )
        )
    return issues


def _lint_bit_transport_spine(
    ctx: _LintContext,
    structure_checked: bool = False,
) -> List[RecipeLintIssue]:
    issues: List[RecipeLintIssue] = []
    payload = _profile_step(
        ctx, issues, "payload_bit_boundary", "channel.bit_boundary", structure_checked
    )
    channel_encoder = _profile_step(
        ctx, issues, "channel_encoder", _channel_encoder_ops(), structure_checked
    )
    tx_boundary = _profile_step(
        ctx, issues, "tx_bit_boundary", "channel.bit_boundary", structure_checked
    )
    wireless = _profile_step(
        ctx,
        issues,
        "wireless_channel",
        BIT_CHANNEL_OPS.union(SYMBOL_CHANNEL_OPS),
        structure_checked,
    )
    rx_boundary = _profile_step(
        ctx, issues, "rx_bit_boundary", "channel.bit_boundary", structure_checked
    )
    match = _profile_step(
        ctx, issues, "channel_bit_count_match", "channel.bit_count_match", structure_checked
    )
    channel_decoder = _profile_step(
        ctx, issues, "channel_decoder", _channel_decoder_ops(), structure_checked
    )
    packetizer = ctx.step("packetizer")
    crc_check = ctx.step("crc_check")

    if packetizer is not None:
        if packetizer.op not in _packetizer_ops():
            issues.append(
                RecipeLintIssue(
                    "error",
                    "packetizer_op_invalid",
                    "Optional `packetizer` must use a registered CRC32 packetizer.",
                    packetizer.id,
                )
            )
        elif packetizer.op == "channel.packetize_crc32.v2":
            _require_input_ref(
                issues,
                packetizer,
                "bits",
                "sender.bits",
                "strict packetizer must consume the sender's semantic payload-bit output directly.",
            )
        elif payload is not None:
            _require_input_ref(
                issues,
                packetizer,
                "bits",
                "payload_bit_boundary.bits",
                "packetizer must consume canonical payload bits from payload_bit_boundary.",
            )
    if crc_check is not None:
        if crc_check.op != "channel.crc32_check":
            issues.append(
                RecipeLintIssue(
                    "error",
                    "crc_check_op_invalid",
                    "Optional `crc_check` must use channel.crc32_check.",
                    crc_check.id,
                )
            )
        elif channel_decoder is not None:
            _require_input_ref(
                issues,
                crc_check,
                "bits",
                "channel_decoder.bits",
                "crc_check must consume decoded packetized payload bits from channel_decoder.",
            )

    if payload is not None and channel_encoder is not None:
        expected_encoder_input = (
            "packetizer.bits"
            if packetizer is not None and packetizer.op in _packetizer_ops()
            else "payload_bit_boundary.bits"
        )
        _require_input_ref(
            issues,
            channel_encoder,
            "bits",
            expected_encoder_input,
            "channel_encoder must consume canonical payload bits, or packetized payload bits for protected digital links.",
        )
    if channel_encoder is not None and tx_boundary is not None:
        _require_input_ref(
            issues,
            tx_boundary,
            "bits",
            "channel_encoder.coded_bits",
            "tx_bit_boundary must checkpoint the channel-coded bitstream.",
        )
    if match is not None:
        _require_input_ref(
            issues,
            match,
            "reference",
            "tx_bit_boundary.bits",
            "channel_bit_count_match.reference must be the fixed modulator-input bitstream.",
        )
        _require_input_ref(
            issues,
            match,
            "candidate",
            "rx_bit_boundary.bits",
            "channel_bit_count_match.candidate must be the fixed receiver-side bitstream.",
        )
    if rx_boundary is not None and channel_decoder is not None:
        if channel_decoder.op == "channel.nr_ldpc_decoder":
            _require_input_ref(
                issues,
                channel_decoder,
                "llr",
                "demodulator.llr",
                "NR LDPC decoding must consume the soft demodulator LLR artifact.",
            )
        else:
            _require_input_ref(
                issues,
                channel_decoder,
                "coded_bits",
                "rx_bit_boundary.bits",
                "channel_decoder must consume the canonical receiver-side bitstream.",
            )

    modulator = ctx.step("modulator")
    demodulator = ctx.step("demodulator")
    carrier_impairment = ctx.step("carrier_impairment")
    receiver_frontend = ctx.step("receiver_frontend")
    tx_power = ctx.step("tx_power") or ctx.step("tx_power_normalize") or ctx.step("tx_power_allocate")
    channel_state = ctx.step("channel_state")
    csi_observation = ctx.step("csi_observation")
    if wireless is not None and wireless.op in SYMBOL_CHANNEL_OPS:
        if modulator is None:
            issues.append(
                RecipeLintIssue(
                    "error",
                    "modulator_missing",
                    "Symbol-based bit transport must include a `modulator` step.",
                    "wireless_channel",
                )
            )
        elif modulator.op not in BIT_TO_SYMBOL_MODULATOR_OPS:
            issues.append(
                RecipeLintIssue(
                    "error",
                    "modulator_op_invalid",
                    "`modulator` must use a registered bit-to-symbol modulation operation.",
                    modulator.id,
                )
            )
        else:
            _require_input_ref(
                issues,
                modulator,
                "bits",
                "tx_bit_boundary.bits",
                "modulator must consume canonical tx_bit_boundary bits.",
            )
        if channel_state is not None:
            if channel_state.op != "wireless.ofdm_channel_state":
                issues.append(
                    RecipeLintIssue(
                        "error",
                        "channel_state_op_invalid",
                        "Optional `channel_state` must use wireless.ofdm_channel_state.",
                        channel_state.id,
                    )
                )
            else:
                _require_input_ref(
                    issues,
                    channel_state,
                    "symbols",
                    "modulator.symbols",
                    "channel_state must size its OFDM realization from symbols emitted by modulator.",
                )
        if csi_observation is not None:
            if csi_observation.op != "wireless.ofdm_delayed_csi":
                issues.append(
                    RecipeLintIssue(
                        "error",
                        "csi_observation_op_invalid",
                        "Optional `csi_observation` must use wireless.ofdm_delayed_csi.",
                        csi_observation.id,
                    )
                )
            elif channel_state is None:
                issues.append(
                    RecipeLintIssue(
                        "error",
                        "csi_observation_state_missing",
                        "Delayed transmitter CSI requires a channel_state trajectory.",
                        csi_observation.id,
                    )
                )
            else:
                _require_input_ref(
                    issues,
                    csi_observation,
                    "state",
                    "channel_state.state",
                    "csi_observation must derive old and current CSI from one channel_state trajectory.",
                )
        if demodulator is None:
            issues.append(
                RecipeLintIssue(
                    "error",
                    "demodulator_missing",
                    "Symbol-based bit transport must include a `demodulator` step.",
                    "wireless_channel",
                )
            )
        elif demodulator.op not in SYMBOL_TO_BIT_DEMODULATOR_OPS:
            issues.append(
                RecipeLintIssue(
                    "error",
                    "demodulator_op_invalid",
                    "`demodulator` must use a registered symbol-to-bit demodulation operation.",
                    demodulator.id,
                )
            )
        else:
            expected_symbols = (
                "receiver_frontend.rx_symbols"
                if receiver_frontend is not None
                and receiver_frontend.op == "hardware.receiver_iq_imbalance"
                else "carrier_impairment.rx_symbols"
                if carrier_impairment is not None
                and carrier_impairment.op == "wireless.carrier_phase_impairment"
                else "wireless_channel.rx_symbols"
                if wireless.op == "wireless.channel"
                else "wireless_channel.symbols"
            )
            _require_input_ref(
                issues,
                demodulator,
                "rx_symbols",
                expected_symbols,
                "demodulator must consume the final receiver-front-end symbol stream.",
            )
        if receiver_frontend is not None:
            if receiver_frontend.op != "hardware.receiver_iq_imbalance":
                issues.append(
                    RecipeLintIssue(
                        "error",
                        "receiver_frontend_op_invalid",
                        "Optional `receiver_frontend` must use hardware.receiver_iq_imbalance.",
                        receiver_frontend.id,
                    )
                )
            elif wireless.op != "wireless.channel":
                issues.append(
                    RecipeLintIssue(
                        "error",
                        "receiver_frontend_channel_invalid",
                        "Receiver I/Q impairment requires complex symbols from wireless.channel.",
                        receiver_frontend.id,
                    )
                )
            else:
                _require_input_ref(
                    issues,
                    receiver_frontend,
                    "rx_symbols",
                    "wireless_channel.rx_symbols",
                    "receiver_frontend must consume symbols emitted by wireless_channel.",
                )
        if receiver_frontend is not None and carrier_impairment is not None:
            issues.append(
                RecipeLintIssue(
                    "error",
                    "receiver_frontend_ambiguous",
                    "Use only one canonical receiver-front-end impairment stage.",
                    receiver_frontend.id,
                )
            )
        if carrier_impairment is not None:
            if carrier_impairment.op != "wireless.carrier_phase_impairment":
                issues.append(
                    RecipeLintIssue(
                        "error",
                        "carrier_impairment_op_invalid",
                        "Optional `carrier_impairment` must use wireless.carrier_phase_impairment.",
                        carrier_impairment.id,
                    )
                )
            elif wireless.op != "wireless.channel":
                issues.append(
                    RecipeLintIssue(
                        "error",
                        "carrier_impairment_channel_invalid",
                        "Carrier phase impairment requires received complex symbols from wireless.channel.",
                        carrier_impairment.id,
                    )
                )
            else:
                _require_input_ref(
                    issues,
                    carrier_impairment,
                    "rx_symbols",
                    "wireless_channel.rx_symbols",
                    "carrier_impairment must consume symbols emitted by wireless_channel.",
                )
        if tx_power is not None:
            if tx_power.op not in TX_POWER_OPS:
                issues.append(
                    RecipeLintIssue(
                        "error",
                        "tx_power_op_invalid",
                        "Optional `tx_power` must use a registered TX power operation.",
                        tx_power.id,
                    )
                )
            else:
                _require_input_ref(
                    issues,
                    tx_power,
                    "symbols",
                    "channel_state.symbols" if channel_state is not None else "modulator.symbols",
                    "tx_power must consume the OFDM-grid-sized symbol stream when channel_state is present, otherwise symbols emitted by modulator.",
                )
                if (
                    channel_state is not None
                    and tx_power.op in CSI_POWER_ALLOCATOR_OPS
                ):
                    _require_input_ref(
                        issues,
                        tx_power,
                        "channel_state",
                        (
                            "csi_observation.transmitter_csi"
                            if csi_observation is not None
                            else "channel_state.state"
                        ),
                        (
                            "CSI-aware tx_power must consume only the delayed/noisy "
                            "transmitter observation when csi_observation is present."
                        ),
                    )
                csi_policy = (
                    str(tx_power.params.get("policy") or "")
                    if tx_power.op in CSI_POWER_ALLOCATOR_OPS
                    else ""
                )
                if csi_observation is not None and csi_policy == "water_filling":
                    issues.append(
                        RecipeLintIssue(
                            "error",
                            "delayed_csi_water_filling_policy_invalid",
                            "A delayed-CSI topology must use "
                            "policy=observed_csi_water_filling for the "
                            "mismatched practical baseline; policy=water_filling "
                            "is reserved for perfect current CSI.",
                            tx_power.id,
                        )
                    )
                if csi_policy in {
                    "water_filling",
                    "observed_csi_water_filling",
                    "robust_csi_water_filling",
                    "causal_ar_water_filling",
                    "causal_ar_box_water_filling",
                    "learned_checkpoint",
                    "learned_artifact",
                } and channel_state is None:
                    issues.append(
                        RecipeLintIssue(
                            "error",
                            "csi_power_policy_state_missing",
                            "%s requires an explicit wireless.ofdm_channel_state step." % csi_policy,
                            tx_power.id,
                        )
                    )
                if csi_policy == "learned_checkpoint":
                    checkpoint_path = str(tx_power.params.get("checkpoint_path") or "").strip()
                    checkpoint_sha = str(tx_power.params.get("checkpoint_sha256") or "").strip().lower()
                    if not checkpoint_path:
                        issues.append(
                            RecipeLintIssue(
                                "error",
                                "learned_power_checkpoint_path_missing",
                                "Learned power allocation requires a frozen checkpoint_path.",
                                tx_power.id,
                            )
                        )
                    if len(checkpoint_sha) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_sha):
                        issues.append(
                            RecipeLintIssue(
                                "error",
                                "learned_power_checkpoint_sha_missing",
                                "Learned power allocation requires the checkpoint's lowercase SHA-256.",
                                tx_power.id,
                            )
                        )
                    if str(tx_power.params.get("granularity") or "") != "per_subcarrier":
                        issues.append(
                            RecipeLintIssue(
                                "error",
                                "learned_power_granularity_invalid",
                                "Learned CSI power allocation requires granularity=per_subcarrier.",
                                tx_power.id,
                            )
                        )
                    if str(tx_power.params.get("budget_mode") or "fixed_average") != "fixed_average":
                        issues.append(
                            RecipeLintIssue(
                                "error",
                                "learned_power_budget_invalid",
                                "Learned CSI power allocation requires budget_mode=fixed_average.",
                                tx_power.id,
                            )
                        )
        if wireless is not None:
            expected_wireless_symbols = (
                "tx_power.symbols"
                if tx_power is not None
                else "channel_state.symbols"
                if channel_state is not None
                else "modulator.symbols"
            )
            _require_input_ref(
                issues,
                wireless,
                "symbols",
                expected_wireless_symbols,
                "wireless_channel must consume symbols emitted by tx_power when present, otherwise by modulator.",
            )
            if channel_state is not None and wireless.op == "wireless.channel":
                _require_input_ref(
                    issues,
                    wireless,
                    "channel_state",
                    (
                        "csi_observation.actual_state"
                        if csi_observation is not None
                        else "channel_state.state"
                    ),
                    (
                        "wireless_channel must apply current CSI; delayed/noisy CSI "
                        "is reserved for the allocator."
                    ),
                )
        if (
            csi_observation is not None
            and channel_decoder is not None
            and channel_decoder.op == "channel.nr_ldpc_decoder"
        ):
            if demodulator is not None:
                _require_input_ref(
                    issues,
                    demodulator,
                    "allocation",
                    "tx_power.allocation",
                    (
                        "Measured delayed-CSI NR decoding requires the demodulator "
                        "to consume the exact transmitter power allocation."
                    ),
                )
                _require_input_ref(
                    issues,
                    demodulator,
                    "channel_state",
                    "csi_observation.actual_state",
                    (
                        "Measured delayed-CSI NR decoding requires perfect current "
                        "receiver CSI; delayed transmitter CSI must not be used for LLRs."
                    ),
                )
            if wireless is not None and str(
                wireless.params.get("receiver_processing") or "matched"
            ) != "matched":
                issues.append(
                    RecipeLintIssue(
                        "error",
                        "delayed_csi_nr_receiver_processing_invalid",
                        (
                            "Measured delayed-CSI NR decoding requires explicit "
                            "receiver_processing=matched for the current-CSI ZF LLR model."
                        ),
                        wireless.id,
                    )
                )
            delivery = ctx.step("delivery_evaluation")
            if delivery is None:
                issues.append(
                    RecipeLintIssue(
                        "error",
                        "delayed_csi_nr_delivery_metric_missing",
                        (
                            "Measured delayed-CSI NR recipes require a "
                            "`delivery_evaluation` step using CRC-authoritative all-attempt goodput."
                        ),
                        channel_decoder.id,
                    )
                )
            elif delivery.op != "metrics.nr_ldpc_ofdm_delivery":
                issues.append(
                    RecipeLintIssue(
                        "error",
                        "delayed_csi_nr_delivery_metric_invalid",
                        (
                            "`delivery_evaluation` must use "
                            "metrics.nr_ldpc_ofdm_delivery."
                        ),
                        delivery.id,
                    )
                )
            else:
                required_delivery_inputs = {
                    "reference_payload": "payload_bit_boundary.bits",
                    "decoded_payload": "channel_decoder.bits",
                    "decoder_report": "channel_decoder.report",
                    "tx_symbols": "tx_power.symbols",
                    "actual_state": "csi_observation.actual_state",
                    "allocation": "tx_power.allocation",
                    "rx_symbols": "wireless_channel.rx_symbols",
                }
                for input_name, reference in required_delivery_inputs.items():
                    _require_input_ref(
                        issues,
                        delivery,
                        input_name,
                        reference,
                        (
                            "CRC-authoritative delayed-CSI delivery input %s must "
                            "consume %s."
                            % (input_name, reference)
                        ),
                    )
        if rx_boundary is not None:
            _require_input_ref(
                issues,
                rx_boundary,
                "bits",
                "demodulator.bits",
                "rx_bit_boundary must checkpoint the demodulated bitstream.",
            )
    else:
        if wireless is not None:
            _require_input_ref(
                issues,
                wireless,
                "bits",
                "tx_bit_boundary.bits",
                "Disabled or bit-level channels must consume canonical tx_bit_boundary bits.",
            )
        if rx_boundary is not None:
            _require_input_ref(
                issues,
                rx_boundary,
                "bits",
                "wireless_channel.bits",
                "rx_bit_boundary must checkpoint the channel output bits.",
            )

    if not str(ctx.recipe.metadata.get("rate_count_fixed_point") or ""):
        issues.append(
            RecipeLintIssue(
                "warning",
                "rate_count_fixed_point_missing",
                "metadata.rate_count_fixed_point is not set; RD plots may not know the canonical transmitted-bit metric.",
            )
        )
    return issues


def _lint_symbol_transport_spine(
    ctx: _LintContext,
    structure_checked: bool = False,
    enforce_joint: bool = False,
) -> List[RecipeLintIssue]:
    issues: List[RecipeLintIssue] = []
    tx_boundary = _profile_step(
        ctx, issues, "tx_symbol_boundary", "channel.symbol_boundary", structure_checked
    )
    wireless = _profile_step(
        ctx, issues, "wireless_channel", SYMBOL_CHANNEL_OPS, structure_checked
    )
    rx_boundary = _profile_step(
        ctx, issues, "rx_symbol_boundary", "channel.symbol_boundary", structure_checked
    )
    match = _profile_step(
        ctx, issues, "channel_symbol_count_match", "channel.symbol_count_match", structure_checked
    )
    if enforce_joint:
        sender = ctx.step("sender")
        receiver = ctx.step("receiver")
        tx_power = ctx.step("tx_power") or ctx.step("tx_power_normalize")
        channel_state = ctx.step("channel_state")
        if channel_state is not None:
            if channel_state.op != "wireless.ofdm_channel_state":
                issues.append(
                    RecipeLintIssue(
                        "error",
                        "channel_state_op_invalid",
                        "Optional `channel_state` must use wireless.ofdm_channel_state.",
                        channel_state.id,
                    )
                )
            elif sender is not None:
                _require_input_ref(
                    issues,
                    channel_state,
                    "symbols",
                    "sender.symbols",
                    "channel_state must size its realization from symbols emitted by the joint sender.",
                )
        if tx_power is not None and sender is not None:
            _require_input_ref(
                issues,
                tx_power,
                "symbols",
                "channel_state.symbols" if channel_state is not None else "sender.symbols",
                "tx_power must consume symbols emitted by the joint sender, or the shared channel-state sizing step.",
            )
            if (
                channel_state is not None
                and tx_power.op in CSI_POWER_ALLOCATOR_OPS
            ):
                _require_input_ref(
                    issues,
                    tx_power,
                    "channel_state",
                    "channel_state.state",
                    "CSI-aware tx_power must consume the same explicit channel state as the channel.",
                )
        if tx_boundary is not None and tx_power is not None:
            _require_input_ref(
                issues,
                tx_boundary,
                "symbols",
                "%s.symbols" % tx_power.id,
                "tx_symbol_boundary must consume the power-constrained joint-sender symbols.",
            )
        if receiver is not None:
            _require_input_ref(
                issues,
                receiver,
                "symbols",
                "rx_symbol_boundary.symbols",
                "The joint receiver must consume the canonical receiver-side symbols.",
            )
    if wireless is not None:
        _require_input_ref(
            issues,
            wireless,
            "symbols",
            "tx_symbol_boundary.symbols",
            "wireless_channel must consume canonical tx_symbol_boundary symbols.",
        )
    if rx_boundary is not None:
        expected = "wireless_channel.rx_symbols" if wireless is not None and wireless.op == "wireless.channel" else "wireless_channel.symbols"
        _require_input_ref(
            issues,
            rx_boundary,
            "symbols",
            expected,
            "rx_symbol_boundary must checkpoint the channel output symbols.",
        )
    if match is not None:
        _require_input_ref(
            issues,
            match,
            "reference",
            "tx_symbol_boundary.symbols",
            "channel_symbol_count_match.reference must be the fixed transmitter-side symbols.",
        )
        _require_input_ref(
            issues,
            match,
            "candidate",
            "rx_symbol_boundary.symbols",
            "channel_symbol_count_match.candidate must be the fixed receiver-side symbols.",
        )
    if tx_boundary is not None and not str(ctx.recipe.metadata.get("rate_count_fixed_point") or ""):
        issues.append(
            RecipeLintIssue(
                "warning",
                "rate_count_fixed_point_missing",
                "metadata.rate_count_fixed_point is not set; channel-use plots may not know the canonical symbol metric.",
            )
        )
    return issues


def _lint_runtime_artifacts(ctx: _LintContext) -> List[RecipeLintIssue]:
    runner = _runner(ctx.recipe.metadata)
    if runner in NATIVE_RUNNERS:
        return []
    issues: List[RecipeLintIssue] = []
    if runner not in NON_NATIVE_RUNNERS:
        issues.append(
            RecipeLintIssue(
                "warning",
                "runtime_unknown",
                "Runtime `%s` is not one of the known native or conversion-backed runners." % runner,
            )
        )
        return issues

    model_consumers = [
        step for step in ctx.recipe.steps if "model" in ctx.input_kinds_by_step.get(step.id, {})
    ]
    model_producers = [
        step
        for step in ctx.recipe.steps
        if any(kind.startswith("model.") for kind in ctx.output_kinds_by_step.get(step.id, {}).values())
    ]
    if not model_consumers:
        has_model_ops = any(step.op.startswith("model.") for step in ctx.recipe.steps)
        if has_model_ops:
            issues.append(
                RecipeLintIssue(
                    "error",
                    "runtime_adapter_missing",
                    "Runtime `%s` is selected but no runtime-specific model-consuming adapter step is present."
                    % runner,
                )
            )
        return issues

    if not model_producers:
        issues.append(
            RecipeLintIssue(
                "error",
                "runtime_export_missing",
                "Runtime `%s` requires an explicit export/conversion step that produces a model artifact."
                % runner,
            )
        )

    accepted_model_kinds = RUNNER_MODEL_KINDS.get(runner) or set()
    for step in model_consumers:
        reference = step.inputs.get("model")
        if not reference:
            issues.append(
                RecipeLintIssue(
                    "error",
                    "runtime_model_input_missing",
                    "Runtime-specific step must receive a model artifact input.",
                    step.id,
                )
            )
            continue
        producer_kind = ctx.model_producer_kind(reference)
        if accepted_model_kinds and producer_kind not in accepted_model_kinds:
            issues.append(
                RecipeLintIssue(
                    "error",
                    "runtime_model_kind_invalid",
                    "Runtime `%s` expects model artifact kind %s but %s produces `%s`."
                    % (runner, sorted(accepted_model_kinds), reference, producer_kind),
                    step.id,
                )
            )
    return issues


def _require_step(
    ctx: _LintContext,
    issues: List[RecipeLintIssue],
    step_id: str,
    expected_ops: Union[Iterable[str], str],
) -> Optional[RecipeStep]:
    expected = {expected_ops} if isinstance(expected_ops, str) else set(expected_ops)
    step = ctx.step(step_id)
    if step is None:
        issues.append(
            RecipeLintIssue(
                "error",
                "required_step_missing",
                "Recipe profile requires `%s` using %s." % (step_id, sorted(expected)),
                step_id,
            )
        )
        return None
    if step.op not in expected:
        issues.append(
            RecipeLintIssue(
                "error",
                "required_step_op_invalid",
                "`%s` must use one of %s, got `%s`." % (step_id, sorted(expected), step.op),
                step_id,
            )
        )
    return step


def _profile_step(
    ctx: _LintContext,
    issues: List[RecipeLintIssue],
    step_id: str,
    expected_ops: Union[Iterable[str], str],
    structure_checked: bool,
) -> Optional[RecipeStep]:
    if structure_checked:
        return ctx.step(step_id)
    return _require_step(ctx, issues, step_id, expected_ops)


def _require_input_ref(
    issues: List[RecipeLintIssue],
    step: RecipeStep,
    input_name: str,
    expected_ref: str,
    message: str,
) -> None:
    actual_ref = step.inputs.get(input_name)
    if actual_ref != expected_ref:
        issues.append(
            RecipeLintIssue(
                "error",
                "input_reference_invalid",
                "%s Expected `%s`, got `%s`." % (message, expected_ref, actual_ref),
                step.id,
            )
        )


def _runner(metadata: Mapping[str, Any]) -> str:
    timing = metadata.get("codec_timing") if isinstance(metadata, Mapping) else None
    if isinstance(timing, Mapping) and timing.get("runner") is not None:
        return str(timing.get("runner") or "").strip().lower()
    for key in ("runner", "codec_runner", "runtime"):
        if metadata.get(key) is not None:
            return str(metadata.get(key) or "").strip().lower()
    return ""


def _channel_encoder_ops() -> Set[str]:
    return {
        "channel.identity_encoder",
        "channel.repetition_encoder",
        "channel.nr_ldpc_encoder",
    }


def _packetizer_ops() -> Set[str]:
    return {
        "channel.packetize_crc32",
        "channel.packetize_crc32.v2",
    }


def _channel_decoder_ops() -> Set[str]:
    return {
        "channel.identity_decoder",
        "channel.repetition_decoder",
        "channel.nr_ldpc_decoder",
    }
