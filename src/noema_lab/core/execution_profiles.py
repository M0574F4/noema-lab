from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


JsonDict = Dict[str, Any]

EXECUTION_PROFILE_VERSION = 1
LAYERED_DIGITAL_PROFILE_ID = "layered_digital"
JOINT_SOURCE_CHANNEL_SYMBOLS_PROFILE_ID = "joint_source_channel_symbols"
CSI_FEEDBACK_DOWNLINK_PROFILE_ID = "csi_feedback_downlink"
PILOT_CHANNEL_ESTIMATION_PROFILE_ID = "pilot_channel_estimation"
MIMO_OFDM_CHANNEL_ESTIMATION_PROFILE_ID = "mimo_ofdm_channel_estimation"
BEAMFORMING_LINK_EVALUATION_PROFILE_ID = "beamforming_link_evaluation"
RANGE_LOCALIZATION_PROFILE_ID = "range_localization"
AOA_ARRAY_ESTIMATION_PROFILE_ID = "aoa_array_estimation"
TASK_INFERENCE_PROFILE_ID = "task_inference"
TASK_EVALUATION_PROFILE_ID = "task_evaluation"
CUSTOM_EXECUTION_PROFILE_ID = "custom"


class ExecutionProfileDeclarationError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionProfileRef:
    id: str
    version: int = EXECUTION_PROFILE_VERSION
    based_on: Optional["ExecutionProfileRef"] = None

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {"id": self.id, "version": self.version}
        if self.based_on is not None:
            payload["based_on"] = self.based_on.to_dict()
        return payload


@dataclass(frozen=True)
class ExecutionStageDefinition:
    role: str
    label: str
    presence: str
    step_ids: Tuple[str, ...] = ()
    allowed_ops: Tuple[str, ...] = ()
    required_params: Tuple[Tuple[str, Any], ...] = ()
    fused_into: Optional[str] = None
    summary: str = ""

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "role": self.role,
            "label": self.label,
            "presence": self.presence,
        }
        if self.step_ids:
            payload["step_ids"] = list(self.step_ids)
        if self.allowed_ops:
            payload["allowed_ops"] = list(self.allowed_ops)
        if self.required_params:
            payload["required_params"] = dict(self.required_params)
        if self.fused_into:
            payload["fused_into"] = self.fused_into
        if self.summary:
            payload["summary"] = self.summary
        return payload


@dataclass(frozen=True)
class ExecutionProfileDefinition:
    id: str
    version: int
    label: str
    summary: str
    stages: Tuple[ExecutionStageDefinition, ...]

    @property
    def required_spine(self) -> Tuple[ExecutionStageDefinition, ...]:
        return tuple(stage for stage in self.stages if stage.presence == "required")

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "version": self.version,
            "label": self.label,
            "summary": self.summary,
            "required_ordered_spine": [stage.role for stage in self.required_spine],
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass(frozen=True)
class ExecutionProfileIssue:
    code: str
    message: str
    step_id: Optional[str] = None

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {"code": self.code, "message": self.message}
        if self.step_id:
            payload["step_id"] = self.step_id
        return payload


@dataclass(frozen=True)
class ExecutionProfileInspection:
    reference: ExecutionProfileRef
    status: str
    definition: Optional[ExecutionProfileDefinition]
    issues: Tuple[ExecutionProfileIssue, ...] = ()
    stage_bindings: Tuple[Tuple[str, str], ...] = ()

    def to_dict(self) -> JsonDict:
        return {
            "reference": self.reference.to_dict(),
            "status": self.status,
            "profile": self.definition.to_dict() if self.definition is not None else None,
            "stage_bindings": {role: step_id for role, step_id in self.stage_bindings},
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class ExecutionProfileCatalog:
    profiles: Tuple[ExecutionProfileDefinition, ...]
    schema_version: int = 1

    def get(self, profile_id: str, version: int = EXECUTION_PROFILE_VERSION) -> Optional[ExecutionProfileDefinition]:
        return next(
            (
                profile
                for profile in self.profiles
                if profile.id == str(profile_id) and profile.version == int(version)
            ),
            None,
        )

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "custom": {
                "id": CUSTOM_EXECUTION_PROFILE_ID,
                "version": EXECUTION_PROFILE_VERSION,
                "label": "Custom topology",
                "summary": "The recipe DAG is authoritative and does not claim a standard execution spine.",
            },
        }


_BIT_CHANNEL_OPS = (
    "channel.identity_link",
    "channel.capacity_oracle_digital_link",
    "wireless.digital_link",
    "channel.identity_symbol_link",
    "wireless.channel",
)
_CHANNEL_ENCODER_OPS = (
    "channel.identity_encoder",
    "channel.repetition_encoder",
    "channel.nr_ldpc_encoder",
)
_CHANNEL_DECODER_OPS = (
    "channel.identity_decoder",
    "channel.repetition_decoder",
    "channel.nr_ldpc_decoder",
)
_TX_POWER_OPS = (
    "channel.symbol_power_identity",
    "channel.symbol_power_normalize",
    "model.symbol_power_allocator",
    "model.causal_csi_power_allocator",
)


_LAYERED_DIGITAL = ExecutionProfileDefinition(
    id=LAYERED_DIGITAL_PROFILE_ID,
    version=EXECUTION_PROFILE_VERSION,
    label="Layered digital",
    summary=(
        "A bit-transport spine with separate payload and channel-code fixed points; modulation, "
        "power control, and symbol transport are explicit when a symbol channel is used."
    ),
    stages=(
        ExecutionStageDefinition("source", "Data source", "optional", ("data",)),
        ExecutionStageDefinition("sender", "Source encoder", "optional", ("sender",)),
        ExecutionStageDefinition("payload_encoder", "Payload encoder", "optional", ("payload_encoder",)),
        ExecutionStageDefinition(
            "payload_bit_boundary",
            "Payload-bit boundary",
            "required",
            ("payload_bit_boundary",),
            ("channel.bit_boundary",),
        ),
        ExecutionStageDefinition("packetizer", "Packetizer", "optional", ("packetizer",)),
        ExecutionStageDefinition(
            "channel_encoder",
            "Channel encoder",
            "required",
            ("channel_encoder",),
            _CHANNEL_ENCODER_OPS,
        ),
        ExecutionStageDefinition(
            "tx_bit_boundary",
            "Transmitter-bit boundary",
            "required",
            ("tx_bit_boundary",),
            ("channel.bit_boundary",),
        ),
        ExecutionStageDefinition("modulator", "Modulator", "optional", ("modulator",)),
        ExecutionStageDefinition("channel_state", "Channel state", "optional", ("channel_state",)),
        ExecutionStageDefinition(
            "tx_power",
            "Transmit-power control",
            "optional",
            ("tx_power", "tx_power_normalize", "tx_power_allocate"),
            _TX_POWER_OPS,
        ),
        ExecutionStageDefinition(
            "wireless_channel",
            "Communication channel",
            "required",
            ("wireless_channel",),
            _BIT_CHANNEL_OPS,
        ),
        ExecutionStageDefinition(
            "receiver_frontend",
            "Receiver front end",
            "optional",
            ("receiver_frontend", "carrier_impairment"),
            (
                "hardware.receiver_iq_imbalance",
                "wireless.carrier_phase_impairment",
            ),
        ),
        ExecutionStageDefinition("demodulator", "Demodulator", "optional", ("demodulator",)),
        ExecutionStageDefinition(
            "rx_bit_boundary",
            "Receiver-bit boundary",
            "required",
            ("rx_bit_boundary",),
            ("channel.bit_boundary",),
        ),
        ExecutionStageDefinition(
            "channel_bit_count_match",
            "Channel bit-count check",
            "required",
            ("channel_bit_count_match",),
            ("channel.bit_count_match",),
        ),
        ExecutionStageDefinition(
            "channel_decoder",
            "Channel decoder",
            "required",
            ("channel_decoder",),
            _CHANNEL_DECODER_OPS,
        ),
        ExecutionStageDefinition("payload_decoder", "Payload decoder", "optional", ("payload_decoder",)),
        ExecutionStageDefinition("receiver", "Source decoder / receiver", "optional", ("receiver",)),
        ExecutionStageDefinition("evaluation", "Evaluation", "optional", ("evaluation",)),
    ),
)


_JOINT_SOURCE_CHANNEL_SYMBOLS = ExecutionProfileDefinition(
    id=JOINT_SOURCE_CHANNEL_SYMBOLS_PROFILE_ID,
    version=EXECUTION_PROFILE_VERSION,
    label="Joint source-channel symbols",
    summary=(
        "A symbol-native JSCC spine whose sender and receiver jointly realize the source, payload, "
        "channel-code, and modulation layers around an explicit power constraint and channel."
    ),
    stages=(
        ExecutionStageDefinition("source", "Data source", "optional", ("data",)),
        ExecutionStageDefinition("sender", "Joint sender", "required", ("sender",)),
        ExecutionStageDefinition(
            "source_encoder",
            "Source encoder",
            "fused",
            ("source_encoder",),
            fused_into="sender",
        ),
        ExecutionStageDefinition(
            "payload_encoder",
            "Payload encoder",
            "fused",
            ("payload_encoder",),
            fused_into="sender",
        ),
        ExecutionStageDefinition(
            "channel_encoder",
            "Channel encoder",
            "fused",
            ("channel_encoder",),
            fused_into="sender",
        ),
        ExecutionStageDefinition(
            "modulator",
            "Modulator",
            "fused",
            ("modulator",),
            fused_into="sender",
        ),
        ExecutionStageDefinition(
            "tx_power",
            "Transmit-power constraint",
            "required",
            ("tx_power", "tx_power_normalize"),
            _TX_POWER_OPS,
        ),
        ExecutionStageDefinition(
            "tx_symbol_boundary",
            "Transmitter-symbol boundary",
            "required",
            ("tx_symbol_boundary",),
            ("channel.symbol_boundary",),
        ),
        ExecutionStageDefinition("channel_state", "Channel state", "optional", ("channel_state",)),
        ExecutionStageDefinition(
            "wireless_channel",
            "Symbol channel",
            "required",
            ("wireless_channel",),
            ("channel.identity_symbol_link", "wireless.channel"),
        ),
        ExecutionStageDefinition(
            "rx_symbol_boundary",
            "Receiver-symbol boundary",
            "required",
            ("rx_symbol_boundary",),
            ("channel.symbol_boundary",),
        ),
        ExecutionStageDefinition(
            "channel_symbol_count_match",
            "Channel symbol-count check",
            "required",
            ("channel_symbol_count_match",),
            ("channel.symbol_count_match",),
        ),
        ExecutionStageDefinition(
            "demodulator",
            "Demodulator",
            "fused",
            ("demodulator",),
            fused_into="receiver",
        ),
        ExecutionStageDefinition(
            "channel_decoder",
            "Channel decoder",
            "fused",
            ("channel_decoder",),
            fused_into="receiver",
        ),
        ExecutionStageDefinition(
            "payload_decoder",
            "Payload decoder",
            "fused",
            ("payload_decoder",),
            fused_into="receiver",
        ),
        ExecutionStageDefinition(
            "source_decoder",
            "Source decoder",
            "fused",
            ("source_decoder",),
            fused_into="receiver",
        ),
        ExecutionStageDefinition("receiver", "Joint receiver", "required", ("receiver",)),
        ExecutionStageDefinition("evaluation", "Evaluation", "optional", ("evaluation",)),
    ),
)


_CSI_FEEDBACK_DOWNLINK = ExecutionProfileDefinition(
    id=CSI_FEEDBACK_DOWNLINK_PROFILE_ID,
    version=EXECUTION_PROFILE_VERSION,
    label="CSI feedback and downlink precoding",
    summary=(
        "A causal FDD downlink spine in which user-side channel state is compressed, "
        "transported over an explicit feedback link, reconstructed at the transmitter, "
        "and used for downlink precoding on the same realized channel."
    ),
    stages=(
        ExecutionStageDefinition(
            "channel_state",
            "Downlink channel state",
            "required",
            ("channel_state",),
            ("wireless.miso_ofdm_csi",),
        ),
        ExecutionStageDefinition(
            "csi_observation",
            "User CSI observation",
            "optional",
            ("csi_observation",),
        ),
        ExecutionStageDefinition(
            "feedback_encoder",
            "CSI feedback encoder",
            "required",
            ("feedback_encoder",),
            ("model.csi_feedback_encoder",),
        ),
        ExecutionStageDefinition(
            "feedback_link",
            "CSI feedback link",
            "required",
            ("feedback_link",),
            ("channel.csi_feedback_link",),
        ),
        ExecutionStageDefinition(
            "feedback_decoder",
            "CSI feedback decoder",
            "required",
            ("feedback_decoder",),
            ("model.csi_feedback_decoder",),
        ),
        ExecutionStageDefinition(
            "precoder",
            "Reconstructed-CSI precoder",
            "required",
            ("precoder",),
            ("model.csi_mrt_precoder",),
        ),
        ExecutionStageDefinition(
            "evaluation",
            "CSI feedback and downlink evaluation",
            "required",
            ("evaluation",),
            ("metrics.csi_feedback",),
        ),
    ),
)


_PILOT_CHANNEL_ESTIMATION = ExecutionProfileDefinition(
    id=PILOT_CHANNEL_ESTIMATION_PROFILE_ID,
    version=EXECUTION_PROFILE_VERSION,
    label="Pilot-aided channel estimation",
    summary="Channel realization and pilot pattern produce observations for channel estimation.",
    stages=(
        ExecutionStageDefinition(
            "channel_state",
            "Channel realization",
            "required",
            ("data",),
            ("source.ai_phy_channel_realization",),
            (("scenario", "flat_siso"),),
        ),
        ExecutionStageDefinition(
            "pilot_pattern",
            "Pilot pattern",
            "required",
            ("pilots",),
            ("source.ai_phy_pilot_pattern",),
            (("scenario", "unit"),),
        ),
        ExecutionStageDefinition(
            "pilot_observation",
            "Received pilots",
            "required",
            ("pilot_observation",),
            ("wireless.pilot_observation",),
        ),
        ExecutionStageDefinition(
            "channel_estimator",
            "Channel estimator",
            "required",
            ("estimator",),
        ),
        ExecutionStageDefinition(
            "evaluation",
            "Estimation evaluation",
            "required",
            ("evaluation",),
            ("metrics.channel_estimation",),
        ),
    ),
)


_MIMO_OFDM_CHANNEL_ESTIMATION = ExecutionProfileDefinition(
    id=MIMO_OFDM_CHANNEL_ESTIMATION_PROFILE_ID,
    version=EXECUTION_PROFILE_VERSION,
    label="MIMO-OFDM channel estimation",
    summary="MIMO-OFDM channel and pilot-grid branches feed observation, estimation, and evaluation.",
    stages=(
        ExecutionStageDefinition(
            "mimo_ofdm_channel",
            "MIMO-OFDM channel realization",
            "required",
            ("data",),
            ("source.ai_phy_channel_realization",),
            (("scenario", "mimo_ofdm"),),
        ),
        ExecutionStageDefinition(
            "pilot_grid",
            "OFDM pilot grid",
            "required",
            ("pilots",),
            ("source.ai_phy_pilot_pattern",),
            (("scenario", "comb"),),
        ),
        ExecutionStageDefinition(
            "pilot_observation",
            "Received pilot grid",
            "required",
            ("pilot_observation",),
            ("wireless.pilot_observation",),
        ),
        ExecutionStageDefinition(
            "channel_estimator",
            "MIMO-OFDM estimator",
            "required",
            ("estimator",),
        ),
        ExecutionStageDefinition(
            "evaluation",
            "Estimation evaluation",
            "required",
            ("evaluation",),
            ("metrics.channel_estimation",),
        ),
    ),
)


_BEAMFORMING_LINK_EVALUATION = ExecutionProfileDefinition(
    id=BEAMFORMING_LINK_EVALUATION_PROFILE_ID,
    version=EXECUTION_PROFILE_VERSION,
    label="Beamforming link evaluation",
    summary="A realized link is mapped to beamforming weights and evaluated on that same link.",
    stages=(
        ExecutionStageDefinition(
            "link_scenario",
            "Link scenario",
            "required",
            ("data",),
            ("source.beamforming_scenario",),
        ),
        ExecutionStageDefinition(
            "beamformer",
            "Beamformer",
            "required",
            ("beamformer",),
        ),
        ExecutionStageDefinition(
            "evaluation",
            "Link evaluation",
            "required",
            ("evaluation",),
            ("metrics.beamforming",),
        ),
    ),
)


_RANGE_LOCALIZATION = ExecutionProfileDefinition(
    id=RANGE_LOCALIZATION_PROFILE_ID,
    version=EXECUTION_PROFILE_VERSION,
    label="Range-based localization",
    summary="Anchor geometry produces noisy ranges for localization and position-error evaluation.",
    stages=(
        ExecutionStageDefinition(
            "geometry",
            "Anchor and tag geometry",
            "required",
            ("data",),
            ("source.localization_geometry",),
        ),
        ExecutionStageDefinition(
            "range_observation",
            "Range observations",
            "required",
            ("range_observation",),
            ("wireless.range_observation",),
        ),
        ExecutionStageDefinition(
            "localizer",
            "Position estimator",
            "required",
            ("localizer",),
        ),
        ExecutionStageDefinition(
            "evaluation",
            "Position evaluation",
            "required",
            ("evaluation",),
            ("metrics.localization",),
        ),
    ),
)


_AOA_ARRAY_ESTIMATION = ExecutionProfileDefinition(
    id=AOA_ARRAY_ESTIMATION_PROFILE_ID,
    version=EXECUTION_PROFILE_VERSION,
    label="Array-based AoA estimation",
    summary="An angular scene produces array snapshots for direction estimation and error evaluation.",
    stages=(
        ExecutionStageDefinition(
            "angular_scene",
            "Angular scene",
            "required",
            ("data",),
            ("source.aoa_scene",),
        ),
        ExecutionStageDefinition(
            "array_observation",
            "Array observations",
            "required",
            ("array_observation",),
            ("wireless.ula_array_observation",),
        ),
        ExecutionStageDefinition(
            "angle_estimator",
            "Angle estimator",
            "required",
            ("estimator",),
        ),
        ExecutionStageDefinition(
            "evaluation",
            "Angle evaluation",
            "required",
            ("evaluation",),
            ("metrics.aoa_estimation",),
        ),
    ),
)


_TASK_INFERENCE = ExecutionProfileDefinition(
    id=TASK_INFERENCE_PROFILE_ID,
    version=EXECUTION_PROFILE_VERSION,
    label="Task inference",
    summary="Task data flows through an inference or decision stage before task metrics.",
    stages=(
        ExecutionStageDefinition(
            "source",
            "Task data",
            "required",
            ("data",),
        ),
        ExecutionStageDefinition(
            "inference",
            "Inference or decision",
            "required",
            ("receiver", "inference", "answerer"),
        ),
        ExecutionStageDefinition(
            "evaluation",
            "Task evaluation",
            "required",
            ("evaluation",),
        ),
    ),
)


_TASK_EVALUATION = ExecutionProfileDefinition(
    id=TASK_EVALUATION_PROFILE_ID,
    version=EXECUTION_PROFILE_VERSION,
    label="Direct task evaluation",
    summary="Reference and candidate data are scored directly, without a required inference stage.",
    stages=(
        ExecutionStageDefinition(
            "source",
            "References and candidates",
            "required",
            ("data",),
        ),
        ExecutionStageDefinition(
            "evaluation",
            "Direct task evaluation",
            "required",
            ("evaluation",),
        ),
    ),
)


_CATALOG = ExecutionProfileCatalog(
    (
        _LAYERED_DIGITAL,
        _JOINT_SOURCE_CHANNEL_SYMBOLS,
        _CSI_FEEDBACK_DOWNLINK,
        _PILOT_CHANNEL_ESTIMATION,
        _MIMO_OFDM_CHANNEL_ESTIMATION,
        _BEAMFORMING_LINK_EVALUATION,
        _RANGE_LOCALIZATION,
        _AOA_ARRAY_ESTIMATION,
        _TASK_INFERENCE,
        _TASK_EVALUATION,
    )
)


def execution_profile_catalog() -> ExecutionProfileCatalog:
    return _CATALOG


def custom_execution_profile_ref() -> ExecutionProfileRef:
    return ExecutionProfileRef(CUSTOM_EXECUTION_PROFILE_ID, EXECUTION_PROFILE_VERSION)


def execution_profile_ref_from_value(value: Any) -> ExecutionProfileRef:
    if value is None:
        return custom_execution_profile_ref()
    if not isinstance(value, Mapping):
        raise ExecutionProfileDeclarationError("Recipe execution_profile must be a mapping")
    unknown = set(value) - {"id", "version", "based_on"}
    if unknown:
        raise ExecutionProfileDeclarationError(
            "Recipe execution_profile has unknown field(s): %s"
            % ", ".join(sorted(str(field) for field in unknown))
        )
    profile_id = value.get("id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ExecutionProfileDeclarationError("Recipe execution_profile requires a non-empty string 'id'")
    raw_version = value["version"] if "version" in value else EXECUTION_PROFILE_VERSION
    try:
        version = int(raw_version)
    except (TypeError, ValueError) as exc:
        raise ExecutionProfileDeclarationError("Recipe execution_profile.version must be an integer") from exc
    if profile_id == CUSTOM_EXECUTION_PROFILE_ID:
        if version != EXECUTION_PROFILE_VERSION:
            raise ExecutionProfileDeclarationError(
                "Unsupported execution profile custom version: %s" % version
            )
        raw_base = value.get("based_on")
        based_on = None
        if raw_base is not None:
            if not isinstance(raw_base, Mapping):
                raise ExecutionProfileDeclarationError(
                    "Recipe execution_profile.based_on must be a mapping"
                )
            base_unknown = set(raw_base) - {"id", "version"}
            if base_unknown:
                raise ExecutionProfileDeclarationError(
                    "Recipe execution_profile.based_on has unknown field(s): %s"
                    % ", ".join(sorted(str(field) for field in base_unknown))
                )
            base_id = raw_base.get("id")
            if not isinstance(base_id, str) or not base_id.strip():
                raise ExecutionProfileDeclarationError(
                    "Recipe execution_profile.based_on requires a non-empty string 'id'"
                )
            raw_base_version = (
                raw_base["version"] if "version" in raw_base else EXECUTION_PROFILE_VERSION
            )
            try:
                base_version = int(raw_base_version)
            except (TypeError, ValueError) as exc:
                raise ExecutionProfileDeclarationError(
                    "Recipe execution_profile.based_on.version must be an integer"
                ) from exc
            if _CATALOG.get(base_id, base_version) is None:
                raise ExecutionProfileDeclarationError(
                    "Unknown base execution profile: %s version %s"
                    % (base_id, base_version)
                )
            based_on = ExecutionProfileRef(base_id, base_version)
        return ExecutionProfileRef(profile_id, version, based_on)
    if value.get("based_on") is not None:
        raise ExecutionProfileDeclarationError(
            "Recipe execution_profile.based_on is only valid when id is custom"
        )
    if _CATALOG.get(profile_id, version) is None:
        raise ExecutionProfileDeclarationError(
            "Unknown execution profile: %s version %s" % (profile_id, version)
        )
    return ExecutionProfileRef(profile_id, version)


def inspect_execution_profile(recipe: Any) -> ExecutionProfileInspection:
    reference = getattr(recipe, "execution_profile", None) or custom_execution_profile_ref()
    if not isinstance(reference, ExecutionProfileRef):
        reference = execution_profile_ref_from_value(reference)
    if reference.id == CUSTOM_EXECUTION_PROFILE_ID:
        return ExecutionProfileInspection(reference, "custom", None)
    definition = _CATALOG.get(reference.id, reference.version)
    if definition is None:
        issue = ExecutionProfileIssue(
            "execution_profile_unknown",
            "Unknown execution profile `%s` version %s." % (reference.id, reference.version),
        )
        return ExecutionProfileInspection(reference, "invalid", None, (issue,))

    steps = list(getattr(recipe, "steps", ()) or ())
    by_id = {str(getattr(step, "id", "")): step for step in steps}
    positions = {str(getattr(step, "id", "")): index for index, step in enumerate(steps)}
    issues = []
    bindings = []
    last_position = -1
    for stage in definition.required_spine:
        matched_ids = [step_id for step_id in stage.step_ids if step_id in by_id]
        if not matched_ids:
            issues.append(
                ExecutionProfileIssue(
                    "required_step_missing",
                    "Execution profile `%s` requires stage `%s` using step ID %s."
                    % (definition.id, stage.role, list(stage.step_ids)),
                    stage.step_ids[0] if stage.step_ids else None,
                )
            )
            continue
        if len(matched_ids) > 1:
            issues.append(
                ExecutionProfileIssue(
                    "execution_profile_stage_ambiguous",
                    "Execution profile stage `%s` is realized more than once: %s."
                    % (stage.role, ", ".join(matched_ids)),
                    matched_ids[0],
                )
            )
        step_id = matched_ids[0]
        step = by_id[step_id]
        bindings.append((stage.role, step_id))
        operation_id = str(getattr(step, "op", ""))
        if stage.allowed_ops and operation_id not in stage.allowed_ops:
            issues.append(
                ExecutionProfileIssue(
                    "required_step_op_invalid",
                    "Execution profile stage `%s` requires one of %s, got `%s`."
                    % (stage.role, list(stage.allowed_ops), operation_id),
                    step_id,
                )
            )
        params = getattr(step, "params", {}) or {}
        for param_name, expected_value in stage.required_params:
            actual_value = params.get(param_name) if isinstance(params, Mapping) else None
            if actual_value != expected_value:
                issues.append(
                    ExecutionProfileIssue(
                        "required_step_param_invalid",
                        "Execution profile stage `%s` requires parameter `%s=%r`, got %r."
                        % (stage.role, param_name, expected_value, actual_value),
                        step_id,
                    )
                )
        position = positions[step_id]
        if position <= last_position:
            issues.append(
                ExecutionProfileIssue(
                    "execution_profile_stage_order_invalid",
                    "Execution profile stage `%s` is out of its required spine order."
                    % stage.role,
                    step_id,
                )
            )
        last_position = max(last_position, position)

    for stage in definition.stages:
        if stage.presence != "fused":
            continue
        exposed = [step_id for step_id in stage.step_ids if step_id in by_id]
        for step_id in exposed:
            issues.append(
                ExecutionProfileIssue(
                    "execution_profile_fused_stage_exposed",
                    "Execution profile `%s` fuses stage `%s` into `%s`; `%s` must not be a separate block."
                    % (definition.id, stage.role, stage.fused_into, step_id),
                    step_id,
                )
            )

    return ExecutionProfileInspection(
        reference,
        "invalid" if issues else "conformant",
        definition,
        tuple(issues),
        tuple(bindings),
    )
