from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Dict, Optional


JsonDict = Dict[str, Any]

BIT_PER_SOURCE_PIXEL = "bit/source_pixel"
COMPLEX_CHANNEL_USE_PER_SOURCE_PIXEL = "complex_channel_use/source_pixel"

_PROXY_KIND = (
    "noema.resource_conversion.idealized_native_payload_use_proxy"
)
_PROXY_NAME = "idealized native-payload use proxy"


class ResourceUnitError(ValueError):
    """A resource value cannot be compared under the declared unit contract."""


def _unit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ResourceUnitError(
            "%s must be a non-empty unit string without surrounding whitespace"
            % field
        )
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ResourceUnitError("%s must be a real number" % field)
    rendered = float(value)
    if not math.isfinite(rendered):
        raise ResourceUnitError("%s must be finite" % field)
    return rendered


def _modulation_order(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ResourceUnitError("%s must be an integer" % field)
    order = int(value)
    if order < 2 or order & (order - 1):
        raise ResourceUnitError(
            "%s must be a power-of-two integer greater than or equal to 2"
            % field
        )
    return order


@dataclass(frozen=True)
class ResourceQuantity:
    """A non-negative scalar whose physical unit participates in comparison.

    Provenance such as ``native payload`` or ``measured protected use`` belongs
    in the conversion/admission evidence, not in the dimensional unit string.
    This keeps an idealized use proxy and a measured use commensurate without
    pretending they were obtained by the same accounting path.
    """

    value: float
    unit: str

    def __post_init__(self) -> None:
        value = _finite_float(self.value, field="resource value")
        if value < 0.0:
            raise ResourceUnitError("resource value must be non-negative")
        object.__setattr__(self, "value", 0.0 if value == 0.0 else value)
        object.__setattr__(self, "unit", _unit(self.unit, field="resource unit"))

    def to_evidence(self) -> JsonDict:
        return {"value": self.value, "unit": self.unit}


@dataclass(frozen=True)
class ExecutedCodedModulationBindings:
    """Optional executed settings against which a nominal proxy is checked."""

    modulation_order: int
    code_rate: float
    source_unit: Optional[str] = None
    output_unit: Optional[str] = None
    binding_identity: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "modulation_order",
            _modulation_order(
                self.modulation_order,
                field="executed modulation_order",
            ),
        )
        code_rate = _finite_float(
            self.code_rate,
            field="executed code_rate",
        )
        if code_rate <= 0.0 or code_rate > 1.0:
            raise ResourceUnitError(
                "executed code_rate must be greater than 0 and at most 1"
            )
        object.__setattr__(self, "code_rate", code_rate)
        if self.source_unit is not None:
            object.__setattr__(
                self,
                "source_unit",
                _unit(self.source_unit, field="executed source_unit"),
            )
        if self.output_unit is not None:
            object.__setattr__(
                self,
                "output_unit",
                _unit(self.output_unit, field="executed output_unit"),
            )
        if self.binding_identity is not None:
            if (
                not isinstance(self.binding_identity, str)
                or not self.binding_identity
                or self.binding_identity != self.binding_identity.strip()
            ):
                raise ResourceUnitError(
                    "executed binding_identity must be a non-empty string "
                    "without surrounding whitespace"
                )

    def to_evidence(self) -> JsonDict:
        evidence: JsonDict = {
            "modulation_order": self.modulation_order,
            "code_rate": self.code_rate,
        }
        if self.source_unit is not None:
            evidence["source_unit"] = self.source_unit
        if self.output_unit is not None:
            evidence["output_unit"] = self.output_unit
        if self.binding_identity is not None:
            evidence["binding_identity"] = self.binding_identity
        return evidence


@dataclass(frozen=True)
class IdealizedNativePayloadUseProxy:
    """Convert native payload bits using one declared nominal link rate.

    This proxy deliberately excludes framing, padding, finite-block effects,
    pilots, cyclic prefixes, control traffic, and retransmissions.  It is not
    evidence of physical or full-link resource use.
    """

    source_unit: str
    output_unit: str
    modulation_order: int
    nominal_code_rate: float
    executed_bindings: Optional[ExecutedCodedModulationBindings] = None

    def __post_init__(self) -> None:
        source_unit = _unit(self.source_unit, field="proxy source_unit")
        output_unit = _unit(self.output_unit, field="proxy output_unit")
        if source_unit == output_unit:
            raise ResourceUnitError(
                "proxy source_unit and output_unit must be distinct"
            )
        order = _modulation_order(
            self.modulation_order,
            field="proxy modulation_order",
        )
        code_rate = _finite_float(
            self.nominal_code_rate,
            field="proxy nominal_code_rate",
        )
        if code_rate <= 0.0 or code_rate > 1.0:
            raise ResourceUnitError(
                "proxy nominal_code_rate must be greater than 0 and at most 1"
            )
        coefficient = 1.0 / (code_rate * math.log2(order))
        if not math.isfinite(coefficient):
            raise ResourceUnitError(
                "proxy parameters do not produce a finite conversion coefficient"
            )
        bindings = self.executed_bindings
        if bindings is not None and not isinstance(
            bindings,
            ExecutedCodedModulationBindings,
        ):
            raise ResourceUnitError(
                "executed_bindings must be ExecutedCodedModulationBindings"
            )
        if bindings is not None:
            if bindings.modulation_order != order:
                raise ResourceUnitError(
                    "executed modulation_order does not match the nominal proxy"
                )
            if bindings.code_rate != code_rate:
                raise ResourceUnitError(
                    "executed code_rate does not match nominal_code_rate"
                )
            if (
                bindings.source_unit is not None
                and bindings.source_unit != source_unit
            ):
                raise ResourceUnitError(
                    "executed source_unit does not match the nominal proxy"
                )
            if (
                bindings.output_unit is not None
                and bindings.output_unit != output_unit
            ):
                raise ResourceUnitError(
                    "executed output_unit does not match the nominal proxy"
                )
        object.__setattr__(self, "source_unit", source_unit)
        object.__setattr__(self, "output_unit", output_unit)
        object.__setattr__(self, "modulation_order", order)
        object.__setattr__(self, "nominal_code_rate", code_rate)

    @property
    def bits_per_modulation_use(self) -> int:
        return self.modulation_order.bit_length() - 1

    @property
    def coefficient(self) -> float:
        return 1.0 / (
            self.nominal_code_rate * float(self.bits_per_modulation_use)
        )

    def convert(self, quantity: ResourceQuantity) -> ResourceQuantity:
        if not isinstance(quantity, ResourceQuantity):
            raise ResourceUnitError(
                "proxy input must be a typed ResourceQuantity"
            )
        if quantity.unit != self.source_unit:
            raise ResourceUnitError(
                "observed resource unit %r does not match proxy source_unit %r"
                % (quantity.unit, self.source_unit)
            )
        transformed = quantity.value * self.coefficient
        if not math.isfinite(transformed):
            raise ResourceUnitError(
                "proxy conversion did not produce a finite resource value"
            )
        return ResourceQuantity(transformed, self.output_unit)

    def to_evidence(self) -> JsonDict:
        evidence: JsonDict = {
            "kind": _PROXY_KIND,
            "name": _PROXY_NAME,
            "interpretation": (
                "idealized native-payload use proxy; not measured physical "
                "or full-link use"
            ),
            "formula": (
                "bits_per_source_unit / "
                "(nominal_code_rate * log2(modulation_order))"
            ),
            "source_unit": self.source_unit,
            "output_unit": self.output_unit,
            "modulation_order": self.modulation_order,
            "bits_per_modulation_use": self.bits_per_modulation_use,
            "nominal_code_rate": self.nominal_code_rate,
            "coefficient": self.coefficient,
        }
        if self.executed_bindings is not None:
            evidence["executed_bindings"] = (
                self.executed_bindings.to_evidence()
            )
        return evidence


def evaluate_resource_admission(
    *,
    observed: ResourceQuantity,
    budget: ResourceQuantity,
    aggregation_policy: str,
    conversion: Optional[IdealizedNativePayloadUseProxy] = None,
    tolerance: Optional[ResourceQuantity] = None,
) -> JsonDict:
    """Convert an already-aggregated observation and compare it to a budget.

    Unit equality is exact.  Different units require the explicit idealized
    native-payload proxy; arbitrary conversion callbacks are not accepted.
    The caller remains responsible for performing the declared aggregation.
    """

    if not isinstance(observed, ResourceQuantity):
        raise ResourceUnitError("observed must be a typed ResourceQuantity")
    if not isinstance(budget, ResourceQuantity):
        raise ResourceUnitError("budget must be a typed ResourceQuantity")
    if tolerance is None:
        tolerance = ResourceQuantity(0.0, budget.unit)
    elif not isinstance(tolerance, ResourceQuantity):
        raise ResourceUnitError("tolerance must be a typed ResourceQuantity")
    elif tolerance.unit != budget.unit:
        raise ResourceUnitError(
            "tolerance unit %r does not match budget unit %r"
            % (tolerance.unit, budget.unit)
        )
    if (
        not isinstance(aggregation_policy, str)
        or not aggregation_policy
        or aggregation_policy != aggregation_policy.strip()
    ):
        raise ResourceUnitError(
            "aggregation_policy must be a non-empty string without "
            "surrounding whitespace"
        )

    if conversion is None:
        if observed.unit != budget.unit:
            raise ResourceUnitError(
                "observed and budget units differ; an explicit "
                "IdealizedNativePayloadUseProxy is required"
            )
        transformed = observed
        conversion_evidence: JsonDict = {
            "kind": "noema.resource_conversion.identity",
            "source_unit": observed.unit,
            "output_unit": observed.unit,
            "coefficient": 1.0,
        }
    else:
        if not isinstance(conversion, IdealizedNativePayloadUseProxy):
            raise ResourceUnitError(
                "the only supported non-identity conversion is "
                "IdealizedNativePayloadUseProxy"
            )
        if conversion.output_unit != budget.unit:
            raise ResourceUnitError(
                "proxy output_unit %r does not match budget unit %r"
                % (conversion.output_unit, budget.unit)
            )
        transformed = conversion.convert(observed)
        conversion_evidence = conversion.to_evidence()

    margin = budget.value - transformed.value
    admission_margin = budget.value + tolerance.value - transformed.value
    return {
        "schema_version": 1,
        "kind": "noema.resource_conversion_admission",
        "aggregation_policy": aggregation_policy,
        "observed": observed.to_evidence(),
        "conversion": {
            **conversion_evidence,
            "input_value": observed.value,
            "transformed_value": transformed.value,
            "transformed_unit": transformed.unit,
        },
        "budget": {
            "maximum": budget.value,
            "tolerance": tolerance.value,
            "unit": budget.unit,
        },
        "admission": {
            "admitted": transformed.value <= budget.value + tolerance.value,
            "comparison": (
                "transformed_value <= budget_maximum + budget_tolerance"
            ),
            "margin": margin,
            "margin_unit": budget.unit,
            "admission_boundary_margin": admission_margin,
        },
    }
