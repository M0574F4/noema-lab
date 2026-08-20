from __future__ import annotations

import math
import unittest

from noema_lab.core.resource_units import (
    BIT_PER_SOURCE_PIXEL,
    COMPLEX_CHANNEL_USE_PER_SOURCE_PIXEL,
    ExecutedCodedModulationBindings,
    IdealizedNativePayloadUseProxy,
    ResourceQuantity,
    ResourceUnitError,
    evaluate_resource_admission,
)


AGGREGATION = "mean_over_declared_source_items"


class ResourceUnitTests(unittest.TestCase):
    def test_qpsk_rate_one_half_proxy_is_identity_valued_not_identity_typed(
        self,
    ) -> None:
        proxy = IdealizedNativePayloadUseProxy(
            source_unit=BIT_PER_SOURCE_PIXEL,
            output_unit=COMPLEX_CHANNEL_USE_PER_SOURCE_PIXEL,
            modulation_order=4,
            nominal_code_rate=0.5,
            executed_bindings=ExecutedCodedModulationBindings(
                modulation_order=4,
                code_rate=0.5,
                source_unit=BIT_PER_SOURCE_PIXEL,
                output_unit=COMPLEX_CHANNEL_USE_PER_SOURCE_PIXEL,
                binding_identity="retained-plan:coder-and-modulator",
            ),
        )

        evidence = evaluate_resource_admission(
            observed=ResourceQuantity(
                0.4,
                BIT_PER_SOURCE_PIXEL,
            ),
            budget=ResourceQuantity(
                0.5,
                COMPLEX_CHANNEL_USE_PER_SOURCE_PIXEL,
            ),
            aggregation_policy=AGGREGATION,
            conversion=proxy,
        )

        self.assertEqual(evidence["aggregation_policy"], AGGREGATION)
        self.assertEqual(
            evidence["conversion"]["name"],
            "idealized native-payload use proxy",
        )
        self.assertNotEqual(
            evidence["observed"]["unit"],
            evidence["budget"]["unit"],
        )
        self.assertEqual(evidence["conversion"]["coefficient"], 1.0)
        self.assertEqual(evidence["conversion"]["transformed_value"], 0.4)
        self.assertAlmostEqual(evidence["admission"]["margin"], 0.1)
        self.assertTrue(evidence["admission"]["admitted"])
        self.assertEqual(
            evidence["conversion"]["executed_bindings"][
                "binding_identity"
            ],
            "retained-plan:coder-and-modulator",
        )
        self.assertIn(
            "not measured physical or full-link use",
            evidence["conversion"]["interpretation"],
        )

    def test_16qam_rate_three_quarters_proxy_is_nonidentity(self) -> None:
        proxy = IdealizedNativePayloadUseProxy(
            source_unit="bit/source_sample",
            output_unit="complex_channel_use/source_sample",
            modulation_order=16,
            nominal_code_rate=0.75,
        )

        evidence = evaluate_resource_admission(
            observed=ResourceQuantity(
                0.9,
                "bit/source_sample",
            ),
            budget=ResourceQuantity(
                0.25,
                "complex_channel_use/source_sample",
            ),
            aggregation_policy="maximum_over_declared_source_items",
            conversion=proxy,
        )

        self.assertAlmostEqual(evidence["conversion"]["coefficient"], 1.0 / 3.0)
        self.assertAlmostEqual(
            evidence["conversion"]["transformed_value"],
            0.3,
        )
        self.assertAlmostEqual(evidence["admission"]["margin"], -0.05)
        self.assertFalse(evidence["admission"]["admitted"])

    def test_same_unit_admission_uses_identity_without_a_transform(self) -> None:
        unit = COMPLEX_CHANNEL_USE_PER_SOURCE_PIXEL
        evidence = evaluate_resource_admission(
            observed=ResourceQuantity(0.3, unit),
            budget=ResourceQuantity(0.5, unit),
            aggregation_policy=AGGREGATION,
        )
        self.assertEqual(
            evidence["conversion"]["kind"],
            "noema.resource_conversion.identity",
        )
        self.assertEqual(evidence["conversion"]["coefficient"], 1.0)
        self.assertTrue(evidence["admission"]["admitted"])

    def test_tolerance_is_typed_and_applied_only_at_admission_boundary(
        self,
    ) -> None:
        unit = COMPLEX_CHANNEL_USE_PER_SOURCE_PIXEL
        evidence = evaluate_resource_admission(
            observed=ResourceQuantity(0.5005, unit),
            budget=ResourceQuantity(0.5, unit),
            tolerance=ResourceQuantity(0.001, unit),
            aggregation_policy=AGGREGATION,
        )
        self.assertTrue(evidence["admission"]["admitted"])
        self.assertAlmostEqual(evidence["admission"]["margin"], -0.0005)
        self.assertAlmostEqual(
            evidence["admission"]["admission_boundary_margin"],
            0.0005,
        )
        with self.assertRaisesRegex(
            ResourceUnitError,
            "tolerance unit",
        ):
            evaluate_resource_admission(
                observed=ResourceQuantity(0.5, unit),
                budget=ResourceQuantity(0.5, unit),
                tolerance=ResourceQuantity(0.001, "bit/source_pixel"),
                aggregation_policy=AGGREGATION,
            )

    def test_different_units_without_explicit_transform_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            ResourceUnitError,
            "units differ",
        ):
            evaluate_resource_admission(
                observed=ResourceQuantity(
                    0.4,
                    BIT_PER_SOURCE_PIXEL,
                ),
                budget=ResourceQuantity(
                    0.5,
                    COMPLEX_CHANNEL_USE_PER_SOURCE_PIXEL,
                ),
                aggregation_policy=AGGREGATION,
            )

    def test_proxy_rejects_source_and_budget_unit_mismatches(self) -> None:
        proxy = IdealizedNativePayloadUseProxy(
            source_unit=BIT_PER_SOURCE_PIXEL,
            output_unit=COMPLEX_CHANNEL_USE_PER_SOURCE_PIXEL,
            modulation_order=4,
            nominal_code_rate=0.5,
        )
        with self.assertRaisesRegex(
            ResourceUnitError,
            "does not match proxy source_unit",
        ):
            evaluate_resource_admission(
                observed=ResourceQuantity(
                    0.4,
                    "bit/source_sample",
                ),
                budget=ResourceQuantity(
                    0.5,
                    COMPLEX_CHANNEL_USE_PER_SOURCE_PIXEL,
                ),
                aggregation_policy=AGGREGATION,
                conversion=proxy,
            )
        with self.assertRaisesRegex(
            ResourceUnitError,
            "does not match budget unit",
        ):
            evaluate_resource_admission(
                observed=ResourceQuantity(
                    0.4,
                    BIT_PER_SOURCE_PIXEL,
                ),
                budget=ResourceQuantity(
                    0.5,
                    "complex_channel_use/source_item",
                ),
                aggregation_policy=AGGREGATION,
                conversion=proxy,
            )

    def test_proxy_rejects_invalid_modulation_and_rate(self) -> None:
        base = {
            "source_unit": BIT_PER_SOURCE_PIXEL,
            "output_unit": COMPLEX_CHANNEL_USE_PER_SOURCE_PIXEL,
            "nominal_code_rate": 0.5,
        }
        for invalid_order in (1, 3, 6, 4.0, True):
            with self.subTest(modulation_order=invalid_order):
                with self.assertRaises(ResourceUnitError):
                    IdealizedNativePayloadUseProxy(
                        modulation_order=invalid_order,
                        **base,
                    )
        for invalid_rate in (0.0, -0.5, 1.1, math.inf, math.nan, True):
            with self.subTest(nominal_code_rate=invalid_rate):
                with self.assertRaises(ResourceUnitError):
                    IdealizedNativePayloadUseProxy(
                        source_unit=base["source_unit"],
                        output_unit=base["output_unit"],
                        modulation_order=4,
                        nominal_code_rate=invalid_rate,
                    )

    def test_proxy_rejects_executed_binding_mismatches(self) -> None:
        common = {
            "source_unit": BIT_PER_SOURCE_PIXEL,
            "output_unit": COMPLEX_CHANNEL_USE_PER_SOURCE_PIXEL,
            "modulation_order": 4,
            "nominal_code_rate": 0.5,
        }
        mismatches = (
            ExecutedCodedModulationBindings(
                modulation_order=16,
                code_rate=0.5,
            ),
            ExecutedCodedModulationBindings(
                modulation_order=4,
                code_rate=0.75,
            ),
            ExecutedCodedModulationBindings(
                modulation_order=4,
                code_rate=0.5,
                source_unit="bit/source_sample",
            ),
            ExecutedCodedModulationBindings(
                modulation_order=4,
                code_rate=0.5,
                output_unit="complex_channel_use/source_sample",
            ),
        )
        for bindings in mismatches:
            with self.subTest(bindings=bindings):
                with self.assertRaises(ResourceUnitError):
                    IdealizedNativePayloadUseProxy(
                        executed_bindings=bindings,
                        **common,
                    )

    def test_nonfinite_quantities_and_blank_aggregation_are_rejected(
        self,
    ) -> None:
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ResourceUnitError, "finite"):
                    ResourceQuantity(value, "resource/source_item")
        with self.assertRaisesRegex(ResourceUnitError, "non-negative"):
            ResourceQuantity(-0.1, "resource/source_item")
        with self.assertRaisesRegex(ResourceUnitError, "aggregation_policy"):
            evaluate_resource_admission(
                observed=ResourceQuantity(0.1, "resource/source_item"),
                budget=ResourceQuantity(0.2, "resource/source_item"),
                aggregation_policy=" ",
            )

    def test_arbitrary_conversion_objects_are_not_accepted(self) -> None:
        with self.assertRaisesRegex(
            ResourceUnitError,
            "only supported non-identity conversion",
        ):
            evaluate_resource_admission(
                observed=ResourceQuantity(0.1, "a/source_item"),
                budget=ResourceQuantity(0.2, "b/source_item"),
                aggregation_policy=AGGREGATION,
                conversion=object(),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
