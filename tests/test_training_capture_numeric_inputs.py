from __future__ import annotations

import unittest
from pathlib import Path

from noema_lab.core.recipes import load_recipe
from noema_lab.training.csi_feedback_export import (
    CsiFeedbackExportError,
    suggested_csi_feedback_capture_total,
)
from noema_lab.training.resource_allocation_export import (
    ResourceAllocationExportError,
    resource_capture_single_run_capacity,
    suggested_resource_capture_total,
)


ROOT = Path(__file__).resolve().parents[1]


class TrainingCaptureNumericInputTests(unittest.TestCase):
    def test_csi_feedback_rejects_invalid_explicit_capture_counts(self) -> None:
        cases = (
            (
                {"split_plan": {"total_samples": "768"}},
                "dataset_capture.split_plan.total_samples",
            ),
            ({"split_plan": {"total_samples": 2}}, "total_samples"),
            ({"samples": False}, "dataset_capture.samples"),
            ({"samples": 0}, "dataset_capture.samples"),
            ({"split_plan": []}, "split_plan must be a mapping"),
        )
        for capture, expected in cases:
            with self.subTest(capture=capture):
                recipe = load_recipe(
                    ROOT / "recipes" / "csi_feedback_sionna_train.yaml"
                )
                recipe.dataset_capture = capture
                with self.assertRaisesRegex(CsiFeedbackExportError, expected):
                    suggested_csi_feedback_capture_total(recipe)

    def test_csi_feedback_rejects_invalid_source_sample_count(self) -> None:
        for value in ("512", 0, None, True):
            with self.subTest(value=value):
                recipe = load_recipe(
                    ROOT / "recipes" / "csi_feedback_sionna_train.yaml"
                )
                source = next(
                    step
                    for step in recipe.steps
                    if step.op == "wireless.miso_ofdm_csi"
                )
                source.params["sample_count"] = value
                with self.assertRaisesRegex(
                    CsiFeedbackExportError,
                    "wireless.miso_ofdm_csi.sample_count",
                ):
                    suggested_csi_feedback_capture_total(recipe)

    def test_resource_capture_counts_are_strict_and_precedence_is_explicit(self) -> None:
        recipe = load_recipe(
            ROOT / "recipes" / "resource_water_filling_baseline.yaml"
        )
        self.assertEqual(resource_capture_single_run_capacity(recipe), 256)
        self.assertEqual(suggested_resource_capture_total(recipe), 192)

        recipe.dataset_capture = {
            "samples": 9,
            "split_plan": {"total_samples": 12},
        }
        self.assertEqual(suggested_resource_capture_total(recipe), 12)

    def test_resource_capture_rejects_invalid_explicit_counts(self) -> None:
        cases = (
            (
                {"split_plan": {"total_samples": "12"}},
                "dataset_capture.split_plan.total_samples",
            ),
            ({"split_plan": {"total_samples": 2}}, "total_samples"),
            ({"samples": 3.0}, "dataset_capture.samples"),
            ({"samples": -1}, "dataset_capture.samples"),
            ({"split_plan": None}, "split_plan must be a mapping"),
        )
        for capture, expected in cases:
            with self.subTest(capture=capture):
                recipe = load_recipe(
                    ROOT / "recipes" / "resource_water_filling_baseline.yaml"
                )
                recipe.dataset_capture = capture
                with self.assertRaisesRegex(ResourceAllocationExportError, expected):
                    suggested_resource_capture_total(recipe)

    def test_resource_capture_rejects_invalid_source_batch_size(self) -> None:
        for value in ("256", 0, None, False):
            with self.subTest(value=value):
                recipe = load_recipe(
                    ROOT / "recipes" / "resource_water_filling_baseline.yaml"
                )
                source = next(
                    step for step in recipe.steps if step.op == "source.random_bits"
                )
                source.params["batch_size"] = value
                with self.assertRaisesRegex(
                    ResourceAllocationExportError,
                    "source.random_bits.batch_size",
                ):
                    resource_capture_single_run_capacity(recipe)


if __name__ == "__main__":
    unittest.main()
