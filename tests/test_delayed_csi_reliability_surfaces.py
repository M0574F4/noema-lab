from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
TUTORIAL = ROOT / "docs" / "tutorials" / "reliability_aware_ofdm_allocation_demo.md"


class DelayedCsiReliabilitySurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_js = APP_JS.read_text(encoding="utf-8")
        cls.tutorial = TUTORIAL.read_text(encoding="utf-8")

    def test_overview_and_table_use_finite_blocklength_metrics(self) -> None:
        overview_start = self.app_js.index("function resultOverviewPrimaryFigures(rows)")
        overview_end = self.app_js.index("function resultExamplesPanel(rows", overview_start)
        overview_source = self.app_js[overview_start:overview_end]
        self.assertIn("resourceFiniteBlocklengthOverviewFigures", overview_source)

        figure_start = self.app_js.index("function resourceFiniteBlocklengthOverviewFigures(rows)")
        figure_end = self.app_js.index("function resourceDeliveryFigure(rows)", figure_start)
        figure_source = self.app_js[figure_start:figure_end]
        self.assertIn("finiteBlocklengthGoodput", figure_source)
        self.assertIn("finiteBlocklengthBler", figure_source)
        self.assertIn("Finite-Blocklength Expected Goodput", figure_source)
        self.assertIn("Finite-Blocklength Predicted BLER", figure_source)
        self.assertIn("data-log-axes", figure_source)
        self.assertIn("data-domain-chart", figure_source)

        table_start = self.app_js.index("function resourceFiniteBlocklengthComparisonTable(rows)")
        table_end = self.app_js.index("function resourceAllocationCountCell", table_start)
        table_source = self.app_js[table_start:table_end]
        self.assertIn("resource.finite_blocklength.expected_goodput_bps_hz", table_source)
        self.assertIn("resource.finite_blocklength.predicted_bler", table_source)
        self.assertIn("resource.csi.observed_actual_gain_correlation", table_source)
        self.assertIn("resource.csi.observed_actual_complex_nmse", table_source)

    def test_modeled_finite_blocklength_runs_skip_uncoded_error_figures(self) -> None:
        panel_start = self.app_js.index("function resultCommunicationPanel(rows")
        panel_end = self.app_js.index("function receiverDecisionPreview", panel_start)
        panel_source = self.app_js[panel_start:panel_end]
        self.assertIn("modeledFiniteBlocklengthRow", panel_source)
        self.assertIn(
            "!modeledFiniteBlocklengthRow(row) && rowHasCommunicationErrorEvidence(row)",
            panel_source,
        )
        self.assertIn(
            "!modeledFiniteBlocklengthRow(row) &&",
            panel_source,
        )

    def test_delayed_csi_water_filling_is_not_labeled_as_an_oracle(self) -> None:
        start = self.app_js.index("function resourceAllocationPolicyName(item)")
        end = self.app_js.index("function resourcePolicySeries", start)
        source = self.app_js[start:end]
        self.assertLess(source.index('normalized.includes("observed")'), source.index('if (normalized.includes("water")) return "Water filling (Shannon oracle)"'))
        self.assertIn('return "Water filling on delayed CSI"', source)
        self.assertIn('return "Uncertainty-shrunk water filling"', source)

    def test_allocation_preview_can_show_observed_and_current_csi(self) -> None:
        start = self.app_js.index("function resourceAllocationFigure(rows)")
        end = self.app_js.index("function resourcePerformanceItems", start)
        source = self.app_js[start:end]
        self.assertIn("observed_unit_power_snr_db", source)
        self.assertIn("observed_channel_gain", source)
        self.assertIn("actual_trace_label", source)
        self.assertIn("observed_trace_label", source)
        self.assertIn("snr:actual:", source)
        self.assertIn("snr:observed:", source)

    def test_causal_allocator_uses_the_same_managed_artifact_picker(self) -> None:
        self.assertIn(
            '"model.causal_csi_power_allocator": Object.freeze({',
            self.app_js,
        )
        start = self.app_js.index(
            '"model.causal_csi_power_allocator": Object.freeze({'
        )
        source = self.app_js[start : start + 1200]
        self.assertIn('artifactActivation: "explicit"', source)
        self.assertIn("aggregateArtifactControls: true", source)
        self.assertIn('canonicalValue: "learned_artifact"', source)

    def test_tutorial_has_one_complete_cli_block_and_exact_ui_contract(self) -> None:
        self.assertIn("## CLI training summary", self.tutorial)
        cli = self.tutorial.split("## CLI training summary", 1)[1].split("## Scenario", 1)[0]
        self.assertEqual(cli.count("```bash"), 1)
        self.assertIn("delayed-csi-resource-allocation", cli)
        self.assertIn("resource_allocation_delayed_csi_finite_blocklength/training_plan.yaml", cli)
        self.assertIn("capture_train_recipe.yaml", cli)
        self.assertIn("capture_validation_recipe.yaml", cli)
        self.assertIn("capture_test_recipe.yaml", cli)
        self.assertIn("train_demo.py", cli)
        self.assertIn("evaluate_demo.py", cli)
        self.assertIn("build_benchmark.py", cli)

        self.assertIn("Reliability-aware OFDM allocation with delayed CSI", self.tutorial)
        self.assertIn("csi_observation_transmitter_csi", self.tutorial)
        self.assertIn("csi_observation_actual_state", self.tutorial)
        self.assertIn("current channel never enters the returned model ABI", self.tutorial)
        self.assertIn(
            "20260726T192946Z_resource_allocation."
            "delayed_csi_finite_blocklength_post_training_v2",
            self.tutorial,
        )
        self.assertIn(
            'data-noema-chart="delayed-csi-goodput"',
            self.tutorial,
        )
        self.assertIn(
            'data-noema-chart="delayed-csi-predicted-bler"',
            self.tutorial,
        )
        self.assertIn(
            "ordinary uncoded-QPSK payload BER and BLER",
            self.tutorial,
        )
        self.assertIn(
            "--slug reliability-aware-delayed-csi-ofdm-allocation",
            self.tutorial,
        )

    def test_tutorial_is_registered_in_both_navigation_pages(self) -> None:
        tutorials = (ROOT / "docs" / "tutorials.md").read_text(encoding="utf-8")
        workflow = (
            ROOT / "docs" / "tutorials" / "physical_layer_demo_workflow.md"
        ).read_text(encoding="utf-8")
        self.assertIn("tutorials/reliability_aware_ofdm_allocation_demo.md", tutorials)
        self.assertIn("reliability_aware_ofdm_allocation_demo", workflow)


if __name__ == "__main__":
    unittest.main()
