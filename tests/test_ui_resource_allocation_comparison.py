from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"


class UiResourceAllocationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.app_js = APP_JS.read_text(encoding="utf-8")
        start = self.app_js.index("function resourceAllocationComparisonTable(rows)")
        end = self.app_js.index("function resourceAllocationDiagnosticsTable(rows)", start)
        self.table_source = self.app_js[start:end]
        diagnostics_start = end
        diagnostics_end = self.app_js.index("function resourceAllocationPolicyLabel(op)", diagnostics_start)
        self.diagnostics_source = self.app_js[diagnostics_start:diagnostics_end]

    def test_concluding_table_uses_complete_run_delivery_and_capacity_metrics(self):
        self.assertIn("Delivered payload bits ↑", self.table_source)
        self.assertIn("channel.delivered_payload_bit_count", self.table_source)
        self.assertIn("channel.tx_energy_for_payload", self.table_source)
        self.assertIn("channel.achieved_payload_goodput_bits_per_resource_element", self.table_source)
        self.assertIn("resource.theoretical_shannon_spectral_efficiency_bps_hz", self.table_source)
        self.assertIn("resource.theoretical_p05_state_spectral_efficiency_bps_hz", self.table_source)
        self.assertIn("resource.theoretical_outage_probability", self.table_source)
        self.assertIn("channel.payload_delivery_block_error_rate", self.table_source)
        self.assertIn("resource.power_constraint.max_abs_error", self.table_source)

    def test_concluding_table_is_oracle_free_but_raw_metrics_remain_available(self):
        self.assertNotIn("water_filling_oracle", self.table_source)
        self.assertNotIn("optimality_gap", self.table_source)
        self.assertNotIn("KKT", self.table_source)
        self.assertNotIn("RMSE", self.table_source)

        # Oracle diagnostics remain in the run's raw metrics and all-metrics export;
        # they are simply not used to rank methods in the concluding table.
        self.assertIn("resource.water_filling_oracle_spectral_efficiency_bps_hz", self.app_js)
        self.assertIn("resource.water_filling_relative_optimality_gap", self.app_js)

    def test_oracle_and_constraint_diagnostics_have_their_own_table(self):
        self.assertIn("Policy / WF SE ↑", self.diagnostics_source)
        self.assertIn("WF oracle SE ↑", self.diagnostics_source)
        self.assertIn("Gap to WF ↓", self.diagnostics_source)
        self.assertIn("Relative gap ↓", self.diagnostics_source)
        self.assertIn("WF allocation RMSE ↓", self.diagnostics_source)
        self.assertIn("WF KKT residual ↓", self.diagnostics_source)
        self.assertIn("Relative power error ↓", self.diagnostics_source)
        self.assertIn("Negative-power violation ↓", self.diagnostics_source)
        self.assertIn("resource.water_filling_oracle_spectral_efficiency_bps_hz", self.diagnostics_source)
        self.assertIn("resource.water_filling_optimality_gap_bps_hz", self.diagnostics_source)
        self.assertIn("resource.water_filling_power_normalized_rmse", self.diagnostics_source)
        self.assertIn("resource.water_filling_kkt_normalized_residual", self.diagnostics_source)
        self.assertIn("resource.power_constraint.max_relative_error", self.diagnostics_source)
        self.assertIn("resource.power_constraint.max_negative_violation", self.diagnostics_source)


if __name__ == "__main__":
    unittest.main()
