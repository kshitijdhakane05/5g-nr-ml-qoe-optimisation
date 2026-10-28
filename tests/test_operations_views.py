from __future__ import annotations

import unittest

from dashboard.operations_views import (
    kpi_trend_figure, topology_figure, ue_measurement_table, valid_measurements,
)
from dashboard.config_explanations import configuration_change_rows


class OperationsViewDataTests(unittest.TestCase):
    def test_topology_uses_configured_gnb_and_ue_count(self):
        figure = topology_figure({"ue_count": 3, "gnb_count": 2,
                                  "bandwidth_band1_mhz": 50, "bandwidth_band2_mhz": 50})
        names = [trace.name for trace in figure.data]
        self.assertEqual(names, ["Low-latency UDP workload", "Voice QoS UDP workload", "gNB / cell"])
        self.assertEqual(len(figure.data[0].x), 2)
        self.assertEqual(len(figure.data[1].x), 1)
        self.assertEqual(len(figure.data[2].x), 2)
        self.assertIn("not to scale", figure.layout.title.text)

    def test_topology_draws_measured_cell_associations(self):
        figure = topology_figure(
            {"ue_count": 2, "gnb_count": 2, "bandwidth_band1_mhz": 50, "bandwidth_band2_mhz": 50},
            ue_measurements=[{"ue_id": 1, "serving_cell_id": 1}, {"ue_id": 2, "serving_cell_id": 2}],
        )
        self.assertEqual(figure.data[0].name, "Measured UE association")
        self.assertNotEqual(figure.data[0].x[0], figure.data[0].x[3])

    def test_kpi_plot_preserves_failed_run_as_gap_and_uses_loss_percent(self):
        events = [
            {"epoch": 1, "run_status": "COMPLETED", "metrics": {"mean_packet_loss": 0.125}},
            {"epoch": 2, "run_status": "TIMED_OUT", "metrics": {"mean_packet_loss": 1.0}},
            {"epoch": 3, "run_status": "COMPLETED", "metrics": {"mean_packet_loss": 0.0}},
        ]
        figure = kpi_trend_figure(events, "Packet loss")
        self.assertEqual(list(figure.data[0].x), [1, 2, 3])
        self.assertEqual(list(figure.data[0].y), [12.5, None, 0.0])
        self.assertFalse(figure.data[0].connectgaps)
        self.assertEqual(figure.layout.yaxis.title.text, "%")

    def test_valid_measurements_excludes_failed_and_missing_score_records(self):
        events = [
            {"run_status": "COMPLETED", "measured_qoe_proxy": 0.0, "metrics": {"is_outage": True}},
            {"run_status": "FAILED", "measured_qoe_proxy": 0.7, "metrics": {}},
            {"run_status": "COMPLETED", "metrics": {}},
        ]
        self.assertEqual(valid_measurements(events), [events[0]])

    def test_per_ue_profiles_do_not_claim_unconfigured_sla_compliance(self):
        table = ue_measurement_table([], session_target=0.8, ue_count=2)
        self.assertEqual(table["Traffic class"].tolist(), ["low_latency", "voice_qos"])
        self.assertTrue((table["Individual service SLA"] == "Not defined by this scenario").all())
        self.assertTrue((table["Comparison"] == "Insufficient per-UE measurements").all())
        self.assertTrue(table["Throughput (Mbps)"].isna().all())

    def test_configuration_diff_uses_recorded_reason_and_marks_prediction(self):
        rows = configuration_change_rows(
            {"bandwidth_mhz": 20, "total_tx_power_dbm": 35},
            {"bandwidth_band1_mhz": 50, "total_tx_power_dbm": 35},
            reason="Highest model-ranked candidate", status="Proposed",
            expected_effect="Model estimate only; pending ns-3 run.",
            evaluation_result="Not measured.",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Setting"], "Band 1 bandwidth")
        self.assertEqual(rows[0]["Before"], "20 MHz")
        self.assertEqual(rows[0]["After"], "50 MHz")
        self.assertEqual(rows[0]["Why it changed"], "Highest model-ranked candidate")
        self.assertIn("pending ns-3", rows[0]["Expected effect"])
        self.assertEqual(rows[0]["Status"], "Proposed")
        self.assertEqual(rows[0]["Evaluation result"], "Not measured.")


if __name__ == "__main__":
    unittest.main()
