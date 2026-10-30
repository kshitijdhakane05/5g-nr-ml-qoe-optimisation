import unittest

from simulation.hybrid_advisor import compare_rate_candidate, load_bundle, predict_ue


class HybridAdvisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()

    def test_versioned_artifact_schema_loads(self):
        self.assertIsNotNone(self.bundle)
        self.assertEqual(self.bundle["metadata"]["artifact_version"], "live-ns3-kpi-v1")

    def test_supported_context_predicts_kpis_and_service_qoe(self):
        result = predict_ue(self.bundle, config={
            "ue_count": 2, "gnb_count": 1, "bandwidth_mhz": 50,
            "total_tx_power_dbm": 35, "scenario_size_m": 30,
            "traffic_mix": "legacy", "lambda_ull": 10000,
        }, traffic_class="low_latency", lambda_be=20000)
        self.assertTrue(result["supported"])
        self.assertGreaterEqual(result["predicted_qoe"], 0)
        self.assertLessEqual(result["predicted_qoe"], 1)
        self.assertIn("throughput_mbps", result["predicted_metrics"])
        self.assertGreaterEqual(result["predicted_receive_probability"], 0)

    def test_unsupported_traffic_and_radio_context_are_rejected(self):
        config = {"ue_count": 2, "gnb_count": 2, "bandwidth_mhz": 50,
                  "total_tx_power_dbm": 35, "scenario_size_m": 30,
                  "traffic_mix": "gaming_video", "lambda_ull": 10000}
        self.assertFalse(predict_ue(self.bundle, config=config,
                                    traffic_class="gaming", lambda_be=10000)["supported"])
        config["traffic_mix"] = "legacy"
        self.assertFalse(predict_ue(self.bundle, config=config,
                                    traffic_class="low_latency", lambda_be=10000)["supported"])

    def test_model_compares_candidate_and_protects_worst_class(self):
        config = {"ue_count": 2, "gnb_count": 1, "bandwidth_mhz": 50,
                  "total_tx_power_dbm": 35, "scenario_size_m": 30,
                  "traffic_mix": "legacy", "lambda_ull": 10000, "lambda_be": 20000}
        result = compare_rate_candidate({
            "config": config, "active_parameters": {"lambda_be_pps": 20000},
            "latest": {"ues": [{"traffic_class": "low_latency", "qoe": .9},
                                {"traffic_class": "voice_qos", "qoe": .6}]},
        }, 10000)
        self.assertTrue(result["available"])
        self.assertTrue(result["candidate_supported"])
        self.assertGreater(result["predicted_mean_delta"], 0)
        self.assertGreaterEqual(result["predicted_worst_delta"], -0.01)
        self.assertEqual(result["measured_mean_qoe_before"], .75)
        self.assertGreater(result["hybrid_estimated_mean_qoe_after"], .75)
        self.assertIn("not a live measurement", result["source"])


if __name__ == "__main__":
    unittest.main()
