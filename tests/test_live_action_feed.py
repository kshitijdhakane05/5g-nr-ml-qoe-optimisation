import unittest

from dashboard.live_simulation import _event_summary, format_action_feed


class LiveActionFeedTests(unittest.TestCase):
    def test_feed_labels_action_triggered_before_target_breach_as_predictive(self):
        event = {"type": "action_pending", "timestamp": "2026-10-30T12:00:00Z",
                 "action": "Best-effort offered packet rate", "previous_pps": 10000,
                 "new_pps": 5000, "decision_basis": "predictive_preemptive",
                 "measured_qoe_at_decision": .81, "target_qoe": .8,
                 "trigger_forecast": {"qoe": .62, "horizon_s": 5},
                 "reason": "Two users are forecast below target."}
        row = _event_summary(event)
        self.assertIn("PRE-EMPTIVE FORECAST ACTION", row["Details"])
        self.assertIn("still above target 0.800", row["Details"])
        self.assertIn("+5s", row["Details"])

    def test_feed_separates_forecasts_from_measured_outcomes_and_groups_no_action(self):
        events = [
            {"type": "controller_decision", "timestamp": "2026-10-30T12:00:00Z",
             "action": "No action", "reason": "Holding healthy configuration"},
            {"type": "controller_decision", "timestamp": "2026-10-30T12:00:01Z",
             "action": "No action", "reason": "Holding healthy configuration"},
            {"type": "restart_epoch_outcome", "timestamp": "2026-10-30T12:00:02Z",
             "pre_action_qoe_mean": .7, "post_action_qoe_mean": .8},
            {"type": "controller_decision", "timestamp": "2026-10-30T12:00:03Z",
             "action": "Increase bandwidth", "forecast": {"qoe": .65}},
        ]
        frame, grouped_count = format_action_feed(events)
        self.assertEqual(grouped_count, 1)
        self.assertIn("Trend forecast (not measured)", frame.iloc[0]["Details"])
        self.assertIn("Measured mean UE QoE", frame.iloc[1]["Details"])
        self.assertIn("2 no-action decisions", " ".join(frame["Event"].tolist()))


if __name__ == "__main__":
    unittest.main()
