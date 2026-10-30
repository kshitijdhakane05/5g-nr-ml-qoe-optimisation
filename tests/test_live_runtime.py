from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from simulation import live_runtime as live


class LiveForecastTests(unittest.TestCase):
    def test_paired_trial_uses_complete_post_onset_windows(self):
        from simulation.predictive_demand import _point_has_requested_demand
        at_command_tick = {"sim_time_s": 9.4, "window_start_s": 8.4,
                           "requested_lambda_be_pps": 20000, "lambda_be_pps": 20000}
        full_window = {"sim_time_s": 10.4, "window_start_s": 9.4,
                       "requested_lambda_be_pps": 20000, "lambda_be_pps": 20000}
        self.assertFalse(_point_has_requested_demand(at_command_tick, 20000, 9.4))
        self.assertTrue(_point_has_requested_demand(full_window, 20000, 9.4))

    def test_overall_user_qoe_is_equal_weight_mean_and_zero_rx_counts(self):
        self.assertAlmostEqual(live.mean_user_qoe([{"qoe": .9}, {"qoe": .5}, {"qoe": 0}]), 1.4 / 3)
        self.assertEqual(live.mean_user_qoe([]), 0.0)
        self.assertEqual(live.mean_user_qoe([{"qoe": 1.4}, {"qoe": -.2}]), .5)

    def test_forecast_requires_four_measurements(self):
        self.assertFalse(live.forecast([])["available"])
        self.assertFalse(live.forecast([{"sim_time_s": i} for i in range(3)])["available"])

    def test_forecast_extrapolates_falling_qoe_and_reports_low_confidence(self):
        pts = []
        for i in range(5):
            pts.append({"sim_time_s": i, "qoe": .9 - .02*i, "throughput_mbps": 40,
                        "delay_ms": 10 + 2*i, "jitter_ms": 1, "packet_loss_fraction": .01})
        result = live.forecast(pts)
        self.assertTrue(result["available"])
        self.assertTrue(result["risk"])
        self.assertEqual(result["horizon_s"], 5)
        self.assertEqual(result["confidence"], "low; uncalibrated short-history baseline")

    def test_forecast_marks_target_violation_even_without_a_falling_trend(self):
        pts = [{"sim_time_s": i, "qoe": .6, "throughput_mbps": 30, "delay_ms": 10,
                "jitter_ms": 2, "packet_loss_fraction": .01} for i in range(5)]
        self.assertFalse(live.forecast(pts)["risk"])
        result = live.forecast(pts, target_qoe=.7)
        self.assertTrue(result["risk"])
        self.assertIn("below the configured target", result["reason"])


class LiveActionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sid = "live_test"
        folder = self.root / self.sid
        folder.mkdir()
        self.folder = folder
        (folder / "session.json").write_text(json.dumps({
            "session_id": self.sid, "created_at": live._utc(), "status": "Running", "mode": "Recommend",
            "active_parameters": {"lambda_be_pps": 10000}, "last_control_at": 0,
            "action_pending": False,
        }))
        (folder / "events.jsonl").touch()
        self.patch = patch.object(live, "LIVE_ROOT", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_monitor_mode_cannot_apply_action(self):
        state = json.loads((self.folder / "session.json").read_text())
        state["mode"] = "Monitor"
        (self.folder / "session.json").write_text(json.dumps(state))
        with self.assertRaises(PermissionError):
            live.request_rate_change(self.sid, 5000, approved=True)
        self.assertFalse((self.folder / "command.txt").exists())

    def test_recommend_mode_requires_explicit_approval_and_bounds(self):
        with self.assertRaises(PermissionError):
            live.request_rate_change(self.sid, 5000, approved=False)
        with self.assertRaises(ValueError):
            live.request_rate_change(self.sid, 2000, approved=True)
        live.request_rate_change(self.sid, 5000, approved=True)
        self.assertEqual((self.folder / "command.txt").read_text(), "5000")
        event = json.loads((self.folder / "events.jsonl").read_text().splitlines()[0])
        self.assertEqual(event["type"], "action_pending")

    def test_stale_session_rejects_action(self):
        state = json.loads((self.folder / "session.json").read_text())
        state["status"] = "Stale"
        (self.folder / "session.json").write_text(json.dumps(state))
        with self.assertRaises(PermissionError):
            live.request_rate_change(self.sid, 5000, approved=True)

    def test_scheduled_demand_protection_uses_matched_reference_and_separate_cap(self):
        state = json.loads((self.folder / "session.json").read_text())
        state.update({"mode": "Automatic", "target_qoe": .85,
            "config": {"admission_control": True, "ue_count": 2, "traffic_mix": "legacy"},
            "latest": {"sim_time_s": 5.0, "qoe": .91,
                "ues": [{"ue_id": 1, "qoe": .90}, {"ue_id": 2, "qoe": .92}]},
            "active_parameters": {"lambda_be_pps": 7500, "lambda_be_cap_pps": 20000}})
        (self.folder / "session.json").write_text(json.dumps(state))
        reference = {"session_id": "matched_no_control", "per_ue_qoe": {"1": .62, "2": .74}}
        live.schedule_demand_preview(self.sid, requested_pps=20000, onset_sim_time_s=10,
            preemptive_cap_pps=5000, reference=reference)
        request_id = live.request_demand_change(self.sid, 20000, onset_sim_time_s=10)
        self.assertTrue(request_id)
        demand_file = self.folder / "segments/segment_001/demand.txt"
        if not demand_file.exists():  # compatibility path for an unsegmented fixture
            demand_file = self.folder / "demand.txt"
        self.assertEqual(demand_file.read_text(), "10.000000 20000")
        saved = json.loads((self.folder / "session.json").read_text())
        self.assertEqual(saved["scheduled_demand_preview"]["preemptive_cap_pps"], 5000)
        self.assertTrue(saved["demand_change_pending"])

    def test_scheduled_reference_triggers_cap_while_live_ues_are_still_healthy(self):
        state = json.loads((self.folder / "session.json").read_text())
        state.update({"mode": "Automatic", "target_qoe": .85,
            "config": {"admission_control": True, "ue_count": 2, "traffic_mix": "legacy"},
            "active_parameters": {"lambda_be_pps": 7500, "lambda_be_cap_pps": 20000},
            "scheduled_demand_preview": {"status": "scheduled", "requested_pps": 20000,
                "onset_sim_time_s": 10, "preemptive_cap_pps": 5000,
                "reference": {"session_id": "matched_no_control", "per_ue_qoe": {"1": .62, "2": .74}}}})
        (self.folder / "session.json").write_text(json.dumps(state))
        latest = {"sim_time_s": 6.0, "qoe": .91,
            "ues": [{"ue_id": 1, "qoe": .90}, {"ue_id": 2, "qoe": .92}]}
        with patch.object(live, "request_rate_change", return_value="action-test") as apply_action:
            self.assertTrue(live._apply_scheduled_demand_preview(self.sid, state, latest))
        self.assertEqual(apply_action.call_args.args[:2], (self.sid, 5000))
        self.assertEqual(apply_action.call_args.kwargs["decision_basis"], "predictive_preemptive")
        self.assertEqual(apply_action.call_args.kwargs["counterfactual_reference"]["session_id"], "matched_no_control")
        self.assertGreater(latest["ues"][0]["qoe"], .85)

    def test_admission_cap_action_is_distinct_from_requested_source_rate(self):
        state = json.loads((self.folder / "session.json").read_text())
        state.update({"mode": "Automatic", "config": {"traffic_mix": "legacy", "admission_control": True},
            "active_parameters": {"lambda_be_pps": 7500, "lambda_be_cap_pps": 20000},
            "latest": {"sim_time_s": 5.0, "ues": []}})
        (self.folder / "session.json").write_text(json.dumps(state))
        live.request_rate_change(self.sid, 5000, approved=True,
            decision_basis="predictive_preemptive",
            counterfactual_reference={"session_id": "matched_no_control"})
        saved = json.loads((self.folder / "session.json").read_text())
        self.assertEqual(saved["pending_control_action"], "set_best_effort_admission_cap")
        self.assertEqual(saved["pending_action_kind"], "admission_cap")
        self.assertEqual((self.folder / "command.txt").read_text(), "5000")

    def test_restart_epoch_request_validates_config_and_persists_transition(self):
        state = json.loads((self.folder / "session.json").read_text())
        state.update({"segments": [{"segment": 1}], "config": {"ue_count": 2, "gnb_count": 1,
            "ue_association_policy": "closest", "bandwidth_mhz": 50.0, "total_tx_power_dbm": 35.0,
            "scenario_size_m": 30, "lambda_be": 10000, "lambda_ull": 2500},
            "active_parameters": {"ue_count": 2, "gnb_count": 1, "ue_association_policy": "closest",
                "bandwidth_mhz": 50.0, "total_tx_power_dbm": 35.0, "lambda_be_pps": 10000,
                "lambda_ull_pps": 2500}})
        (self.folder / "session.json").write_text(json.dumps(state))
        with self.assertRaises(ValueError):
            live.request_structural_change(self.sid, {"gnb_count": 5}, approved=True)
        live.request_structural_change(self.sid, {"gnb_count": 2}, approved=True,
                                       rationale="Test restart epoch")
        request = json.loads((self.folder / "structural_request.json").read_text())
        self.assertEqual(request["config"]["gnb_count"], 2)
        saved = json.loads((self.folder / "session.json").read_text())
        self.assertTrue(saved["structural_action_pending"])
        self.assertEqual(saved["active_parameters"]["gnb_count"], 1)

    def test_startup_rejects_non_closest_association_with_single_gnb_before_launch(self):
        binary = self.root / "simulator"
        binary.write_text("placeholder")
        binary.chmod(0o755)
        with patch.object(live, "LIVE_BINARY", binary), patch.object(live, "list_sessions", return_value=[]):
            with self.assertRaisesRegex(ValueError, "round_robin association requires at least two gNBs"):
                live.start_session({"ue_count": 10, "gnb_count": 1,
                                    "ue_association_policy": "round_robin"})
        self.assertEqual([p.name for p in self.root.glob("live_*/")], ["live_test"])

    def test_structural_change_rejects_non_closest_single_gnb_configuration(self):
        state = json.loads((self.folder / "session.json").read_text())
        state.update({"segments": [{"segment": 1}], "config": {"ue_count": 2, "gnb_count": 1,
            "ue_association_policy": "closest", "bandwidth_mhz": 50.0, "total_tx_power_dbm": 35.0,
            "scenario_size_m": 30, "lambda_be": 10000, "lambda_ull": 2500},
            "active_parameters": {"ue_count": 2, "gnb_count": 1, "ue_association_policy": "closest",
                "bandwidth_mhz": 50.0, "total_tx_power_dbm": 35.0, "lambda_be_pps": 10000,
                "lambda_ull_pps": 2500}})
        (self.folder / "session.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "round_robin association requires at least two gNBs"):
            live.request_structural_change(self.sid,
                {"ue_association_policy": "round_robin"}, approved=True)

    def test_controller_holds_configuration_when_target_is_attained_and_forecast_is_stable(self):
        meta = {"target_qoe": .8, "active_parameters": {"lambda_be_pps": 10000}}
        latest = {"qoe": .82, "delay_ms": 10, "jitter_ms": 1, "packet_loss_fraction": .01}
        forecast = {"risk": False, "qoe": .81, "throughput_mbps": 40,
                    "delay_ms": 10, "jitter_ms": 1, "packet_loss_fraction": .01}
        self.assertIsNone(live.recommend_control_action(meta, latest, forecast))

    def test_healthy_controller_holds_for_one_forecast_crossing_but_acts_for_multiple(self):
        latest_ues = [{"ue_id": 1, "qoe": .84}, {"ue_id": 2, "qoe": .84}]
        def history(ue_1_values, ue_2_values):
            return [{"sim_time_s": index, "qoe": .9, "throughput_mbps": 30,
                     "delay_ms": 70, "jitter_ms": 2, "packet_loss_fraction": .01,
                     "ues": [{"ue_id": 1, "qoe": ue_1_values[index], "throughput_mbps": 10,
                              "delay_ms": 70, "jitter_ms": 2, "packet_loss_fraction": .01},
                             {"ue_id": 2, "qoe": ue_2_values[index], "throughput_mbps": 10,
                              "delay_ms": 70, "jitter_ms": 2, "packet_loss_fraction": .01}]} for index in range(4)]
        meta = {"target_qoe": .8, "config": {"ue_count": 2},
                "active_parameters": {"lambda_be_pps": 10000}}
        one_crossing = history([.90, .87, .84, .81], [.84, .84, .84, .84])
        one_forecast = live.forecast_ue_states(one_crossing, target_qoe=.8)
        state = live.assess_ue_target_state(latest_ues, one_forecast, .8, expected_ue_count=2)
        self.assertEqual(state["forecast_target_crossing_ue_ids"], [1])
        latest = {"qoe": .84, "ues": latest_ues}
        meta["recent_points"] = one_crossing
        proposal = live.recommend_control_action(meta, latest, {"available": True, "risk": True})
        self.assertEqual(proposal["kind"], "none")
        self.assertIn("at least two UEs", proposal["reason"])

        both_cross = history([.90, .87, .84, .81], [.90, .87, .84, .81])
        both_forecast = live.forecast_ue_states(both_cross, target_qoe=.8)
        both_forecast[1]["forecast"].update({"available": True, "qoe": .78, "risk": True})
        both_forecast[2]["forecast"].update({"available": True, "qoe": .77, "risk": True})
        meta["recent_points"] = both_cross
        proposal = live.recommend_control_action(meta, {"qoe": .84, "ues": latest_ues},
                                                  {"available": True, "risk": True})
        self.assertNotEqual(proposal["kind"], "none")

    def test_per_ue_forecast_and_action_prioritize_user_below_target_even_when_network_is_above(self):
        points = []
        for t in range(5):
            weak_qoe = .66 + .005 * t
            strong_qoe = .91
            points.append({"sim_time_s": t, "qoe": .90, "throughput_mbps": 80,
                "delay_ms": 8, "jitter_ms": 1, "packet_loss_fraction": .01,
                "ues": [
                    {"ue_id": 1, "qoe": weak_qoe, "throughput_mbps": 20,
                     "delay_ms": 70 + t, "jitter_ms": 2, "packet_loss_fraction": .01},
                    {"ue_id": 2, "qoe": strong_qoe, "throughput_mbps": 40,
                     "delay_ms": 8, "jitter_ms": 1, "packet_loss_fraction": .01},
                ]})
        forecasts = live.forecast_ue_states(points, target_qoe=.8)
        self.assertTrue(forecasts[1]["forecast"]["risk"])
        self.assertFalse(forecasts[2]["forecast"]["risk"])
        meta = {"target_qoe": .8, "active_parameters": {"lambda_be_pps": 10000},
                "recent_points": points}
        aggregate_forecast = live.forecast(points, target_qoe=.8)
        proposal = live.recommend_control_action(meta, points[-1], aggregate_forecast)
        self.assertIsNotNone(proposal)
        self.assertIn(1, proposal["affected_ue_ids"])

    def test_automatic_controller_increases_load_when_throughput_is_limiting_qoe(self):
        state = {"meta": {"session_id": self.sid, "status": "Running", "mode": "Automatic",
                           "active_parameters": {"lambda_be_pps": 10000}, "last_control_at": 0,
                           "last_decision_sim_time_s": 0, "action_pending": False,
                           "target_qoe": .85, "current_segment": 1, "segments": [{"segment": 1}]},
                 "points": [{"segment_id": "segment_001", "sim_time_s": i, "qoe": .9-.02*i, "delay_ms": 10,
                             "packet_loss_fraction": .01, "throughput_mbps": 30,
                             "jitter_ms": 2} for i in range(5)]}
        with patch.object(live, "read_session", return_value=state), \
             patch.object(live, "request_rate_change") as apply_action:
            live.control_cycle(self.sid)
        apply_action.assert_called_once()
        self.assertEqual(apply_action.call_args.args[1], 12500)
        events = [json.loads(row) for row in (self.folder / "events.jsonl").read_text().splitlines()]
        self.assertIn("Increase best-effort offered load", events[-1]["action"])

    def test_automatic_controller_reduces_load_for_congestion_when_target_is_threatened(self):
        state = {"meta": {"session_id": self.sid, "status": "Running", "mode": "Automatic",
                           "active_parameters": {"lambda_be_pps": 10000}, "last_control_at": 0,
                           "last_decision_sim_time_s": 0, "action_pending": False,
                           "target_qoe": .85, "current_segment": 1, "segments": [{"segment": 1}]},
                 "points": [{"segment_id": "segment_001", "sim_time_s": i, "qoe": .9-.02*i,
                             "delay_ms": 20 + 10*i, "packet_loss_fraction": .01,
                             "throughput_mbps": 30, "jitter_ms": 2} for i in range(5)]}
        with patch.object(live, "read_session", return_value=state), \
             patch.object(live, "request_rate_change") as apply_action:
            live.control_cycle(self.sid)
        apply_action.assert_called_once()
        self.assertEqual(apply_action.call_args.args[1], 5000)

    def test_automatic_controller_acts_on_forecast_before_any_ue_crosses_target(self):
        points = []
        qoe_values = (.90, .87, .84, .81)
        for index, qoe in enumerate(qoe_values, 1):
            points.append({"segment_id": "segment_001", "sim_time_s": float(index),
                "qoe": qoe, "throughput_mbps": 30, "delay_ms": 60 + index,
                "jitter_ms": 2, "packet_loss_fraction": .01, "lambda_be_pps": 10000,
                "ues": [
                    {"ue_id": ue_id, "qoe": qoe, "throughput_mbps": 10,
                     "delay_ms": 60 + index, "jitter_ms": 2,
                     "packet_loss_fraction": .01}
                    for ue_id in (1, 2)
                ]})
        meta = {"session_id": self.sid, "status": "Running", "mode": "Automatic",
            "active_parameters": {"lambda_be_pps": 10000}, "last_control_at": 0,
            "last_decision_sim_time_s": 0, "action_pending": False,
            "target_qoe": .8, "current_segment": 1,
            "config": {"ue_count": 2, "traffic_mix": "legacy"},
            "segments": [{"segment": 1}]}
        state = {"meta": meta, "points": points}
        (self.folder / "session.json").write_text(json.dumps(meta))
        with patch.object(live, "read_session", return_value=state), \
             patch.object(live, "request_rate_change") as apply_action:
            live.control_cycle(self.sid)
        self.assertEqual(points[-1]["qoe"], .81)
        self.assertGreater(points[-1]["qoe"], meta["target_qoe"])
        apply_action.assert_called_once()
        self.assertEqual(apply_action.call_args.kwargs["decision_basis"], "predictive_preemptive")
        decision = json.loads((self.folder / "events.jsonl").read_text().splitlines()[0])
        self.assertEqual(decision["decision_basis"], "predictive_preemptive")
        self.assertGreater(decision["measured_qoe_at_decision"], decision["target_qoe"])

    def test_automatic_controller_uses_restart_epoch_when_rate_is_maxed(self):
        meta = {"session_id": self.sid, "status": "Running", "mode": "Automatic",
            "active_parameters": {"lambda_be_pps": live.MAX_RATE, "gnb_count": 1,
                "bandwidth_mhz": 50.0, "ue_association_policy": "closest"},
            "config": {"ue_count": 2, "gnb_count": 1, "bandwidth_mhz": 50.0,
                "ue_association_policy": "closest", "total_tx_power_dbm": 35.0,
                "scenario_size_m": 30, "lambda_be": live.MAX_RATE, "lambda_ull": 500},
            "segments": [{"segment": 1, "config": {"ue_count": 2, "gnb_count": 1,
                "bandwidth_mhz": 50.0, "ue_association_policy": "closest"}}],
            "target_qoe": .95, "current_segment": 1, "last_control_at": 0,
            "last_structural_control_at": 0, "structural_actions": 0,
            "last_decision_sim_time_s": 0, "action_pending": False}
        points = [{"segment_id": "segment_001", "sim_time_s": i, "qoe": .80,
                   "throughput_mbps": 60, "delay_ms": 10, "jitter_ms": 1,
                   "packet_loss_fraction": .01} for i in range(5)]
        state = {"meta": meta, "points": points}
        with patch.object(live, "read_session", return_value=state), \
             patch.object(live, "request_rate_change") as rate_action, \
             patch.object(live, "request_structural_change") as restart_action:
            live.control_cycle(self.sid)
        rate_action.assert_not_called()
        restart_action.assert_called_once()
        self.assertEqual(restart_action.call_args.args[1]["gnb_count"], 2)
        self.assertTrue(restart_action.call_args.kwargs["approved"])

    def test_action_acknowledgement_and_measurement_are_persisted_once(self):
        state = json.loads((self.folder / "session.json").read_text())
        state.update({"action_pending": True, "pending_new_pps": 5000,
                      "pending_controls_seen": 0, "controls_seen": 0, "action_verified": False,
                      "pending_action_id": "test-action", "config": {"ue_count": 0}})
        (self.folder / "session.json").write_text(json.dumps(state))
        (self.folder / "controls.jsonl").write_text(json.dumps({"sim_time_s": 1, "value": 5000}) + "\n")
        rows = []
        for sim_time, qoe in ((1, .8), (2, .81), (3, .82)):
            rows.append({"sim_time_s": sim_time, "qoe": qoe, "throughput_mbps": 20,
                         "delay_ms": 10, "jitter_ms": 1, "packet_loss_fraction": .01,
                         "lambda_be_pps": 5000})
        (self.folder / "telemetry.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        live.refresh_action_status(self.sid)
        live.refresh_action_status(self.sid)
        events = [json.loads(row) for row in (self.folder / "events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(e["type"] == "action_applied" for e in events), 1)
        self.assertEqual(sum(e["type"] == "post_action_measurement" for e in events), 1)

    def _prepare_test_shock_session(self, mode="Automatic", qoes=(.86, .88, .87)):
        segment = self.folder / "segments" / "segment_001"
        segment.mkdir(parents=True, exist_ok=True)
        meta = json.loads((self.folder / "session.json").read_text())
        meta.update({"status": "Running", "mode": mode, "target_qoe": .8,
            "current_segment": 1, "config": {"ue_count": 2, "gnb_count": 1,
                "traffic_mix": "legacy", "lambda_be": 2500},
            "segments": [{"segment": 1, "status": "Running"}],
            "active_parameters": {"ue_count": 2, "lambda_be_pps": 2500,
                "lambda_ull_pps": 2500}, "controller_rate_actions": 0,
            "last_control_at": 0, "action_pending": False})
        (self.folder / "session.json").write_text(json.dumps(meta))
        points = []
        for index, score in enumerate(qoes):
            points.append({"sim_time_s": index + 1, "qoe": .87, "throughput_mbps": 30,
                "delay_ms": 10, "jitter_ms": 1, "packet_loss_fraction": .01,
                "lambda_be_pps": 2500, "ues": [
                    {"ue_id": 1, "qoe": score, "throughput_mbps": 2, "delay_ms": 10,
                     "jitter_ms": 1, "packet_loss_fraction": .01},
                    {"ue_id": 2, "qoe": .86, "throughput_mbps": 1, "delay_ms": 10,
                     "jitter_ms": 1, "packet_loss_fraction": .01}]})
        (segment / "telemetry.jsonl").write_text("".join(json.dumps(p) + "\n" for p in points))
        (segment / "controls.jsonl").write_text("")
        return segment

    def test_test_load_shock_requires_three_complete_above_target_windows(self):
        self._prepare_test_shock_session(qoes=(.81, .81, .79))
        status = live.test_load_shock_eligibility(self.sid)
        self.assertFalse(status["allowed"])
        self.assertIn("Every UE must meet", status["reason"])
        with self.assertRaises(PermissionError):
            live.request_test_load_shock(self.sid)

        self._prepare_test_shock_session(qoes=(.81, .82, .83))
        status = live.test_load_shock_eligibility(self.sid)
        self.assertTrue(status["allowed"])
        self._prepare_test_shock_session(mode="Recommend", qoes=(.81, .82, .83))
        status = live.test_load_shock_eligibility(self.sid)
        self.assertFalse(status["allowed"])
        self.assertIn("Automatic mode", status["reason"])

        self._prepare_test_shock_session(qoes=(.81, .82, .83))
        status = live.test_load_shock_eligibility(self.sid)
        self.assertTrue(status["allowed"])
        action_id = live.request_test_load_shock(self.sid)
        self.assertEqual((self.folder / "segments/segment_001/command.txt").read_text(), "20000")
        meta = json.loads((self.folder / "session.json").read_text())
        self.assertTrue(meta["test_impairment_pending"])
        self.assertIsNone(meta.get("action_applied_sim_time_s"))
        (self.folder / "segments/segment_001/controls.jsonl").write_text(
            json.dumps({"sim_time_s": 4, "value": 20000}) + "\n")
        live.refresh_action_status(self.sid)
        meta = json.loads((self.folder / "session.json").read_text())
        self.assertFalse(meta["test_impairment_pending"])
        self.assertEqual(meta["active_parameters"]["lambda_be_pps"], 20000)
        events = [json.loads(line) for line in (self.folder / "events.jsonl").read_text().splitlines()]
        applied = next(event for event in events if event["type"] == "test_impairment_applied")
        self.assertEqual(applied["action_id"], action_id)

    def test_new_rate_action_ignores_old_ack_and_verifies_its_own_post_windows(self):
        segment = self._prepare_test_shock_session(qoes=(.99, .99, .99))
        meta = json.loads((self.folder / "session.json").read_text())
        meta["active_parameters"]["lambda_be_pps"] = 20000
        meta.update({"action_verified": True, "action_applied_sim_time_s": 4.0,
                     "action_previous_pps": 2500, "controls_seen": 1})
        (self.folder / "session.json").write_text(json.dumps(meta))
        controls = segment / "controls.jsonl"
        controls.write_text(json.dumps({"sim_time_s": 4.0, "value": 20000,
                                        "action": "set_best_effort_packets_per_second"}) + "\n")
        action_id = live.request_rate_change(self.sid, 10000, approved=True,
                                              rationale="Test next action correlation")
        meta = json.loads((self.folder / "session.json").read_text())
        self.assertIsNone(meta["action_applied_sim_time_s"])
        self.assertFalse(meta["action_verified"])
        self.assertEqual(meta["pending_controls_seen"], 1)

        controls.write_text(controls.read_text() + json.dumps({"sim_time_s": 5.0, "value": 10000,
            "action": "set_best_effort_packets_per_second"}) + "\n")
        live.refresh_action_status(self.sid)
        meta = json.loads((self.folder / "session.json").read_text())
        self.assertEqual(meta["action_applied_sim_time_s"], 5.0)
        self.assertFalse(meta["action_verified"], "old windows must not verify the new action")

        telemetry = segment / "telemetry.jsonl"
        rows = [json.loads(line) for line in telemetry.read_text().splitlines()]
        for sim_time, voice_qoe in ((6.0, .68), (7.0, .69)):
            rows.append({"sim_time_s": sim_time, "qoe": .835, "throughput_mbps": 30,
                "delay_ms": 30, "jitter_ms": 1, "packet_loss_fraction": .1,
                "lambda_be_pps": 10000, "ues": [
                    {"ue_id": 1, "qoe": .99, "throughput_mbps": 2, "delay_ms": 10,
                     "jitter_ms": 1, "packet_loss_fraction": .01},
                    {"ue_id": 2, "qoe": voice_qoe, "throughput_mbps": 1, "delay_ms": 60,
                     "jitter_ms": 1, "packet_loss_fraction": .2}]})
        telemetry.write_text("".join(json.dumps(row) + "\n" for row in rows))
        live.refresh_action_status(self.sid)
        live.refresh_action_status(self.sid)
        meta = json.loads((self.folder / "session.json").read_text())
        self.assertTrue(meta["action_verified"])
        events = [json.loads(line) for line in (self.folder / "events.jsonl").read_text().splitlines()]
        outcomes = [event for event in events if event["type"] == "post_action_measurement"]
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["action_id"], action_id)
        self.assertFalse(outcomes[0]["target_reached"])
        self.assertTrue(outcomes[0]["network_mean_target_reached"])


if __name__ == "__main__":
    unittest.main()
