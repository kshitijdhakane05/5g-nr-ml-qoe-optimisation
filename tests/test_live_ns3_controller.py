from __future__ import annotations

import csv
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from dashboard.live_ns3_control import (
    activate_session, build_qoe_figure, candidate_and_outcome_summary, prepare_session_review_state,
    legacy_candidate_scope_note, recovery_session_parameters, recovery_summary,
    session_status_explanation, session_status_label, _history_table,
    session_run_counts,
)
from simulation.live_control import (
    _run_epoch,
    _mark_worker_failure,
    mark_worker_lost,
    build_epoch_command,
    classify_action_outcome,
    assess_candidate_risk,
    configuration_fingerprint,
    controller_phase,
    target_range_status,
    recommended_timeout_seconds,
    apply_target_attainment_record,
    stability_epoch_passes,
    empirical_qoe_prediction,
    PREDICTION_FEATURES,
    decide_next_epoch,
    decide_post_epoch_exploration,
    run_slot_category,
    load_verified_configurations,
    load_candidate_performance,
    load_pilot_candidate_evidence,
    known_all_outage_evidence,
    production_feature_contract_audit,
    verified_initial_configs,
    supported_radio_grid,
    supported_topology_candidates,
    validate_custom_config,
)


class LiveNs3ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = load_verified_configurations()
        cls.ue2 = verified_initial_configs(2)
        cls.ue6 = verified_initial_configs(6)
        cls.initial = next(c for c in cls.ue2 if c["config_id"] == "L9-01")

    def _history(self, qoe=0.4, config=None, action=None):
        return [{
            "epoch": 1, "config": config or self.initial,
            "measured_qoe_proxy": qoe, "controller_action": action,
        }]

    def _make_post_epoch_session(self, folder: Path):
        from simulation.live_control import HISTORY_COLUMNS
        folder.mkdir()
        (folder / "epochs").mkdir()
        initial = next(c for c in self.ue2 if c["config_id"] == "L9-06")
        state = {
            "session_id": folder.name, "session_dir": str(folder), "status": "QUEUED",
            "initial_config": initial, "current_config": initial, "target_qoe": 0.9,
            "max_epochs": 1, "controller_mode": "adaptive", "max_actions": 3,
            "allow_exploratory_candidates": False,
            "run_remaining_experimental_candidates": True,
            "max_additional_exploratory_epochs": 1,
            "no_improvement_tolerance": 0.005, "rng_seed": 20261027,
            "rng_run_start": 1, "timeout_seconds": 30,
            "scenario_id_base": 12345, "binary": "/mock/ns3",
            "validated_candidate_scope": "campaign workload/traffic semantics match",
            "preset_name": "test", "post_epoch_exploration_count": 0,
        }
        (folder / "session.json").write_text(json.dumps(state))
        (folder / "session_config.json").write_text(json.dumps({
            "feasible_configurations": self.ue2,
            "exploratory_configurations": supported_radio_grid(initial, self.ue2),
            "candidate_performance": [{"key": list(k), **v} for k, v in load_candidate_performance().items()],
            "pilot_candidate_performance": [], "live_candidate_performance": [],
            "run_remaining_experimental_candidates": True,
            "max_additional_exploratory_epochs": 1,
        }))
        with (folder / "epoch_history.csv").open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS).writeheader()
        return initial

    def test_recovery_summary_uses_best_valid_measurement_and_session_tolerance(self):
        with tempfile.TemporaryDirectory() as temp:
            config = dict(self.initial)
            events = [
                {"epoch": 1, "run_status": "COMPLETED", "measured_qoe_proxy": 0.72,
                 "config": config, "metrics": {"is_outage": False}},
                {"epoch": 2, "run_status": "FAILED", "measured_qoe_proxy": None,
                 "config": self.ue2[1]},
                {"epoch": 3, "run_status": "COMPLETED", "measured_qoe_proxy": 0.78,
                 "config": self.ue2[2], "metrics": {"is_outage": False}},
            ]
            state = {"session_dir": temp, "target_qoe": 0.8, "target_tolerance": 0.01,
                     "max_epochs": 3, "status": "EPOCH_LIMIT_REACHED",
                     "final_outcome": "Normal run budget reached."}
            summary = recovery_summary(events, state)
            self.assertAlmostEqual(summary["latest_qoe"], 0.78)
            self.assertAlmostEqual(summary["best_qoe"], 0.78)
            self.assertAlmostEqual(summary["shortfall"], 0.01)
            self.assertFalse(summary["target_met"])
            self.assertEqual(summary["epochs_used"], 3)
            self.assertEqual(summary["stop_reason"], "Normal run budget reached.")

    def test_recovery_session_is_new_bounded_and_preserves_source_context(self):
        summary = {
            "target_qoe": 0.8, "target_tolerance": 0.01, "best_qoe": 0.74,
            "best_config": dict(self.ue2[1]), "original_config": dict(self.initial),
            "stop_reason": "Normal run budget reached.",
        }
        parent_state = {"rng_seed": 9, "rng_run_start": 2, "timeout_seconds": 90,
                        "max_actions": 2, "stable_epochs_required": 2}
        events = [{"rng_run": 4}]
        params = recovery_session_parameters(
            "parent-session", summary, parent_state, events,
            run_budget=2, start_from_best=True, allow_exploratory=False,
        )
        self.assertEqual(params["mode"], "adaptive")
        self.assertEqual(params["max_epochs"], 2)
        self.assertEqual(params["initial_config"], self.ue2[1])
        self.assertEqual(params["rng_run_start"], 5)
        self.assertFalse(params["allow_exploratory_candidates"])
        self.assertFalse(params["run_remaining_experimental_candidates"])
        self.assertEqual(params["recovery_context"]["parent_session_id"], "parent-session")
        with self.assertRaises(ValueError):
            recovery_session_parameters(
                "parent-session", summary, parent_state, events,
                run_budget=7, start_from_best=True,
            )

    def test_below_target_chooses_nearest_untried_verified_configuration(self):
        decision = decide_next_epoch(
            self._history(0.4), self.configs, target_qoe=0.8, max_epochs=3,
        )
        self.assertEqual(decision["action"], "CHANGE_CONFIGURATION")
        self.assertEqual(decision["next_config"]["config_id"], "ANCHOR-UE2")
        self.assertIn("Prediction unavailable", decision["rationale"])
        self.assertIn("Measured QoE", decision["rationale"])
        rejected = decision["rejected_alternatives"]
        self.assertTrue(any(item.get("kind") == "no_action" for item in rejected))

    def test_next_action_uses_previous_measured_improvement_and_changes_configuration(self):
        first = {
            "epoch": 1, "config": self.initial, "measured_qoe_proxy": 0.0,
            "controller_action": "CHANGE_CONFIGURATION",
        }
        second_config = next(c for c in self.ue2 if c["config_id"] == "ANCHOR-UE2")
        second = {"epoch": 2, "config": second_config, "measured_qoe_proxy": 0.72}
        decision = decide_next_epoch(
            [first, second], self.configs, target_qoe=0.8, max_epochs=3,
        )
        self.assertEqual(decision["action"], "CHANGE_CONFIGURATION")
        self.assertEqual(decision["next_config"]["config_id"], "L9-06")

    def test_at_or_above_target_makes_no_action(self):
        for qoe in (0.8,):
            decision = decide_next_epoch(self._history(qoe), self.configs, target_qoe=0.8, max_epochs=3)
            self.assertEqual(decision["action"], "CONTINUE_STABILITY")
        outside_range = decide_next_epoch(self._history(0.95), self.configs, target_qoe=0.8, max_epochs=3)
        self.assertNotEqual(outside_range["action"], "CONTINUE_STABILITY")

    def test_no_measurement_and_invalid_values_stop_without_action(self):
        for qoe in (None, float("nan"), 1.1):
            decision = decide_next_epoch(self._history(qoe), self.configs, target_qoe=0.8, max_epochs=3)
            self.assertEqual(decision["action"], "STOP_INVALID_MEASUREMENT")

    def test_maximum_epoch_limit_is_enforced(self):
        history = self._history(0.3)
        history.extend([
            {"epoch": 2, "config": self.ue2[1], "measured_qoe_proxy": 0.4, "controller_action": "CHANGE_CONFIGURATION"},
        ])
        decision = decide_next_epoch(history, self.configs, target_qoe=0.8, max_epochs=2)
        self.assertEqual(decision["action"], "STOP_EPOCH_LIMIT")

    def test_screenshot_regression_rolls_back_to_best_measured_nonoutage_configuration(self):
        anchor6 = next(c for c in self.ue6 if c["config_id"] == "ANCHOR-UE6")
        l9_02 = next(c for c in self.ue6 if c["config_id"] == "L9-02")
        history = [
            {"epoch": 1, "config": anchor6, "measured_qoe_proxy": 0.6664,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 2, "config": l9_02, "measured_qoe_proxy": 0.0,
             "controller_action": None, "is_outage": True},
        ]
        performance = load_candidate_performance()
        decision = decide_next_epoch(history, self.ue6, target_qoe=0.9, max_epochs=3,
                                     candidate_performance=performance)
        self.assertEqual(decision["action"], "CHANGE_CONFIGURATION")
        self.assertEqual(decision["next_config"]["config_id"], "ANCHOR-UE6")
        self.assertTrue(decision["is_recovery"])

    def test_known_outage_configuration_is_not_selected(self):
        anchor6 = next(c for c in self.ue6 if c["config_id"] == "ANCHOR-UE6")
        decision = decide_next_epoch(
            self._history(0.4, anchor6), self.ue6, target_qoe=0.9, max_epochs=3,
            candidate_performance=load_candidate_performance(),
        )
        self.assertNotEqual(decision.get("next_config", {}).get("config_id"), "L9-02")
        self.assertIn("Excluded from automatic selection", decision["rationale"])
        self.assertIn("2 campaign repetitions; all rows outage", decision["rationale"])

    def test_tiny_improvement_is_not_meaningful_but_does_not_claim_outage(self):
        self.assertEqual(classify_action_outcome(0.00000558284), "NO_MEANINGFUL_CHANGE")
        self.assertEqual(classify_action_outcome(-0.6664), "WORSENED")

    def test_screenshot_candidate_exhaustion_reports_validated_scope_precisely(self):
        configs = {c["config_id"]: c for c in self.ue2}
        history = [
            {"epoch": 1, "config": configs["L9-06"], "measured_qoe_proxy": 0.7123,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 2, "config": configs["ANCHOR-UE2"], "measured_qoe_proxy": 0.7875,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 3, "config": configs["L9-08"], "measured_qoe_proxy": 0.78750558284,
             "controller_action": None},
        ]
        decision = decide_next_epoch(
            history, self.ue2, target_qoe=0.9, max_epochs=6,
            candidate_performance=load_candidate_performance(),
        )
        self.assertEqual(decision["action"], "STOP_NO_SAFE_CANDIDATES")
        self.assertIn("3 of 4 conditions attempted", decision["rationale"])
        self.assertIn("1 excluded", decision["rationale"])
        self.assertIn("other supported ns-3 settings", decision["rationale"])

    def test_previously_successful_campaign_configs_remain_eligible(self):
        performance = load_candidate_performance()
        for config_id in ("L9-06", "L9-08", "ANCHOR-UE2"):
            config = next(c for c in self.ue2 if c["config_id"] == config_id)
            self.assertFalse(performance[tuple((config["ue_count"], config["bandwidth_band1_mhz"],
                                                config["total_tx_power_dbm"]))]["all_rows_outage"])

    def test_opt_in_selects_supported_untried_candidate_not_known_all_outage(self):
        configs = {c["config_id"]: c for c in self.ue2}
        history = [
            {"epoch": 1, "config": configs["L9-06"], "measured_qoe_proxy": 0.7123,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 2, "config": configs["ANCHOR-UE2"], "measured_qoe_proxy": 0.7875,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 3, "config": configs["L9-08"], "measured_qoe_proxy": 0.78750558284,
             "controller_action": None},
        ]
        grid = supported_radio_grid(configs["L9-08"], self.ue2)
        pilot = load_pilot_candidate_evidence()
        decision = decide_next_epoch(
            history, self.ue2, target_qoe=0.9, max_epochs=6,
            candidate_performance=load_candidate_performance(), exploratory_configs=grid,
            allow_exploratory_candidates=True, pilot_performance=pilot,
        )
        self.assertEqual(decision["action"], "CHANGE_CONFIGURATION")
        self.assertTrue(decision["next_config"]["config_id"].startswith("EXPLORATORY-") or
                        decision["next_config"]["config_id"] == "L9-01")
        self.assertNotEqual(decision["next_config"]["config_id"], "L9-01")
        self.assertIn("exploratory", decision["rationale"])
        self.assertIn("prior low-load pilot", decision["rationale"])
        self.assertIn("ranking basis: prior lower-load pilot feasibility evidence", decision["rationale"])
        self.assertIn("QoE under this session load is unknown", decision["rationale"])

    def test_exploration_disabled_stops_after_validated_candidates_are_exhausted(self):
        configs = {c["config_id"]: c for c in self.ue2}
        history = [
            {"epoch": 1, "config": configs["L9-06"], "measured_qoe_proxy": 0.70,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 2, "config": configs["ANCHOR-UE2"], "measured_qoe_proxy": 0.71,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 3, "config": configs["L9-08"], "measured_qoe_proxy": 0.72},
        ]
        decision = decide_next_epoch(
            history, self.ue2, target_qoe=0.8, max_epochs=5,
            candidate_performance=load_candidate_performance(),
            exploratory_configs=supported_radio_grid(configs["L9-08"], self.ue2),
            allow_exploratory_candidates=False,
        )
        self.assertEqual(decision["action"], "STOP_NO_SAFE_CANDIDATES")

    def test_exploration_selects_a_unique_supported_untried_candidate(self):
        configs = {c["config_id"]: c for c in self.ue2}
        history = [
            {"epoch": 1, "config": configs["L9-06"], "measured_qoe_proxy": 0.70,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 2, "config": configs["ANCHOR-UE2"], "measured_qoe_proxy": 0.71,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 3, "config": configs["L9-08"], "measured_qoe_proxy": 0.72},
        ]
        grid = supported_radio_grid(configs["L9-08"], self.ue2)
        decision = decide_next_epoch(
            history, self.ue2, target_qoe=0.8, max_epochs=5,
            candidate_performance=load_candidate_performance(), exploratory_configs=grid + [grid[-1]],
            allow_exploratory_candidates=True, pilot_performance=load_pilot_candidate_evidence(),
        )
        self.assertEqual(decision["action"], "CHANGE_CONFIGURATION")
        self.assertTrue(decision["is_exploratory"])
        self.assertNotIn((2, decision["next_config"]["bandwidth_band1_mhz"],
                          decision["next_config"]["total_tx_power_dbm"]),
                         {(2, 20.0, 30.0), (2, 50.0, 35.0), (2, 50.0, 40.0), (2, 100.0, 35.0)})
        self.assertIn("QoE under this session load is unknown", decision["rationale"])

    def test_known_outage_evidence_is_reported_and_excluded_from_exploration(self):
        configs = {c["config_id"]: c for c in self.ue2}
        history = [
            {"epoch": 1, "config": configs["L9-06"], "measured_qoe_proxy": 0.70,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 2, "config": configs["ANCHOR-UE2"], "measured_qoe_proxy": 0.71,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 3, "config": configs["L9-08"], "measured_qoe_proxy": 0.72},
        ]
        key = (2, 20.0, 40.0)
        live_bad = {key: {"all_runs_outage": True, "run_count": 1}}
        self.assertIn("all 1 prior matching-semantics", known_all_outage_evidence(key, live=live_bad))
        decision = decide_next_epoch(
            history, self.ue2, target_qoe=0.8, max_epochs=5,
            candidate_performance=load_candidate_performance(),
            exploratory_configs=supported_radio_grid(configs["L9-08"], self.ue2),
            allow_exploratory_candidates=True, live_performance=live_bad,
        )
        self.assertNotEqual((2, decision.get("next_config", {}).get("bandwidth_band1_mhz"),
                             decision.get("next_config", {}).get("total_tx_power_dbm")), key)

    def test_one_related_outage_is_a_signal_not_a_family_exclusion(self):
        assessment = assess_candidate_risk(
            (2, 20.0, 35.0),
            campaign={(2, 20.0, 30.0): {"outage_repetition_count": 1}},
        )
        self.assertEqual(assessment["level"], "RELATED_OUTAGE_SIGNAL")
        self.assertFalse(assessment["excluded_by_default"])

    def test_repeated_outages_across_multiple_radio_tuples_trigger_family_policy(self):
        assessment = assess_candidate_risk(
            (2, 20.0, 35.0),
            campaign={(2, 20.0, 30.0): {"outage_repetition_count": 2}},
            live={(2, 20.0, 40.0): {"outage_run_count": 1}},
        )
        self.assertTrue(assessment["repeated_band_pattern"])
        self.assertTrue(assessment["excluded_by_default"])
        self.assertEqual(assessment["related_outage_config_count"], 2)

    def test_empirical_predictor_uses_exact_runs_and_marks_unobserved_configs_unavailable(self):
        config = next(c for c in self.ue2 if c["config_id"] == "L9-06")
        prediction = empirical_qoe_prediction(config, target_qoe=0.8,
                                              campaign=load_candidate_performance())
        self.assertTrue(prediction["available"])
        self.assertEqual(prediction["run_count"], 2)
        self.assertEqual(prediction["domain"], "VALIDATED_EXACT_CONDITION")
        self.assertIsNotNone(prediction["lower"])
        self.assertEqual(prediction["outage_fraction"], 0.0)
        unknown = dict(config, bandwidth_band1_mhz=20.0, total_tx_power_dbm=40.0)
        unavailable = empirical_qoe_prediction(unknown, target_qoe=0.8)
        self.assertFalse(unavailable["available"])
        self.assertIsNone(unavailable["qoe"])
        self.assertEqual(unavailable["domain"], "OUT_OF_DOMAIN_OR_UNOBSERVED")
        self.assertIn('"lambda_be_pps":20000', configuration_fingerprint(config))

    def test_empirical_predictor_deduplicates_reused_rng_repetitions(self):
        config = next(c for c in self.ue2 if c["config_id"] == "L9-06")
        key = (2, 50.0, 40.0)
        campaign = {key: {"run_qoe_values": [0.70, 0.72], "run_outage_values": [False, False],
                          "run_keys": [(20261026, 1), (20261026, 2)]}}
        live = {key: {"run_qoe_values": [0.70, 0.90], "run_outage_values": [False, False],
                      "run_keys": [(20261026, 1), (20261026, 3)]}}
        session = [{"run_status": "COMPLETED", "measured_qoe_proxy": 0.90, "config": config,
                    "rng_seed": 20261026, "rng_run": 3}]
        prediction = empirical_qoe_prediction(config, target_qoe=0.8, campaign=campaign,
                                              live=live, session_history=session)
        self.assertEqual(prediction["run_count"], 3)
        self.assertAlmostEqual(prediction["qoe"], (0.70 + 0.72 + 0.90) / 3)

    def test_prediction_features_are_available_before_run_only(self):
        self.assertIn("lambda_be_pps", PREDICTION_FEATURES)
        self.assertIn("packet_size_be_bytes", PREDICTION_FEATURES)
        self.assertNotIn("tx_packets", PREDICTION_FEATURES)
        self.assertNotIn("rx_packets", PREDICTION_FEATURES)
        self.assertNotIn("throughput_mbps", PREDICTION_FEATURES)

    def test_target_seeking_prefers_candidate_with_better_empirical_outcomes(self):
        configs = {c["config_id"]: c for c in self.ue2}
        current = configs["L9-01"]
        history = [{"config": current, "run_status": "COMPLETED", "measured_qoe_proxy": 0.0,
                    "metrics": {"is_outage": True}}]
        decision = decide_next_epoch(history, self.ue2, target_qoe=0.9, max_epochs=4,
                                     candidate_performance=load_candidate_performance())
        self.assertEqual(decision["action"], "CHANGE_CONFIGURATION")
        self.assertEqual(decision["next_config"]["config_id"], "L9-06")
        self.assertAlmostEqual(decision["predicted_qoe"], 0.740881, places=5)
        self.assertEqual(decision["predicted_outage_fraction"], 0.0)

    def test_target_phase_requires_consecutive_validated_measurements(self):
        config = self.ue2[1]
        one = [{"config": config, "run_status": "COMPLETED", "measured_qoe_proxy": 0.80}]
        two = one + [{"config": config, "run_status": "COMPLETED", "measured_qoe_proxy": 0.80}]
        self.assertEqual(controller_phase(one, target_qoe=0.8, stable_epochs_required=2), "TARGET_SEEKING")
        self.assertEqual(controller_phase(two, target_qoe=0.8, stable_epochs_required=2), "TARGET_MAINTENANCE")
        failed_gap = two + [{"config": config, "run_status": "TIMEOUT", "measured_qoe_proxy": None}]
        declined = two + [{"config": config, "run_status": "COMPLETED", "measured_qoe_proxy": 0.70}]
        self.assertEqual(controller_phase(failed_gap, target_qoe=0.8, stable_epochs_required=2), "TARGET_SEEKING")
        self.assertEqual(controller_phase(declined, target_qoe=0.8, stable_epochs_required=2), "TARGET_SEEKING")

    def test_maintenance_retains_current_setting_when_prediction_meets_margin(self):
        config = next(c for c in self.ue2 if c["config_id"] == "L9-06")
        history = [{"epoch": i, "config": config, "run_status": "COMPLETED",
                    "measured_qoe_proxy": 0.78 + (i - 1) * 0.01,
                    "metrics": {"is_outage": False}} for i in (1, 2)]
        decision = decide_next_epoch(history, self.ue2, target_qoe=0.7, max_epochs=4,
                                     candidate_performance=load_candidate_performance(),
                                     target_tolerance=0.1,
                                     maintenance_margin=0.0, maintenance_hysteresis=0.02)
        self.assertEqual(decision["controller_phase"], "TARGET_MAINTENANCE")
        self.assertEqual(decision["action"], "CONTINUE_MAINTENANCE")
        self.assertEqual(decision["next_config"], config)

    def test_maintenance_switches_when_predicted_alternative_has_material_margin(self):
        current = next(c for c in self.ue2 if c["config_id"] == "L9-06")
        alternative = next(c for c in self.ue2 if c["config_id"] == "ANCHOR-UE2")
        history = [{"epoch": i, "config": current, "run_status": "COMPLETED",
                    "measured_qoe_proxy": 0.80, "metrics": {"is_outage": False}}
                   for i in (1, 2)]
        performance = load_candidate_performance()
        performance[(2, 50.0, 35.0)] = {
            "run_qoe_values": [0.98, 0.99], "run_outage_values": [False, False],
            "mean_qoe_proxy": 0.985, "repetition_count": 2, "all_rows_outage": False,
            "outage_repetition_count": 0,
        }
        decision = decide_next_epoch(history, self.ue2, target_qoe=0.8, max_epochs=4,
                                     candidate_performance=performance,
                                     target_tolerance=0.01,
                                     maintenance_margin=0.1, maintenance_hysteresis=0.02)
        self.assertEqual(decision["controller_phase"], "TARGET_MAINTENANCE")
        self.assertEqual(decision["action"], "CHANGE_CONFIGURATION")
        self.assertEqual(decision["next_config"]["config_id"], alternative["config_id"])
        self.assertGreaterEqual(decision["predicted_qoe"], 0.9)

    def test_measured_outage_rolls_back_once_to_best_safe_configuration(self):
        safe = next(c for c in self.ue2 if c["config_id"] == "L9-06")
        outage = dict(safe, config_id="EXPLORATORY-OUTAGE", bandwidth_band1_mhz=20.0,
                      total_tx_power_dbm=40.0)
        history = [
            {"epoch": 1, "config": safe, "run_status": "COMPLETED", "measured_qoe_proxy": 0.76,
             "metrics": {"is_outage": False}},
            {"epoch": 2, "config": outage, "run_status": "COMPLETED", "measured_qoe_proxy": 0.0,
             "metrics": {"is_outage": True}},
        ]
        decision = decide_next_epoch(history, [], target_qoe=0.8, max_epochs=4,
                                     exploratory_configs=[], allow_exploratory_candidates=True)
        self.assertTrue(decision["is_recovery"])
        self.assertEqual(decision["next_config"], safe)
        self.assertIn("Measured outage", decision["rationale"])
        history[-1]["recovery_target_key"] = decision["recovery_target_key"]
        no_retry = decide_next_epoch(history, [], target_qoe=0.8, max_epochs=4,
                                     exploratory_configs=[], allow_exploratory_candidates=True)
        self.assertFalse(no_retry.get("is_recovery", False))

    def test_exact_outage_is_skipped_but_advanced_override_can_select_it(self):
        current = next(c for c in self.ue2 if c["config_id"] == "L9-06")
        history = self._history(0.6, current)
        failed = dict(current, config_id="EXPLORATORY-B120-P30-UE2",
                      bandwidth_band1_mhz=20.0, total_tx_power_dbm=30.0)
        campaign = {(2, 20.0, 30.0): {"outage_repetition_count": 2, "all_rows_outage": True,
                                      "mean_qoe_proxy": 0.0, "repetition_count": 2}}
        excluded = decide_next_epoch(
            history, [], target_qoe=0.9, max_epochs=4, exploratory_configs=[failed],
            allow_exploratory_candidates=True, candidate_performance=campaign,
        )
        self.assertEqual(excluded["action"], "STOP_NO_SAFE_CANDIDATES")
        permitted = decide_next_epoch(
            history, [], target_qoe=0.9, max_epochs=4, exploratory_configs=[failed],
            allow_exploratory_candidates=True, candidate_performance=campaign,
            allow_high_risk_candidates=True,
        )
        self.assertEqual(permitted["action"], "CHANGE_CONFIGURATION")
        self.assertTrue(permitted["high_risk_override"])
        self.assertIn("outage", permitted["candidate_risk_reason"])

    def test_risk_filter_prefers_safe_candidate_and_logs_skipped_risk(self):
        current = next(c for c in self.ue2 if c["config_id"] == "ANCHOR-UE2")
        history = self._history(0.6, current)
        grid = [dict(current, config_id="RISKY20", bandwidth_band1_mhz=20.0, total_tx_power_dbm=30.0),
                dict(current, config_id="SAFE100", bandwidth_band1_mhz=100.0, total_tx_power_dbm=30.0)]
        campaign = {(2, 20.0, 30.0): {"outage_repetition_count": 2, "all_rows_outage": True,
                                      "mean_qoe_proxy": 0.0, "repetition_count": 2}}
        decision = decide_next_epoch(
            history, [], target_qoe=0.9, max_epochs=4, exploratory_configs=grid,
            allow_exploratory_candidates=True, candidate_performance=campaign,
        )
        self.assertEqual(decision["next_config"]["config_id"], "SAFE100")
        self.assertEqual(decision["risk_excluded_count"], 1)
        self.assertEqual(decision["candidate_risk_level"], "NO_EXACT_OUTCOME")

    def test_repeat_trials_are_opt_in_and_still_bounded_by_epoch_limit(self):
        current = self.ue2[0]
        history = self._history(0.6, current)
        no_repeat = decide_next_epoch(history, [current], target_qoe=0.9, max_epochs=3)
        repeat = decide_next_epoch(history, [current], target_qoe=0.9, max_epochs=3,
                                   allow_repeat_trials=True)
        limited = decide_next_epoch(history, [current], target_qoe=0.9, max_epochs=1,
                                    allow_repeat_trials=True)
        self.assertNotEqual(no_repeat["action"], "CHANGE_CONFIGURATION")
        self.assertEqual(repeat["action"], "CHANGE_CONFIGURATION")
        self.assertEqual(repeat["next_config"], current)
        self.assertEqual(limited["action"], "STOP_EPOCH_LIMIT")

    def test_no_eligible_exploratory_candidate_reports_outage_exclusion(self):
        configs = {c["config_id"]: c for c in self.ue2}
        grid = supported_radio_grid(configs["L9-08"], self.ue2)
        history = [
            {"epoch": 1, "config": configs["L9-06"], "measured_qoe_proxy": 0.60,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 2, "config": configs["ANCHOR-UE2"], "measured_qoe_proxy": 0.61,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 3, "config": configs["L9-08"], "measured_qoe_proxy": 0.62},
        ]
        campaign = load_candidate_performance()
        live = {key: {"all_runs_outage": True, "run_count": 1}
                for key in {(2, 20.0, 35.0), (2, 20.0, 40.0), (2, 50.0, 30.0),
                            (2, 100.0, 30.0), (2, 100.0, 40.0)}}
        decision = decide_next_epoch(
            history, self.ue2, target_qoe=0.9, max_epochs=6,
            candidate_performance=campaign, exploratory_configs=grid,
            allow_exploratory_candidates=True, live_performance=live,
        )
        self.assertEqual(decision["action"], "STOP_NO_SAFE_CANDIDATES")
        self.assertIn("Conservative risk exclusions", decision["rationale"])

    def test_exploration_rejects_candidate_that_changes_session_invariants(self):
        configs = {c["config_id"]: c for c in self.ue2}
        history = [
            {"epoch": 1, "config": configs["L9-06"], "measured_qoe_proxy": 0.70,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 2, "config": configs["ANCHOR-UE2"], "measured_qoe_proxy": 0.71,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 3, "config": configs["L9-08"], "measured_qoe_proxy": 0.72},
        ]
        invalid = dict(configs["L9-08"], config_id="BAD-B2", bandwidth_band1_mhz=100.0,
                       bandwidth_band2_mhz=100.0, total_tx_power_dbm=30.0)
        decision = decide_next_epoch(
            history, self.ue2, target_qoe=0.9, max_epochs=5,
            candidate_performance=load_candidate_performance(), exploratory_configs=[invalid],
            allow_exploratory_candidates=True,
        )
        self.assertEqual(decision["action"], "STOP_NO_SAFE_CANDIDATES")
        self.assertIn("failed validation", decision["rationale"])

    def test_exploration_respects_epoch_and_action_limits(self):
        configs = {c["config_id"]: c for c in self.ue2}
        history = self._history(0.7, configs["L9-06"], "CHANGE_CONFIGURATION")
        grid = supported_radio_grid(configs["L9-06"], self.ue2)
        at_epoch_limit = decide_next_epoch(
            history, self.ue2, target_qoe=0.9, max_epochs=1, max_actions=1, exploratory_configs=grid,
            allow_exploratory_candidates=True,
        )
        at_action_limit = decide_next_epoch(
            history, self.ue2, target_qoe=0.9, max_epochs=5, max_actions=1,
            exploratory_configs=grid, allow_exploratory_candidates=True,
        )
        self.assertEqual(at_epoch_limit["action"], "STOP_EPOCH_LIMIT")
        self.assertTrue(at_epoch_limit["action_budget_exhausted"])
        self.assertEqual(at_action_limit["action"], "CHANGE_CONFIGURATION")
        self.assertTrue(at_action_limit["is_exploratory"])
        self.assertTrue(at_action_limit["action_budget_exhausted"])

    def test_six_epoch_session_continues_exploration_after_action_budget_at_epoch_four(self):
        configs = {c["config_id"]: c for c in self.ue2}
        history = [
            {"epoch": 1, "config": configs["L9-06"], "measured_qoe_proxy": 0.70,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 2, "config": configs["ANCHOR-UE2"], "measured_qoe_proxy": 0.71,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 3, "config": configs["L9-08"], "measured_qoe_proxy": 0.72,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 4, "config": configs["L9-01"], "measured_qoe_proxy": 0.0,
             "is_outage": True, "controller_action": None},
        ]
        full_grid = supported_radio_grid(configs["L9-08"], self.ue2)
        three_remaining = [c for c in full_grid if
                           (c["bandwidth_band1_mhz"], c["total_tx_power_dbm"])
                           in {(50.0, 30.0), (100.0, 30.0), (100.0, 40.0)}]
        decision = decide_next_epoch(
            history, self.ue2, target_qoe=0.9, max_epochs=6, max_actions=3,
            candidate_performance=load_candidate_performance(),
            exploratory_configs=three_remaining, allow_exploratory_candidates=True,
            pilot_performance=load_pilot_candidate_evidence(),
        )
        self.assertEqual(len(history), 4)
        self.assertEqual(len(three_remaining), 3)
        self.assertEqual(decision["action"], "CHANGE_CONFIGURATION")
        self.assertTrue(decision["is_recovery"])
        self.assertTrue(decision["action_budget_exhausted"])
        self.assertIn("best prior non-outage", decision["rationale"])

    def test_post_epoch_exploration_is_opt_in_and_selects_untried_candidate(self):
        configs = {c["config_id"]: c for c in self.ue2}
        history = [{"epoch": 1, "config": configs["L9-06"], "measured_qoe_proxy": 0.72,
                    "controller_action": "STOP_EPOCH_LIMIT"}]
        grid = supported_radio_grid(configs["L9-06"], self.ue2)
        disabled = decide_next_epoch(history, self.ue2, target_qoe=0.85, max_epochs=1)
        enabled = decide_post_epoch_exploration(
            history, self.ue2, grid, target_qoe=0.85, max_additional_epochs=2,
            candidate_performance=load_candidate_performance(),
            pilot_performance=load_pilot_candidate_evidence(),
        )
        self.assertEqual(disabled["action"], "STOP_EPOCH_LIMIT")
        self.assertEqual(enabled["action"], "CONTINUE_POST_EPOCH_EXPLORATION")
        self.assertTrue(enabled["is_exploratory"])
        self.assertEqual(enabled["post_epoch_exploration_count"], 0)
        self.assertIn("Normal epoch limit reached", enabled["continuation_reason"])

    def test_post_epoch_exploration_stops_when_candidates_are_exhausted(self):
        grid = supported_radio_grid(self.ue2[0], self.ue2)
        extra = [c for c in grid if (c["ue_count"], c["bandwidth_band1_mhz"], c["total_tx_power_dbm"])
                 not in {(c["ue_count"], c["bandwidth_band1_mhz"], c["total_tx_power_dbm"])
                         for c in self.ue2}]
        history = [{"epoch": i + 1, "config": c, "measured_qoe_proxy": 0.7,
                    "post_epoch_exploration": True}
                   for i, c in enumerate(extra)]
        decision = decide_post_epoch_exploration(
            history, self.ue2, grid, target_qoe=0.9, max_additional_epochs=9,
            candidate_performance=load_candidate_performance(),
        )
        self.assertEqual(decision["action"], "STOP_NO_EXPERIMENTAL_CANDIDATES")
        self.assertIn("No eligible untried experimental configuration remains", decision["rationale"])

    def test_post_epoch_exploration_has_separate_additional_epoch_limit(self):
        history = [{"epoch": 2, "config": self.ue2[0], "measured_qoe_proxy": 0.7,
                    "post_epoch_exploration": True}]
        decision = decide_post_epoch_exploration(
            history, self.ue2, supported_radio_grid(self.ue2[0], self.ue2),
            target_qoe=0.9, max_additional_epochs=1,
        )
        self.assertEqual(decision["action"], "STOP_POST_EPOCH_LIMIT")
        self.assertEqual(decision["post_epoch_exploration_count"], 1)

    def test_post_epoch_exploration_respects_target_and_fixed_mode_guard(self):
        target_history = [{"epoch": 2, "config": self.ue2[0], "measured_qoe_proxy": 0.9}]
        decision = decide_post_epoch_exploration(
            target_history, self.ue2, supported_radio_grid(self.ue2[0], self.ue2),
            target_qoe=0.9, max_additional_epochs=1,
        )
        self.assertEqual(decision["action"], "CONTINUE_POST_EPOCH_EXPLORATION")
        with self.assertRaisesRegex(ValueError, "adaptive mode"):
            decide_next_epoch(self._history(0.4), self.ue2, target_qoe=0.9,
                              max_epochs=3, mode="fixed", allow_exploratory_candidates=True)

    def test_exhausted_opt_in_grid_with_only_tiny_gain_stops_as_no_meaningful_improvement(self):
        configs = {c["config_id"]: c for c in self.ue2}
        grid = [c for c in supported_radio_grid(configs["L9-08"], self.ue2)
                if c["bandwidth_band1_mhz"] == 20.0 and c["total_tx_power_dbm"] in {35.0, 40.0}]
        history = []
        for epoch, config in enumerate(grid, start=1):
            history.append({"epoch": epoch, "config": config,
                            "measured_qoe_proxy": 0.700 + (epoch - 1) * 0.0001,
                            "controller_action": None})
        decision = decide_next_epoch(
            history, [], target_qoe=0.9, max_epochs=3,
            candidate_performance=load_candidate_performance(), exploratory_configs=grid,
            allow_exploratory_candidates=True, pilot_performance=load_pilot_candidate_evidence(),
        )
        self.assertEqual(decision["action"], "STOP_NO_MEANINGFUL_IMPROVEMENT")
        self.assertIn("within the 0.005000 tolerance", decision["rationale"])

    def test_supported_grid_is_distinct_from_fractional_validated_registry(self):
        grid = supported_radio_grid(self.ue2[0], self.ue2)
        self.assertEqual(len(grid), 9)
        keys = {(c["ue_count"], c["bandwidth_band1_mhz"], c["total_tx_power_dbm"]) for c in grid}
        self.assertEqual(len(keys), 9)
        self.assertTrue(any(c["config_id"].startswith("EXPLORATORY-") for c in grid))

    def test_repeated_candidates_are_never_selected_again(self):
        anchor6 = next(c for c in self.ue6 if c["config_id"] == "ANCHOR-UE6")
        l9_02 = next(c for c in self.ue6 if c["config_id"] == "L9-02")
        l9_09 = next(c for c in self.ue6 if c["config_id"] == "L9-09")
        history = [
            {"epoch": 1, "config": anchor6, "measured_qoe_proxy": 0.66, "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 2, "config": l9_02, "measured_qoe_proxy": 0.0, "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 3, "config": l9_09, "measured_qoe_proxy": 0.65},
        ]
        decision = decide_next_epoch(history, self.ue6, target_qoe=0.9, max_epochs=5,
                                     candidate_performance=load_candidate_performance())
        self.assertEqual(decision["action"], "CHANGE_CONFIGURATION")
        self.assertEqual(decision["next_config"]["config_id"], "L9-04")

    def test_no_alternative_and_action_limit_are_explicit(self):
        current_only = decide_next_epoch(
            self._history(0.4), [self.initial], target_qoe=0.8, max_epochs=3,
        )
        self.assertEqual(current_only["action"], "STOP_NO_VALIDATED_CANDIDATES")
        history = self._history(0.4, action="CHANGE_CONFIGURATION")
        history.extend([
            {"epoch": 2, "config": self.ue2[1], "measured_qoe_proxy": 0.6,
             "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 3, "config": self.ue2[2], "measured_qoe_proxy": 0.7,
             "controller_action": "CHANGE_CONFIGURATION"},
        ])
        decision = decide_next_epoch(history, self.configs, target_qoe=0.8, max_epochs=6, max_actions=2)
        self.assertEqual(decision["action"], "CHANGE_CONFIGURATION")
        self.assertTrue(decision["action_budget_exhausted"])

    def test_all_available_alternatives_exhausted_reports_no_improvement(self):
        history = []
        configs = self.ue2
        for index, config in enumerate(configs):
            history.append({"epoch": index + 1, "config": config, "measured_qoe_proxy": 0.4 + index * 0.01,
                            "controller_action": "CHANGE_CONFIGURATION" if index < len(configs) - 1 else None})
        decision = decide_next_epoch(history, configs, target_qoe=0.9, max_epochs=len(configs) + 2,
                                     max_actions=10)
        self.assertEqual(decision["action"], "STOP_NO_VALIDATED_CANDIDATES")

    def test_fixed_mode_repeats_same_verified_config(self):
        decision = decide_next_epoch(self._history(0.4), self.configs, target_qoe=0.8,
                                     max_epochs=3, mode="fixed")
        self.assertEqual(decision["action"], "CONTINUE_FIXED")
        self.assertEqual(decision["next_config"], self.initial)

    def test_explicit_epoch_command_has_bandwidth_power_traffic_and_rng(self):
        command = build_epoch_command(
            self.initial, binary="/tmp/ns3", run_dir="/tmp/live-run",
            scenario_id=88, rng_run=2,
        )
        for expected in (
            "--bandwidthBand1=20000000", "--bandwidthBand2=50000000", "--totalTxPower=30.0",
            "--ueNumPergNb=2", "--lambdaBe=20000", "--lambdaUll=10000",
            "--ueCount=2", "--gNbNum=1",
            "--packetSizeBe=1252", "--packetSizeUll=100", "--RngSeed=20261026", "--RngRun=2",
        ):
            self.assertIn(expected, command)

    def test_topology_alternative_keeps_ue_population_and_has_no_fabricated_prediction(self):
        base = dict(self.initial, gnb_count=1)
        topologies = supported_topology_candidates(
            base, self.ue2, candidate_budget=1, candidate_performance=load_candidate_performance(),
        )
        self.assertEqual(len(topologies), 1)
        candidate = topologies[0]
        self.assertEqual(candidate["gnb_count"], 2)
        self.assertEqual(candidate["ue_association_policy"], "round_robin")
        self.assertEqual(candidate["ue_count"], base["ue_count"])
        self.assertEqual(candidate["bandwidth_band1_mhz"], base["bandwidth_band1_mhz"])
        self.assertEqual(candidate["total_tx_power_dbm"], base["total_tx_power_dbm"])
        self.assertTrue(candidate["topology_candidate"])
        history = [{"epoch": 1, "config": base, "measured_qoe_proxy": 0.65,
                    "run_status": "COMPLETED", "metrics": {"is_outage": False}}]
        decision = decide_next_epoch(
            history, [], target_qoe=0.9, max_epochs=3, mode="adaptive",
            exploratory_configs=topologies, allow_topology_exploration=True,
        )
        self.assertEqual(decision["action"], "CHANGE_CONFIGURATION")
        self.assertTrue(decision["topology_change"])
        self.assertEqual(decision["next_config"]["gnb_count"], 2)
        self.assertEqual(decision["next_config"]["ue_count"], base["ue_count"])
        self.assertIsNone(decision["predicted_qoe"])
        self.assertIn("requires actual ns-3 evaluation", decision["candidate_evidence"])

    def test_topology_candidate_identity_includes_gnb_count(self):
        one = dict(self.initial, gnb_count=1)
        two = dict(self.initial, gnb_count=2)
        self.assertNotEqual(configuration_fingerprint(one), configuration_fingerprint(two))

    def test_topology_search_generates_distinct_workload_grouping(self):
        base = dict(self.initial, gnb_count=1)
        candidates = supported_topology_candidates(base, self.ue2, candidate_budget=2)
        self.assertEqual([(c["gnb_count"], c["ue_association_policy"]) for c in candidates],
                         [(2, "round_robin"), (2, "by_traffic_class")])
        self.assertEqual({c["ue_count"] for c in candidates}, {base["ue_count"]})
        self.assertEqual(len({configuration_fingerprint(c) for c in candidates}), 2)

    def test_absolute_target_range_and_validation_outage_rules(self):
        self.assertTrue(target_range_status(0.7486, 0.75, 0.03))
        self.assertFalse(target_range_status(0.79, 0.75, 0.03))
        passed, reason = stability_epoch_passes(
            {"run_status": "COMPLETED", "measured_qoe_proxy": 0.7486, "ue_measurements": []},
            target_qoe=0.75, tolerance=0.03,
        )
        self.assertTrue(passed, reason)
        passed, reason = stability_epoch_passes(
            {"run_status": "COMPLETED", "measured_qoe_proxy": 0.7486,
             "ue_measurements": [{"is_outage": True}]},
            target_qoe=0.75, tolerance=0.03,
        )
        self.assertFalse(passed)
        self.assertIn("outage", reason)
        passed, _ = stability_epoch_passes(
            {"run_status": "INVALID_OUTPUT", "measured_qoe_proxy": 0.0},
            target_qoe=0.75, tolerance=0.03,
        )
        self.assertFalse(passed)

    def test_custom_timeout_recommendation_scales_from_measured_packet_workload(self):
        standard = dict(self.initial, ue_count=2, sim_time="1s", lambda_be_pps=10000,
                        lambda_ull_pps=5000, app_start_seconds=0.4)
        self.assertEqual(recommended_timeout_seconds(standard), (120, 9000))
        high_load = dict(self.initial, ue_count=6, sim_time="3s", lambda_be_pps=20000,
                         lambda_ull_pps=10000, app_start_seconds=0.4)
        # This workload matches the successful 115.42-second run used for calibration.
        self.assertEqual(recommended_timeout_seconds(high_load), (145, 234000))
        maximum = dict(self.initial, ue_count=10, sim_time="3s", lambda_be_pps=20000,
                       lambda_ull_pps=10000, app_start_seconds=0.4)
        recommendation, packet_count = recommended_timeout_seconds(maximum)
        self.assertEqual(packet_count, 390000)
        self.assertGreaterEqual(recommendation, 145)
        self.assertLessEqual(recommendation, 300)

    def test_target_entry_persists_explanation_when_run_record_has_no_rationale(self):
        event = {"epoch": 1, "run_status": "COMPLETED", "measured_qoe_proxy": 0.7486}
        config = dict(self.initial, config_id="TARGET-CANDIDATE")
        result = apply_target_attainment_record(
            event, config=config, previous_config=None, target_qoe=0.75,
            tolerance=0.03, validation_epochs=5, epoch=1,
        )
        self.assertEqual(result["controller_action"], "TARGET_REACHED_START_VALIDATION")
        self.assertIn("absolute target range", result["action_rationale"])
        self.assertEqual(result["decision_record"]["reason"], result["action_rationale"])
        self.assertEqual(result["decision_record"]["resulting_config"], config)

    def test_custom_target_hit_run_completes_validation_without_worker_key_error(self):
        from types import SimpleNamespace
        from simulation.live_control import create_live_session, run_live_session

        with tempfile.TemporaryDirectory() as temp:
            with patch("simulation.live_control.verify_ns3_runtime", return_value=(True, "ok")), \
                 patch("simulation.live_control.subprocess.Popen", return_value=SimpleNamespace(pid=1001)):
                _session_id, folder, _worker = create_live_session(
                    initial_config=self.initial, target_qoe=0.70, target_tolerance=0.01,
                    max_epochs=1, mode="fixed", stability_validation_epochs=1,
                    output_root=Path(temp), binary="/tmp/ns3",
                )

            def fake_run(_session, config, *, epoch, validation_epoch=False, **_kwargs):
                return {
                    "epoch": epoch, "run_status": "COMPLETED", "config": config,
                    "measured_qoe_proxy": 0.70,
                    "metrics": {"is_outage": False, "mean_qoe_proxy": 0.70},
                    "duration_seconds": 0.1, "exit_code": 0, "command": [],
                    "ue_measurements_status": "UNAVAILABLE_OLD_BINARY",
                    "is_validation_epoch": validation_epoch,
                }

            with patch("simulation.live_control._run_epoch", side_effect=fake_run):
                run_live_session(folder)

            state = json.loads((folder / "session.json").read_text())
            events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
            self.assertEqual(state["status"], "SUCCESS_STABLE")
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["controller_action"], "TARGET_REACHED_START_VALIDATION")
            self.assertTrue(events[0]["decision_record"]["reason"])
            self.assertTrue(events[1]["validation_pass"])

    def test_controller_phase_requires_same_configuration_and_absolute_range(self):
        a = dict(self.initial, config_id="A")
        b = dict(self.initial, config_id="B", bandwidth_band1_mhz=50)
        history = [
            {"run_status": "COMPLETED", "config": a, "measured_qoe_proxy": 0.75},
            {"run_status": "COMPLETED", "config": b, "measured_qoe_proxy": 0.75},
        ]
        self.assertEqual(controller_phase(history, target_qoe=0.75, target_tolerance=0.03,
                                          stable_epochs_required=2), "TARGET_SEEKING")
        history[-1]["config"] = a
        history[-1]["measured_qoe_proxy"] = 0.79
        self.assertEqual(controller_phase(history, target_qoe=0.75, target_tolerance=0.03,
                                          stable_epochs_required=2), "TARGET_SEEKING")

    def test_association_policy_is_validated_and_part_of_identity(self):
        with self.assertRaisesRegex(ValueError, "requires at least two"):
            validate_custom_config(dict(self.initial, ue_association_policy="round_robin"))
        two_cell = validate_custom_config(dict(self.initial, gnb_count=2,
                                               ue_association_policy="round_robin"))
        self.assertEqual(two_cell["ue_association_policy"], "round_robin")
        self.assertNotEqual(configuration_fingerprint(dict(self.initial, gnb_count=2)),
                            configuration_fingerprint(two_cell))
        command = build_epoch_command(two_cell, binary="/tmp/ns3", run_dir="/tmp/rr",
                                      scenario_id=45, rng_run=1)
        self.assertIn("--ueAssociationPolicy=round_robin", command)

    def test_custom_configuration_reaches_all_supported_cli_inputs(self):
        custom = validate_custom_config({
            "config_id": "CUSTOM", "ue_count": 6, "bandwidth_band1_mhz": 100,
            "bandwidth_band2_mhz": 20, "total_tx_power_dbm": 40, "lambda_be_pps": 15000,
            "lambda_ull_pps": 5000, "packet_size_be_bytes": 1200,
            "packet_size_ull_bytes": 256, "sim_time_seconds": 2,
            "scenario_size_m": 60,
        })
        command = build_epoch_command(custom, binary="/tmp/ns3", run_dir="/tmp/custom",
                                      scenario_id=44, rng_run=7, rng_seed=1234)
        for expected in ("--bandwidthBand1=100000000", "--bandwidthBand2=20000000",
                         "--totalTxPower=40.0", "--ueNumPergNb=6", "--lambdaBe=15000",
                         "--ueCount=6", "--gNbNum=1",
                         "--lambdaUll=5000", "--packetSizeBe=1200", "--packetSizeUll=256",
                         "--simTime=2s", "--scenarioSize=60.0", "--RngSeed=1234", "--RngRun=7"):
            self.assertIn(expected, command)

    def test_custom_configuration_rejects_unsupported_ranges(self):
        invalid = dict(self.initial, bandwidth_band1_mhz=40)
        with self.assertRaisesRegex(ValueError, "20, 50 or 100"):
            validate_custom_config(invalid)

    def test_starting_session_b_switches_active_selection_without_losing_a(self):
        state = {"live_active_session_id": "session-A", "live_session_review": "session-A"}
        activate_session(state, "session-B", "/tmp/session-B")
        self.assertEqual(state["live_active_session_id"], "session-B")
        self.assertEqual(state["live_session_review"], "session-A")
        self.assertEqual(state["live_pending_session_review_id"], "session-B")
        self.assertEqual(state["live_session_dir"], "/tmp/session-B")
        prepare_session_review_state(state, ["session-A", "session-B"])
        self.assertEqual(state["live_session_review"], "session-B")
        self.assertNotIn("live_pending_session_review_id", state)
        # Reviewing the older session changes only the widget selection.
        state["live_session_review"] = "session-A"
        prepare_session_review_state(state, ["session-A", "session-B"])
        self.assertEqual(state["live_session_review"], "session-A")
        self.assertEqual(state["live_active_session_id"], "session-B")
        # A later new session again wins the review selector on the next rerun.
        activate_session(state, "session-C", "/tmp/session-C")
        prepare_session_review_state(state, ["session-A", "session-B", "session-C"])
        prepare_session_review_state(state, ["session-A", "session-B", "session-C"])
        self.assertEqual(state["live_session_review"], "session-C")
        self.assertEqual(state["live_active_session_id"], "session-C")

    def test_run_slot_limits_have_exact_normal_and_extra_boundaries(self):
        self.assertIsNone(run_slot_category(1, normal_limit=0, extra_enabled=False, extra_limit=0))
        self.assertEqual(run_slot_category(1, normal_limit=1, extra_enabled=False, extra_limit=2), "normal")
        self.assertIsNone(run_slot_category(2, normal_limit=1, extra_enabled=False, extra_limit=2))
        self.assertEqual(run_slot_category(2, normal_limit=1, extra_enabled=True, extra_limit=1), "experimental")
        self.assertIsNone(run_slot_category(3, normal_limit=1, extra_enabled=True, extra_limit=1))

    def test_run_counts_use_unique_attempt_records_and_do_not_call_rejection_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "epochs" / "normal_1").mkdir(parents=True)
            (folder / "epochs" / "extra_2").mkdir(parents=True)
            (folder / "epochs" / "normal_1" / "run_config.json").write_text(json.dumps({
                "run_id": "normal-1", "epoch": 1, "run_status": "COMPLETED",
            }))
            (folder / "epochs" / "extra_2" / "run_config.json").write_text(json.dumps({
                "run_id": "extra-2", "epoch": 2, "run_status": "STARTING",
                "post_epoch_exploration": True,
            }))
            (folder / "epochs" / "validation_3" ).mkdir()
            (folder / "epochs" / "validation_3" / "run_config.json").write_text(json.dumps({
                "run_id": "validation-3", "epoch": 3, "run_status": "COMPLETED",
                "is_validation_epoch": True,
            }))
            events = [{"run_id": "normal-1", "epoch": 1, "run_status": "COMPLETED"}]
            counts = session_run_counts(folder, events, {
                "max_epochs": 1, "max_additional_exploratory_epochs": 1,
                "run_remaining_experimental_candidates": True,
            })
            self.assertEqual(counts["normal_attempted"], 1)
            self.assertEqual(counts["normal_completed"], 1)
            self.assertEqual(counts["experimental_attempted"], 1)
            self.assertEqual(counts["experimental_completed"], 0)
            self.assertEqual(counts["validation_attempted"], 1)
            self.assertEqual(counts["validation_completed"], 1)

    def test_new_sessions_have_unique_directories_and_independent_history(self):
        from types import SimpleNamespace
        from simulation.live_control import create_live_session, list_live_sessions
        with tempfile.TemporaryDirectory() as temp:
            with patch("simulation.live_control.verify_ns3_runtime", return_value=(True, "ok")), \
                 patch("simulation.live_control.subprocess.Popen", return_value=SimpleNamespace(pid=1234)):
                id_a, dir_a, _ = create_live_session(
                    initial_config=self.initial, target_qoe=0.8, max_epochs=2, mode="fixed",
                    output_root=Path(temp), binary="/tmp/ns3",
                )
                id_b, dir_b, _ = create_live_session(
                    initial_config=self.initial, target_qoe=0.7, max_epochs=1, mode="adaptive",
                    output_root=Path(temp), binary="/tmp/ns3",
                )
            self.assertNotEqual(id_a, id_b)
            self.assertNotEqual(dir_a, dir_b)
            self.assertEqual({p.name for p in list_live_sessions(temp)}, {id_a, id_b})
            state_a = json.loads((dir_a / "session.json").read_text())
            state_b = json.loads((dir_b / "session.json").read_text())
            self.assertEqual(state_a["target_qoe"], 0.8)
            self.assertEqual(state_b["target_qoe"], 0.7)
            self.assertFalse(state_a["run_remaining_experimental_candidates"])
            self.assertFalse(state_a["allow_high_risk_candidates"])
            self.assertFalse(state_a["allow_repeat_trials"])
            config_a = json.loads((dir_a / "session_config.json").read_text())
            self.assertFalse(config_a["run_remaining_experimental_candidates"])
            self.assertFalse(config_a["allow_high_risk_candidates"])
            self.assertFalse(config_a["allow_repeat_trials"])
            self.assertEqual(config_a["max_additional_exploratory_epochs"], 3)
            self.assertTrue((dir_a / "epoch_history.csv").is_file())
            self.assertTrue((dir_b / "epoch_history.csv").is_file())

    def test_custom_traffic_does_not_reuse_campaign_validation_without_opt_in(self):
        from types import SimpleNamespace
        from simulation.live_control import create_live_session
        custom = dict(self.initial, lambda_be_pps=15000)
        with tempfile.TemporaryDirectory() as temp:
            with patch("simulation.live_control.verify_ns3_runtime", return_value=(True, "ok")), \
                 patch("simulation.live_control.subprocess.Popen", return_value=SimpleNamespace(pid=1234)):
                _sid, folder, _worker = create_live_session(
                    initial_config=custom, target_qoe=0.8, max_epochs=2, mode="adaptive",
                    output_root=Path(temp), binary="/tmp/ns3", allow_exploratory_candidates=False,
                )
            state = json.loads((folder / "session.json").read_text())
            session_config = json.loads((folder / "session_config.json").read_text())
            self.assertIn("none:", state["validated_candidate_scope"])
            self.assertEqual(session_config["feasible_configurations"], [])
            self.assertFalse(session_config["allow_exploratory_candidates"])
            self.assertEqual(len(session_config["exploratory_configurations"]), 9)

    def test_worker_launch_failure_is_not_left_queued(self):
        from simulation.live_control import create_live_session, read_session_state
        with tempfile.TemporaryDirectory() as temp:
            with patch("simulation.live_control.verify_ns3_runtime", return_value=(True, "ok")), \
                 patch("simulation.live_control.subprocess.Popen", side_effect=OSError("injected spawn error")):
                with self.assertRaisesRegex(OSError, "injected spawn error"):
                    create_live_session(
                        initial_config=self.initial, output_root=Path(temp), binary="/tmp/ns3",
                    )
            folders = list(Path(temp).iterdir())
            session_folders = [p for p in folders if p.is_dir()]
            self.assertEqual(len(session_folders), 1)
            state = read_session_state(session_folders[0])
            self.assertEqual(state["status"], "FAILED")
            self.assertEqual(state["phase"], "WORKER_START_FAILED")

    def test_production_model_prediction_is_disabled_on_feature_mismatch(self):
        audit = production_feature_contract_audit()
        self.assertFalse(audit["compatible"])
        self.assertTrue(audit["training_uses_postrun_tx_packets"])
        self.assertFalse(audit["prediction_enabled"])

    def test_live_chart_plots_only_measured_points_and_keeps_failed_gap(self):
        events = [
            {"epoch": 1, "run_status": "COMPLETED", "measured_qoe_proxy": 0.0,
             "config": self.initial, "metrics": {"is_outage": True}, "controller_action": "CHANGE_CONFIGURATION"},
            {"epoch": 2, "run_status": "TIMEOUT", "measured_qoe_proxy": None},
            {"epoch": 3, "run_status": "COMPLETED", "measured_qoe_proxy": 0.72},
        ]
        figure = build_qoe_figure(events, 0.8)
        trace = figure.data[0]
        self.assertEqual(list(trace.x), [1, 2, 3])
        self.assertEqual(list(trace.y), [0.0, None, 0.72])
        self.assertEqual(trace.customdata[0][0], self.initial["config_id"])
        self.assertEqual(trace.customdata[0][5], "outage")
        self.assertEqual(trace.customdata[0][6], "CHANGE_CONFIGURATION")
        self.assertFalse(trace.connectgaps)
        self.assertEqual(len(figure.layout.shapes), 1)
        self.assertAlmostEqual(figure.layout.shapes[0].y0, 0.8)
        empty = build_qoe_figure([], 0.8)
        self.assertEqual(list(empty.data[0].x), [])
        self.assertEqual(len(empty.layout.shapes), 1)

    def test_chart_keeps_pre_run_prediction_separate_from_measurement(self):
        events = [{"epoch": 1, "run_status": "COMPLETED", "measured_qoe_proxy": 0.72,
                   "predicted_qoe": 0.75, "predicted_qoe_lower": 0.70,
                   "predicted_qoe_upper": 0.80, "config": self.initial}]
        figure = build_qoe_figure(events, 0.8)
        self.assertEqual(figure.data[0].name, "Measured QoE proxy")
        self.assertEqual(list(figure.data[0].y), [0.72])
        self.assertEqual(figure.data[1].name, "Pre-run empirical prediction")
        self.assertEqual(list(figure.data[1].y), [0.75])

    def test_history_display_keeps_failed_qoe_missing(self):
        history = _history_table([{
            "epoch": 1, "run_status": "TIMEOUT", "measured_qoe_proxy": None,
            "config": self.initial, "metrics": None, "controller_action": "STOP_FAILED_EPOCH",
            "error_message": "timeout",
        }])
        self.assertTrue(history.loc[0, "Measured QoE proxy"] is None or
                        history.loc[0, "Measured QoE proxy"] != history.loc[0, "Measured QoE proxy"])
        self.assertEqual(history.loc[0, "Status"], "TIMEOUT")

    def test_terminal_statuses_explain_target_unmet_and_distinct_stop_reasons(self):
        self.assertEqual(session_status_label({"status": "TARGET_REACHED", "target_qoe": 0.8,
                                               "latest_measured_qoe_proxy": 0.8}), "Target reached")
        self.assertIn("Target unmet", session_status_label({"status": "EPOCH_LIMIT_REACHED", "target_qoe": 0.9,
                                                              "latest_measured_qoe_proxy": 0.0}))
        self.assertIn("Target unmet", session_status_label({
            "status": "EXPLORATORY_EPOCH_LIMIT_REACHED", "target_qoe": 0.9,
            "latest_measured_qoe_proxy": 0.7,
        }))
        self.assertEqual(session_status_label({"status": "NO_EXPERIMENTAL_CANDIDATES"}),
                         "No eligible experimental candidates remain")
        self.assertIn("Target unmet", session_status_label({
            "status": "STOP_NO_SAFE_CANDIDATES", "target_qoe": 0.8,
            "latest_measured_qoe_proxy": 0.7,
        }))
        self.assertIn("No validated candidates", session_status_label({"status": "NO_FEASIBLE_CANDIDATES"}))
        self.assertIn("do not rule out other ns-3-supported settings",
                      legacy_candidate_scope_note({"status": "NO_FEASIBLE_CANDIDATES"}))
        self.assertEqual(session_status_label({"status": "NO_IMPROVEMENT_AFTER_AVAILABLE_TRIALS"}),
                         "No improvement after available trials")
        self.assertEqual(session_status_label({"status": "SIMULATION_FAILURE"}), "Simulation failure")
        self.assertEqual(session_status_label({"status": "USER_CANCELLED"}), "User cancelled")
        self.assertIn("Target unmet", session_status_label({"status": "NO_VALIDATED_CANDIDATES",
                                                              "target_qoe": 0.9,
                                                              "latest_measured_qoe_proxy": 0.7875}))
        self.assertIn("Target unmet", session_status_label({"status": "NO_MEANINGFUL_IMPROVEMENT",
                                                              "target_qoe": 0.9,
                                                              "latest_measured_qoe_proxy": 0.7875}))

    def test_status_explanations_use_plain_language_without_changing_status_codes(self):
        self.assertIn("simulation run is active", session_status_explanation({"status": "RUNNING"}))
        self.assertIn("timeout", session_status_explanation({"status": "SIMULATION_TIMEOUT"}))
        self.assertEqual(session_status_label({"status": "SIMULATION_TIMEOUT"}), "Simulation timeout")
        self.assertEqual(session_status_label({"status": "RUNNING", "stop_requested": True}), "Stopping")
        self.assertIn("safe stop", session_status_explanation({"status": "RUNNING", "stop_requested": True}))

    def test_outcome_summary_shows_gap_best_attempts_and_opt_in_remaining_count(self):
        configs = {c["config_id"]: c for c in self.ue2}
        events = [
            {"epoch": 1, "run_status": "COMPLETED", "measured_qoe_proxy": 0.7123,
             "config": configs["L9-06"]},
            {"epoch": 2, "run_status": "COMPLETED", "measured_qoe_proxy": 0.7875,
             "config": configs["ANCHOR-UE2"]},
            {"epoch": 3, "run_status": "COMPLETED", "measured_qoe_proxy": 0.78750558284,
             "config": configs["L9-08"]},
        ]
        cfg = {
            "feasible_configurations": self.ue2,
            "exploratory_configurations": supported_radio_grid(configs["L9-08"], self.ue2),
            "candidate_performance": [{"key": list(k), **v} for k, v in load_candidate_performance().items()],
        }
        summary = candidate_and_outcome_summary(events, {
            "current_config": configs["L9-08"], "target_qoe": 0.9,
            "allow_exploratory_candidates": False,
        }, cfg)
        self.assertAlmostEqual(summary["target_gap"], 0.11249441716)
        self.assertEqual(summary["best_config"], "L9-08")
        self.assertEqual(len(summary["attempted_configs"]), 3)
        self.assertEqual(summary["remaining_validated_count"], 0)
        self.assertGreater(summary["remaining_exploratory_count"], 0)
        self.assertEqual(summary["remaining_candidate_count"], 0)
        legacy_cfg = dict(cfg)
        legacy_cfg.pop("exploratory_configurations")
        legacy_summary = candidate_and_outcome_summary(events, {
            "current_config": configs["L9-08"], "target_qoe": 0.9,
            "allow_exploratory_candidates": False,
        }, legacy_cfg)
        self.assertEqual(legacy_summary["remaining_exploratory_count"], summary["remaining_exploratory_count"])

    def test_csv_history_contract_contains_required_fields(self):
        from simulation.live_control import HISTORY_COLUMNS, _history_row
        row = _history_row({
            "session_id": "s", "epoch": 1, "config": self.initial,
            "run_status": "FAILED", "measured_qoe_proxy": None, "metrics": None,
            "controller_action": "STOP_FAILED_EPOCH",
        })
        self.assertEqual(set(row), set(HISTORY_COLUMNS))
        self.assertEqual(row["measured_qoe_proxy"], "")
        self.assertEqual(row["predicted_qoe"], "")
        self.assertIn("mean_packet_loss_percent", row)
        self.assertIn("adaptive_action_count", row)
        self.assertIn("exploratory_trial_count", row)
        self.assertIn("action_budget_exhausted", row)

    def test_exploratory_selection_evidence_is_saved_and_shown_with_measured_chart_point(self):
        from simulation.live_control import _history_row
        exploratory = dict(self.initial, config_id="EXPLORATORY-B150-P30-UE2",
                           total_tx_power_dbm=30.0)
        event = {
            "session_id": "explore-session", "epoch": 1, "config": exploratory,
            "run_status": "COMPLETED", "measured_qoe_proxy": 0.73,
            "controller_action": "STOP_EPOCH_LIMIT", "candidate_scope": "exploratory supported configuration",
            "is_exploratory": True, "candidate_selection_rationale": "untried supported tuple",
            "candidate_evidence": "one matching-semantics prior live run without outage",
            "candidate_validation": "PASS: supported values and frozen scenario fields",
            "action_rationale": "epoch limit reached", "metrics": {"is_outage": False},
        }
        saved = _history_row(event)
        self.assertEqual(saved["is_exploratory"], True)
        self.assertEqual(saved["candidate_evidence"], event["candidate_evidence"])
        self.assertEqual(saved["candidate_validation"], event["candidate_validation"])
        table = _history_table([event])
        self.assertEqual(table.loc[0, "Exploration status"], "exploratory supported configuration")
        self.assertEqual(table.loc[0, "Candidate evidence"], event["candidate_evidence"])
        chart = build_qoe_figure([event], 0.8)
        self.assertEqual(list(chart.data[0].y), [0.73])
        self.assertEqual(chart.data[0].customdata[0][0], exploratory["config_id"])

    def test_nonzero_simulator_exit_is_failure_and_has_no_measurement(self):
        class FailedProcess:
            returncode = 7
            def poll(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session"
            (session / "epochs").mkdir(parents=True)
            with patch("simulation.live_control.subprocess.Popen", return_value=FailedProcess()):
                event = _run_epoch(
                    session, self.initial, epoch=1, scenario_id=11, rng_seed=20261026,
                    rng_run=1, timeout_seconds=10, binary="/mock/ns3",
                )
            self.assertEqual(event["run_status"], "FAILED")
            self.assertEqual(event["exit_code"], 7)
            self.assertIsNone(event.get("measured_qoe_proxy"))
            run_config = json.loads((Path(event["raw_csv_path"]).parent / "run_config.json").read_text())
            self.assertEqual(run_config["run_status"], "FAILED")

    def test_user_cancellation_is_preserved_for_post_epoch_sessions(self):
        from simulation.live_control import run_live_session
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "post-limit-cancel"
            self._make_post_epoch_session(folder)
            (folder / "STOP_REQUESTED").write_text("cancel")
            with patch("simulation.live_control._run_epoch") as run_epoch:
                run_live_session(folder)
            state = json.loads((folder / "session.json").read_text())
            self.assertEqual(state["status"], "USER_CANCELLED")
            run_epoch.assert_not_called()

    def test_failed_extra_simulation_terminates_post_epoch_session(self):
        from simulation.live_control import run_live_session
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "post-limit-failure"
            initial = self._make_post_epoch_session(folder)

            def fake_run(_session, config, *, epoch, **_kwargs):
                if epoch == 1:
                    return {
                        "epoch": epoch, "run_status": "COMPLETED", "config": config,
                        "measured_qoe_proxy": 0.72, "metrics": {"is_outage": False},
                        "duration_seconds": 0.1, "exit_code": 0, "command": [],
                    }
                return {
                    "epoch": epoch, "run_status": "FAILED", "config": config,
                    "measured_qoe_proxy": None, "metrics": None,
                    "duration_seconds": 0.1, "exit_code": 7, "command": [],
                    "error_type": "ProcessExit", "error_message": "injected extra-run failure",
                }

            with patch("simulation.live_control._run_epoch", side_effect=fake_run) as run_epoch:
                run_live_session(folder)
            state = json.loads((folder / "session.json").read_text())
            events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
            self.assertEqual(run_epoch.call_count, 2)
            self.assertFalse(run_epoch.call_args_list[0].kwargs["post_epoch_exploration"])
            self.assertTrue(run_epoch.call_args_list[1].kwargs["post_epoch_exploration"])
            self.assertEqual(state["status"], "SIMULATION_FAILURE")
            self.assertEqual(len(events), 2)
            self.assertTrue(events[1]["post_epoch_exploration"])
            self.assertEqual(state["post_epoch_exploration_count"], 1)
            self.assertEqual(events[1]["post_epoch_exploration_count"], 1)
            self.assertEqual(events[1]["exploratory_trial_count"], 1)
            self.assertIsNone(events[1]["measured_qoe_proxy"])
            self.assertIn("injected extra-run failure", events[1]["error_message"])

    def test_successful_post_epoch_trials_stop_at_separate_allowance(self):
        from simulation.live_control import run_live_session
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "post-limit-boundary"
            initial = self._make_post_epoch_session(folder)

            def fake_run(_session, config, *, epoch, **_kwargs):
                qoe = 0.70 + epoch * 0.01
                return {
                    "epoch": epoch, "run_status": "COMPLETED", "config": config,
                    "measured_qoe_proxy": qoe, "metrics": {"is_outage": False},
                    "duration_seconds": 0.1, "exit_code": 0, "command": [],
                }

            with patch("simulation.live_control._run_epoch", side_effect=fake_run) as run_epoch:
                run_live_session(folder)
            state = json.loads((folder / "session.json").read_text())
            events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
            self.assertEqual(run_epoch.call_count, 2)
            self.assertEqual([event["epoch"] for event in events], [1, 2])
            self.assertEqual([event["post_epoch_exploration"] for event in events], [False, True])
            self.assertEqual(state["status"], "EXPLORATORY_EPOCH_LIMIT_REACHED")
            self.assertEqual(state["post_epoch_exploration_count"], 1)

    def test_duplicate_worker_cannot_reserve_or_launch_same_session_twice(self):
        from simulation.live_control import run_live_session
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "duplicate-worker"
            self._make_post_epoch_session(folder)
            entered_run = threading.Event()
            release_run = threading.Event()

            def fake_run(_session, config, *, epoch, **_kwargs):
                entered_run.set()
                self.assertTrue(release_run.wait(3), "test did not release the owned worker")
                return {
                    "epoch": epoch, "run_status": "FAILED", "config": config,
                    "measured_qoe_proxy": None, "metrics": None, "duration_seconds": 0.1,
                    "exit_code": 7, "command": [], "error_type": "ProcessExit",
                    "error_message": "expected test failure",
                }

            errors = []
            with patch("simulation.live_control._run_epoch", side_effect=fake_run) as run_epoch:
                worker = threading.Thread(
                    target=lambda: self._capture_thread_error(errors, run_live_session, folder),
                )
                worker.start()
                self.assertTrue(entered_run.wait(3), "first worker did not launch its epoch")
                # A second scheduling attempt must fail the per-session lock without
                # changing counters or invoking another simulator process.
                run_live_session(folder)
                release_run.set()
                worker.join(5)
                self.assertFalse(worker.is_alive())
                self.assertEqual(run_epoch.call_count, 1)
            self.assertEqual(errors, [])
            state = json.loads((folder / "session.json").read_text())
            events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(state["status"], "SIMULATION_FAILURE")
            self.assertIn("Duplicate controller worker rejected", (folder / "controller.log").read_text())

    @staticmethod
    def _capture_thread_error(errors, function, *args):
        try:
            function(*args)
        except BaseException as exc:
            errors.append(exc)

    def test_timeout_is_recorded_without_a_zero_qoe_point(self):
        class HangingProcess:
            returncode = None
            def poll(self):
                return self.returncode
            def terminate(self):
                self.returncode = -15
            def wait(self, timeout=None):
                return self.returncode

        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session"
            (session / "epochs").mkdir(parents=True)
            with patch("simulation.live_control.subprocess.Popen", return_value=HangingProcess()), \
                 patch("simulation.live_control.time.monotonic", side_effect=[0.0, 0.0, 2.0, 3.0]):
                event = _run_epoch(
                    session, self.initial, epoch=1, scenario_id=12, rng_seed=20261026,
                    rng_run=1, timeout_seconds=1, binary="/mock/ns3",
                )
            self.assertEqual(event["run_status"], "TIMEOUT")
            self.assertEqual(event["exit_code"], -15)
            self.assertIsNone(event.get("measured_qoe_proxy"))
            self.assertEqual(event["timeout_stage"], "ns-3 process execution")
            self.assertTrue(event["process_terminated"])
            self.assertIn("stdout.log", event["error_message"])
            self.assertIn("Partial outputs are retained", event["error_message"])

    def test_real_safe_process_timeout_is_terminated_and_recorded(self):
        import time
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session"
            (session / "epochs").mkdir(parents=True)
            with patch("simulation.live_control.build_epoch_command", return_value=["/bin/sleep", "5"]):
                event = _run_epoch(
                    session, self.initial, epoch=1, scenario_id=13, rng_seed=20261026,
                    rng_run=1, timeout_seconds=0.25, binary="/bin/sleep",
                )
            self.assertEqual(event["run_status"], "TIMEOUT")
            self.assertIsNone(event.get("measured_qoe_proxy"))
            self.assertLess(event["duration_seconds"], 5)
            self.assertTrue(event["process_terminated"])
            self.assertIn("configured limit 0.2s", event["error_message"])

    def test_real_safe_process_failure_is_not_counted_as_measurement(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session"
            (session / "epochs").mkdir(parents=True)
            with patch("simulation.live_control.build_epoch_command", return_value=["/bin/false"]):
                event = _run_epoch(
                    session, self.initial, epoch=1, scenario_id=14, rng_seed=20261026,
                    rng_run=1, timeout_seconds=5, binary="/bin/false",
                )
            self.assertEqual(event["run_status"], "FAILED")
            self.assertEqual(event["exit_code"], 1)
            self.assertIsNone(event.get("measured_qoe_proxy"))

    def test_safe_cancel_terminates_only_the_epoch_child(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session"
            (session / "epochs").mkdir(parents=True)
            with patch("simulation.live_control.build_epoch_command", return_value=["/bin/sleep", "5"]):
                stopper = threading.Thread(target=lambda: (time.sleep(0.3), (session / "STOP_REQUESTED").write_text("stop")))
                stopper.start()
                event = _run_epoch(
                    session, self.initial, epoch=1, scenario_id=15, rng_seed=20261026,
                    rng_run=1, timeout_seconds=20, binary="/bin/sleep",
                )
                stopper.join()
            self.assertEqual(event["run_status"], "STOPPED")
            self.assertIsNone(event.get("measured_qoe_proxy"))
            self.assertIsNotNone(event["exit_code"])

    def test_unexpected_worker_exit_is_persisted_as_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "session"
            folder.mkdir()
            (folder / "session.json").write_text(json.dumps({"status": "RUNNING", "worker_pid": 999999999}))
            state = mark_worker_lost(folder)
            self.assertEqual(state["status"], "FAILED")
            self.assertEqual(state["phase"], "WORKER_EXITED")
            _mark_worker_failure(folder, RuntimeError("injected worker exception"))
            state = json.loads((folder / "session.json").read_text())
            self.assertIn("injected worker exception", state["final_outcome"])
            self.assertTrue((folder / "worker_exception.log").is_file())


if __name__ == "__main__":
    unittest.main()
