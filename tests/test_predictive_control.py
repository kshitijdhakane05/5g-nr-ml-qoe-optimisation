from __future__ import annotations

import json
import csv
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from simulation.live_control import (
    HISTORY_COLUMNS,
    _persist_updated_event,
    request_action_approval,
)
from simulation.qoe_forecast import (
    evaluate_next_forecast,
    forecast_qoe_degradation,
    telemetry_freshness,
)
from simulation.result_parser import calculate_qoe_proxy


class PredictiveControlTests(unittest.TestCase):
    def _event(self, epoch: int, throughput: float, *, fingerprint: str = "cfg-a") -> dict:
        qoe = calculate_qoe_proxy(throughput, 0.1, 0.1, 0.0)
        return {
            "epoch": epoch, "run_status": "COMPLETED", "end_utc": datetime.now(timezone.utc).isoformat(),
            "configuration_fingerprint": fingerprint, "config": {"bandwidth_band1_mhz": 50},
            "measured_qoe_proxy": qoe,
            "ue_measurements": [{"ue_id": 1, "throughput_mbps": throughput, "delay_ms": 0.1,
                                 "jitter_ms": 0.1, "packet_loss_fraction": 0.0, "qoe_proxy": qoe}],
        }

    def test_forecast_requires_three_valid_same_configuration_epochs(self):
        result = forecast_qoe_degradation([self._event(1, 50), self._event(2, 48)], target_qoe=0.96)
        self.assertEqual(result["status"], "INSUFFICIENT_HISTORY")
        self.assertIn("Need 3", result["reason"])

    def test_forecast_flags_user_before_target_crossing_and_reports_cause(self):
        events = [self._event(i, value) for i, value in enumerate((50.0, 48.0, 46.0), 1)]
        target = events[-1]["measured_qoe_proxy"] - 0.003
        result = forecast_qoe_degradation(events, target_qoe=target, horizon_epochs=1)
        self.assertEqual(result["status"], "RISK_PREDICTED")
        self.assertEqual(result["at_risk_ue_ids"], [1])
        self.assertGreater(events[-1]["measured_qoe_proxy"], target)
        self.assertIn("throughput trending down", result["users"][0]["reason"])
        self.assertEqual(result["horizon_epochs"], 1)
        self.assertIn("not calibrated", result["uncertainty"])

    def test_configuration_change_resets_forecast_history(self):
        events = [self._event(1, 50), self._event(2, 48), self._event(3, 46, fingerprint="cfg-b")]
        result = forecast_qoe_degradation(events, target_qoe=0.8)
        self.assertEqual(result["status"], "INSUFFICIENT_HISTORY")
        self.assertIn("found 1", result["reason"])

    def test_forecast_is_scored_only_against_same_configuration_next_measurement(self):
        previous = self._event(4, 46)
        previous["forecast_for_next"] = {"target_epoch": 5, "predicted_qoe": 0.9}
        current = self._event(5, 44)
        result = evaluate_next_forecast(previous, current)
        self.assertEqual(result["status"], "EVALUATED")
        self.assertAlmostEqual(result["absolute_error"], abs(result["observed_qoe"] - 0.9))
        current["configuration_fingerprint"] = "cfg-b"
        result = evaluate_next_forecast(previous, current)
        self.assertEqual(result["status"], "INVALIDATED_CONFIGURATION_CHANGE")

    def test_stale_and_invalid_telemetry_are_not_marked_live(self):
        now = datetime.now(timezone.utc)
        event = {"run_status": "COMPLETED", "end_utc": (now - timedelta(seconds=90)).isoformat()}
        self.assertEqual(telemetry_freshness(event, max_age_seconds=30, now=now)["status"], "STALE")
        self.assertEqual(telemetry_freshness({"run_status": "RUNNING"})["status"], "UNAVAILABLE")
        event["end_utc"] = "not-a-time"
        self.assertEqual(telemetry_freshness(event, now=now)["status"], "INVALID_TIMESTAMP")

    def test_recommendation_approval_requires_exact_pending_fingerprint_and_persists(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            state = {
                "session_id": "s1", "status": "WAITING_FOR_APPROVAL",
                "pending_action": {"epoch": 2, "fingerprint": "sha-config", "action": "Band 1 bandwidth restart",
                                   "previous_config": {"bandwidth_band1_mhz": 20},
                                   "proposed_config": {"bandwidth_band1_mhz": 50}, "rationale": "risk"},
            }
            (folder / "session.json").write_text(json.dumps(state))
            decision = request_action_approval(folder, approve=True)
            self.assertEqual(decision["fingerprint"], "sha-config")
            self.assertTrue((folder / "action_decision_epoch_2.json").is_file())
            ledger = json.loads((folder / "interventions.jsonl").read_text().splitlines()[0])
            self.assertEqual(ledger["status"], "APPROVED")
            with self.assertRaises(ValueError):
                request_action_approval(folder, approve=False)

    def test_verified_action_status_replaces_persisted_epoch_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            event = {
                "session_id": "s1", "epoch": 1, "run_status": "COMPLETED",
                "config": {"bandwidth_band1_mhz": 20, "bandwidth_band2_mhz": 50,
                           "total_tx_power_dbm": 35, "ue_count": 2},
                "action_application_status": "SCHEDULED_RESTART",
                "measured_qoe_proxy": 0.2,
            }
            (folder / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
            with (folder / "epoch_history.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
                writer.writeheader()
                writer.writerow({**dict.fromkeys(HISTORY_COLUMNS, ""), "session_id": "s1", "epoch": 1})
            _persist_updated_event(folder, event)
            event["action_application_status"] = "APPLIED_AND_VERIFIED"
            event["verification_epoch"] = 2
            event["observed_qoe_after"] = 0.63
            _persist_updated_event(folder, event)
            saved = json.loads((folder / "events.jsonl").read_text().splitlines()[0])
            self.assertEqual(saved["action_application_status"], "APPLIED_AND_VERIFIED")
            self.assertEqual(saved["observed_qoe_after"], 0.63)
            self.assertEqual((folder / "events.jsonl").read_text().count("\n"), 1)
            with (folder / "epoch_history.csv").open(encoding="utf-8", newline="") as handle:
                csv_event = next(csv.DictReader(handle))
            self.assertEqual(csv_event["action_application_status"], "APPLIED_AND_VERIFIED")
            self.assertEqual(csv_event["observed_qoe_after"], "0.63")


if __name__ == "__main__":
    unittest.main()
