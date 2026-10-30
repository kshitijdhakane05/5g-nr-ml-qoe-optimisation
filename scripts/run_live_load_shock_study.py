#!/usr/bin/env python3
"""Repeat the guarded live ns-3 load-shock demonstration and save its evidence."""
from __future__ import annotations

import argparse
import csv
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation import live_runtime as live


STUDY_ROOT = ROOT / "results" / "live_control"


def _read_events(session_id: str) -> list[dict[str, Any]]:
    path = live.LIVE_ROOT / session_id / "events.jsonl"
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def _trace_rows(session_id: str, target: float) -> list[dict[str, Any]]:
    rows = []
    for point in live.read_session(session_id)["points"]:
        by_id = {int(ue["ue_id"]): ue for ue in point.get("ues", [])}
        rows.append({
            "session_id": session_id,
            "sim_time_s": point.get("sim_time_s"),
            "mean_ue_qoe_measured": point.get("qoe"),
            "lambda_be_pps_measured": point.get("lambda_be_pps"),
            "ue_count": len(by_id),
            "minimum_ue_qoe_measured": min((float(ue.get("qoe", 0)) for ue in by_id.values()), default=None),
            "below_target_ue_ids": ",".join(str(ue_id) for ue_id, ue in sorted(by_id.items())
                                               if float(ue.get("qoe", 0)) < target),
            "per_ue_qoe_measured": json.dumps({str(i): ue.get("qoe") for i, ue in by_id.items()}),
            "per_ue_delay_ms_measured": json.dumps({str(i): ue.get("delay_ms") for i, ue in by_id.items()}),
            "per_ue_loss_fraction_measured": json.dumps({str(i): ue.get("packet_loss_fraction") for i, ue in by_id.items()}),
        })
    return rows


def _run_trial(index: int, target: float, timeout_seconds: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = {"ue_count": 2, "gnb_count": 1, "ue_association_policy": "closest",
              "traffic_mix": "legacy", "bandwidth_mhz": 50, "total_tx_power_dbm": 35,
              "scenario_size_m": 30, "lambda_be": 2500, "lambda_ull": 2500}
    session_id = live.start_session(config, mode="Automatic", target_qoe=target)
    folder = live.LIVE_ROOT / session_id
    meta = json.loads((folder / "session.json").read_text())
    meta["session_purpose"] = "Repeatable guarded live load-shock study"
    meta["study_replicate"] = index
    live._write_meta(folder, meta)
    shock_requested = False
    failure = None
    started = time.monotonic()
    outcome = None
    trace = []
    try:
        while time.monotonic() - started < timeout_seconds:
            data = live.read_session(session_id)
            if not shock_requested:
                eligibility = live.test_load_shock_eligibility(session_id)
                if eligibility.get("allowed"):
                    live.request_test_load_shock(session_id)
                    shock_requested = True
            events = _read_events(session_id)
            shock_events = [event for event in events if event.get("type") == "test_impairment_applied"]
            corrective_requests = [event for event in events if event.get("type") == "action_pending"
                                   and shock_events and event.get("timestamp", "") >= shock_events[0].get("timestamp", "")]
            request_ids = {event.get("action_id") for event in corrective_requests}
            outcomes = [event for event in events if event.get("type") == "post_action_measurement"
                        and event.get("action_id") in request_ids]
            if outcomes:
                outcome = outcomes[-1]
                break
            if data["meta"].get("status") in {"Failed", "Stopped"}:
                failure = f"Simulation ended with status {data['meta'].get('status')} before action verification."
                break
            time.sleep(.5)
        else:
            failure = f"Timed out after {timeout_seconds:g} seconds without a verified corrective action."
    except Exception as exc:  # keep the failure in the replicate record and still stop the session
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        trace = _trace_rows(session_id, target)
        try:
            live.stop_session(session_id)
            stop_deadline = time.monotonic() + 30
            while time.monotonic() < stop_deadline:
                status = live.read_session(session_id)["meta"].get("status")
                if status == "Stopped":
                    break
                if status == "Failed":
                    break
                time.sleep(.5)
        except Exception as exc:
            failure = failure or f"Could not stop session cleanly: {type(exc).__name__}: {exc}"

    events = _read_events(session_id)
    shock_applied = next((event for event in events if event.get("type") == "test_impairment_applied"), None)
    applied_rate_change = next((event for event in events if event.get("type") == "action_applied"
                                and event.get("action_id") in {e.get("action_id") for e in events
                                    if e.get("type") == "action_pending"}), None)
    first_bad = next((row for row in trace if row.get("below_target_ue_ids")), None)
    record = {
        "replicate": index, "session_id": session_id, "target_qoe": target,
        "shock_requested": shock_requested, "shock_acknowledged": shock_applied is not None,
        "shock_sim_time_s": shock_applied.get("sim_time_s") if shock_applied else None,
        "disturbance_rate_pps": shock_applied.get("new_pps") if shock_applied else None,
        "controller_rate_action_acknowledged": applied_rate_change is not None,
        "controller_previous_pps": applied_rate_change.get("previous_pps") if applied_rate_change else None,
        "controller_new_pps": applied_rate_change.get("new_pps") if applied_rate_change else None,
        "controller_action_sim_time_s": applied_rate_change.get("sim_time_s") if applied_rate_change else None,
        "first_below_target_sim_time_s": first_bad.get("sim_time_s") if first_bad else None,
        "verified_pre_action_mean_qoe": outcome.get("pre_action_qoe_mean") if outcome else None,
        "verified_post_action_mean_qoe": outcome.get("post_action_qoe_mean") if outcome else None,
        "verified_mean_qoe_delta": outcome.get("measured_delta") if outcome else None,
        "all_ues_target_reached": outcome.get("target_reached") if outcome else None,
        "mean_target_reached": outcome.get("network_mean_target_reached") if outcome else None,
        "per_ue_outcome": json.dumps(outcome.get("per_ue_outcome", {})) if outcome else "{}",
        "failure": failure,
    }
    return record, trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="Number of independent ns-3 sessions (default: 3).")
    parser.add_argument("--target", type=float, default=.70, help="Per-UE QoE target (default: 0.70).")
    parser.add_argument("--timeout", type=float, default=180, help="Maximum seconds per replicate (default: 180).")
    args = parser.parse_args()
    if args.runs < 1 or not 0 <= args.target <= 1 or args.timeout <= 0:
        parser.error("runs and timeout must be positive; target must be in [0,1]")
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    study_id = datetime.now(timezone.utc).strftime("load_shock_%Y%m%dT%H%M%SZ")
    output = STUDY_ROOT / study_id
    output.mkdir(exist_ok=False)
    records, traces = [], []
    for index in range(1, args.runs + 1):
        record, trace = _run_trial(index, args.target, args.timeout)
        records.append(record)
        traces.extend(trace)
        print(json.dumps(record, sort_keys=True), flush=True)
    (output / "study.json").write_text(json.dumps({
        "study_id": study_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": {"ue_count": 2, "gnb_count": 1, "traffic_mix": "legacy",
                     "bandwidth_mhz": 50, "total_tx_power_dbm": 35,
                     "initial_lambda_be_pps": 2500, "lambda_ull_pps": 2500,
                     "target_qoe": args.target},
        "method": "Fresh ns-3 session per replicate; test disturbance gated by three fresh, complete above-target UE windows; Automatic controller; post-change ns-3 measurements.",
        "replicates": records,
    }, indent=2))
    if records:
        with (output / "replicates.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    if traces:
        with (output / "telemetry.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(traces[0]))
            writer.writeheader(); writer.writerows(traces)
    print(f"Study artifacts: {output}")
    return 0 if all(record.get("all_ues_target_reached") is True for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
