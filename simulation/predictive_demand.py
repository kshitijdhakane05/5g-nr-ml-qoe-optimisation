"""Opt-in paired ns-3 trial for scheduled-demand QoE protection.

The no-control arm measures a same-scenario reference. The Automatic arm uses
that reference to cap admitted offered load before a scheduled demand increase.
This is an admission-control demonstration, not a claim that suppressed demand
was delivered or that the reference is the controlled run's observed outcome.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simulation import live_runtime as live

ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = ROOT / "results/live_control/predictive_demand_studies"
TRIAL_CONFIG = {
    "ue_count": 4, "gnb_count": 1, "ue_association_policy": "closest",
    "traffic_mix": "legacy", "bandwidth_mhz": 50.0,
    "total_tx_power_dbm": 35.0, "scenario_size_m": 30.0,
    "lambda_be": 7500, "lambda_ull": 2500, "admission_control": True,
}
TARGET_QOE = 0.85
REQUESTED_PPS = 20000
PREEMPTIVE_CAP_PPS = 5000


def latest_study() -> dict[str, Any] | None:
    pointer = STUDY_ROOT / "latest.json"
    try:
        data = json.loads(pointer.read_text())
        return json.loads((STUDY_ROOT / data["study_id"] / "study.json").read_text())
    except (OSError, ValueError, KeyError):
        return None


def start_study() -> str:
    """Start a durable paired study in the background; refuse overlapping runs."""
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    active = next((s for s in live.list_sessions()
                   if live.session_process_is_active(s)), None)
    if active:
        raise RuntimeError(f"Stop active simulation {active.get('session_id')} before starting the paired trial.")
    previous = latest_study()
    if previous and previous.get("status") == "Running":
        raise RuntimeError(f"Paired trial {previous.get('study_id')} is already running.")
    sid = "demand_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    folder = STUDY_ROOT / sid
    folder.mkdir()
    (folder / "study.json").write_text(json.dumps({
        "study_id": sid, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "Running", "phase": "starting_no_control_reference",
        "target_qoe": TARGET_QOE, "requested_pps": REQUESTED_PPS,
        "preemptive_cap_pps": PREEMPTIVE_CAP_PPS,
        "config": TRIAL_CONFIG,
        "interpretation": "Measured matched reference and controlled ns-3 outcomes; no synthetic samples.",
    }, indent=2))
    (STUDY_ROOT / "latest.json").write_text(json.dumps({"study_id": sid}))
    log = (folder / "runner.log").open("ab")
    subprocess.Popen([sys.executable, "-m", "simulation.predictive_demand", "--run", sid],
                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return sid


def _save(folder: Path, state: dict[str, Any]) -> None:
    tmp = folder / "study.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(folder / "study.json")


def _wait_for(folder: Path, sid: str, predicate, timeout_s: float,
              phase: str) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    state = json.loads((folder / "study.json").read_text())
    state["phase"] = phase
    _save(folder, state)
    while time.monotonic() < deadline:
        live.refresh_action_status(sid)
        data = live.read_session(sid)
        state = json.loads((folder / "study.json").read_text())
        state["current_session_status"] = data["meta"].get("status")
        state["current_sim_time_s"] = data["meta"].get("simulation_time_s")
        _save(folder, state)
        if data["meta"].get("status") in {"Failed", "Stopped", "Stale"}:
            raise RuntimeError(f"Session {sid} became {data['meta'].get('status')} during {phase}.")
        # Predicates indicate readiness; callers need the complete session
        # record for points, metadata, and events. Returning a matching point
        # here used to leak that point object and crash the study with
        # KeyError("points").
        if predicate(data):
            return data
        time.sleep(.4)
    raise TimeoutError(f"Timed out during {phase} for session {sid}.")


def _point_has_requested_demand(point: dict[str, Any], requested: int,
                                onset_sim_time_s: float) -> bool:
    return (int(point.get("requested_lambda_be_pps", -1)) == requested
            and int(point.get("lambda_be_pps", -1)) == requested
            and float(point.get("window_start_s", point.get("sim_time_s", 0))) >= onset_sim_time_s - 1e-6)


def _point_has_scheduled_demand(point: dict[str, Any], requested: int,
                                onset_sim_time_s: float) -> bool:
    """Match a complete demand window even when admission control caps delivery."""
    return (int(point.get("requested_lambda_be_pps", -1)) == requested
            and float(point.get("window_start_s", point.get("sim_time_s", 0)))
                >= onset_sim_time_s - 1e-6)


def _stop(sid: str) -> None:
    try:
        live.stop_session(sid)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            data = live.read_session(sid)
            if data["meta"].get("status") in {"Stopped", "Failed"}:
                return
            time.sleep(.3)
    except (OSError, ValueError, KeyError):
        pass


def _run_study(study_id: str) -> None:
    folder = STUDY_ROOT / study_id
    state = json.loads((folder / "study.json").read_text())
    session_ids: list[str] = []
    try:
        # Arm a real demand schedule after four healthy measured windows.
        baseline_id = live.start_session(TRIAL_CONFIG, mode="Monitor", target_qoe=TARGET_QOE)
        session_ids.append(baseline_id)
        state.update({"baseline_session_id": baseline_id, "phase": "warming_no_control"})
        _save(folder, state)
        warm = _wait_for(folder, baseline_id,
            lambda d: d if len([p for p in d["points"] if p.get("segment_id") == "segment_001"]) >= 4 else None,
            90, "warming_no_control")
        baseline_meta = warm["meta"]
        onset = float(baseline_meta.get("simulation_time_s", 0)) + 5.0
        live.request_demand_change(baseline_id, REQUESTED_PPS, onset_sim_time_s=onset,
                                   reason="Paired predictive trial: scheduled demand for no-control reference")
        baseline_data = _wait_for(folder, baseline_id,
            lambda d: next((p for p in reversed(d["points"])
                            if p.get("segment_id") == "segment_001"
                            and _point_has_requested_demand(p, REQUESTED_PPS, onset)), None),
            90, "measuring_no_control_demand")
        reference_point = next(p for p in reversed(baseline_data["points"])
                               if p.get("segment_id") == "segment_001"
                               and _point_has_requested_demand(p, REQUESTED_PPS, onset))
        reference_qoe = {str(int(ue["ue_id"])): float(ue["qoe"])
                         for ue in reference_point.get("ues", [])}
        if len(reference_qoe) != int(TRIAL_CONFIG["ue_count"]):
            raise RuntimeError("No-control reference did not contain a complete measured UE window.")
        reference = {"session_id": baseline_id, "sim_time_s": reference_point["sim_time_s"],
                     "requested_pps": REQUESTED_PPS, "admitted_pps": REQUESTED_PPS,
                     "per_ue_qoe": reference_qoe,
                     "mean_ue_qoe": sum(reference_qoe.values()) / len(reference_qoe),
                     "source": "measured no-control ns-3 outcome"}
        state["no_control_reference"] = reference
        state["baseline_point"] = reference_point
        _save(folder, state)
        _stop(baseline_id)
        if all(value >= TARGET_QOE for value in reference_qoe.values()):
            raise RuntimeError("Scheduled demand did not push any UE below target in the no-control arm; no predictive claim made.")

        state["phase"] = "starting_automatic_controlled_run"
        _save(folder, state)
        controlled_id = live.start_session(TRIAL_CONFIG, mode="Automatic", target_qoe=TARGET_QOE)
        session_ids.append(controlled_id)
        state["controlled_session_id"] = controlled_id
        _save(folder, state)
        warm_controlled = _wait_for(folder, controlled_id,
            lambda d: d if len([p for p in d["points"] if p.get("segment_id") == "segment_001"]) >= 4 else None,
            90, "warming_automatic_run")
        controlled_onset = float(warm_controlled["meta"].get("simulation_time_s", 0)) + 5.0
        live.schedule_demand_preview(controlled_id, requested_pps=REQUESTED_PPS,
            onset_sim_time_s=controlled_onset, preemptive_cap_pps=PREEMPTIVE_CAP_PPS,
            reference=reference)
        live.request_demand_change(controlled_id, REQUESTED_PPS,
            onset_sim_time_s=controlled_onset,
            reason="Known upcoming demand; matched no-control measurement showed target breach")
        state.update({"phase": "waiting_for_preemptive_cap_and_demand",
                      "controlled_onset_sim_time_s": controlled_onset})
        _save(folder, state)
        # Wait until ns-3 confirms both the cap application and the requested
        # demand onset, then capture the first full post-onset measured window.
        controlled_data = _wait_for(folder, controlled_id,
            lambda d: next((p for p in reversed(d["points"])
                            if p.get("segment_id") == "segment_001"
                            and _point_has_scheduled_demand(p, REQUESTED_PPS, controlled_onset)
                            and int(p.get("admission_cap_pps", -1)) == PREEMPTIVE_CAP_PPS), None)
                if d["meta"].get("scheduled_demand_preview", {}).get("status") in {"demand_applied", "cap_verified"}
                else None,
            90, "measuring_controlled_outcome")
        controlled_point = next(p for p in reversed(controlled_data["points"])
            if p.get("segment_id") == "segment_001" and _point_has_scheduled_demand(p, REQUESTED_PPS, controlled_onset)
            and int(p.get("admission_cap_pps", -1)) == PREEMPTIVE_CAP_PPS)
        controlled_qoe = {str(int(ue["ue_id"])): float(ue["qoe"])
                          for ue in controlled_point.get("ues", [])}
        action = next((e for e in controlled_data["events"]
                       if e.get("type") == "action_applied"
                       and e.get("decision_basis") == "predictive_preemptive"), None)
        preview = controlled_data["meta"].get("scheduled_demand_preview", {})
        action_time = action.get("sim_time_s") if action else preview.get("cap_applied_sim_time_s")
        onset_actual = preview.get("actual_onset_sim_time_s")
        action_preceded = action_time is not None and onset_actual is not None and float(action_time) < float(onset_actual)
        outcome = {"session_id": controlled_id, "sim_time_s": controlled_point.get("sim_time_s"),
                   "per_ue_qoe": controlled_qoe,
                   "mean_ue_qoe": sum(controlled_qoe.values()) / len(controlled_qoe) if controlled_qoe else None,
                   "requested_pps": int(controlled_point.get("requested_lambda_be_pps", REQUESTED_PPS)),
                   "admitted_pps": int(controlled_point.get("lambda_be_pps", -1)),
                   "unadmitted_pps": int(controlled_point.get("unadmitted_demand_pps", 0)),
                   "source": "measured controlled ns-3 outcome"}
        passed = (action_preceded and len(controlled_qoe) == int(TRIAL_CONFIG["ue_count"])
                  and all(value >= TARGET_QOE for value in controlled_qoe.values()))
        state.update({"controlled_outcome": outcome, "controlled_point": controlled_point,
                      "controller_action": action, "action_preceded_demand": action_preceded,
                      "actual_demand_onset_sim_time_s": onset_actual,
                      "success": passed,
                      "status": "Completed" if passed else "Completed · criterion not met",
                      "phase": "complete", "ended_at": datetime.now(timezone.utc).isoformat(),
                      "comparison_note": ("Paired measured no-control reference is the displayed estimate of what the controlled scenario would have experienced without intervention. It is not a simultaneous observation."),
                      "limitations": ["Demand increase is scheduled and known in advance; this does not predict an unknown arrival.",
                                      "QoE is calculated on admitted/transmitted traffic; unadmitted requested demand is reported separately.",
                                      "Sequential matched runs may differ due to simulation randomness."]})
        _save(folder, state)
    except Exception as exc:
        state.update({"status": "Failed", "phase": "failed", "error": f"{type(exc).__name__}: {exc}",
                      "ended_at": datetime.now(timezone.utc).isoformat()})
        _save(folder, state)
    finally:
        for sid in session_ids:
            try:
                if live.read_session(sid)["meta"].get("status") not in {"Stopped", "Failed"}:
                    _stop(sid)
            except (OSError, ValueError, KeyError):
                pass


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--run":
    _run_study(sys.argv[2])
