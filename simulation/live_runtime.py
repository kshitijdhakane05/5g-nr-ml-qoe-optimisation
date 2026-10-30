"""Persistent real-time ns-3 process supervision and transparent QoE control."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import fcntl
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simulation.hybrid_advisor import compare_rate_candidate

ROOT = Path(__file__).resolve().parents[1]
LIVE_BINARY = ROOT / "ns-3.48/build-live-release/scratch/ns3.48-qoe-live-control"
LIVE_ROOT = ROOT / "results/live_sessions"
QOE_DEFINITION = (
    "Service QoE v2 (0–1) scores each UE against its traffic profile: 0.40×throughput satisfaction + "
    "0.25×delay satisfaction + 0.20×jitter satisfaction + 0.15×packet-loss satisfaction. "
    "Throughput satisfaction is min(measured/reference, 1); the other terms are max(0, 1−measured/limit). "
    "No received packets scores zero. Overall user QoE is the equal-weight mean across configured UEs; "
    "per-class means are also reported. The legacy pooled-flow proxy remains diagnostic only."
)
MIN_RATE, MAX_RATE = 2500, 20000
LIVE_BANDWIDTHS_MHZ = {20.0, 50.0, 100.0}
LIVE_MAX_GNBS = 4
SAFETY_FLOOR_MARGINS = {
    "low_latency": 0.08,
    "voice_qos": 0.10,
    "gaming": 0.08,
    "video_streaming": 0.12,
}
SAFETY_WARNING_MARGIN = 0.05


def default_safety_floors(target_qoe: float) -> dict[str, float]:
    """Return adjustable project policy floors below the selected QoE target."""
    target = max(0.0, min(1.0, float(target_qoe)))
    return {service: round(max(0.0, target - margin), 2)
            for service, margin in SAFETY_FLOOR_MARGINS.items()}


def normalize_safety_floors(target_qoe: float,
                            configured: dict[str, float] | None = None) -> dict[str, float]:
    """Validate per-service QoE floors; at target zero, protection is inactive."""
    target = max(0.0, min(1.0, float(target_qoe)))
    values = default_safety_floors(target)
    if configured:
        for service in values:
            if service in configured:
                values[service] = float(configured[service])
    for service, floor in values.items():
        if (not 0.0 <= floor <= 1.0 or floor > target
                or (target > 0 and floor >= target)):
            raise ValueError(f"Safety floor for {service} must be in [0, target QoE).")
    return values


def mean_user_qoe(ue_states: list[dict[str, Any]]) -> float:
    """Equal-weight mean QoE across users; missing-reception entries contribute zero."""
    if not ue_states:
        return 0.0
    values = [max(0.0, min(1.0, float(ue.get("qoe", 0.0)))) for ue in ue_states]
    return sum(values) / len(values)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _segment_dir(folder: Path, number: int) -> Path:
    root = folder / "segments"
    if number == 1 and not root.exists() and (folder / "session.json").is_file():
        try:
            saved = json.loads((folder / "session.json").read_text())
            if not saved.get("segments"):
                return folder  # preserve compatibility with an already running pre-segment session
        except (OSError, ValueError):
            pass
    path = root / f"segment_{number:03d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _launch_live_process(folder: Path, config: dict[str, Any], number: int, target_qoe: float) -> subprocess.Popen[Any]:
    segment = _segment_dir(folder, number)
    args = [str(LIVE_BINARY), f"--ueCount={int(config.get('ue_count', 2))}",
            f"--ueNumPergNb={int(config.get('ue_count', 2))}", f"--gNbNum={int(config.get('gnb_count',1))}",
            f"--ueAssociationPolicy={config.get('ue_association_policy','closest')}",
            f"--trafficMix={config.get('traffic_mix', 'legacy')}",
            f"--bandwidthBand1={int(float(config.get('bandwidth_mhz',50))*1e6)}",
            f"--totalTxPower={float(config.get('total_tx_power_dbm',35))}",
            f"--scenarioSize={float(config.get('scenario_size_m',30))}",
            f"--lambdaBe={int(config.get('lambda_be',10000))}",
            f"--lambdaUll={int(config.get('lambda_ull',2500))}", "--simTime=3600s",
            f"--outputDir={segment}", f"--liveTelemetry={segment/'telemetry.jsonl'}",
            f"--liveCommand={segment/'command.txt'}", f"--liveStop={segment/'stop'}"]
    if bool(config.get("admission_control", False)):
        args.extend([f"--liveDemand={segment/'demand.txt'}", "--maxAdmittedBePps=20000",
                     "--rngSeed=1", "--rngRun=1"])
    (segment / "run_config.json").write_text(json.dumps({"segment": number, "config": config,
        "target_qoe": target_qoe, "command": args, "started_at": _utc()}, indent=2))
    with (segment / "simulator.log").open("ab") as log:
        proc = subprocess.Popen(args, cwd=segment, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    return proc


def _active_telemetry_path(folder: Path, meta: dict[str, Any]) -> Path:
    number = int(meta.get("current_segment", 1))
    segmented = folder / "segments" / f"segment_{number:03d}" / "telemetry.jsonl"
    return segmented if segmented.exists() or (folder / "segments").exists() else folder / "telemetry.jsonl"


def list_sessions() -> list[dict[str, Any]]:
    result = []
    if not LIVE_ROOT.exists():
        return result
    for folder in sorted(LIVE_ROOT.iterdir(), reverse=True):
        try:
            meta = json.loads((folder / "session.json").read_text())
            result.append(meta)
        except (OSError, ValueError):
            continue
    return result


def start_session(config: dict[str, Any], mode: str = "Recommend", target_qoe: float = .72,
                  safety_floors: dict[str, float] | None = None) -> str:
    if not LIVE_BINARY.is_file() or not os.access(LIVE_BINARY, os.X_OK):
        raise RuntimeError(
            f"Live ns-3 release binary unavailable: {LIVE_BINARY}. "
            "Build it with the release-profile commands in docs/live-simulation.md."
        )
    if mode not in {"Monitor", "Recommend", "Automatic"}:
        raise ValueError("Mode must be Monitor, Recommend, or Automatic")
    if not 0 <= float(target_qoe) <= 1:
        raise ValueError("QoE target must be in [0,1]")
    validated_floors = normalize_safety_floors(float(target_qoe), safety_floors)
    normalized = {"ue_count": int(config.get("ue_count", 2)),
                 "gnb_count": int(config.get("gnb_count", 1)),
                 "ue_association_policy": str(config.get("ue_association_policy", "closest")),
                 "traffic_mix": str(config.get("traffic_mix", "legacy")),
                 "bandwidth_mhz": float(config.get("bandwidth_mhz", 50)),
                 "total_tx_power_dbm": float(config.get("total_tx_power_dbm", 35)),
                 "scenario_size_m": float(config.get("scenario_size_m", 30)),
                 "lambda_be": int(config.get("lambda_be", 10000)),
                 "lambda_ull": int(config.get("lambda_ull", 2500)),
                 "admission_control": bool(config.get("admission_control", False))}
    if not 2 <= normalized["ue_count"] <= 20:
        raise ValueError("UE count must be between 2 and 20")
    if normalized["bandwidth_mhz"] not in LIVE_BANDWIDTHS_MHZ:
        raise ValueError(f"Live bandwidth must be one of {sorted(LIVE_BANDWIDTHS_MHZ)} MHz")
    if not 1 <= normalized["gnb_count"] <= LIVE_MAX_GNBS:
        raise ValueError(f"Live gNB count must be 1–{LIVE_MAX_GNBS}")
    if normalized["ue_association_policy"] not in {"closest", "by_traffic_class", "round_robin"}:
        raise ValueError("Unsupported UE association policy")
    if normalized["traffic_mix"] not in {"legacy", "gaming_video"}:
        raise ValueError("Unsupported UE traffic mix")
    if normalized["ue_association_policy"] != "closest" and normalized["gnb_count"] < 2:
        raise ValueError(f"{normalized['ue_association_policy']} association requires at least two gNBs")
    if not 10 <= normalized["total_tx_power_dbm"] <= 50:
        raise ValueError("Total transmit power must be between 10 and 50 dBm")
    if not 2500 <= normalized["lambda_be"] <= 20000 or not 500 <= normalized["lambda_ull"] <= 10000:
        raise ValueError("Offered packet rates are outside the supported live ranges")
    for previous in list_sessions():
        if session_process_is_active(previous):
            raise RuntimeError(f"Live ns-3 session {previous.get('session_id')} is still running. Stop it before starting another.")
    sid = "live_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    folder = LIVE_ROOT / sid
    folder.mkdir(parents=True, exist_ok=False)
    # Each structural restart is a separate ns-3 run segment under one logical session.
    # One hour of wall-clock simulation; user stop is supervised through a file.
    meta = {"session_id": sid, "created_at": _utc(), "status": "Starting", "mode": mode,
            "target_qoe": float(target_qoe), "safety_floors": validated_floors,
            "safety_warning_margin": SAFETY_WARNING_MARGIN, "config": normalized,
            "configuration_verified": False,
            "active_parameters": {"ue_count": normalized["ue_count"],
                                  "bandwidth_mhz": normalized["bandwidth_mhz"],
                                  "gnb_count": normalized["gnb_count"],
                                  "ue_association_policy": normalized["ue_association_policy"],
                                  "traffic_mix": normalized["traffic_mix"],
                                  "total_tx_power_dbm": normalized["total_tx_power_dbm"],
                                  "lambda_be_pps": normalized["lambda_be"],
                                  "lambda_ull_pps": normalized["lambda_ull"],
                                  "lambda_be_cap_pps": (MAX_RATE if normalized["admission_control"]
                                                        else normalized["lambda_be"])},
            "action_pending": False, "last_control_at": 0, "pid": None, "current_segment": 1,
            "segments": [{"segment": 1, "config": normalized, "status": "Starting", "started_at": _utc()}],
            "structural_actions": 0, "last_decision_sim_time_s": 0}
    (folder / "session.json").write_text(json.dumps(meta, indent=2))
    (folder / "events.jsonl").touch()
    proc = _launch_live_process(folder, normalized, 1, target_qoe)
    meta["pid"] = proc.pid
    meta["status"] = "Running"
    meta["segments"][0]["status"] = "Running"
    (folder / "session.json").write_text(json.dumps(meta, indent=2))
    worker_log = (folder / "controller.log").open("ab")
    worker = subprocess.Popen([sys.executable, "-m", "simulation.live_runtime", "--worker", sid],
                             cwd=ROOT, stdout=worker_log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    meta["controller_pid"] = worker.pid
    _write_meta(folder, meta)
    return sid


def _write_meta(folder: Path, meta: dict[str, Any]) -> None:
    tmp = folder / f"session.json.{os.getpid()}.{threading.get_ident()}.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(folder / "session.json")


def read_session(session_id: str) -> dict[str, Any]:
    folder = LIVE_ROOT / session_id
    meta = json.loads((folder / "session.json").read_text())
    points = []
    segment_root = folder / "segments"
    telemetry_files = sorted(segment_root.glob("segment_*/telemetry.jsonl")) if segment_root.exists() else []
    if not telemetry_files:
        telemetry_files = [folder / "telemetry.jsonl"]
    for telemetry_file in telemetry_files:
        segment_id = telemetry_file.parent.name if telemetry_file.parent.parent == segment_root else "segment_001"
        try:
            with telemetry_file.open() as f:
                for line in f:
                    try:
                        p = json.loads(line)
                        if all(k in p for k in ("sim_time_s", "qoe", "throughput_mbps")):
                            p.setdefault("segment_id", segment_id)
                            points.append(p)
                    except ValueError:
                        continue  # only ignore a possibly partial last append
        except FileNotFoundError:
            continue
    events = []
    try:
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines() if line]
    except (OSError, ValueError):
        pass
    now = time.time()
    latest = points[-1] if points else None
    active_telemetry = _active_telemetry_path(folder, meta)
    if meta.get("status") in {"Running", "Starting", "Stale"}:
        if meta.get("pid") and not _pid_alive(int(meta["pid"]), str(LIVE_BINARY)):
            if not (folder / "stop").exists() and meta.get("status") != "Restarting":
                meta["status"] = "Failed"
            elif (folder / "stop").exists():
                meta["status"] = "Stopped"
        elif latest and not active_telemetry.exists() and meta.get("status") == "Starting":
            started = meta.get("segments", [{}])[-1].get("started_at", meta.get("created_at"))
            if now - datetime.fromisoformat(started).timestamp() > 15:
                meta["status"] = "Stale"
            else:
                meta["transition_status"] = "Waiting for the first measurement from the new run segment"
        elif latest and (not active_telemetry.exists() or now - active_telemetry.stat().st_mtime > 15):
            meta["status"] = "Stale"
        elif not latest and now - datetime.fromisoformat(
                meta.get("segments", [{}])[-1].get("started_at", meta["created_at"])
        ).timestamp() > 15:
            meta["status"] = "Stale"
        elif latest:
            meta["status"] = "Running"
    if latest:
        if active_telemetry.exists():
            meta["last_measurement_utc"] = datetime.fromtimestamp(active_telemetry.stat().st_mtime, timezone.utc).isoformat()
        meta["simulation_time_s"] = latest["sim_time_s"]
        meta["latest"] = latest
        current_segment_id = f"segment_{int(meta.get('current_segment', 1)):03d}"
        current_segment_points = [p for p in points if p.get("segment_id") == current_segment_id]
        verified = False
        if current_segment_points:
            observed = current_segment_points[-1]
            expected = meta.get("config", {})
            verified = (int(observed.get("ue_count", -1)) == int(expected.get("ue_count", -2))
                        and int(observed.get("gnb_count", -1)) == int(expected.get("gnb_count", -2))
                        and float(observed.get("bandwidth_mhz", -1)) == float(expected.get("bandwidth_mhz", -2))
                        and abs(float(observed.get("total_tx_power_dbm", -999)) - float(expected.get("total_tx_power_dbm", 35))) < 1e-6
                        and str(observed.get("ue_association_policy", "")) == str(expected.get("ue_association_policy", ""))
                        and str(observed.get("traffic_mix", expected.get("traffic_mix", "legacy")))
                            == str(expected.get("traffic_mix", "legacy")))
        meta["configuration_verified"] = verified
        if verified:
            meta["active_parameters"].update({
                "ue_count": int(observed["ue_count"]), "bandwidth_mhz": float(observed["bandwidth_mhz"]),
                "gnb_count": int(observed["gnb_count"]),
                "ue_association_policy": str(observed["ue_association_policy"]),
                "traffic_mix": str(observed.get("traffic_mix", expected.get("traffic_mix", "legacy"))),
                "total_tx_power_dbm": float(observed["total_tx_power_dbm"]),
                "lambda_be_pps": int(observed["lambda_be_pps"]),
                "lambda_be_requested_pps": int(observed.get("requested_lambda_be_pps",
                                                           observed["lambda_be_pps"])),
                "lambda_be_cap_pps": int(observed.get("admission_cap_pps",
                                                      observed["lambda_be_pps"])),
            })
        if meta.get("status") == "Running":
            meta["forecast"] = forecast(current_segment_points, target_qoe=float(meta.get("target_qoe", .72)))
            meta["ue_forecasts"] = forecast_ue_states(
                current_segment_points, target_qoe=float(meta.get("target_qoe", .72)))
        else:
            meta["forecast"] = {"available": False, "reason": f"Live forecast paused because session status is {meta.get('status')}.", "horizon_s": 5}
    _write_meta(folder, meta)
    return {"meta": meta, "points": points[-300:], "events": events, "folder": str(folder)}


def _pid_alive(pid: int, expected_command: str | None = None) -> bool:
    try:
        os.kill(pid, 0)
        if expected_command:
            command = Path(f"/proc/{pid}/cmdline")
            return command.exists() and expected_command.encode() in command.read_bytes()
        return True
    except OSError:
        return False


def session_process_is_active(meta: dict[str, Any]) -> bool:
    """Return whether a session should block a second simulator launch.

    Persisted session status can outlive the simulator process (for example,
    after an application or host restart). Trust the supervised ns-3 PID when
    present instead of treating an old ``Running``/``Stale`` label as proof
    that a simulation is still active. Keep a brief guard for the small window
    between writing a Starting record and saving the launched PID.
    """
    if meta.get("status") not in {"Running", "Starting", "Restarting", "Stale"}:
        return False
    pid = meta.get("pid")
    if pid:
        try:
            return _pid_alive(int(pid), str(LIVE_BINARY))
        except (TypeError, ValueError):
            return False
    if meta.get("status") not in {"Starting", "Restarting"}:
        return False
    started = meta.get("created_at")
    segments = meta.get("segments") or []
    if segments:
        started = segments[-1].get("started_at", started)
    try:
        age = time.time() - datetime.fromisoformat(str(started)).timestamp()
    except (TypeError, ValueError):
        return False
    return 0 <= age <= 30


def forecast(points: list[dict[str, Any]], horizon_s: int = 5,
             target_qoe: float | None = None) -> dict[str, Any]:
    if len(points) < 4:
        return {"available": False, "reason": "Need at least four live measurements.", "horizon_s": horizon_s}
    recent = points[-5:]
    xs = [float(x["sim_time_s"]) for x in recent]
    x0 = xs[-1]
    result: dict[str, Any] = {"available": True, "method": "linear trend over last five 1-second windows",
                              "horizon_s": horizon_s, "at_sim_time_s": x0 + horizon_s,
                              "prediction_timestamp_utc": _utc(),
                              "confidence": "low; uncalibrated short-history baseline"}
    for key in ("qoe", "throughput_mbps", "delay_ms", "jitter_ms", "packet_loss_fraction"):
        ys = [float(p[key]) for p in recent]
        # Least-squares line; clip physical quantities to valid ranges.
        meanx, meany = sum(xs)/len(xs), sum(ys)/len(ys)
        den = sum((x-meanx)**2 for x in xs)
        slope = sum((x-meanx)*(y-meany) for x,y in zip(xs,ys))/den if den else 0.0
        val = ys[-1] + slope * horizon_s
        if key in {"qoe", "packet_loss_fraction"}: val = min(1.0,max(0.0,val))
        else: val = max(0.0,val)
        result[key] = val
    falling = result["qoe"] < points[-1]["qoe"] - .04
    below_target = target_qoe is not None and result["qoe"] < float(target_qoe)
    result["risk"] = falling or below_target
    reasons = []
    if falling: reasons.append("QoE trend projects a drop of at least 0.04")
    if below_target: reasons.append(f"predicted QoE is below the configured target ({float(target_qoe):.2f})")
    result["reason"] = "; ".join(reasons) if reasons else "No projected QoE drop or target violation."
    return result


def forecast_ue_states(points: list[dict[str, Any]], horizon_s: int = 5,
                       target_qoe: float = .72) -> dict[int, dict[str, Any]]:
    """Forecast per-UE QoE from that UE's own chronological telemetry windows."""
    ue_ids = sorted({int(ue["ue_id"]) for point in points for ue in point.get("ues", [])})
    predictions: dict[int, dict[str, Any]] = {}
    for ue_id in ue_ids:
        history = []
        for point in points:
            ue = next((item for item in point.get("ues", []) if int(item.get("ue_id", -1)) == ue_id), None)
            if ue and all(key in ue for key in ("qoe", "throughput_mbps", "delay_ms", "jitter_ms", "packet_loss_fraction")):
                history.append({"sim_time_s": point["sim_time_s"], **ue})
        current = history[-1] if history else None
        prediction = forecast(history, horizon_s=horizon_s, target_qoe=target_qoe) if len(history) >= 4 else {
            "available": False, "risk": bool(current and float(current["qoe"]) < target_qoe),
            "reason": "Insufficient per-UE history", "horizon_s": horizon_s,
        }
        if current and float(current["qoe"]) < target_qoe:
            prediction["risk"] = True
            prediction["reason"] = (prediction.get("reason", "") + "; " if prediction.get("reason") else "") + "measured UE QoE is below target"
        predictions[ue_id] = {"current": current, "forecast": prediction}
    return predictions


def assess_ue_target_state(ue_states: list[dict[str, Any]], forecasts: dict[int, dict[str, Any]],
                           target_qoe: float, expected_ue_count: int | None = None) -> dict[str, Any]:
    """Summarize complete per-UE target status and predicted downward crossings."""
    expected = int(expected_ue_count or len(ue_states))
    complete = bool(ue_states) and len(ue_states) == expected
    current_below = sorted(int(ue["ue_id"]) for ue in ue_states
                           if float(ue.get("qoe", 0.0)) < float(target_qoe))
    all_current_met = complete and not current_below
    crossings = []
    if complete:
        for ue in ue_states:
            ue_id = int(ue["ue_id"])
            fc = forecasts.get(ue_id, {}).get("forecast", {})
            if (float(ue.get("qoe", 0.0)) >= float(target_qoe)
                    and fc.get("available")
                    and fc.get("qoe") is not None
                    and float(fc["qoe"]) < float(target_qoe)):
                crossings.append(ue_id)
    return {"complete": complete, "all_current_met": all_current_met,
            "current_below_ue_ids": current_below,
            "forecast_target_crossing_ue_ids": sorted(crossings),
            "required_crossings": 2}


def assess_safety_floors(points: list[dict[str, Any]],
                         safety_floors: dict[str, float] | None = None,
                         target_qoe: float = .72,
                         warning_margin: float = SAFETY_WARNING_MARGIN) -> dict[str, Any]:
    """Flag service-specific floor breaches or short-horizon downward crossings.

    Forecasts use the existing five-window linear trend. A near-floor warning
    requires a declining trend and is a policy trigger, not a calibrated
    probability of future QoE.
    """
    floors = normalize_safety_floors(target_qoe, safety_floors)
    if not points:
        return {"available": False, "at_risk_ue_ids": [], "users": [],
                "reason": "Waiting for measured UE history."}
    latest = points[-1]
    predictions = forecast_ue_states(points, target_qoe=target_qoe)
    users = []
    for ue in latest.get("ues", []):
        ue_id = int(ue.get("ue_id", -1))
        service = str(ue.get("traffic_class", ""))
        floor = float(floors.get(service, max(0.0, float(target_qoe) - .10)))
        current = float(ue.get("qoe", 0.0))
        item = predictions.get(ue_id, {}).get("forecast", {})
        projected = item.get("qoe") if item.get("available") else None
        breached = current < floor
        projected_crossing = (not breached and projected is not None and float(projected) < floor)
        near_and_declining = (not breached and not projected_crossing and projected is not None
                              and current <= floor + float(warning_margin)
                              and float(projected) < current - .01)
        at_risk = breached or projected_crossing or near_and_declining
        if breached:
            status = "below safety floor"
        elif projected_crossing:
            status = "forecast to cross safety floor"
        elif near_and_declining:
            status = "near floor and declining"
        else:
            status = "within safety margin"
        users.append({"ue_id": ue_id, "traffic_class": service, "floor": floor,
                      "measured_qoe": current, "forecast_qoe": projected,
                      "status": status, "at_risk": at_risk,
                      "forecast_horizon_s": item.get("horizon_s")})
    expected = int(latest.get("ue_count", len(latest.get("ues", []))))
    complete = bool(users) and len(users) == expected
    at_risk_ids = sorted(row["ue_id"] for row in users if row["at_risk"])
    return {"available": complete, "complete": complete, "at_risk_ue_ids": at_risk_ids,
            "users": users, "warning_margin": float(warning_margin),
            "reason": ("One or more service floors are at risk." if at_risk_ids
                       else "No service floor breach or downward crossing is forecast.")
            if complete else "Waiting for a complete per-UE measurement."}


def set_mode(session_id: str, mode: str) -> None:
    if mode not in {"Monitor", "Recommend", "Automatic"}: raise ValueError(mode)
    folder = LIVE_ROOT / session_id
    meta = json.loads((folder / "session.json").read_text())
    meta["mode"] = mode
    _write_meta(folder, meta)
    _event(folder, {"type": "mode_changed", "mode": mode, "timestamp": _utc()})


def stop_session(session_id: str) -> None:
    folder = LIVE_ROOT / session_id
    (folder / "stop").touch()
    _event(folder, {"type": "stop_requested", "timestamp": _utc()})


def request_structural_change(session_id: str, new_config: dict[str, Any], *, approved: bool = False,
                              rationale: str = "", rollback: bool = False) -> None:
    """Request a restart epoch, keeping the logical session and its telemetry history."""
    folder = LIVE_ROOT / session_id
    meta = json.loads((folder / "session.json").read_text())
    if not meta.get("segments"):
        raise PermissionError("This session uses the earlier live runtime. Start a new session to enable restart epochs.")
    if meta.get("status") != "Running":
        raise PermissionError(f"Structural changes require fresh Running telemetry; status is {meta.get('status')}.")
    if meta.get("mode") == "Monitor":
        raise PermissionError("Monitor mode cannot apply configuration changes")
    if meta.get("mode") == "Recommend" and not approved:
        raise PermissionError("Recommend mode requires explicit approval")
    if meta.get("structural_action_pending") or (folder / "structural_request.json").exists():
        raise ValueError("A restart epoch is already pending")
    if time.time() - float(meta.get("last_structural_control_at", 0)) < 30 and not rollback:
        raise ValueError("Structural action cooldown is active")
    config = dict(meta.get("config", {}))
    config.update(new_config)
    config["ue_count"] = int(config.get("ue_count", 2))
    config["gnb_count"] = int(config.get("gnb_count", 1))
    config["bandwidth_mhz"] = float(config.get("bandwidth_mhz", 50))
    config["total_tx_power_dbm"] = float(config.get("total_tx_power_dbm", 35))
    config["ue_association_policy"] = str(config.get("ue_association_policy", "closest"))
    config["traffic_mix"] = str(config.get("traffic_mix", "legacy"))
    config["lambda_be"] = int(meta["active_parameters"]["lambda_be_pps"])
    config["lambda_ull"] = int(meta["active_parameters"]["lambda_ull_pps"])
    if config["bandwidth_mhz"] not in LIVE_BANDWIDTHS_MHZ:
        raise ValueError(f"Bandwidth must be one of {sorted(LIVE_BANDWIDTHS_MHZ)} MHz")
    if not 1 <= config["gnb_count"] <= LIVE_MAX_GNBS:
        raise ValueError(f"gNB count must be 1–{LIVE_MAX_GNBS}")
    if config["ue_association_policy"] not in {"closest", "by_traffic_class", "round_robin"}:
        raise ValueError("Unsupported UE association policy")
    if config["traffic_mix"] not in {"legacy", "gaming_video"}:
        raise ValueError("Unsupported UE traffic mix")
    if config["ue_association_policy"] != "closest" and config["gnb_count"] < 2:
        raise ValueError(f"{config['ue_association_policy']} association requires at least two gNBs")
    if not 2 <= config["ue_count"] <= 20:
        raise ValueError("UE count must be between 2 and 20")
    if not 10 <= config["total_tx_power_dbm"] <= 50:
        raise ValueError("Total transmit power must be between 10 and 50 dBm")
    if not 2500 <= config["lambda_be"] <= 20000 or not 500 <= config["lambda_ull"] <= 10000:
        raise ValueError("Offered packet rates are outside the supported live ranges")
    if all(config.get(k) == meta.get("config", {}).get(k) for k in
           ("gnb_count", "bandwidth_mhz", "total_tx_power_dbm", "ue_association_policy")):
        raise ValueError("Restart configuration is identical to the active topology/radio settings")
    current_latest = meta.get("latest") or {}
    current_ue_forecasts = meta.get("ue_forecasts", {})
    request = {"config": config, "rollback": bool(rollback), "rationale": rationale,
               "requested_at": _utc(),
               "affected_ue_ids": sorted(set(
                   [int(ue["ue_id"]) for ue in current_latest.get("ues", [])
                    if float(ue.get("qoe", 0)) < float(meta.get("target_qoe", .72))]
                   + [int(ue_id) for ue_id, item in current_ue_forecasts.items()
                      if item.get("forecast", {}).get("risk")]))}
    tmp = folder / "structural_request.tmp"
    tmp.write_text(json.dumps(request))
    tmp.replace(folder / "structural_request.json")
    meta["structural_action_pending"] = True
    meta["pending_structural_config"] = config
    meta["pending_structural_rollback"] = bool(rollback)
    meta["pending_structural_rationale"] = rationale
    meta["pending_structural_affected_ue_ids"] = request["affected_ue_ids"]
    meta["last_structural_control_at"] = time.time()
    if not rollback:
        meta["structural_actions"] = int(meta.get("structural_actions", 0)) + 1
    meta["transition_status"] = "Restart requested"
    _write_meta(folder, meta)
    _event(folder, {"type": "restart_epoch_requested", "timestamp": _utc(),
                    "previous_config": meta.get("config"), "new_config": config,
                    "rollback": bool(rollback), "rationale": rationale,
                    "affected_ue_ids": request["affected_ue_ids"],
                    "measured_before": meta.get("latest"),
                    "forecast_before": meta.get("forecast"),
                    "status": "Pending restart; no topology change claimed yet"})


def _apply_structural_request(folder: Path) -> None:
    request_path = folder / "structural_request.json"
    if not request_path.is_file():
        return
    request = json.loads(request_path.read_text())
    request_path.unlink(missing_ok=True)
    meta = json.loads((folder / "session.json").read_text())
    previous_config = dict(meta["config"])
    config = dict(request["config"])
    current_segment = int(meta.get("current_segment", 1))
    active_segment = _segment_dir(folder, current_segment)
    meta.update({"status": "Restarting", "transition_status": "Stopping current ns-3 segment",
                 "structural_action_pending": True, "configuration_verified": False})
    for item in meta.get("segments", []):
        if int(item.get("segment", -1)) == current_segment:
            item["status"] = "Stopping"
    _write_meta(folder, meta)
    _event(folder, {"type": "restart_epoch_started", "timestamp": _utc(),
                    "from_segment": current_segment, "previous_config": previous_config,
                    "new_config": config, "affected_ue_ids": request.get("affected_ue_ids", []),
                    "status": "Restarting ns-3 for startup-only parameters"})
    (active_segment / "stop").touch()
    deadline = time.time() + 180
    while _pid_alive(int(meta.get("pid") or 0), str(LIVE_BINARY)) and time.time() < deadline:
        if (folder / "stop").exists():
            break
        time.sleep(0.5)
    if (folder / "stop").exists():
        meta.update({"status": "Stopped", "transition_status": "Stopped during restart request",
                     "structural_action_pending": False})
        for item in meta.get("segments", []):
            if int(item.get("segment", -1)) == current_segment:
                item["status"] = "Stopped"
        _write_meta(folder, meta)
        _event(folder, {"type": "restart_epoch_cancelled", "timestamp": _utc(),
                        "reason": "User stopped the logical session during transition"})
        return
    if _pid_alive(int(meta.get("pid") or 0), str(LIVE_BINARY)):
        meta.update({"status": "Failed", "transition_status": "Restart failed: old simulator did not stop",
                     "structural_action_pending": False,
                     "last_action_status": "Restart failed; prior ns-3 process did not stop within 180 seconds"})
        _write_meta(folder, meta)
        _event(folder, {"type": "restart_epoch_failed", "timestamp": _utc(),
                        "error": "Old ns-3 process did not stop within 180 seconds"})
        return
    for item in meta.get("segments", []):
        if int(item.get("segment", -1)) == current_segment:
            item["status"] = "Completed"
            item["ended_at"] = _utc()
    next_segment = current_segment + 1
    before_points = read_session(meta["session_id"])["points"][-3:]
    before_ue_values: dict[str, list[float]] = {}
    for point in before_points:
        for ue in point.get("ues", []):
            before_ue_values.setdefault(str(ue["ue_id"]), []).append(float(ue.get("qoe", 0)))
    before_ue_qoe = {ue_id: sum(values) / len(values) for ue_id, values in before_ue_values.items() if values}
    try:
        proc = _launch_live_process(folder, config, next_segment, float(meta.get("target_qoe", .72)))
    except Exception as exc:
        meta.update({"status": "Failed", "transition_status": "Restart failed to launch",
                     "structural_action_pending": False,
                     "last_action_status": f"Restart launch failed: {type(exc).__name__}: {exc}"})
        _write_meta(folder, meta)
        _event(folder, {"type": "restart_epoch_failed", "timestamp": _utc(), "error": str(exc)})
        return
    segment_record = {"segment": next_segment, "config": config, "status": "Starting", "started_at": _utc(),
                      "previous_config": previous_config}
    meta["segments"].append(segment_record)
    meta.update({"status": "Starting", "transition_status": "Waiting for first telemetry from restart epoch",
                 "current_segment": next_segment, "config": config, "pid": proc.pid,
                 "configuration_verified": False,
                 "structural_trial": {"segment": next_segment, "previous_config": previous_config,
                    "new_config": config, "before_qoe": [p["qoe"] for p in before_points],
                    "before_ue_qoe": before_ue_qoe,
                    "rollback": bool(request.get("rollback")), "rollback_attempted": bool(request.get("rollback"))},
                 "structural_action_pending": True, "last_decision_sim_time_s": 0,
                 "controller_rate_actions": 0})
    _write_meta(folder, meta)
    _event(folder, {"type": "restart_epoch_launched", "timestamp": _utc(),
                    "segment": next_segment, "pid": proc.pid, "previous_config": previous_config,
                    "new_config": config, "rollback": bool(request.get("rollback")),
                    "status": "Process launched; waiting for telemetry to verify active parameters"})


def request_rate_change(session_id: str, new_rate: int, approved: bool = False,
                        rollback: bool = False, rationale: str | None = None,
                        hybrid_evaluation: dict[str, Any] | None = None,
                        decision_basis: str | None = None,
                        counterfactual_reference: dict[str, Any] | None = None) -> str:
    folder = LIVE_ROOT / session_id
    with (folder / "controller.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _queue_rate_change(session_id, new_rate, approved, rollback, rationale,
                                  hybrid_evaluation, decision_basis, counterfactual_reference)


def _control_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def request_demand_change(session_id: str, requested_pps: int, *,
                          onset_sim_time_s: float | None = None,
                          reason: str = "Scheduled offered-demand request") -> str:
    """Change requested demand in admission-control sessions, separately from the admitted-rate cap."""
    if not MIN_RATE <= int(requested_pps) <= MAX_RATE:
        raise ValueError(f"Requested demand must be {MIN_RATE}–{MAX_RATE} pps")
    folder = LIVE_ROOT / session_id
    meta = json.loads((folder / "session.json").read_text())
    if not meta.get("config", {}).get("admission_control"):
        raise PermissionError("Separate demand requests require an admission-control test session.")
    preview = meta.get("scheduled_demand_preview")
    if preview:
        if preview.get("status") not in {"scheduled", "cap_pending_acknowledgement", "cap_applied", "cap_verified"}:
            raise PermissionError("The matched demand preview is not active.")
        if int(requested_pps) != int(preview.get("requested_pps", -1)):
            raise ValueError("Requested rate does not match the armed demand preview.")
    latest_time = float((meta.get("latest") or {}).get("sim_time_s", 0))
    requested_onset = (float(onset_sim_time_s) if onset_sim_time_s is not None
                       else (float(preview.get("onset_sim_time_s", latest_time))
                             if preview else latest_time))
    if requested_onset < latest_time:
        raise ValueError("Scheduled demand onset cannot be in the past.")
    if preview and abs(requested_onset - float(preview.get("onset_sim_time_s", -1))) > 1e-6:
        raise ValueError("Demand onset does not match the armed preview.")
    if preview and requested_onset < float(preview.get("onset_sim_time_s", 0)):
        raise ValueError("Demand onset must match the armed preview.")
    if meta.get("status") != "Running":
        raise PermissionError(f"Demand changes require a running simulator; status is {meta.get('status')}.")
    # The demand schedule is an independent ns-3 input. In predictive trials it
    # must be armed while the controller is still applying its cap, before the
    # future onset. The simulator applies the cap and demand at separate ticks.
    if meta.get("demand_change_pending") or (meta.get("action_pending") and not preview):
        raise PermissionError("Wait for the current ns-3 control command to be acknowledged.")
    segment = _segment_dir(folder, int(meta.get("current_segment", 1)))
    control_path = segment / "controls.jsonl"
    seen = len(_control_rows(control_path))
    request_id = uuid.uuid4().hex
    demand_path = segment / "demand.txt"
    tmp = segment / "demand.tmp"
    tmp.write_text(f"{requested_onset:.6f} {int(requested_pps)}")
    tmp.replace(demand_path)
    meta.update({"demand_change_pending": True, "pending_demand_id": request_id,
                 "pending_demand_pps": int(requested_pps), "pending_demand_controls_seen": seen})
    _write_meta(folder, meta)
    latest = meta.get("latest") or {}
    _event(folder, {"type": "scheduled_demand_pending", "timestamp": _utc(),
                    "demand_id": request_id, "sim_time_s": latest.get("sim_time_s"),
                    "onset_sim_time_s": requested_onset,
                    "requested_pps": int(requested_pps),
                    "admission_cap_pps": meta.get("active_parameters", {}).get("lambda_be_cap_pps"),
                    "reason": reason,
                    "status": "Demand request sent to ns-3; waiting for acknowledgement"})
    return request_id


def schedule_demand_preview(session_id: str, *, requested_pps: int, onset_sim_time_s: float,
                            preemptive_cap_pps: int, reference: dict[str, Any]) -> None:
    """Arm an opt-in forecast-led admission action from a matched no-action reference run."""
    folder = LIVE_ROOT / session_id
    meta = json.loads((folder / "session.json").read_text())
    if meta.get("mode") != "Automatic" or meta.get("status") != "Running":
        raise PermissionError("Predictive demand protection requires a fresh Automatic session.")
    if not meta.get("config", {}).get("admission_control"):
        raise PermissionError("Predictive demand protection requires admission-control mode.")
    if meta.get("scheduled_demand_preview"):
        raise PermissionError("A scheduled demand preview is already recorded for this session.")
    if not MIN_RATE <= int(preemptive_cap_pps) < int(requested_pps) <= MAX_RATE:
        raise ValueError("The pre-emptive cap must be within bounds and below requested demand.")
    if not reference.get("session_id") or not reference.get("per_ue_qoe"):
        raise ValueError("A measured no-action ns-3 reference is required.")
    below = sorted(int(ue_id) for ue_id, value in reference["per_ue_qoe"].items()
                   if float(value) < float(meta.get("target_qoe", .72)))
    if not below:
        raise ValueError("The matched no-action reference did not show QoE degradation below target.")
    latest = meta.get("latest") or {}
    current_ues = latest.get("ues", [])
    expected = int(meta.get("config", {}).get("ue_count", 0))
    if len(current_ues) != expected or any(float(ue.get("qoe", 0)) < float(meta.get("target_qoe", .72))
                                           for ue in current_ues):
        raise PermissionError("Every configured UE must currently be measured above target before arming protection.")
    if float(onset_sim_time_s) < float(latest.get("sim_time_s", 0)) + 3.0:
        raise ValueError("Scheduled demand must begin at least 3 simulated seconds in the future.")
    preview = {"status": "scheduled", "requested_pps": int(requested_pps),
               "onset_sim_time_s": float(onset_sim_time_s),
               "preemptive_cap_pps": int(preemptive_cap_pps),
               "reference": reference, "reference_at_risk_ue_ids": below,
               "created_at": _utc()}
    meta["scheduled_demand_preview"] = preview
    _write_meta(folder, meta)
    _event(folder, {"type": "scheduled_demand_preview", "timestamp": _utc(),
                    "sim_time_s": latest.get("sim_time_s"), "onset_sim_time_s": float(onset_sim_time_s),
                    "requested_pps": int(requested_pps), "preemptive_cap_pps": int(preemptive_cap_pps),
                    "reference_session_id": reference["session_id"],
                    "reference_at_risk_ue_ids": below,
                    "reference_per_ue_qoe": reference["per_ue_qoe"],
                    "status": "Known upcoming demand risk armed; waiting for automatic controller decision"})


def _queue_rate_change(session_id: str, new_rate: int, approved: bool = False,
                       rollback: bool = False, rationale: str | None = None,
                       hybrid_evaluation: dict[str, Any] | None = None,
                       decision_basis: str | None = None,
                       counterfactual_reference: dict[str, Any] | None = None) -> str:
    """Queue a rate change while the caller holds controller.lock."""
    if not MIN_RATE <= int(new_rate) <= MAX_RATE: raise ValueError(f"Rate must be {MIN_RATE}–{MAX_RATE} pps")
    folder = LIVE_ROOT / session_id
    meta = json.loads((folder / "session.json").read_text())
    if meta.get("config", {}).get("traffic_mix", "legacy") != "legacy":
        raise PermissionError("Live packet pacing is supported only for the legacy constant-rate UE mix; NGMN traffic uses restart-based network actions.")
    if meta.get("status") != "Running":
        raise PermissionError(f"Live action requires a running, fresh simulator; status is {meta.get('status')}.")
    admission_control = bool(meta.get("config", {}).get("admission_control", False))
    current = int(meta["active_parameters"].get(
        "lambda_be_cap_pps" if admission_control else "lambda_be_pps",
        meta["active_parameters"]["lambda_be_pps"]))
    if int(new_rate) == current: raise ValueError("New offered load equals current value")
    if meta.get("action_pending") or meta.get("test_impairment_pending"):
        raise PermissionError("Another simulator control command is still awaiting acknowledgement")
    if meta.get("action_applied_sim_time_s") is not None and not meta.get("action_verified"):
        raise PermissionError("Wait for the previous rate change to be verified before requesting another")
    if meta["mode"] == "Monitor": raise PermissionError("Monitor mode never proposes or applies actions")
    if meta["mode"] == "Recommend" and not approved: raise PermissionError("Recommend mode requires explicit approval")
    if time.time() - float(meta.get("last_control_at",0)) < 10 and not rollback: raise ValueError("10-second action cooldown is active")
    active_segment = _segment_dir(folder, int(meta.get("current_segment", 1)))
    control_path = active_segment / "controls.jsonl"
    controls_seen = len(_control_rows(control_path))
    action_id = uuid.uuid4().hex
    control_action = ("set_best_effort_admission_cap" if admission_control
                      else "set_best_effort_packets_per_second")
    meta.update({"action_pending": True, "pending_action_id": action_id,
        "pending_action_kind": "admission_cap" if admission_control else "rate_change",
        "pending_control_action": control_action, "pending_controls_seen": controls_seen,
        "pending_previous_pps": current, "pending_new_pps": int(new_rate),
        "pending_is_rollback": bool(rollback), "action_verified": False,
        "pending_decision_basis": decision_basis or "operator_approved",
        "pending_counterfactual_reference": counterfactual_reference,
        "action_applied_sim_time_s": None, "action_verification_status": "pending_acknowledgement",
        "last_action_status": "Rate change requested; waiting for ns-3 acknowledgement"})
    _write_meta(folder, meta)
    command = active_segment / "command.txt"
    tmp = active_segment / "command.tmp"
    tmp.write_text(str(int(new_rate)))
    tmp.replace(command)
    meta["last_control_at"] = time.time()
    _write_meta(folder, meta)
    live_data = read_session(session_id)
    segment_id = f"segment_{int(meta.get('current_segment', 1)):03d}"
    segment_points = [p for p in live_data["points"] if p.get("segment_id") == segment_id]
    ue_forecasts = forecast_ue_states(segment_points, target_qoe=float(meta.get("target_qoe", .72)))
    affected_ue_ids = [ue_id for ue_id, item in ue_forecasts.items()
                       if item.get("current") and (float(item["current"].get("qoe", 0)) < float(meta.get("target_qoe", .72))
                           or item.get("forecast", {}).get("risk"))]
    reason = rationale or ("Restore the last pre-action offered load at the user's request."
                           if rollback else "User-approved recommendation to reduce offered load; QoE impact is uncertain.")
    _event(folder, {"type": "rollback_pending" if rollback else "action_pending",
                    "action_id": action_id,
                    "action": "Best-effort admission cap" if admission_control else "Best-effort offered packet rate",
                    "previous_pps": current, "new_pps": int(new_rate), "timestamp": _utc(),
                    "rollback": bool(rollback),
                    "affected_ue_ids": affected_ue_ids,
                    "reason": reason, "trigger_measurement": live_data["meta"].get("latest"),
                    "trigger_forecast": live_data["meta"].get("forecast"),
                    "decision_basis": decision_basis or "operator_approved",
                    "measured_qoe_at_decision": ((live_data["meta"].get("latest") or {}).get("qoe")),
                    "target_qoe": float(meta.get("target_qoe", .72)),
                    "counterfactual_reference": counterfactual_reference,
                    "hybrid_evaluation": hybrid_evaluation,
                    "prediction": ("Requested demand is admitted up to this cap; unadmitted demand is reported separately."
                                   if admission_control else
                                   "Traffic source pacing changes live; QoE effect is not assumed.")})
    meta["rate_action_before_ue_qoe"] = {str(ue_id): float(item["current"].get("qoe", 0))
        for ue_id, item in ue_forecasts.items() if item.get("current")}
    meta["rate_action_affected_ue_ids"] = affected_ue_ids
    _write_meta(folder, meta)
    return action_id


def test_load_shock_eligibility(session_id: str) -> dict[str, Any]:
    """Return whether a one-shot test load shock is safe to request now."""
    folder = LIVE_ROOT / session_id
    try:
        data = read_session(session_id)
        meta = data["meta"]
    except (OSError, ValueError, KeyError) as exc:
        return {"allowed": False, "reason": f"Session data unavailable: {exc}"}
    if meta.get("status") != "Running":
        return {"allowed": False, "reason": "The simulation must be Running with fresh telemetry."}
    if meta.get("mode") != "Automatic":
        return {"allowed": False, "reason": "Select Automatic mode so the controller can respond to the test disturbance."}
    if meta.get("config", {}).get("traffic_mix", "legacy") != "legacy":
        return {"allowed": False, "reason": "The test load shock is supported only for the legacy constant-rate traffic mix."}
    if meta.get("test_impairment_triggered") or meta.get("test_impairment_pending"):
        return {"allowed": False, "reason": "A test load shock has already been triggered in this session."}
    if meta.get("action_pending") or meta.get("structural_action_pending"):
        return {"allowed": False, "reason": "Wait for the current configuration action to finish first."}
    if meta.get("action_applied_sim_time_s") is not None and not meta.get("action_verified"):
        return {"allowed": False, "reason": "Wait for the previous rate change to be verified first."}
    if int(meta.get("controller_rate_actions", 0)) >= 1:
        return {"allowed": False, "reason": "The controller's per-segment live rate action has already been used."}
    if int(meta.get("active_parameters", {}).get("lambda_be_pps", MAX_RATE)) >= MAX_RATE:
        return {"allowed": False, "reason": f"The offered rate is already at its supported maximum ({MAX_RATE:,} packets/s per UE)."}
    expected = int(meta.get("config", {}).get("ue_count", 0))
    target = float(meta.get("target_qoe", .72))
    segment_id = f"segment_{int(meta.get('current_segment', 1)):03d}"
    points = [point for point in data["points"] if point.get("segment_id") == segment_id]
    if len(points) < 3:
        return {"allowed": False, "reason": "Need three complete measured windows before triggering the test disturbance."}
    expected_ids = set(range(1, expected + 1))
    for point in points[-3:]:
        ues = point.get("ues", [])
        ids = [int(ue.get("ue_id", -1)) for ue in ues]
        if len(ids) != expected or set(ids) != expected_ids:
            return {"allowed": False, "reason": "The latest three windows must each contain one measurement for every configured UE."}
        if any(float(ue.get("qoe", 0.0)) < target for ue in ues):
            return {"allowed": False, "reason": f"Every UE must meet the measured QoE target ({target:.3f}) for three consecutive windows."}
        measured_rate = point.get("lambda_be_pps")
        active_rate = int(meta.get("active_parameters", {}).get("lambda_be_pps", -1))
        if measured_rate is None or int(measured_rate) != active_rate:
            return {"allowed": False, "reason": "The offered rate must be stable and confirmed in each of the three above-target windows."}
    if time.time() - float(meta.get("last_control_at", 0)) < 10:
        return {"allowed": False, "reason": "Wait for the controller rate-action cooldown before the test disturbance."}
    return {"allowed": True, "reason": "All UEs have met target in three consecutive fresh measurement windows.",
            "target_qoe": target, "window_count": 3, "new_rate_pps": MAX_RATE}


def request_test_load_shock(session_id: str) -> str:
    """Raise legacy offered load once, after all UEs have stably met target.

    This is a test disturbance, not a network impairment or controller action.
    The increase is acknowledged separately so it cannot be mistaken for a
    controller rate-change acknowledgement or QoE verification.
    """
    folder = LIVE_ROOT / session_id
    with (folder / "controller.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        eligibility = test_load_shock_eligibility(session_id)
        if not eligibility.get("allowed"):
            raise PermissionError(eligibility.get("reason", "Test load shock is not currently allowed"))
        meta = json.loads((folder / "session.json").read_text())
        active_segment = _segment_dir(folder, int(meta.get("current_segment", 1)))
        previous = int(meta["active_parameters"]["lambda_be_pps"])
        new_rate = int(eligibility["new_rate_pps"])
        action_id = uuid.uuid4().hex
        controls_seen = len(_control_rows(active_segment / "controls.jsonl"))
        meta.update({"test_impairment_triggered": True, "test_impairment_pending": True,
            "test_impairment_action_id": action_id, "test_impairment_controls_seen": controls_seen,
            "test_impairment_previous_pps": previous, "test_impairment_new_pps": new_rate,
            "test_impairment_status": "pending_acknowledgement",
            "last_action_status": "Test-only load shock requested; waiting for ns-3 acknowledgement"})
        _write_meta(folder, meta)
        command = active_segment / "command.txt"
        tmp = active_segment / "command.tmp"
        tmp.write_text(str(new_rate))
        tmp.replace(command)
        data = read_session(session_id)
        latest = data["points"][-1]
        _event(folder, {"type": "test_impairment_pending", "action_id": action_id,
            "timestamp": _utc(), "sim_time_s": latest.get("sim_time_s"),
            "target_qoe": eligibility["target_qoe"], "eligible_windows": eligibility["window_count"],
            "previous_pps": previous, "new_pps": new_rate,
            "triggering_ue_measurements": latest.get("ues", []),
            "status": "Test-only workload shock requested; waiting for ns-3 acknowledgement",
            "description": "Offered-load increase; this changes traffic demand, not channel conditions."})
        return action_id


def _event(folder: Path, event: dict[str, Any]) -> None:
    with (folder / "events.jsonl").open("a") as f:
        f.write(json.dumps(event) + "\n")


def _apply_scheduled_demand_preview(session_id: str, meta: dict[str, Any],
                                    latest: dict[str, Any]) -> bool:
    """Apply a tested admission cap before a known demand request, using a matched no-action run."""
    preview = meta.get("scheduled_demand_preview") or {}
    if preview.get("status") != "scheduled":
        return False
    folder = LIVE_ROOT / session_id
    sim_time = float(latest.get("sim_time_s", 0))
    onset = float(preview.get("onset_sim_time_s", 0))
    if sim_time >= onset:
        preview.update({"status": "missed", "error": "Controller did not apply a cap before scheduled demand onset"})
        meta["scheduled_demand_preview"] = preview
        _write_meta(folder, meta)
        _event(folder, {"type": "scheduled_demand_missed", "timestamp": _utc(),
                        "sim_time_s": sim_time, "onset_sim_time_s": onset,
                        "reason": preview["error"]})
        return True
    target = float(meta.get("target_qoe", .72))
    expected = int(meta.get("config", {}).get("ue_count", 0))
    ues = latest.get("ues", [])
    if len(ues) != expected or any(float(ue.get("qoe", 0)) < target for ue in ues):
        return False
    reference = preview.get("reference") or {}
    per_ue_reference = {str(key): float(value)
                        for key, value in reference.get("per_ue_qoe", {}).items()}
    threatened = sorted(int(ue_id) for ue_id, value in per_ue_reference.items()
                        if value < target)
    if not threatened:
        preview.update({"status": "rejected", "error": "Matched no-action reference did not breach the target"})
        meta["scheduled_demand_preview"] = preview
        _write_meta(folder, meta)
        return True
    cap = int(preview.get("preemptive_cap_pps", -1))
    current_cap = int(meta.get("active_parameters", {}).get("lambda_be_cap_pps",
                    meta.get("active_parameters", {}).get("lambda_be_pps", 0)))
    if (not meta.get("config", {}).get("admission_control") or cap < MIN_RATE
            or cap >= current_cap or int(meta.get("controller_rate_actions", 0)) >= 1):
        preview.update({"status": "rejected", "error": "The proposed admission cap is unsupported or the action budget is used"})
        meta["scheduled_demand_preview"] = preview
        _write_meta(folder, meta)
        return True
    reference_record = {
        "session_id": reference.get("session_id"),
        "source": "matched no-action ns-3 run; measured outcome, not a model estimate",
        "requested_pps": int(preview.get("requested_pps", 0)),
        "per_ue_qoe": per_ue_reference,
        "target_qoe": target,
        "at_risk_ue_ids": threatened,
        "mean_ue_qoe": (sum(per_ue_reference.values()) / len(per_ue_reference)
                        if per_ue_reference else None),
    }
    rationale = (
        f"PRE-EMPTIVE: matched no-action ns-3 session {reference_record['session_id']} measured "
        f"UE(s) {', '.join(map(str, threatened))} below QoE target {target:.3f} when offered demand "
        f"reached {int(preview['requested_pps']):,} pps/UE. Current measured mean UE QoE is "
        f"{mean_user_qoe(ues):.3f}; apply admission cap {cap:,} pps/UE before the request at "
        f"simulation time {onset:.1f}s. Unadmitted requested packets will be shown separately."
    )
    action_id = request_rate_change(
        session_id, cap, approved=True, rationale=rationale,
        decision_basis="predictive_preemptive", counterfactual_reference=reference_record)
    meta = json.loads((folder / "session.json").read_text())
    preview = meta.get("scheduled_demand_preview", preview)
    preview.update({"status": "cap_pending_acknowledgement", "action_id": action_id,
                    "preemptive_action_sim_time_s": sim_time,
                    "measured_qoe_at_action": mean_user_qoe(ues),
                    "counterfactual_reference": reference_record,
                    "affected_ue_ids": threatened})
    meta["scheduled_demand_preview"] = preview
    meta["controller_rate_actions"] = int(meta.get("controller_rate_actions", 0)) + 1
    meta["last_decision_sim_time_s"] = sim_time
    _write_meta(folder, meta)
    _event(folder, {"type": "controller_decision", "timestamp": _utc(),
                    "sim_time_s": sim_time, "action": f"Set admission cap to {cap:,} pps/UE",
                    "decision_basis": "predictive_preemptive",
                    "measured_qoe_at_decision": mean_user_qoe(ues), "target_qoe": target,
                    "scheduled_demand_onset_sim_time_s": onset,
                    "requested_demand_pps": int(preview["requested_pps"]),
                    "counterfactual_reference": reference_record,
                    "affected_ue_ids": threatened, "reason": rationale,
                    "objective": "prevent the measured no-action reference from crossing per-UE targets"})
    return True


def control_cycle(session_id: str) -> None:
    """Choose a target-directed live action, then use a restart epoch for structural controls."""
    data = read_session(session_id)
    meta, points = data["meta"], data["points"]
    current_segment = f"segment_{int(meta.get('current_segment', 1)):03d}"
    points = [p for p in points if p.get("segment_id") == current_segment]
    awaiting_rate_verification = (meta.get("action_applied_sim_time_s") is not None
                                  and not meta.get("action_verified"))
    if (meta.get("mode") != "Automatic" or meta.get("status") != "Running" or len(points) < 4
            or meta.get("action_pending") or meta.get("structural_action_pending")
            or meta.get("test_impairment_pending")
            or awaiting_rate_verification):
        return
    fc = forecast(points, target_qoe=float(meta.get("target_qoe", .72)))
    target = float(meta.get("target_qoe", .72))
    latest = points[-1]
    if _apply_scheduled_demand_preview(session_id, meta, latest):
        return
    meta["recent_points"] = points
    ue_forecasts = forecast_ue_states(points, target_qoe=target)
    ue_target_state = assess_ue_target_state(
        latest.get("ues", []), ue_forecasts, target,
        expected_ue_count=int(meta.get("config", {}).get("ue_count", 0)) or None)
    floor_state = assess_safety_floors(
        points, meta.get("safety_floors"), target,
        float(meta.get("safety_warning_margin", SAFETY_WARNING_MARGIN)))
    floor_risk_ids = set(floor_state.get("at_risk_ue_ids", []))
    meta["safety_floor_assessment"] = floor_state
    _write_meta(LIVE_ROOT / session_id, meta)
    # Do not make decisions from a partial UE snapshot, or disturb a healthy
    # configuration for one user's non-target forecast fluctuation.
    if latest.get("ues") and not ue_target_state["complete"]:
        return
    if ue_target_state["all_current_met"]:
        if (len(ue_target_state["forecast_target_crossing_ue_ids"]) < ue_target_state["required_crossings"]
                and not floor_risk_ids):
            return
    ue_risk = any(item.get("forecast", {}).get("risk") for item in ue_forecasts.values())
    below_target = any(item.get("current") and float(item["current"].get("qoe", 0)) < target
                       for item in ue_forecasts.values()) or float(latest["qoe"]) < target
    if not fc.get("available") or (not fc.get("risk") and not below_target and not ue_risk and not floor_risk_ids):
        return
    decision_at = float(latest["sim_time_s"])
    if decision_at <= float(meta.get("last_decision_sim_time_s", 0)):
        return
    rate = int(meta["active_parameters"]["lambda_be_pps"])
    proposal = recommend_control_action(meta, latest, fc,
                                        safety_floor_risk_ue_ids=sorted(floor_risk_ids))
    measured_below_target = (float(latest.get("qoe", 0.0)) < target
                             or bool(ue_target_state["current_below_ue_ids"]))
    floor_users = {int(row["ue_id"]): row for row in floor_state.get("users", [])}
    measured_below_floor = any(bool(floor_users.get(ue_id, {}).get("status") == "below safety floor")
                               for ue_id in floor_risk_ids)
    predictive_before_breach = bool(
        not measured_below_target
        and (fc.get("risk") or ue_target_state["forecast_target_crossing_ue_ids"])
    )
    decision_basis = ("safety_floor_recovery" if measured_below_floor else
                      "safety_floor_preemptive" if floor_risk_ids and not measured_below_floor else
                      "predictive_preemptive" if predictive_before_breach
                      else "reactive_target_recovery" if measured_below_target
                      else "trend_warning")
    rate_cooldown = time.time() - float(meta.get("last_control_at", 0)) >= 10
    if proposal and proposal["kind"] == "rate" and rate_cooldown:
        new_rate = int(proposal["value"])
        reason = proposal["reason"]
        request_rate_change(session_id, new_rate, approved=True, rationale=reason,
                            hybrid_evaluation=proposal.get("hybrid_evaluation"),
                            decision_basis=decision_basis)
        meta = json.loads((LIVE_ROOT / session_id / "session.json").read_text())
        meta["controller_rate_actions"] = int(meta.get("controller_rate_actions", 0)) + 1
        _write_meta(LIVE_ROOT / session_id, meta)
        action = proposal["action"]
    else:
        candidate = proposal.get("config") if proposal and proposal["kind"] == "restart" else None
        candidate_reason = proposal.get("reason", "No action has a defensible measured basis") if proposal else "No forecasted target threat; retain the measured-safe configuration"
        reason = candidate_reason
        if candidate and int(meta.get("structural_actions", 0)) < 3 and time.time() - float(meta.get("last_structural_control_at", 0)) >= 30:
            request_structural_change(session_id, candidate, approved=True,
                rationale=f"QoE {float(latest['qoe']):.3f} is below target {target:.3f} or forecast to cross it; {candidate_reason}. This is a measured restart epoch, not a predicted gain.")
            action = f"Restart epoch: {candidate_reason}"
        else:
            meta["last_decision_sim_time_s"] = decision_at
            _write_meta(LIVE_ROOT/session_id, meta)
            reason = candidate_reason if not candidate else "No additional restart epoch is permitted by the action budget/cooldown."
            _event(LIVE_ROOT/session_id, {"type": "controller_decision", "timestamp": _utc(),
                "sim_time_s": decision_at, "forecast": fc, "action": "No action",
                "decision_basis": decision_basis,
                "measured_qoe_at_decision": float(latest.get("qoe", 0.0)),
                "target_qoe": target,
                "safety_floor_risk_ue_ids": sorted(floor_risk_ids),
                "safety_floors": meta.get("safety_floors", default_safety_floors(target)),
                "reason": reason, "objective": "reach target" if below_target else "maintain target",
                "hybrid_evaluation": proposal.get("hybrid_evaluation") if proposal else None,
                "rejected_alternatives": [{"action": "no action", "reason": "Only when no bounded action can address the observed QoE component."}]})
            return
    meta = json.loads((LIVE_ROOT / session_id / "session.json").read_text())
    meta["last_decision_sim_time_s"] = decision_at
    _write_meta(LIVE_ROOT / session_id, meta)
    _event(LIVE_ROOT/session_id, {"type": "controller_decision", "timestamp": _utc(),
        "sim_time_s": decision_at, "forecast": fc, "action": action,
        "decision_basis": decision_basis,
        "measured_qoe_at_decision": float(latest.get("qoe", 0.0)),
        "target_qoe": target,
        "objective": "reach target" if below_target else "maintain target",
        "affected_ue_ids": proposal.get("affected_ue_ids", []) if proposal else [],
        "safety_floor_risk_ue_ids": sorted(floor_risk_ids),
        "safety_floors": meta.get("safety_floors", default_safety_floors(target)),
        "ue_forecasts": {str(k): v["forecast"] for k, v in ue_forecasts.items()},
        "reason": reason, "hybrid_evaluation": proposal.get("hybrid_evaluation") if proposal else None,
        "rejected_alternatives": [
            {"action": "structural restart", "reason": "Considered after the per-segment live rate adjustment has been tried or is unavailable."},
            {"action": "no action", "reason": "QoE is below target or forecast to violate maintenance margin."}]})


def recommend_control_action(meta: dict[str, Any], latest: dict[str, Any],
                             forecast_data: dict[str, Any], *,
                             safety_floor_risk_ue_ids: list[int] | None = None) -> dict[str, Any] | None:
    """Protect the worst-scoring UEs; after target attainment act on per-user forecast risk."""
    target = float(meta.get("target_qoe", .72))
    ue_predictions = forecast_ue_states(meta.get("recent_points", [latest]), target_qoe=target)
    ue_target_state = assess_ue_target_state(
        latest.get("ues", []), ue_predictions, target,
        expected_ue_count=int(meta.get("config", {}).get("ue_count", 0)) or None)
    if latest.get("ues") and not ue_target_state["complete"]:
        return {"kind": "none", "affected_ue_ids": [],
                "reason": "Waiting for a complete, consistent per-UE measurement before deciding."}
    healthy_maintenance = ue_target_state["all_current_met"]
    crossing_ids = set(ue_target_state["forecast_target_crossing_ue_ids"])
    floor_ids = set(int(uid) for uid in (safety_floor_risk_ue_ids or []))
    if (healthy_maintenance and len(crossing_ids) < ue_target_state["required_crossings"]
            and not floor_ids):
        return {"kind": "none", "affected_ue_ids": [],
                "reason": ("All UEs currently meet target. Holding this configuration; "
                           "automatic action requires two target crossings or a service safety-floor warning.")}
    at_risk = [(ue_id, item) for ue_id, item in ue_predictions.items()
               if float(item["current"].get("qoe", 0)) < target
               or bool(item["forecast"].get("risk"))]
    if healthy_maintenance:
        at_risk = [(ue_id, item) for ue_id, item in at_risk if ue_id in crossing_ids | floor_ids]
    below_ue_ids = [ue_id for ue_id, item in at_risk if float(item["current"].get("qoe", 0)) < target]
    forecast_ue_ids = [ue_id for ue_id, item in at_risk if item["forecast"].get("available") and item["forecast"].get("risk")]
    below = bool(below_ue_ids) or float(latest.get("qoe", 0)) < target
    threatened = bool(at_risk) or bool(forecast_data.get("risk"))
    if not below and not threatened:
        return None
    affected_ids = sorted(set(below_ue_ids + forecast_ue_ids))
    risk_items = [item["current"] for ue_id, item in at_risk if ue_id in affected_ids and item.get("current")]
    if risk_items:
        delay_rising = any(float(item.get("delay_ms", 0)) >= 50 for item in risk_items)
        loss_rising = any(float(item.get("packet_loss_fraction", 0)) >= .05 for item in risk_items)
        jitter_rising = any(float(item.get("jitter_ms", 0)) >= 20 for item in risk_items)
        ue_throughput = min(float(item.get("throughput_mbps", 0)) for item in risk_items)
    else:
        delay_rising = float(forecast_data.get("delay_ms", 0)) >= 50
        loss_rising = float(forecast_data.get("packet_loss_fraction", 0)) >= .05
        jitter_rising = float(forecast_data.get("jitter_ms", 0)) >= 20
        ue_throughput = float(forecast_data.get("throughput_mbps", 0))
    congested = delay_rising or loss_rising or jitter_rising
    rate = int(meta.get("active_parameters", {}).get("lambda_be_pps", 10000))
    user_context = (f"Affected UE(s): {', '.join(map(str, affected_ids))}. " if affected_ids else "")
    if floor_ids:
        floors = normalize_safety_floors(target, meta.get("safety_floors"))
        current_by_id = {int(ue.get("ue_id", -1)): ue for ue in latest.get("ues", [])}
        floor_details = []
        for ue_id in sorted(floor_ids):
            ue = current_by_id.get(ue_id, {})
            service = str(ue.get("traffic_class", "unknown"))
            item = ue_predictions.get(ue_id, {}).get("forecast", {})
            projected = item.get("qoe")
            projection_text = f", +{item.get('horizon_s', 5)} s trend {float(projected):.3f}" if projected is not None else ""
            floor_details.append(
                f"UE {ue_id} ({service}) measured {float(ue.get('qoe', 0)):.3f}, "
                f"floor {floors.get(service, max(0.0, target - .10)):.3f}{projection_text}"
            )
        user_context += "Service safety-floor warning: " + "; ".join(floor_details) + ". "
    if meta.get("config", {}).get("traffic_mix", "legacy") == "gaming_video":
        candidate, reason = _structural_candidate(meta, latest, forecast_data, congested)
        if candidate:
            reason = "The NGMN gaming/video generators have no verified live-rate control; " + reason
            return {"kind": "restart", "config": candidate, "reason": user_context + reason,
                    "affected_ue_ids": affected_ids, "action": f"Restart epoch: {reason}"}
        return {"kind": "none", "affected_ue_ids": affected_ids,
                "reason": user_context + reason}
    rate_action_used = int(meta.get("controller_rate_actions", 0)) >= 1
    if congested and rate > MIN_RATE and not rate_action_used:
        new = max(MIN_RATE, rate // 2)
        hybrid = compare_rate_candidate(meta, new)
        if hybrid.get("available") and not hybrid.get("candidate_supported"):
            return {"kind": "none", "affected_ue_ids": affected_ids,
                    "reason": user_context + "Live trends suggested reducing offered load, but the hybrid model rejected it: " + hybrid["reason"],
                    "hybrid_evaluation": hybrid}
        if hybrid.get("available"):
            if hybrid.get("hybrid_estimated_mean_qoe_after") is not None:
                hybrid_summary = (f" Hybrid estimate (not measurement): current measured mean QoE "
                    f"{hybrid['measured_mean_qoe_before']:.3f}; applying the dataset model's "
                    f"{hybrid['predicted_mean_delta']:+.3f} predicted change implies "
                    f"{hybrid['hybrid_estimated_mean_qoe_after']:.3f}; worst-profile change "
                    f"{hybrid['predicted_worst_delta']:+.3f}.")
            else:
                hybrid_summary = (f" Hybrid model estimate (not measurement): per-profile mean QoE "
                    f"{hybrid['predicted_mean_qoe_before']:.3f}→{hybrid['predicted_mean_qoe_after']:.3f}; "
                    f"worst-profile change {hybrid['predicted_worst_delta']:+.3f}.")
        else:
            hybrid_summary = f" Hybrid model not applied: {hybrid.get('reason', 'unavailable')}"
        return {"kind": "rate", "value": new, "action": f"Reduce best-effort offered load to {new} packets/s per UE",
                "affected_ue_ids": affected_ids,
                "hybrid_evaluation": hybrid, "reason": user_context + "The lowest-scoring UE(s) are below target or forecast at risk, with delay/loss/jitter indicating congestion." + hybrid_summary}
    if not congested and ue_throughput < 50 and rate < MAX_RATE and not rate_action_used:
        new = min(MAX_RATE, max(rate + 2500, int(rate * 1.25)))
        hybrid = compare_rate_candidate(meta, new)
        if hybrid.get("available") and not hybrid.get("candidate_supported"):
            return {"kind": "none", "affected_ue_ids": affected_ids,
                    "reason": user_context + "Live trends suggested increasing offered load, but the hybrid model rejected it: " + hybrid["reason"],
                    "hybrid_evaluation": hybrid}
        if hybrid.get("available"):
            if hybrid.get("hybrid_estimated_mean_qoe_after") is not None:
                hybrid_summary = (f" Hybrid estimate (not measurement): current measured mean QoE "
                    f"{hybrid['measured_mean_qoe_before']:.3f}; applying the dataset model's "
                    f"{hybrid['predicted_mean_delta']:+.3f} predicted change implies "
                    f"{hybrid['hybrid_estimated_mean_qoe_after']:.3f}; worst-profile change "
                    f"{hybrid['predicted_worst_delta']:+.3f}.")
            else:
                hybrid_summary = (f" Hybrid model estimate (not measurement): per-profile mean QoE "
                    f"{hybrid['predicted_mean_qoe_before']:.3f}→{hybrid['predicted_mean_qoe_after']:.3f}; "
                    f"worst-profile change {hybrid['predicted_worst_delta']:+.3f}.")
        else:
            hybrid_summary = f" Hybrid model not applied: {hybrid.get('reason', 'unavailable')}"
        return {"kind": "rate", "value": new, "action": f"Increase best-effort offered load to {new} packets/s per UE",
                "affected_ue_ids": affected_ids,
                "reason": user_context + "QoE is below target or forecast to cross it; congestion indicators are controlled and the weakest UE's throughput component is below its 50 Mbps cap." + hybrid_summary,
                "hybrid_evaluation": hybrid}
    candidate, reason = _structural_candidate(meta, latest, forecast_data, congested)
    if candidate:
        if rate_action_used:
            reason = "The live rate adjustment has been applied but QoE risk remains; " + reason
        return {"kind": "restart", "config": candidate, "reason": user_context + reason,
                "affected_ue_ids": affected_ids,
                "action": f"Restart epoch: {reason}"}
    return {"kind": "none", "affected_ue_ids": affected_ids, "reason": reason}


def _structural_candidate(meta: dict[str, Any], latest: dict[str, Any], forecast_data: dict[str, Any],
                          congested: bool) -> tuple[dict[str, Any] | None, str]:
    if not meta.get("segments"):
        return None, "This older session must be restarted before structural epochs are available"
    config = dict(meta.get("config", {}))
    tried = {tuple(sorted((k, v) for k, v in item.get("config", {}).items()
                           if k in {"gnb_count", "bandwidth_mhz", "ue_association_policy"}))
             for item in meta.get("segments", [])}
    candidates: list[tuple[dict[str, Any], str]] = []
    next_count = int(config.get("gnb_count", 1)) + 1
    if congested and next_count <= LIVE_MAX_GNBS:
        candidates.append(({"gnb_count": next_count, "ue_association_policy": "closest"},
                           f"increase gNB count to {next_count} and rebuild serving topology"))
    if ((not congested and float(forecast_data.get("throughput_mbps", 0)) < 50)
            or (congested and next_count > LIVE_MAX_GNBS)) and float(config.get("bandwidth_mhz", 50)) < max(LIVE_BANDWIDTHS_MHZ):
        wider = min(v for v in LIVE_BANDWIDTHS_MHZ if v > float(config.get("bandwidth_mhz", 50)))
        candidates.append(({"bandwidth_mhz": wider}, f"increase Band 1 bandwidth to {wider:g} MHz"))
    if not congested and next_count <= LIVE_MAX_GNBS:
        candidates.append(({"gnb_count": next_count, "ue_association_policy": "closest"},
                           f"increase gNB count to {next_count} and rebuild serving topology"))
    if int(config.get("gnb_count", 1)) >= 2 and config.get("ue_association_policy") == "closest":
        candidates.append(({"ue_association_policy": "round_robin"},
                           "rebuild UE association using supported round-robin placement"))
    for candidate, reason in candidates:
        complete = dict(config)
        complete.update(candidate)
        fingerprint = tuple(sorted((k, v) for k, v in complete.items()
                                   if k in {"gnb_count", "bandwidth_mhz", "ue_association_policy"}))
        if fingerprint not in tried:
            return candidate, reason
    return None, "No untried, bounded structural candidate remains; retaining the last measured configuration"


def refresh_action_status(session_id: str) -> None:
    folder = LIVE_ROOT / session_id
    with (folder / "controller.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _refresh_action_status_locked(session_id)


def _refresh_action_status_locked(session_id: str) -> None:
    folder = LIVE_ROOT / session_id
    meta = json.loads((folder / "session.json").read_text())
    active_segment = _segment_dir(folder, int(meta.get("current_segment", 1)))
    control_file = active_segment / "controls.jsonl"
    try:
        rows = [json.loads(line) for line in control_file.read_text().splitlines() if line]
    except (OSError, ValueError): rows=[]
    if meta.get("demand_change_pending"):
        if meta.get("status") in {"Failed", "Stopped"}:
            meta["demand_change_pending"] = False
            _write_meta(folder, meta)
            _event(folder, {"type": "scheduled_demand_failed", "timestamp": _utc(),
                            "demand_id": meta.get("pending_demand_id"),
                            "error": f"Simulator status is {meta.get('status')} before demand acknowledgement"})
            return
        demand_start = int(meta.get("pending_demand_controls_seen", 0))
        demand_rows = rows[demand_start:]
        expected_demand = int(meta.get("pending_demand_pps", -1))
        demand_ack = next((row for row in demand_rows
                           if row.get("action") == "set_requested_best_effort_demand"
                           and int(row.get("value", -2)) == expected_demand), None)
        if demand_ack is not None:
            index = demand_start + demand_rows.index(demand_ack)
            cap = int(meta.get("active_parameters", {}).get("lambda_be_cap_pps", MAX_RATE))
            admitted = min(expected_demand, cap)
            meta["demand_change_pending"] = False
            meta["controls_seen"] = index + 1
            meta["active_parameters"]["lambda_be_requested_pps"] = expected_demand
            preview = meta.get("scheduled_demand_preview")
            if preview:
                preview["status"] = "demand_applied"
                preview["actual_onset_sim_time_s"] = float(demand_ack.get("sim_time_s", 0))
                preview["admitted_rate_pps"] = admitted
            _write_meta(folder, meta)
            _event(folder, {"type": "scheduled_demand_applied", "timestamp": _utc(),
                            "demand_id": meta.get("pending_demand_id"),
                            "sim_time_s": float(demand_ack.get("sim_time_s", 0)),
                            "requested_pps": expected_demand, "admitted_pps": admitted,
                            "unadmitted_pps": max(0, expected_demand - admitted),
                            "admission_cap_pps": cap,
                            "status": "Requested demand applied through the configured admission cap"})
        # Keep processing a pending admission-cap acknowledgement while the
        # scheduled demand file waits for its future onset.
    if meta.get("action_pending") and meta.get("status") in {"Failed", "Stopped"}:
        meta["action_pending"] = False
        meta["action_verification_status"] = "application_failed"
        meta["last_action_status"] = f"Action failed: simulator status is {meta['status']} before acknowledgement"
        _write_meta(folder, meta)
        _event(folder, {"type":"action_failed", "action_id": meta.get("pending_action_id"),
                        "timestamp":_utc(), "error":meta["last_action_status"]})
        return
    if meta.get("test_impairment_pending"):
        if meta.get("status") in {"Failed", "Stopped"}:
            meta["test_impairment_pending"] = False
            meta["test_impairment_status"] = "failed"
            meta["last_action_status"] = f"Test load shock failed: simulator status is {meta['status']} before acknowledgement"
            _write_meta(folder, meta)
            _event(folder, {"type": "test_impairment_failed",
                "action_id": meta.get("test_impairment_action_id"), "timestamp": _utc(),
                "error": meta["last_action_status"]})
            return
        start = int(meta.get("test_impairment_controls_seen", 0))
        unseen = rows[start:]
        expected = int(meta.get("test_impairment_new_pps", -1))
        ack = next((row for row in unseen if int(row.get("value", -2)) == expected), None)
        if ack is not None:
            meta["test_impairment_pending"] = False
            meta["test_impairment_status"] = "applied"
            meta["active_parameters"]["lambda_be_pps"] = expected
            meta["test_impairment_sim_time_s"] = float(ack["sim_time_s"])
            meta["last_action_status"] = "Test load shock applied by ns-3; controller response is being observed"
            meta["controls_seen"] = start + unseen.index(ack) + 1
            _write_meta(folder, meta)
            _event(folder, {"type": "test_impairment_applied",
                "action_id": meta.get("test_impairment_action_id"), "timestamp": _utc(),
                "sim_time_s": float(ack["sim_time_s"]),
                "previous_pps": int(meta.get("test_impairment_previous_pps", -1)),
                "new_pps": expected, "status": "Applied and acknowledged by ns-3"})
        elif unseen:
            meta["test_impairment_pending"] = False
            meta["test_impairment_status"] = "failed"
            meta["last_action_status"] = f"Test load shock failed: ns-3 acknowledged {unseen[0].get('value')} pps, expected {expected} pps"
            _write_meta(folder, meta)
            _event(folder, {"type": "test_impairment_failed",
                "action_id": meta.get("test_impairment_action_id"), "timestamp": _utc(),
                "error": meta["last_action_status"]})
        else:
            return
    if meta.get("action_pending") and rows and len(rows) > int(meta.get("pending_controls_seen", meta.get("controls_seen", 0))):
        # read_session() refreshes active_parameters from the latest telemetry.
        # The observed rate may therefore already equal the command before this
        # acknowledgement check runs. Compare with the saved pending command,
        # not the telemetry-refreshed active value, or the action stays pending
        # forever and control_cycle() can never consider a structural action.
        control_start = int(meta.get("pending_controls_seen", meta.get("controls_seen", 0)))
        unseen_rows = rows[control_start:]
        pending_rate = int(meta.get("pending_new_pps", -1))
        if pending_rate < 0 and unseen_rows:
            # Compatibility with pending actions persisted before the requested
            # value was stored explicitly in session metadata.
            pending_rate = int(unseen_rows[-1].get("value", -1))
        pending_old = int(meta.get("pending_previous_pps",
                                   meta.get("active_parameters", {}).get("lambda_be_pps", pending_rate)))
        expected_control_action = meta.get("pending_control_action")
        applied_row = next((row for row in unseen_rows
                            if int(row.get("value", -2)) == pending_rate
                            and (not expected_control_action
                                 or row.get("action") == expected_control_action)), None)
        if applied_row is not None:
            acknowledged_index = control_start + unseen_rows.index(applied_row)
            meta["controls_seen"] = acknowledged_index + 1
            if meta.get("pending_action_kind") == "admission_cap":
                meta["active_parameters"]["lambda_be_cap_pps"] = pending_rate
            else:
                meta["active_parameters"]["lambda_be_pps"] = pending_rate
            meta["action_pending"] = False
            meta["action_previous_pps"] = pending_old
            meta["action_new_pps"] = pending_rate
            meta["action_is_rollback"] = bool(meta.get("pending_is_rollback", False))
            meta["action_id"] = meta.get("pending_action_id")
            meta["last_action_status"] = "Applied by ns-3; awaiting enough post-change samples for verification"
            meta["action_applied_sim_time_s"] = float(applied_row["sim_time_s"])
            meta["action_verification_status"] = "pending_measurements"
            preview = meta.get("scheduled_demand_preview")
            if preview and preview.get("action_id") == meta.get("action_id"):
                preview.update({"status": "cap_applied",
                                "cap_applied_sim_time_s": float(applied_row["sim_time_s"])})
                meta["scheduled_demand_preview"] = preview
            _write_meta(folder, meta)
            _event(folder, {"type": "action_applied", "timestamp": _utc(),
                            "action_id": meta.get("action_id"),
                            "sim_time_s": applied_row["sim_time_s"], "previous_pps": pending_old,
                            "new_pps": pending_rate, "rollback": meta["action_is_rollback"],
                            "decision_basis": meta.get("pending_decision_basis"),
                            "counterfactual_reference": meta.get("pending_counterfactual_reference"),
                            "status": "Applied; effect verification pending"})
        else:
            meta["controls_seen"] = len(rows)
            meta["action_pending"] = False
            meta["action_verification_status"] = "application_failed"
            meta["last_action_status"] = f"Action failed: ns-3 acknowledged rate {rows[-1].get('value')}, expected {pending_rate}"
            _write_meta(folder, meta)
            _event(folder, {"type": "action_failed", "timestamp": _utc(),
                            "action_id": meta.get("pending_action_id"),
                            "error": meta["last_action_status"], "observed_control": rows[-1]})
            # Do not let samples associated with an earlier action verify this request.
            return
    if meta.get("action_applied_sim_time_s") is not None and not meta.get("action_verified"):
        data = read_session(session_id)
        try:
            meta = json.loads((folder / "session.json").read_text())
        except (OSError, ValueError):
            pass
        active_segment_id = f"segment_{int(meta.get('current_segment', 1)):03d}"
        segment_points = [p for p in data["points"] if p.get("segment_id") == active_segment_id]
        pts = [p for p in segment_points if float(p["sim_time_s"]) > float(meta["action_applied_sim_time_s"])]
        finished = meta.get("status") in {"Stopped", "Failed"}
        if len(pts) >= 2 or (finished and pts):
            before = [p for p in segment_points if float(p["sim_time_s"]) <= float(meta["action_applied_sim_time_s"])]
            pre = sum(float(p["qoe"]) for p in before[-2:]) / len(before[-2:]) if before else None
            observed = pts[:2]
            post = sum(float(p["qoe"]) for p in observed) / len(observed)
            post_ue = {}
            for point in observed:
                for ue in point.get("ues", []):
                    post_ue.setdefault(str(ue["ue_id"]), []).append(float(ue.get("qoe", 0)))
            per_ue_outcome = {}
            for ue_id, values in post_ue.items():
                after_ue = sum(values) / len(values)
                before_ue = meta.get("rate_action_before_ue_qoe", {}).get(ue_id)
                per_ue_outcome[ue_id] = {"before_qoe": before_ue, "after_qoe": after_ue,
                    "delta_qoe": after_ue - float(before_ue) if before_ue is not None else None,
                    "target_reached": after_ue >= float(meta.get("target_qoe", .72))}
            meta["action_verified"] = True
            meta["action_verification_status"] = "verified"
            preview = meta.get("scheduled_demand_preview")
            if preview and preview.get("action_id") == meta.get("action_id"):
                preview.update({"status": "cap_verified",
                                "pre_demand_verification": {
                                    "pre_action_mean_qoe": pre, "post_action_mean_qoe": post,
                                    "post_action_window_count": len(observed),
                                    "source": "measured ns-3 windows before scheduled demand onset"}})
                meta["scheduled_demand_preview"] = preview
            delta = post - pre if pre is not None else None
            target = float(meta.get("target_qoe", .72))
            user_worsened = any(value.get("before_qoe") is not None
                and value.get("after_qoe", 0) < target
                and value.get("delta_qoe", 0) < -0.01 for value in per_ue_outcome.values())
            clearly_worse = (delta is not None and delta < -0.01) or user_worsened
            meta["last_action_status"] = ("Applied and observed; QoE change is descriptive, not causal" +
                (f" (measured delta {delta:+.3f})" if delta is not None else ""))
            _write_meta(folder, meta)
            expected_ue_count = int(meta.get("config", {}).get("ue_count", len(per_ue_outcome)))
            per_ue_complete = len(per_ue_outcome) == expected_ue_count
            all_ues_target = per_ue_complete and all(
                bool(value["target_reached"]) for value in per_ue_outcome.values())
            _event(folder, {"type":"post_action_measurement", "timestamp":_utc(),
                            "action_id": meta.get("action_id"),
                            "sim_time_s":observed[-1]["sim_time_s"], "pre_action_qoe_mean":pre,
                            "post_action_qoe_mean":post, "measured_delta":delta,
                            "target_qoe": target, "target_reached": all_ues_target,
                            "network_mean_target_reached": post >= target,
                            "per_ue_verification_complete": per_ue_complete,
                            "affected_ue_ids": meta.get("rate_action_affected_ue_ids", []),
                            "per_ue_outcome": per_ue_outcome,
                            "measured_outcome":observed, "causal_claim":False})
            if (clearly_worse and post < target and meta.get("mode") == "Automatic"
                    and not bool(meta.get("action_is_rollback"))
                    and not bool(meta.get("rate_rollback_attempted", False))):
                try:
                    _queue_rate_change(session_id, int(meta["action_previous_pps"]), approved=True,
                        rollback=True, rationale="Automatic recovery: the measured rate-change outcome reduced QoE while still below target.")
                    meta = json.loads((folder / "session.json").read_text())
                    meta["rate_rollback_attempted"] = True
                    _write_meta(folder, meta)
                except Exception as exc:
                    _event(folder, {"type": "rate_rollback_failed", "timestamp": _utc(), "error": str(exc)})

    data = read_session(session_id)
    meta = json.loads((folder / "session.json").read_text())
    current_segment_id = f"segment_{int(meta.get('current_segment', 1)):03d}"
    current_points = [p for p in data["points"] if p.get("segment_id") == current_segment_id]
    if meta.get("structural_action_pending") and current_points:
        latest = current_points[-1]
        expected = meta.get("pending_structural_config") or meta.get("config", {})
        matches = (int(latest.get("gnb_count", -1)) == int(expected.get("gnb_count", -2))
                   and float(latest.get("bandwidth_mhz", -1)) == float(expected.get("bandwidth_mhz", -2))
                   and int(latest.get("ue_count", -1)) == int(expected.get("ue_count", -2))
                   and abs(float(latest.get("total_tx_power_dbm", -999)) - float(expected.get("total_tx_power_dbm", 35))) < 1e-6
                   and str(latest.get("ue_association_policy", "")) == str(expected.get("ue_association_policy", ""))
                   and str(latest.get("traffic_mix", expected.get("traffic_mix", "legacy")))
                       == str(expected.get("traffic_mix", "legacy")))
        if matches:
            meta["structural_action_pending"] = False
            meta["transition_status"] = "Restart epoch active; collecting verification windows"
            meta["last_action_status"] = "Restart configuration confirmed by live ns-3 telemetry; awaiting outcome windows"
            meta["active_parameters"].update({
                "ue_count": int(latest["ue_count"]),
                "bandwidth_mhz": float(latest["bandwidth_mhz"]),
                "gnb_count": int(latest["gnb_count"]),
                "ue_association_policy": str(latest["ue_association_policy"]),
                "traffic_mix": str(latest.get("traffic_mix", expected.get("traffic_mix", "legacy"))),
                "total_tx_power_dbm": float(latest["total_tx_power_dbm"]),
                "lambda_be_pps": int(latest["lambda_be_pps"]),
            })
            for segment in meta.get("segments", []):
                if int(segment.get("segment", -1)) == int(meta["current_segment"]):
                    segment["status"] = "Running"
                    segment["verified_by_telemetry"] = True
            _write_meta(folder, meta)
            _event(folder, {"type": "restart_epoch_verified", "timestamp": _utc(),
                            "segment": meta["current_segment"], "sim_time_s": latest["sim_time_s"],
                            "observed_config": {k: latest.get(k) for k in
                                ("ue_count", "gnb_count", "bandwidth_mhz", "total_tx_power_dbm", "ue_association_policy")},
                            "status": "Applied; measuring QoE outcome"})
    trial = meta.get("structural_trial")
    if trial and not trial.get("outcome_verified") and len(current_points) >= 3:
        pre = [float(v) for v in trial.get("before_qoe", [])]
        post = [float(p["qoe"]) for p in current_points[:3]]
        if pre and post:
            before_qoe, after_qoe = sum(pre) / len(pre), sum(post) / len(post)
            target = float(meta.get("target_qoe", .72))
            was_below_target = before_qoe < target
            after_ue_values: dict[str, list[float]] = {}
            for point in current_points[:3]:
                for ue in point.get("ues", []):
                    after_ue_values.setdefault(str(ue["ue_id"]), []).append(float(ue.get("qoe", 0)))
            per_ue_outcome = {}
            for ue_id, values in after_ue_values.items():
                after_ue = sum(values) / len(values)
                before_ue = trial.get("before_ue_qoe", {}).get(ue_id)
                per_ue_outcome[ue_id] = {"before_qoe": before_ue, "after_qoe": after_ue,
                    "delta_qoe": after_ue - float(before_ue) if before_ue is not None else None,
                    "target_reached": after_ue >= target}
            user_below_before = [float(v) for v in trial.get("before_ue_qoe", {}).values() if float(v) < target]
            expected_user_ids = set(trial.get("before_ue_qoe", {}))
            expected_user_ids.update(map(str, meta.get("pending_structural_affected_ue_ids", [])))
            outcomes_complete = bool(expected_user_ids) and expected_user_ids.issubset(per_ue_outcome)
            users_all_reached = outcomes_complete and all(
                per_ue_outcome[user_id]["target_reached"] for user_id in expected_user_ids)
            before_all_users_met = (bool(trial.get("before_ue_qoe"))
                and len(trial.get("before_ue_qoe", {})) == len(expected_user_ids)
                and all(float(value) >= target for value in trial.get("before_ue_qoe", {}).values()))
            weakest_before = min(trial.get("before_ue_qoe", {}).values(), default=before_qoe)
            weakest_after = min((v["after_qoe"] for v in per_ue_outcome.values()), default=after_qoe)
            if before_all_users_met:
                # Once healthy, staying above target is success. A score dip that
                # remains above target is not by itself grounds to undo a change.
                improved = outcomes_complete and users_all_reached
            elif user_below_before:
                deficit_ids = [ue_id for ue_id, value in trial.get("before_ue_qoe", {}).items()
                               if float(value) < target]
                improved = outcomes_complete and all(
                    per_ue_outcome[ue_id]["target_reached"]
                    or float(per_ue_outcome[ue_id]["after_qoe"]) >= float(trial["before_ue_qoe"][ue_id]) + .005
                    for ue_id in deficit_ids)
            elif trial.get("before_ue_qoe"):
                improved = users_all_reached and weakest_after >= float(weakest_before) - .005
            else:
                # Older/aggregate-only segments cannot establish per-user improvement.
                improved = (after_qoe >= target if before_qoe >= target else
                            after_qoe >= target and after_qoe >= before_qoe - .005)
            materially_worse = (after_qoe < before_qoe - .01 or any(
                value.get("before_qoe") is not None and
                (value["after_qoe"] < target <= float(value["before_qoe"])
                 or value.get("delta_qoe", 0) < -0.02)
                for value in per_ue_outcome.values()))
            target_regressions = [ue_id for ue_id, value in per_ue_outcome.items()
                if value.get("before_qoe") is not None
                and float(value["before_qoe"]) >= target
                and float(value["after_qoe"]) < target]
            if not trial.get("before_ue_qoe") and before_qoe >= target and after_qoe < target:
                target_regressions.append("network")
            trial.update({"outcome_verified": True, "before_qoe_mean": before_qoe,
                          "after_qoe_mean": after_qoe, "target_reached": after_qoe >= target,
                          "improved_toward_target": improved, "materially_worse": materially_worse,
                          "per_ue_outcome": per_ue_outcome})
            trial["all_users_target_reached"] = users_all_reached
            trial["per_ue_verification_complete"] = outcomes_complete
            trial["weakest_ue_qoe_before"] = weakest_before
            trial["weakest_ue_qoe_after"] = weakest_after
            rollback_reason = None
            if target_regressions:
                rollback_reason = ("The change pushed previously target-meeting UE(s) below target: "
                                   + ", ".join(map(str, target_regressions)) + ".")
            elif not improved:
                rollback_reason = ("The measured outcome did not improve below-target UE QoE toward target, "
                                   "or complete per-UE verification was unavailable.")
            trial["rollback_reason"] = rollback_reason
            trial["rollback_recommended"] = bool(rollback_reason) and not trial.get("rollback_attempted")
            if trial["rollback_recommended"]:
                meta["last_action_status"] = (f"Restart trial measured QoE {after_qoe:.3f} vs {before_qoe:.3f} before; "
                    f"rollback recommended. {rollback_reason}")
            elif before_all_users_met and users_all_reached:
                meta["last_action_status"] = (f"Measured mean QoE {after_qoe:.3f}; every UE remains at or above "
                    f"target {target:.3f}. Configuration retained despite individual score variation.")
            else:
                meta["last_action_status"] = (f"Restart trial measured QoE {after_qoe:.3f} vs {before_qoe:.3f} before; "
                    + ("target reached" if after_qoe >= target else "improved toward target"))
            _write_meta(folder, meta)
            _event(folder, {"type": "restart_epoch_outcome", "timestamp": _utc(),
                            "segment": trial["segment"], "pre_action_qoe_mean": before_qoe,
                            "post_action_qoe_mean": after_qoe, "target_qoe": target,
                            "target_reached": users_all_reached, "network_aggregate_target_reached": after_qoe >= target,
                            "target_was_below_before": was_below_target,
                            "improved_toward_target_or_maintained": improved,
                            "rollback_recommended": trial["rollback_recommended"],
                            "rollback_reason": rollback_reason,
                            "materially_worse_some_ue_but_still_target_met": bool(materially_worse and not rollback_reason),
                            "affected_ue_ids": meta.get("pending_structural_affected_ue_ids", []),
                            "per_ue_outcome": per_ue_outcome,
                            "measured_outcome": post, "causal_claim": False})
            if trial["rollback_recommended"] and meta.get("mode") == "Automatic":
                try:
                    request_structural_change(session_id, dict(trial["previous_config"]), approved=True,
                        rollback=True, rationale="Automatic recovery: restart epoch failed to improve QoE toward target.")
                    meta = json.loads((folder / "session.json").read_text())
                    meta["structural_trial"]["rollback_attempted"] = True
                    _write_meta(folder, meta)
                except Exception as exc:
                    _event(folder, {"type": "rollback_failed", "timestamp": _utc(), "error": str(exc)})


def run_worker(session_id: str) -> None:
    folder = LIVE_ROOT / session_id
    audited_keys: set[tuple[str, float]] = set()
    while True:
        try:
            snapshot = read_session(session_id)
            state = snapshot["meta"]
            # Persist one forecast per newly observed measurement, in every mode.
            # This is an audit trail; it does not affect controller decisions.
            current_segment_id = f"segment_{int(state.get('current_segment', 1)):03d}"
            segment_points = [p for p in snapshot["points"] if p.get("segment_id") == current_segment_id]
            if state.get("status") == "Running" and segment_points:
                point = segment_points[-1]
                audit_key = (current_segment_id, float(point.get("sim_time_s", -1)))
                if audit_key not in audited_keys and len(segment_points) >= 4:
                    ue_fc = forecast_ue_states(segment_points, target_qoe=float(state.get("target_qoe", .72)))
                    audit_row = {
                        "record_type": "forecast", "session_id": session_id,
                        "segment_id": current_segment_id, "mode": state.get("mode"),
                        "created_at": _utc(), "sim_time_s": audit_key[1], "horizon_s": 5,
                        "target_qoe": float(state.get("target_qoe", .72)),
                        "safety_floors": state.get("safety_floors", {}),
                        "method": "linear trend over last five 1-second windows",
                        "measured_ues": [{"ue_id": int(u["ue_id"]),
                                          "traffic_class": str(u.get("traffic_class", "unknown")),
                                          **{key: u.get(key) for key in (
                                              "qoe", "throughput_mbps", "delay_ms", "jitter_ms",
                                              "packet_loss_fraction")}}
                                         for u in point.get("ues", []) if "ue_id" in u and "qoe" in u],
                        "predictions": {str(uid): pred.get("forecast", {})
                                        for uid, pred in ue_fc.items()},
                        "configuration": {key: point.get(key) for key in (
                            "lambda_be_pps", "bandwidth_mhz", "gnb_count", "total_tx_power_dbm",
                            "ue_association_policy", "traffic_mix")},
                    }
                    with (folder / "forecast_audit.jsonl").open("a") as audit_file:
                        audit_file.write(json.dumps(audit_row, separators=(",", ":")) + "\n")
                    audited_keys.add(audit_key)
            if (folder / "stop").exists():
                segment = _segment_dir(folder, int(state.get("current_segment", 1)))
                (segment / "stop").touch()
                deadline = time.time() + 180
                while _pid_alive(int(state.get("pid") or 0), str(LIVE_BINARY)) and time.time() < deadline:
                    time.sleep(.5)
                state = json.loads((folder / "session.json").read_text())
                if _pid_alive(int(state.get("pid") or 0), str(LIVE_BINARY)):
                    state.update({"status": "Failed", "transition_status": "Stop failed: simulator did not exit"})
                else:
                    state.update({"status": "Stopped", "transition_status": "Stopped by user"})
                for item in state.get("segments", []):
                    if int(item.get("segment", -1)) == int(state.get("current_segment", 1)):
                        item["status"] = state["status"]
                        item["ended_at"] = _utc()
                state["ended_at"] = _utc()
                _write_meta(folder, state)
                _event(folder, {"type": "session_stopped", "timestamp": _utc(), "status": state["status"]})
                return
            if (folder / "structural_request.json").exists() and state.get("status") == "Running":
                _apply_structural_request(folder)
            refresh_action_status(session_id)
            control_cycle(session_id)
            state = json.loads((folder / "session.json").read_text())
            if state.get("status") in {"Stopped", "Failed"}:
                return
        except (OSError, ValueError, KeyError) as exc:
            try:
                meta = json.loads((folder / "session.json").read_text())
                message = f"{type(exc).__name__}: {exc}"
                if meta.get("controller_last_error") != message:
                    meta["controller_last_error"] = message
                    _write_meta(folder, meta)
                    _event(folder, {"type": "controller_error", "timestamp": _utc(), "error": message,
                                    "status": "No action claimed; controller will retry on the next cycle"})
            except (OSError, ValueError):
                pass
        time.sleep(1)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--worker":
    run_worker(sys.argv[2])
