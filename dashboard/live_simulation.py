"""Human-readable dashboard for the persistent real-time ns-3 session."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation.live_runtime import (
    QOE_DEFINITION, assess_safety_floors, assess_ue_target_state, default_safety_floors,
    forecast_ue_states, list_sessions, read_session, recommend_control_action,
    refresh_action_status, request_rate_change, request_structural_change, request_test_load_shock,
    session_process_is_active, test_load_shock_eligibility,
    set_mode, start_session, stop_session,
)
from simulation.result_parser import QOE_FORMULA_VERSION, calculate_qoe_components
from simulation.predictive_demand import latest_study, start_study


def _event_summary(event: dict) -> dict[str, str]:
    """Render an event as a concise human-readable feed row without changing its record."""
    kind = str(event.get("type", "event"))
    labels = {
        "controller_decision": "Controller decision", "action_pending": "Rate change requested",
        "rollback_pending": "Rollback requested", "action_applied": "Rate change applied",
        "action_failed": "Action failed", "post_action_measurement": "Measured verification",
        "restart_epoch_requested": "Restart requested", "restart_epoch_started": "Restart starting",
        "restart_epoch_launched": "New segment launched", "restart_epoch_verified": "Configuration verified",
        "restart_epoch_outcome": "Measured restart outcome", "restart_epoch_failed": "Restart failed",
        "test_impairment_pending": "Test load shock requested", "test_impairment_applied": "Test load shock applied",
        "test_impairment_failed": "Test load shock failed",
        "scheduled_demand_preview": "Upcoming demand risk armed",
        "scheduled_demand_pending": "Demand change scheduled",
        "scheduled_demand_applied": "Demand change applied",
        "scheduled_demand_failed": "Demand change failed",
        "scheduled_demand_missed": "Demand onset missed",
        "rollback_failed": "Rollback failed", "rate_rollback_failed": "Rate rollback failed",
        "mode_changed": "Controller mode changed", "stop_requested": "Stop requested",
        "session_stopped": "Session stopped", "controller_error": "Controller error",
    }
    action = str(event.get("action") or "")
    details = str(event.get("reason") or event.get("rationale") or event.get("error") or "")
    decision_basis = event.get("decision_basis")
    if decision_basis == "safety_floor_preemptive":
        action_prefix = ("SAFETY-FLOOR PRE-EMPTIVE ACTION. "
                         if event.get("action") and str(event.get("action")).lower() != "no action"
                         else "SAFETY-FLOOR EARLY WARNING; no action applied. ")
        details = action_prefix + "A service-specific floor breach was forecast before it was measured. " + details
    elif decision_basis == "predictive_preemptive":
        measured = event.get("measured_qoe_at_decision")
        target = event.get("target_qoe")
        state = (f"Measured QoE {float(measured):.3f} was still above target {float(target):.3f}. "
                 if measured is not None and target is not None else "The target had not yet been breached. ")
        action_prefix = (("SAFETY-FLOOR PRE-EMPTIVE ACTION. " if decision_basis == "safety_floor_preemptive"
                          else "PRE-EMPTIVE FORECAST ACTION. ")
                         if event.get("action") and str(event.get("action")).lower() != "no action"
                         else "PRE-EMPTIVE FORECAST WARNING; no action applied. ")
        details = action_prefix + state + details
    elif decision_basis == "safety_floor_recovery":
        details = "SAFETY-FLOOR RECOVERY DECISION. " + details
    elif decision_basis == "reactive_target_recovery":
        details = "REACTIVE TARGET-RECOVERY DECISION. " + details
    if event.get("previous_pps") is not None or event.get("new_pps") is not None:
        unit = "packets/s per UE"
        if "admission" in str(event.get("action", "")).lower() or event.get("decision_basis") == "predictive_preemptive":
            unit = "admitted packets/s per UE"
        change = f"{event.get('previous_pps', '—')} → {event.get('new_pps', '—')} {unit}"
    elif kind == "scheduled_demand_applied":
        change = (f"Requested {event.get('requested_pps', '—')} · admitted {event.get('admitted_pps', '—')} · "
                  f"unadmitted {event.get('unadmitted_pps', '—')} packets/s per UE")
    elif event.get("new_config") or event.get("previous_config"):
        old, new = event.get("previous_config") or {}, event.get("new_config") or {}
        keys = ("gnb_count", "bandwidth_mhz", "total_tx_power_dbm", "ue_association_policy")
        changes = [f"{key.replace('_', ' ')}: {old.get(key, '—')} → {new.get(key, '—')}"
                   for key in keys if old.get(key) != new.get(key)]
        change = "; ".join(changes) or action or "Configuration evaluated"
    elif action:
        change = action
    else:
        change = str(event.get("status") or "—")
    measured_before = event.get("pre_action_qoe_mean", event.get("before_qoe_mean"))
    measured_after = event.get("post_action_qoe_mean", event.get("after_qoe_mean"))
    if measured_before is not None or measured_after is not None:
        details = f"Measured mean UE QoE {measured_before if measured_before is not None else '—'} → {measured_after if measured_after is not None else '—'}. {details}".strip()
    forecast_data = event.get("forecast") or event.get("trigger_forecast")
    if forecast_data:
        predicted = forecast_data.get("qoe")
        horizon = forecast_data.get("horizon_s")
        horizon_text = f" in +{horizon}s" if horizon is not None else ""
        details = (f"Trend forecast (not measured): QoE {predicted:.3f}{horizon_text}. "
                   if isinstance(predicted, (int, float)) else "Trend forecast (not measured). ") + details
    affected = event.get("affected_ue_ids") or []
    if affected:
        details = f"Affected UEs: {', '.join(map(str, affected))}. {details}".strip()
    status = str(event.get("status") or ("Verified" if kind in {"action_applied", "restart_epoch_verified", "post_action_measurement", "restart_epoch_outcome"} else "Recorded"))
    if kind == "controller_decision" and action.lower() == "no action":
        status = "Monitoring"
    return {"Time (UTC)": str(event.get("timestamp", "—")), "Event": labels.get(kind, kind.replace("_", " ").capitalize()),
            "Change / decision": change, "Status": status, "Details": details or "—"}


def format_action_feed(events: list[dict], limit: int = 40) -> tuple[pd.DataFrame, int]:
    """Condense repeated no-action decisions while preserving all rows for the raw ledger."""
    rows = []
    skipped_no_action = 0
    no_action_count = 0
    last_no_action = None
    for event in reversed(events):
        if event.get("type") == "controller_decision" and str(event.get("action", "")).lower() == "no action":
            no_action_count += 1
            last_no_action = last_no_action or event
            continue
        rows.append(_event_summary(event))
    if no_action_count:
        summary = _event_summary(last_no_action)
        summary.update({"Event": f"Monitoring · {no_action_count} no-action decisions",
                       "Change / decision": "No configuration change", "Status": "Monitoring"})
        rows.append(summary)
    skipped_no_action = max(0, len(events) - len(rows))
    rows.sort(key=lambda row: row["Time (UTC)"], reverse=True)
    return pd.DataFrame(rows[:limit]), skipped_no_action


def _render_predictive_demand_trial(st_module, active):
    # Separate comparison experiment; not part of the live control workflow.
    with st_module.expander("Optional experiment · scheduled-demand comparison", expanded=False):
        st_module.caption(
            "Runs two real, sequential ns-3 sessions with the same scenario and random seed. "
            "The first measures the no-control result for a scheduled demand increase; the second "
            "uses that measured reference to apply an admission cap before the same kind of increase. "
            "This tests known upcoming demand, not prediction of an unknown arrival."
        )
        st_module.caption(
            "Trial profile: 4 UEs, 1 gNB, 50 MHz, 7,500 packets/s/UE initially; demand requests "
            "20,000 packets/s/UE and Automatic protection caps admitted best-effort load at 5,000. "
            "Any unadmitted requested rate is shown separately; QoE describes transmitted traffic."
        )
        study = latest_study()
        busy_trial = bool(study and study.get("status") == "Running")
        if st_module.button("Run matched predictive demand trial", key="run_predictive_demand_trial",
                            type="secondary", disabled=bool(active) or busy_trial):
            try:
                study_id = start_study()
                st_module.success(f"Started paired trial {study_id}. It continues in the background and this view refreshes automatically.")
                st_module.rerun()
            except Exception as exc:
                st_module.error(f"Could not start paired trial: {exc}")
        if active:
            st_module.caption(
                f"Stop the active ns-3 session ({active.get('session_id', 'unknown')}) before starting this separate paired trial."
            )
        if study:
            st_module.write(f"**Trial {study.get('status', 'Unknown')} · {study.get('phase', '—')}**")
            if study.get("error"):
                st_module.error(study["error"])
            reference = study.get("no_control_reference")
            outcome = study.get("controlled_outcome")
            if reference:
                left, right = st_module.columns(2)
                left.metric("Matched no-control QoE · measured",
                            f"{reference['mean_ue_qoe']:.3f}",
                            help="Measured in the separate Monitor session at the first complete window after the scheduled 20,000 pps/UE demand request.")
                if outcome and outcome.get("mean_ue_qoe") is not None:
                    right.metric("Automatic run QoE · measured",
                                 f"{outcome['mean_ue_qoe']:.3f}",
                                 delta=f"{outcome['mean_ue_qoe'] - reference['mean_ue_qoe']:+.3f} vs matched reference")
                else:
                    right.metric("Automatic run QoE · measured", "Waiting")
                at_risk = [int(uid) for uid, value in reference.get("per_ue_qoe", {}).items()
                           if float(value) < float(study.get("target_qoe", .85))]
                st_module.markdown(
                    f"**What QoE would have dropped to without control:** the matched no-control run "
                    f"measured mean UE QoE **{reference['mean_ue_qoe']:.3f}**; "
                    f"{len(at_risk)} of {len(reference.get('per_ue_qoe', {}))} UEs were below the "
                    f"{float(study.get('target_qoe', .85)):.2f} target. This is a separately measured "
                    "counterfactual reference, not a measurement from the controlled run."
                )
                st_module.caption("No-control measured per-UE QoE: " + ", ".join(
                    f"UE {uid}: {float(value):.3f}" for uid, value in reference.get("per_ue_qoe", {}).items()))
            if outcome:
                st_module.caption("Controlled measured per-UE QoE: " + ", ".join(
                    f"UE {uid}: {float(value):.3f}" for uid, value in outcome.get("per_ue_qoe", {}).items()))
                common_ues = sorted(set(reference.get("per_ue_qoe", {}))
                                    & set(outcome.get("per_ue_qoe", {})), key=int)
                if common_ues:
                    comparison = go.Figure()
                    comparison.add_bar(name="Matched no-control · measured",
                        x=[f"UE {uid}" for uid in common_ues],
                        y=[float(reference["per_ue_qoe"][uid]) for uid in common_ues],
                        marker_color="#ff8b7a")
                    comparison.add_bar(name="Automatic · measured",
                        x=[f"UE {uid}" for uid in common_ues],
                        y=[float(outcome["per_ue_qoe"][uid]) for uid in common_ues],
                        marker_color="#42d392")
                    comparison.add_hline(y=float(study.get("target_qoe", .85)),
                        line_dash="dot", line_color="#f6c85f",
                        annotation_text=f"Target {float(study.get('target_qoe', .85)):.2f}")
                    comparison.update_layout(template="plotly_dark", barmode="group",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#e6edf3"}, yaxis={"title": "Measured per-UE QoE (0–1)", "range": [0, 1]},
                        xaxis_title="User equipment", height=300,
                        margin={"t": 25, "b": 35, "l": 45, "r": 20}, legend={"orientation": "h", "y": -0.2})
                    st_module.plotly_chart(comparison, use_container_width=True,
                        key=f"predictive_demand_comparison_{study.get('study_id')}")
                st_module.write(
                    f"Demand requested: {int(outcome.get('requested_pps', 0)):,} pps/UE · "
                    f"admitted: {int(outcome.get('admitted_pps', 0)):,} · "
                    f"unadmitted: {int(outcome.get('unadmitted_pps', 0)):,} pps/UE"
                )
                action = study.get("controller_action") or {}
                if action:
                    st_module.write(
                        f"Controller action: **{action.get('previous_pps', '—'):,} → "
                        f"{action.get('new_pps', '—'):,} pps/UE admission cap** · "
                        f"applied at sim {action.get('sim_time_s', '—')} s · "
                        f"demand onset at sim {study.get('actual_demand_onset_sim_time_s', '—')} s"
                    )
                if study.get("success"):
                    st_module.success("Pre-emptive cap was acknowledged before demand onset and every measured UE stayed at or above target.")
                elif study.get("status", "").startswith("Completed"):
                    st_module.warning("Trial completed, but one or more acceptance conditions were not met. Inspect the session records; no prevention claim is made.")
            st_module.caption(f"Baseline session: {study.get('baseline_session_id', 'pending')} · Controlled session: {study.get('controlled_session_id', 'pending')} · Durable study record: results/live_control/predictive_demand_studies/{study.get('study_id', '')}/study.json")



@st.fragment(run_every="1s")
def render_live_simulation(st_module=st) -> None:
    st_module.title("Live network simulation")
    st_module.caption("One persistent, real-time ns-3 process. Measured telemetry, short-horizon forecasts, and controller actions are shown separately.")
    with st_module.expander("QoE score and measurement units", expanded=False):
        st_module.markdown(QOE_DEFINITION)
        st_module.markdown(
            "**Profile references:** low-latency uses 2 Mbps, 50 ms delay, 5 ms jitter, and 0.1% loss; "
            "voice-QoS uses a 0.1 Mbps project service floor, 100 ms delay, 30 ms jitter, and 1% loss. "
            "Gaming uses a 0.05 Mbps project floor, 50 ms delay, 10 ms jitter, and 0.1% loss; video streaming "
            "uses the NGMN model's 0.064 Mbps source reference, 300 ms delay, 50 ms jitter, and 0.0001% loss. "
            "The low-latency throughput reference comes from its default offered workload (100 B × 2,500 pps). "
            "Delay/loss references are illustrative 3GPP 5QI 3/5QI 1 limits; jitter values are project policy. "
            "These profiles do not mean the simulator configures those exact 5QIs, and this remains a network proxy, not a user survey score."
        )
        st_module.caption("The voice-QoS flow currently uses a high-rate 1,252-byte UDP stress generator. Its 0.1 Mbps score floor is a project service assumption, not a calibrated codec model; the generated load is not a realistic voice source.")
        st_module.caption("Gaming and video use the bundled NGMN packet traffic generators. Video QoE here is a network proxy; playback buffering, startup delay, and resolution are not modeled.")

    sessions = list_sessions()
    active = next((s for s in sessions if session_process_is_active(s)), None)
    with st_module.expander("Start a simulation", expanded=not bool(active)):
        with st_module.form("live_sim_start"):
            cols = st_module.columns(5)
            ues = cols[0].number_input("UE count", min_value=2, max_value=20, value=2, step=1)
            bws = cols[1].selectbox("Band 1 bandwidth · startup only", [20, 50, 100], index=1)
            gnbs = cols[2].number_input("gNB count · startup only", min_value=1, max_value=4, value=1, step=1)
            load = cols[3].number_input("Best-effort packet rate · packets/s per UE", min_value=2500, max_value=20000, value=10000, step=2500)
            ull = cols[4].number_input("Low-latency packet rate · packets/s per UE", min_value=500, max_value=10000, value=2500, step=500)
            association = st_module.selectbox("UE association · startup only",
                ["closest", "round_robin", "by_traffic_class"],
                help="round_robin and by_traffic_class both require at least two gNBs in this ns-3 scenario. Structural changes apply through a restart epoch.")
            traffic_profile_label = st_module.selectbox(
                "UE traffic profile",
                ["Current low-latency + voice mix", "NGMN gaming + video mix"],
                help="The existing mix remains the default. The NGMN option assigns alternating UEs to gaming and video traffic generators.")
            power = st_module.number_input("Total transmit power · dBm · startup only", min_value=10.0, max_value=50.0, value=35.0, step=1.0)
            mode = st_module.selectbox("Controller mode", ["Recommend", "Monitor", "Automatic"], index=0,
                                       help="Recommend is the default. Automatic uses live packet pacing for the current constant-rate mix; NGMN gaming/video traffic uses supported restart epochs for network changes.")
            target = st_module.slider("QoE target", 0.0, 1.0, 0.72, 0.01)
            floor_defaults = default_safety_floors(target)
            st_module.caption("Per-service safety floors trigger protection when the five-second trend crosses the floor, or when a UE is within 0.05 QoE and still declining. Defaults are adjustable project policy values, not standardized service guarantees.")
            floor_cols = st_module.columns(4)
            safety_floors = {
                "low_latency": floor_cols[0].number_input("Low-latency floor", 0.0, 1.0,
                    floor_defaults["low_latency"], 0.01, key="start_floor_low_latency"),
                "voice_qos": floor_cols[1].number_input("Voice floor", 0.0, 1.0,
                    floor_defaults["voice_qos"], 0.01, key="start_floor_voice"),
                "gaming": floor_cols[2].number_input("Gaming floor", 0.0, 1.0,
                    floor_defaults["gaming"], 0.01, key="start_floor_gaming"),
                "video_streaming": floor_cols[3].number_input("Video floor", 0.0, 1.0,
                    floor_defaults["video_streaming"], 0.01, key="start_floor_video"),
            }
            invalid_association = int(gnbs) < 2 and association != "closest"
            if invalid_association:
                st_module.warning(f"{association} association requires at least two gNBs. Select closest or increase the gNB count.")
            invalid_floors = any(float(value) > float(target)
                                 or (float(target) > 0 and float(value) >= float(target))
                                 for value in safety_floors.values())
            if invalid_floors:
                st_module.warning("Each service safety floor must be lower than the QoE target.")
            submitted = st_module.form_submit_button("Start live simulation", type="primary",
                disabled=invalid_association or invalid_floors)
        if submitted:
            try:
                sid = start_session({"ue_count": int(ues), "gnb_count": int(gnbs), "bandwidth_mhz": float(bws),
                                     "lambda_be": int(load), "lambda_ull": int(ull), "total_tx_power_dbm": float(power),
                                     "scenario_size_m": 30, "ue_association_policy": association,
                                     "traffic_mix": "gaming_video" if traffic_profile_label.startswith("NGMN") else "legacy"},
                                    mode, target, safety_floors=safety_floors)
                st_module.session_state["live_selected_session"] = sid
                st_module.success(f"Started ns-3 session {sid}")
                st_module.rerun()
            except Exception as exc:
                st_module.error(f"Could not start ns-3: {exc}")

    if not sessions and not active:
        st_module.info("No simulation sessions yet. Choose startup settings above and start a real ns-3 run.")
        _render_predictive_demand_trial(st_module, active)
        return
    session_ids = [s["session_id"] for s in sessions]
    preferred = st_module.session_state.get("live_selected_session", session_ids[0])
    selected = st_module.selectbox("Session", session_ids, index=session_ids.index(preferred) if preferred in session_ids else 0)
    st_module.session_state["live_selected_session"] = selected
    refresh_action_status(selected)
    data = read_session(selected)
    meta = data["meta"]
    folder = data["folder"]
    status = meta.get("status", "Unknown")
    if status == "Stale": st_module.warning("Telemetry is stale. Automatic control is paused until fresh measurements resume.")
    elif status == "Failed": st_module.error("Simulator exited unexpectedly. Inspect the session log below; no live values are being advanced.")
    elif status == "Running": st_module.success("Simulation running · real-time clock")
    elif status in {"Starting", "Restarting"}: st_module.info(meta.get("transition_status", "Simulation segment is starting."))
    elif status == "Stopped": st_module.info("Simulation stopped. Historical measurements remain available.")

    controls = st_module.columns([2, 2, 1])
    modes = ["Monitor", "Recommend", "Automatic"]
    current_mode = meta.get("mode", "Recommend")
    chosen_mode = controls[0].selectbox("Controller mode", modes, index=modes.index(current_mode), key=f"mode_{selected}")
    if chosen_mode != current_mode:
        set_mode(selected, chosen_mode)
        st_module.rerun()
    if status in {"Running", "Stale", "Starting", "Restarting"} and controls[2].button("Stop simulation", type="secondary", key=f"stop_{selected}"):
        stop_session(selected)
        st_module.info("Stop requested; ns-3 will stop at its next one-second control tick.")

    points = data["points"]
    latest = points[-1] if points else None
    forecast = meta.get("forecast", {})
    fresh = bool(latest and meta.get("status") == "Running")
    current_ue_states = latest.get("ues", []) if latest else []
    current_segment_points = [p for p in points if p.get("segment_id") == f"segment_{int(meta.get('current_segment', 1)):03d}"]
    ue_forecasts = forecast_ue_states(current_segment_points,
        target_qoe=float(meta.get("target_qoe", .72))) if current_segment_points else {}
    floor_assessment = (assess_safety_floors(
        current_segment_points, meta.get("safety_floors"),
        float(meta.get("target_qoe", .72)),
        float(meta.get("safety_warning_margin", .05))) if current_segment_points else {
            "available": False, "at_risk_ue_ids": [], "users": [],
            "reason": "Waiting for measured UE history."})
    safety_floor_risk_ids = floor_assessment.get("at_risk_ue_ids", [])
    below_target_ues = sum(float(ue.get("qoe", 0)) < float(meta.get("target_qoe", .72)) for ue in current_ue_states)
    ue_target_state = assess_ue_target_state(
        current_ue_states, ue_forecasts, float(meta.get("target_qoe", .72)),
        expected_ue_count=int(meta.get("config", {}).get("ue_count", 0)) or None)
    forecast_at_risk_ues = (ue_target_state["forecast_target_crossing_ue_ids"]
        if ue_target_state["all_current_met"] else
        [ue_id for ue_id, item in ue_forecasts.items()
         if item.get("forecast", {}).get("available") and item["forecast"].get("risk")])
    k = st_module.columns(4)
    k[0].metric("Session", status)
    legacy_pooled = bool(latest and latest.get("qoe_basis") != "equal_weight_mean_per_ue")
    legacy_formula = bool(latest and latest.get("qoe_formula_version") != "service_v2")
    qoe_label = "Legacy pooled QoE" if legacy_pooled else "Mean UE QoE"
    k[1].metric(qoe_label, f"{latest['qoe']:.3f}" if fresh else "—",
                help="Equal-weight mean of service-profile per-UE QoE v2 scores; a UE with no received packets scores zero." if not legacy_pooled and not legacy_formula
                     else "This session predates service-profile QoE v2. Its score uses the earlier class-agnostic formula; start a new session for the current formula." if not legacy_pooled
                     else "This historical telemetry predates the equal-weight per-UE QoE definition and contains the pooled-flow score.")
    k[2].metric("Pooled network QoE", f"{latest['network_qoe']:.3f}" if fresh and latest.get("network_qoe") is not None else "—",
                help="Legacy class-agnostic pooled-flow v1 diagnostic. It can hide individual users and is not directly comparable to the service-profile v2 user score or the controller's target.")
    k[3].metric("QoE forecast (+5 s)", f"{forecast['qoe']:.3f}" if forecast.get("available") else "Warming up",
                help="Trend forecast of the equal-weight per-UE mean for current-format sessions.")
    k = st_module.columns(4)
    if ue_target_state["all_current_met"]:
        forecast_risk_label = ("Multiple UEs at risk" if len(forecast_at_risk_ues) >= 2 else
                               "1 UE · monitoring" if forecast_at_risk_ues else "No target breach forecast")
    else:
        forecast_risk_label = "At risk" if forecast.get("risk") else ("No risk" if forecast.get("available") else "Insufficient history")
    k[0].metric("Forecast risk", forecast_risk_label)
    k[1].metric("UEs below target", str(below_target_ues) if fresh and current_ue_states else "—",
                delta=(f"{len(forecast_at_risk_ues)} forecast at risk" if forecast_at_risk_ues else None),
                help="Current measured count below target. One target crossing is usually monitored; a forecast safety-floor breach can trigger protection for that UE.")
    worst_qoe = min((float(ue.get("qoe", 0)) for ue in current_ue_states), default=None)
    k[2].metric("Worst UE QoE", f"{worst_qoe:.3f}" if fresh and worst_qoe is not None else "—",
                help="Minimum measured per-UE QoE in the latest window; protects against the mean concealing one poor user.")
    k[3].metric("Simulation time", f"{latest['sim_time_s']:.1f} s" if latest else "—")
    if legacy_pooled:
        st_module.caption("This selected historical session uses the former pooled-flow QoE definition. New sessions report the equal-weight mean of per-UE scores.")
    elif legacy_formula:
        st_module.caption("This session uses the previous class-agnostic UE formula. Start a new session to use service-profile QoE v2; scores across formula versions are not directly comparable.")
    st_module.caption(f"Last measurement: {meta.get('last_measurement_utc', 'waiting for first sample')} · Active controller: {meta.get('mode')} · Data status: {'fresh' if fresh else status.lower()}")
    if not fresh:
        st_module.caption("Safety-floor assessment reflects the last recorded telemetry; this session is not live.")
    elif floor_assessment.get("available") and safety_floor_risk_ids:
        risk_text = "; ".join(
            f"UE {row['ue_id']} ({row['traffic_class']}): measured {row['measured_qoe']:.3f}, floor {row['floor']:.3f}"
            + (f", forecast {row['forecast_qoe']:.3f} in +{row['forecast_horizon_s']} s"
               if row.get("forecast_qoe") is not None else "")
            for row in floor_assessment["users"] if row["at_risk"])
        st_module.warning(f"Safety-floor warning · {risk_text}. Forecasts are trend estimates, not measurements.")
    elif floor_assessment.get("available"):
        st_module.caption("Safety floors: no UE is below, forecast to cross, or near a declining service floor.")
    else:
        st_module.caption("Safety-floor assessment is waiting for complete UE measurements and enough trend history.")

    active_params = meta.get("active_parameters", {})
    st_module.subheader("Active network and traffic settings" if meta.get("configuration_verified")
                        else "Last verified network settings")
    if not meta.get("configuration_verified"):
        st_module.caption(meta.get("transition_status") or
            "Waiting for simulator telemetry to confirm the requested configuration; values below are the last verified settings when available.")
    p = st_module.columns(4)
    p[0].metric("UEs", active_params.get("ue_count", "—"))
    p[1].metric("gNB count", active_params.get("gnb_count", "—"), help="Topology setting; applied at session start or by a restart epoch.")
    p[2].metric("UE association", active_params.get("ue_association_policy", "—"), help="Serving-cell assignment policy; applied at session start or by a restart epoch.")
    p[3].metric("Band 1 bandwidth", f"{active_params.get('bandwidth_mhz','—')} MHz", help="Restart-only radio setting.")
    p = st_module.columns(3)
    p[0].metric("Total TX power", f"{active_params.get('total_tx_power_dbm','—')} dBm", help="Restart-only radio setting.")
    active_floors = meta.get("safety_floors", default_safety_floors(float(meta.get("target_qoe", .72))))
    st_module.caption("QoE safety floors · " + " · ".join(
        f"{service.replace('_', ' ')} {float(floor):.2f}"
        for service, floor in active_floors.items()))
    if meta.get("config", {}).get("traffic_mix", "legacy") == "gaming_video":
        p[1].metric("UE traffic", "NGMN gaming + video")
        p[2].metric("Live packet pacing", "Unavailable", help="These burst traffic models have no verified live rate control; QoE-triggered network changes use restart epochs.")
        st_module.caption("Gaming/video UEs use NGMN application traffic models. Bandwidth, topology, association, and radio power changes take effect through a verified restart epoch; there is no live source-rate action for these generators.")
    else:
        p[1].metric("Best-effort rate", f"{active_params.get('lambda_be_pps','—')} pps / UE", help="Live adjustable offered packet rate.")
        p[2].metric("Low-latency rate", f"{active_params.get('lambda_ull_pps','—')} pps / UE", help="Offered packet rate, fixed for this run segment.")
        st_module.caption("Packet pacing is adjustable in the running process. Bandwidth, topology, UE association, and radio power use a restart epoch: the current process stops, a new one starts with the selected settings, and telemetry continues under this session with a visible gap.")

    with st_module.expander("Controlled test disturbance · optional", expanded=False):
        st_module.caption("One-shot workload shock for a live demonstration. It raises the legacy voice offered rate to 20,000 packets/s per UE; this changes traffic demand, not radio conditions. The controller remains responsible for any response.")
        shock_eligibility = test_load_shock_eligibility(selected)
        if st_module.button("Inject test load shock", key=f"load_shock_{selected}",
                            type="secondary", disabled=not shock_eligibility.get("allowed")):
            try:
                request_test_load_shock(selected)
                st_module.success("Test load shock queued. Waiting for ns-3 acknowledgement; the controller will observe subsequent telemetry.")
                st_module.rerun()
            except Exception as exc:
                st_module.error(f"Could not trigger test load shock: {exc}")
        st_module.caption(shock_eligibility.get("reason", "Test disturbance unavailable."))

    st_module.subheader("Live UE topology and state")
    st_module.caption("Select a UE marker on the topology to inspect its individual QoE history and controller changes related to it.")
    ue_states = latest.get("ues", []) if latest else []
    gnb_states = latest.get("gnbs", []) if latest else []
    latest_segment = str(latest.get("segment_id", "segment_001")) if latest else ""
    expected_segment = f"segment_{int(meta.get('current_segment', 1)):03d}"
    if ue_states and gnb_states and latest_segment == expected_segment:
        st_module.caption(
            f"Measured per-UE FlowMonitor windows at simulation time {latest['sim_time_s']:.1f} s · "
            f"QoE target {float(meta.get('target_qoe', .72)):.2f}. "
            f"Telemetry status: {status}. Marker color shows measured QoE (red low, green high); "
            "circles are low-latency UEs, diamonds are voice UEs. Faint links show serving cells. "
            "Hover for UE or cell IDs; select a UE for its history. "
            "Values are measurements, not forecasts."
        )
        if latest.get("qoe_formula_version") == "service_v2":
            class_scores = latest.get("qoe_by_traffic_class", {})
            if latest.get("traffic_mix") == "gaming_video":
                class_metrics = (("Gaming mean QoE", "gaming"), ("Video mean QoE", "video_streaming"))
            else:
                class_metrics = (("Low-latency mean QoE", "low_latency"), ("Voice-QoS mean QoE", "voice_qos"))
            class_cols = st_module.columns(len(class_metrics))
            for column, (label, key) in zip(class_cols, class_metrics):
                value = class_scores.get(key)
                column.metric(label, f"{float(value):.3f}" if value is not None else "—")
        topology = go.Figure()
        # Draw muted cell-colored links first so they remain visible behind nodes
        # without competing with the QoE colors or marker hover targets.
        cell_palette = ["#9b8afb", "#41c7c7", "#e3a64b", "#ed7d9a", "#82bd65", "#5f9ee8"]
        cells_by_id = {str(g["cell_id"]): g for g in gnb_states}
        for index, cell_id in enumerate(sorted({str(u["serving_cell_id"]) for u in ue_states})):
            cell = cells_by_id.get(cell_id)
            if cell is None:
                continue
            assigned = [u for u in ue_states if str(u["serving_cell_id"]) == cell_id]
            link_x, link_y = [], []
            for ue in assigned:
                link_x.extend([cell["x_m"], ue["x_m"], None])
                link_y.extend([cell["y_m"], ue["y_m"], None])
            topology.add_trace(go.Scatter(
                x=link_x, y=link_y, mode="lines", showlegend=False, hoverinfo="skip",
                line={"color": cell_palette[index % len(cell_palette)], "width": 1.2, "dash": "dot"},
            ))
        topology.add_trace(go.Scatter(
            x=[g["x_m"] for g in gnb_states], y=[g["y_m"] for g in gnb_states],
            mode="markers", name="gNB / serving cell", marker={"symbol": "square", "size": 15,
                "color": "#4aa3ff", "line": {"color": "#f8fafc", "width": 1.5}},
            customdata=[[g["gnb_id"], g["cell_id"]] for g in gnb_states],
            hovertemplate="gNB %{customdata[0]} · cell %{customdata[1]}<br>x=%{x:.1f} m, y=%{y:.1f} m<extra></extra>",
        ))
        populated_classes = [traffic_class for traffic_class in ("low_latency", "voice_qos", "gaming", "video_streaming")
                             if any(u.get("traffic_class") == traffic_class for u in ue_states)]
        for traffic_class, symbol, display in (
            ("low_latency", "circle", "Low-latency UE"),
            ("voice_qos", "diamond", "Voice UE"),
            ("gaming", "triangle-up", "Gaming UE"),
            ("video_streaming", "hexagon", "Video UE"),
        ):
            users = [u for u in ue_states if u.get("traffic_class") == traffic_class]
            if not users:
                continue
            topology.add_trace(go.Scatter(
                x=[u["x_m"] for u in users], y=[u["y_m"] for u in users],
                mode="markers", name=display,
                marker={"symbol": symbol, "size": 15,
                    "color": [u["qoe"] for u in users], "cmin": 0, "cmax": 1,
                    "colorscale": [[0, "#ff6b6b"], [0.72, "#ffb454"], [1, "#42d392"]],
                    "showscale": traffic_class == populated_classes[0],
                    **({"colorbar": {"title": "QoE<br>(0–1)", "thickness": 14, "len": 0.78}}
                       if traffic_class == populated_classes[0] else {}),
                    "line": {"color": "#f8fafc", "width": 1}},
                customdata=[[u["ue_id"], u["traffic_class"], u["serving_cell_id"], u["throughput_mbps"],
                             u["delay_ms"], u["jitter_ms"], 100 * u["packet_loss_fraction"], u["qoe"]]
                            for u in users],
                hovertemplate=("<b>UE %{customdata[0]}</b><br>Class: %{customdata[1]}<br>Serving cell: %{customdata[2]}"
                               "<br>Throughput: %{customdata[3]:.2f} Mbps<br>Delay: %{customdata[4]:.2f} ms"
                               "<br>Jitter: %{customdata[5]:.2f} ms<br>Loss: %{customdata[6]:.2f}%"
                               "<br>Measured QoE: %{customdata[7]:.3f}<extra></extra>"),
            ))
        topology.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e6edf3"},
            title="Network layout · hover or select a UE for details", xaxis_title="Position x (m)", yaxis_title="Position y (m)",
            height=520, margin={"t": 45, "b": 55, "l": 55, "r": 35},
            legend={"orientation": "h", "y": -0.16, "x": 0}, hovermode="closest")
        topology.update_yaxes(scaleanchor="x", scaleratio=1)
        selection = st_module.plotly_chart(topology, use_container_width=True,
                               key=f"topology_{selected}_{latest.get('segment_id', 'segment_001')}",
                               on_select="rerun", selection_mode="points")
        selection_key = f"{selected}:{latest_segment}"
        if st_module.session_state.get("live_ue_selection_key") != selection_key:
            st_module.session_state["live_ue_selection_key"] = selection_key
            st_module.session_state.pop("live_selected_ue_id", None)
        selected_ue_id = st_module.session_state.get("live_selected_ue_id")
        try:
            for point in selection.selection.points:
                custom = point.get("customdata")
                if custom and len(custom) > 1:
                    selected_ue_id = int(custom[0])
                    st_module.session_state["live_selected_ue_id"] = selected_ue_id
                    break
        except (AttributeError, TypeError, ValueError):
            selected_ue_id = None
        ue_frame = pd.DataFrame(ue_states).rename(columns={
            "ue_id": "UE", "traffic_class": "Traffic class", "serving_cell_id": "Serving cell",
            "throughput_mbps": "Throughput (Mbps)", "delay_ms": "Delay (ms)",
            "jitter_ms": "Jitter (ms)", "packet_loss_fraction": "Packet loss (fraction)",
            "qoe": "Measured QoE", "tx_packets": "TX packets in window", "rx_packets": "RX packets in window",
        })
        st_module.dataframe(ue_frame[["UE", "Traffic class", "Serving cell", "Throughput (Mbps)",
            "Delay (ms)", "Jitter (ms)", "Packet loss (fraction)", "Measured QoE",
            "TX packets in window", "RX packets in window"]], hide_index=True, use_container_width=True)
        st_module.caption("UE-to-cell assignments are listed in the table and hover details. Muted dotted links show serving cells; positions and assignments are measured from the active simulation segment.")
        if selected_ue_id is not None:
            selected_history = []
            for point in points:
                if point.get("segment_id") != latest_segment:
                    continue
                ue = next((item for item in point.get("ues", []) if int(item.get("ue_id", -1)) == selected_ue_id), None)
                if ue:
                    selected_history.append({"Simulation time (s)": point["sim_time_s"],
                        "Measured QoE": ue.get("qoe"), "Throughput (Mbps)": ue.get("throughput_mbps"),
                        "Delay (ms)": ue.get("delay_ms"), "Jitter (ms)": ue.get("jitter_ms"),
                        "Packet loss (%)": float(ue.get("packet_loss_fraction", 0)) * 100})
            selected_ue = next((item for item in ue_states if int(item["ue_id"]) == selected_ue_id), None)
            related_events = [event for event in data["events"]
                if selected_ue_id in [int(v) for v in event.get("affected_ue_ids", [])]
                or event.get("type") in {"session_started", "restart_epoch_launched", "restart_epoch_verified"}
                and selected_ue_id in [int(v) for v in event.get("affected_ue_ids", [])]]
            with st_module.expander(f"UE {selected_ue_id} details and QoE history", expanded=True):
                if selected_ue:
                    st_module.write(f"Traffic class: {selected_ue.get('traffic_class')} · Serving cell: {selected_ue.get('serving_cell_id')} · Latest measured QoE: {float(selected_ue.get('qoe', 0)):.3f} · Target: {float(meta.get('target_qoe', .72)):.3f}")
                    if selected_ue.get("qoe_formula_version") == QOE_FORMULA_VERSION:
                        scores = calculate_qoe_components(
                            float(selected_ue.get("throughput_mbps", 0)),
                            float(selected_ue.get("delay_ms", 0)),
                            float(selected_ue.get("jitter_ms", 0)),
                            float(selected_ue.get("packet_loss_fraction", 0)),
                            str(selected_ue.get("traffic_class", "")),
                            int(selected_ue.get("rx_packets", 0)),
                        )
                        component_cols = st_module.columns(4)
                        component_cols[0].metric("Throughput score", f"{scores['throughput']:.2f}", help="Contribution factor before its 0.40 QoE weight.")
                        component_cols[1].metric("Delay score", f"{scores['delay']:.2f}", help="Contribution factor before its 0.25 QoE weight.")
                        component_cols[2].metric("Jitter score", f"{scores['jitter']:.2f}", help="Contribution factor before its 0.20 QoE weight.")
                        component_cols[3].metric("Loss score", f"{scores['packet_loss']:.2f}", help="Contribution factor before its 0.15 QoE weight.")
                ue_forecast = ue_forecasts.get(selected_ue_id, {}).get("forecast", {})
                if ue_forecast.get("available"):
                    st_module.write(f"Individual forecast (+{ue_forecast.get('horizon_s')} s): QoE {ue_forecast.get('qoe', 0):.3f} · {'at risk' if ue_forecast.get('risk') else 'not at risk'} · {ue_forecast.get('confidence', 'low confidence')}")
                if selected_history:
                    ue_history = pd.DataFrame(selected_history)
                    ue_fig = go.Figure(go.Scatter(x=ue_history["Simulation time (s)"], y=ue_history["Measured QoE"],
                        mode="lines+markers", name=f"UE {selected_ue_id} measured QoE", line={"color": "#4aa3ff"}))
                    ue_fig.add_hline(y=float(meta.get("target_qoe", .72)), line_dash="dot", line_color="#ff6b6b", annotation_text="UE target")
                    selected_service = str((selected_ue or {}).get("traffic_class", ""))
                    service_floor = active_floors.get(selected_service)
                    if service_floor is not None:
                        ue_fig.add_hline(y=float(service_floor), line_dash="dash", line_color="#ffb454",
                                         annotation_text=f"{selected_service.replace('_', ' ')} safety floor")
                    ue_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e6edf3"},
                        xaxis_title="Simulation time within segment (s)", yaxis_title="Measured UE QoE (0–1)",
                        yaxis={"range": [0, 1]}, height=260, margin={"t": 20, "b": 30})
                    st_module.plotly_chart(ue_fig, use_container_width=True, key=f"ue_history_{selected}_{latest_segment}_{selected_ue_id}")
                    st_module.dataframe(ue_history.tail(20), hide_index=True, use_container_width=True)
                if related_events:
                    st_module.markdown("**Controller changes related to this UE**")
                    st_module.dataframe(pd.DataFrame([{"Time": e.get("timestamp"), "Action": e.get("action", e.get("type")),
                        "Status": e.get("status", "Recorded"), "Reason": e.get("reason", e.get("rationale", "")),
                        "Affected UEs": ", ".join(map(str, e.get("affected_ue_ids", [])))}
                        for e in related_events]), hide_index=True, use_container_width=True)
                else:
                    st_module.caption("No controller intervention is recorded as targeting this UE in the selected session.")
    elif latest and latest.get("ues") and latest.get("gnbs"):
        st_module.info(f"The topology from {latest_segment} is retained as history. Waiting for the first measurements from {expected_segment}; no previous-segment topology is presented as current.")
    elif latest:
        st_module.info("This session’s simulator build does not include live per-UE telemetry. Restart with the current live ns-3 build to view UE state.")
    else:
        st_module.info("Waiting for a measured per-UE telemetry window from ns-3.")

    if forecast.get("available"):
        st_module.info(f"Forecast (+{forecast['horizon_s']} s, low-confidence trend): {forecast.get('reason')} Predicted QoE {forecast['qoe']:.3f}. {forecast['method']}.")
    below_target = bool(latest and float(latest.get("qoe", 0)) < float(meta.get("target_qoe", .72)))
    controller_meta = dict(meta)
    controller_meta["recent_points"] = current_segment_points
    proposal = (recommend_control_action(controller_meta, latest, forecast,
        safety_floor_risk_ue_ids=safety_floor_risk_ids)
        if latest and forecast.get("available") else None)
    hybrid_evaluation = proposal.get("hybrid_evaluation") if proposal else None
    per_ue_risk = any(item.get("forecast", {}).get("risk") for item in ue_forecasts.values())
    healthy_maintenance = ue_target_state["all_current_met"]
    actionable_forecast = (bool(safety_floor_risk_ids) or len(forecast_at_risk_ues) >= ue_target_state["required_crossings"]
                           if healthy_maintenance else (forecast.get("risk") or per_ue_risk))
    actionable_forecast = bool(actionable_forecast or safety_floor_risk_ids)
    if (actionable_forecast or below_target_ues) and meta.get("mode") == "Recommend" and not meta.get("action_pending") and not meta.get("structural_action_pending"):
        if proposal and proposal.get("kind") == "rate":
            st_module.warning(f"Target-directed recommendation: {proposal['action']}. {proposal['reason']}")
            current_rate = int(active_params.get("lambda_be_pps", 10000))
            a, b = st_module.columns([1, 3])
            if a.button(f"Approve rate change · {int(proposal['value'])} pps / UE", key=f"approve_rate_{selected}", disabled=status != "Running"):
                try:
                    request_rate_change(selected, int(proposal["value"]), approved=True,
                        rationale=proposal["reason"], hybrid_evaluation=hybrid_evaluation)
                    st_module.success("Live rate change queued; ns-3 acknowledgment is required before it is shown as applied.")
                    st_module.rerun()
                except Exception as exc: st_module.error(str(exc))
            b.caption(f"Current rate: {current_rate} pps/UE · supported bounds 2,500–20,000 · cooldown 10 seconds.")
        elif proposal and proposal.get("kind") == "restart":
            requested = dict(meta.get("config", {})); requested.update(proposal["config"])
            st_module.warning(f"Structural recommendation: {proposal['action']}. This candidate has no guaranteed gain; the new segment will be measured and rolled back if it fails to improve QoE toward the target.")
            st_module.json({"current": meta.get("config"), "proposed restart epoch": requested})
            if st_module.button("Approve structural restart epoch", key=f"approve_restart_{selected}", disabled=status != "Running"):
                try:
                    request_structural_change(selected, proposal["config"], approved=True, rationale=proposal["reason"])
                    st_module.success("Restart epoch queued. The dashboard will show the transition and verify the new configuration from ns-3 telemetry.")
                    st_module.rerun()
                except Exception as exc: st_module.error(str(exc))
        elif proposal:
            st_module.info(f"No safe candidate available: {proposal.get('reason')}")
    elif latest and ue_forecasts:
        suffix = (" A single target crossing is normally monitored, but a service-floor warning can trigger protection earlier."
                  if healthy_maintenance and forecast_at_risk_ues else "")
        st_module.caption(f"Per-UE trends are calculated independently; {len(forecast_at_risk_ues)} UE(s) are forecast below target.{suffix}")

    if hybrid_evaluation:
        with st_module.expander("Hybrid model comparison · estimate, not measurement"):
            if hybrid_evaluation.get("available"):
                st_module.write(hybrid_evaluation.get("reason", ""))
                model_cols = st_module.columns(3)
                current_mean = hybrid_evaluation.get("measured_mean_qoe_before")
                hybrid_mean = hybrid_evaluation.get("hybrid_estimated_mean_qoe_after")
                model_cols[0].metric("Measured mean UE QoE now",
                    f"{current_mean:.3f}" if current_mean is not None else "—")
                model_cols[1].metric("Hybrid estimate after candidate",
                    f"{hybrid_mean:.3f}" if hybrid_mean is not None else f"{hybrid_evaluation['predicted_mean_qoe_after']:.3f}",
                    delta=f"Model KPI change {hybrid_evaluation['predicted_mean_delta']:+.3f}")
                model_cols[2].metric("Predicted worst-profile change",
                    f"{hybrid_evaluation['predicted_worst_delta']:+.3f}")
                st_module.caption(f"{hybrid_evaluation['model_version']} · {hybrid_evaluation['source']} · "
                    f"paired model spread of predicted change {hybrid_evaluation['uncertainty_qoe_delta_std']:.3f}. "
                    f"Raw model-only profile means: {hybrid_evaluation['predicted_mean_qoe_before']:.3f}→"
                    f"{hybrid_evaluation['predicted_mean_qoe_after']:.3f}. A subsequent ns-3 measurement is required to verify the outcome.")
            else:
                st_module.info(f"Model advice unavailable: {hybrid_evaluation.get('reason', 'unsupported context')}")

    if meta.get("action_pending"):
        st_module.warning("Action pending simulator confirmation. It is not yet reported as applied.")
    elif meta.get("last_action_status"):
        st_module.info(meta["last_action_status"])
    if (meta.get("action_verified") and not meta.get("action_is_rollback")
            and meta.get("mode") in {"Recommend", "Automatic"}):
        previous_rate = int(meta.get("action_previous_pps", active_params.get("lambda_be_pps", 10000)))
        if st_module.button(f"Restore previous offered load · {previous_rate} packets/s per UE", key=f"rollback_{selected}"):
            try:
                request_rate_change(selected, previous_rate, approved=True, rollback=True)
                st_module.info("Restore request queued; ns-3 must acknowledge it before it is shown as applied.")
                st_module.rerun()
            except Exception as exc: st_module.error(f"Could not restore the previous load: {exc}")
    structural_trial = meta.get("structural_trial", {})
    if structural_trial.get("rollback_recommended") and meta.get("mode") == "Recommend":
        st_module.warning("The restart epoch did not improve measured QoE toward target. The controller recommends restoring the prior configuration.")
        if st_module.button("Approve structural rollback", key=f"rollback_structural_{selected}", disabled=status != "Running"):
            try:
                request_structural_change(selected, dict(structural_trial["previous_config"]),
                    approved=True, rollback=True,
                    rationale="User-approved recovery: restart epoch failed to improve QoE toward target.")
                st_module.info("Rollback restart queued; the restored configuration will be checked from new telemetry.")
                st_module.rerun()
            except Exception as exc: st_module.error(f"Could not queue structural rollback: {exc}")

    segments = meta.get("segments", [])
    if segments:
        st_module.subheader("Configuration restart epochs")
        st_module.caption("Each row is a separate ns-3 process segment within this logical session. Simulated time restarts at zero within each segment.")
        segment_rows = [{"Segment": f"{item.get('segment', '?'):03d}", "Status": item.get("status", "Unknown"),
                         "UEs": item.get("config", {}).get("ue_count"), "gNBs": item.get("config", {}).get("gnb_count"),
                         "Association": item.get("config", {}).get("ue_association_policy"),
                         "Bandwidth (MHz)": item.get("config", {}).get("bandwidth_mhz"),
                         "Power (dBm)": item.get("config", {}).get("total_tx_power_dbm"),
                         "Started": item.get("started_at", "")}
                        for item in segments]
        st_module.dataframe(pd.DataFrame(segment_rows), hide_index=True, use_container_width=True)

    st_module.subheader("Live network feed" if status == "Running" else "Recent measurement history")
    if points:
        frame = pd.DataFrame(points)
        available_segments = sorted(frame["segment_id"].dropna().astype(str).unique())
        expected_segment_id = f"segment_{int(meta.get('current_segment', 1)):03d}"
        default_segment = (expected_segment_id if expected_segment_id in available_segments
                           else available_segments[-1])
        selected_segment = st_module.selectbox(
            "Chart segment", available_segments, index=available_segments.index(default_segment),
            key=f"history_chart_segment_{selected}",
            help="Show one ns-3 process segment at a time. Simulation time restarts at zero for each segment.")
        chart_frame = frame[frame["segment_id"].astype(str) == selected_segment]
        with st_module.container(border=True):
            st_module.subheader("QoE trend")
            fig = go.Figure()
            basis = ("mean per-UE" if "qoe_basis" in chart_frame
                     and (chart_frame.qoe_basis == "equal_weight_mean_per_ue").all()
                     else "pooled legacy")
            fig.add_trace(go.Scatter(x=chart_frame.sim_time_s, y=chart_frame.qoe, mode="lines",
                    name=f"Measured QoE · {basis}", line={"color":"#4aa3ff", "width":2}))
            if forecast.get("available") and selected_segment == expected_segment_id:
                fig.add_trace(go.Scatter(x=[chart_frame.sim_time_s.iloc[-1], forecast["at_sim_time_s"]],
                                         y=[chart_frame.qoe.iloc[-1], forecast["qoe"]], mode="lines+markers",
                                         name="Trend forecast · not measured", line={"dash":"dash","color":"#ffb454"}))
            fig.add_hline(y=float(meta.get("target_qoe",.72)), line_dash="dot", line_color="#ff6b6b", annotation_text="QoE target")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"color":"#e6edf3"}, xaxis_title="Simulation time within segment (s)", yaxis_title="Mean per-UE QoE proxy (0–1)",
                              yaxis={"range":[0,1]}, height=320, margin={"t":25,"b":30},
                              legend={"orientation":"h", "y":-0.22})
            st_module.plotly_chart(fig, use_container_width=True)

        with st_module.container(border=True):
            st_module.subheader("Throughput")
            throughput_fig = go.Figure()
            throughput_fig.add_trace(go.Scatter(x=chart_frame.sim_time_s, y=chart_frame.throughput_mbps,
                name="Measured throughput", mode="lines", line={"color":"#4aa3ff", "width":2}))
            throughput_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                         font={"color":"#e6edf3"}, xaxis_title="Simulation time within segment (s)",
                                         yaxis_title="Throughput (Mbps)", height=300, margin={"t":20,"b":30},
                                         showlegend=False)
            st_module.plotly_chart(throughput_fig, use_container_width=True)

        with st_module.container(border=True):
            st_module.subheader("Delay and jitter")
            latency_fig = go.Figure()
            latency_fig.add_trace(go.Scatter(x=chart_frame.sim_time_s, y=chart_frame.delay_ms,
                name="Measured delay", mode="lines", line={"color":"#ffb454", "width":2}))
            latency_fig.add_trace(go.Scatter(x=chart_frame.sim_time_s, y=chart_frame.jitter_ms,
                name="Measured jitter", mode="lines", line={"color":"#b69cff", "width":2}))
            latency_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font={"color":"#e6edf3"}, xaxis_title="Simulation time within segment (s)",
                                      yaxis_title="Time (ms)", height=300, margin={"t":20,"b":30},
                                      legend={"orientation":"h", "y":-0.25})
            st_module.plotly_chart(latency_fig, use_container_width=True)

        with st_module.container(border=True):
            st_module.subheader("Packet loss")
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(x=chart_frame.sim_time_s,
                    y=chart_frame.packet_loss_fraction * 100, name="Measured packet loss",
                    mode="lines", line={"color":"#ff7d8a", "width":2}))
            loss_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font={"color":"#e6edf3"}, xaxis_title="Simulation time within segment (s)",
                                   yaxis_title="Packet loss (%)", height=300, margin={"t":20,"b":30},
                                   showlegend=False)
            st_module.plotly_chart(loss_fig, use_container_width=True)
        cols = ["segment_id", "window_start_s", "sim_time_s", "throughput_mbps", "delay_ms", "jitter_ms", "packet_loss_fraction", "qoe", "lambda_be_pps"]
        if "window_start_s" not in frame:
            cols.remove("window_start_s")
        st_module.dataframe(chart_frame[cols].tail(15).rename(columns={"segment_id":"Run segment", "window_start_s":"Window start (s)", "sim_time_s":"Window end (s)", "throughput_mbps":"Throughput (Mbps)", "delay_ms":"Delay (ms)", "jitter_ms":"Jitter (ms)", "packet_loss_fraction":"Packet loss (fraction)", "qoe":"Measured QoE", "lambda_be_pps":"Active offered load (pps/UE)"}), hide_index=True, use_container_width=True)
        st_module.caption("Charts show one selected ns-3 segment at a time; measurements cover each displayed window. A just-applied traffic-rate change may first appear in the following full window.")
    else:
        st_module.info("Waiting for the first real ns-3 telemetry window. No sample data is shown.")

    st_module.subheader("Live controller and action feed")
    if data["events"]:
        feed, condensed_count = format_action_feed(data["events"])
        if condensed_count:
            st_module.caption(f"Showing concise event summaries; {condensed_count} repeated no-action decisions are grouped. The full event ledger remains available below.")
        st_module.dataframe(feed, hide_index=True, use_container_width=True, column_config={
            "Time (UTC)": st_module.column_config.TextColumn(width="medium"),
            "Event": st_module.column_config.TextColumn(width="medium"),
            "Change / decision": st_module.column_config.TextColumn(width="medium"),
            "Status": st_module.column_config.TextColumn(width="small"),
            "Details": st_module.column_config.TextColumn(width="large"),
        })
        with st_module.expander("Full event ledger · raw recorded details"):
            st_module.dataframe(pd.DataFrame(data["events"]).iloc[::-1], hide_index=True, use_container_width=True)
    else:
        st_module.caption("No intervention has been recorded. Decisions, simulator acknowledgements, and follow-up verification are stored with this session.")
    with st_module.expander("Session diagnostics"):
        st_module.caption(f"Persistent session directory: {folder}")
        try:
            from pathlib import Path
            segment_path = Path(folder, "segments", f"segment_{int(meta.get('current_segment', 1)):03d}")
            if not segment_path.exists():
                segment_path = Path(folder)
            run_config = segment_path / "run_config.json"
            if run_config.is_file():
                st_module.json(json.loads(run_config.read_text()))
            log = (segment_path / "simulator.log").read_text(errors="replace")[-5000:]
            st_module.text_area("Recent ns-3 log", log, height=180)
        except Exception as exc:
            st_module.caption(f"Log not available: {exc}")


    _render_predictive_demand_trial(st_module, active)
