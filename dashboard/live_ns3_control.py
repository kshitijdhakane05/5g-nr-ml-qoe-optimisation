"""Streamlit page for live, epoch-based ns-3 QoE measurements and control."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from dashboard.components import page_header

from simulation.live_control import (
    EXPANSION_RESULTS,
    LIVE_RESULTS,
    PRESET_DEFINITIONS,
    assess_candidate_risk,
    configuration_fingerprint,
    create_live_session,
    list_live_sessions,
    load_verified_configurations,
    mark_worker_lost,
    production_feature_contract_audit,
    read_session_events,
    read_session_state,
    request_session_stop,
    recommended_timeout_seconds,
    supported_radio_grid,
    validate_exploratory_candidate,
    validate_custom_config,
)


def build_qoe_figure(events: list[dict[str, Any]], target_qoe: float) -> go.Figure:
    """Plot measured outcomes and pre-run empirical estimates as separate traces."""
    epochs = [int(event["epoch"]) for event in events]
    customdata = []
    for event in events:
        cfg = event.get("config", {})
        metrics = event.get("metrics") or {}
        customdata.append([
            cfg.get("config_id", "unknown"), cfg.get("ue_count", "?"),
            cfg.get("bandwidth_band1_mhz", "?"), cfg.get("bandwidth_band2_mhz", "?"),
            cfg.get("total_tx_power_dbm", "?"),
            "outage" if metrics.get("is_outage") else event.get("run_status", "unknown"),
            event.get("controller_action", "pending"),
        ])
    measured = [
        float(event["measured_qoe_proxy"])
        if event.get("run_status") == "COMPLETED" and event.get("measured_qoe_proxy") is not None
        else None
        for event in events
    ]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=epochs,
        y=measured,
        mode="lines+markers",
        name="Measured QoE proxy",
        customdata=customdata,
        connectgaps=False,
        marker={"size": 10, "color": "#1769aa"},
        line={"color": "#1769aa", "width": 3},
        hovertemplate=("Epoch %{x}<br>Measured QoE %{y:.4f}<br>Config %{customdata[0]}"
                       "<br>UE %{customdata[1]} · B1/B2 %{customdata[2]}/%{customdata[3]} MHz"
                       "<br>Power %{customdata[4]} dBm · %{customdata[5]}"
                       "<br>Decision %{customdata[6]}<extra></extra>"),
    ))
    predicted = [float(event["predicted_qoe"]) if event.get("predicted_qoe") is not None else None
                 for event in events]
    prediction_errors = []
    has_error_band = False
    for event, value in zip(events, predicted):
        lo, hi = event.get("predicted_qoe_lower"), event.get("predicted_qoe_upper")
        if value is not None and lo is not None and hi is not None:
            prediction_errors.append([max(0.0, value - float(lo)), max(0.0, float(hi) - value)])
            has_error_band = True
        else:
            prediction_errors.append([0.0, 0.0])
    figure.add_trace(go.Scatter(
        x=epochs, y=predicted, mode="lines+markers", name="Pre-run empirical prediction",
        customdata=customdata, connectgaps=False,
        marker={"size": 9, "symbol": "diamond", "color": "#a855f7"},
        line={"color": "#a855f7", "width": 2, "dash": "dash"},
        error_y={"type": "data", "symmetric": False, "array": [v[1] for v in prediction_errors],
                 "arrayminus": [v[0] for v in prediction_errors], "visible": has_error_band},
        hovertemplate=("Epoch %{x}<br>Pre-run estimate %{y:.4f}<br>Config %{customdata[0]}"
                       "<br>Domain estimate only; see run count/support in history<extra></extra>"),
    ))
    figure.add_hline(y=float(target_qoe), line_dash="dash", line_color="#d62728",
                     annotation_text=f"Target {float(target_qoe):.2f}", annotation_position="top left")
    figure.update_layout(
        title="Simulated network quality score: estimate and measured outcome",
        xaxis_title="Simulation epoch / iteration",
        yaxis_title="Simulated network quality score (0–1)",
        yaxis={"range": [0, 1.02]},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
        height=430,
        margin={"l": 40, "r": 20, "t": 75, "b": 40},
    )
    return figure


def _config_caption(config: dict[str, Any]) -> str:
    return (
        f"{config.get('config_id', '—')} | UEs={config.get('ue_count', '—')} | "
        f"gNBs={config.get('gnb_count', 1)} ({config.get('ue_association_policy', 'closest')} UE assignment) | "
        f"Band 1={config.get('bandwidth_band1_mhz', '—')} MHz, "
        f"Band 2={config.get('bandwidth_band2_mhz', '—')} MHz | "
        f"requested total power={config.get('total_tx_power_dbm', '—')} dBm"
    )


def _history_table(events: list[dict[str, Any]]) -> pd.DataFrame:
    def optional_int(value: Any) -> int | None:
        return None if value in (None, "") else int(value)

    rows = []
    for event in events:
        config = event.get("config", {})
        metrics = event.get("metrics") or {}
        rows.append({
            "Epoch": event.get("epoch"),
            "Status": event.get("run_status"),
            "Config": config.get("config_id"),
            "Exploration status": event.get("candidate_scope", event.get("configuration_scope")),
            "UEs": config.get("ue_count"),
            "gNB cells": config.get("gnb_count", 1),
            "gNBs": config.get("gnb_count", 1),
            "Band 1 MHz": config.get("bandwidth_band1_mhz"),
            "Band 2 MHz": config.get("bandwidth_band2_mhz"),
            "Total power dBm": config.get("total_tx_power_dbm"),
            "Throughput Mbps": metrics.get("mean_throughput_mbps"),
            "Latency ms": metrics.get("mean_delay_ms"),
            "Jitter ms": metrics.get("mean_jitter_ms"),
            "Packet loss %": (
                float(metrics["mean_packet_loss"]) * 100 if "mean_packet_loss" in metrics else None
            ),
            "Measured QoE proxy": event.get("measured_qoe_proxy"),
            "Prior QoE proxy": event.get("prior_qoe_proxy"),
            "Current QoE proxy": event.get("current_qoe_proxy", event.get("measured_qoe_proxy")),
            "Pre-run predicted QoE": event.get("predicted_qoe"),
            "Prediction error": event.get("prediction_error"),
            "Prediction support n": event.get("prediction_run_count"),
            "Prediction domain": event.get("prediction_domain"),
            "Controller phase": event.get("controller_phase"),
            "Outage": metrics.get("is_outage"),
            "QoE change vs previous valid": event.get("previous_valid_qoe_delta"),
            "Action": event.get("controller_action"),
            "Action outcome": event.get("action_outcome_from_previous"),
            "Candidate evidence": event.get("candidate_evidence"),
            "UE detail status": event.get("ue_measurements_status", "Unavailable"),
            "UE detail rows": len(event.get("ue_measurements") or []),
            "Risk level": event.get("candidate_risk_level"),
            "Risk evidence": event.get("candidate_risk_reason"),
            "Selected config risk": event.get("selected_candidate_risk_level", event.get("candidate_risk_level")),
            "Selected config risk evidence": event.get("selected_candidate_risk_reason", event.get("candidate_risk_reason")),
            "Next config risk": event.get("next_candidate_risk_level"),
            "High-risk override": event.get("high_risk_override"),
            "Adaptive actions used": optional_int(event.get("adaptive_action_count")),
            "Exploratory trials used": optional_int(event.get("exploratory_trial_count")),
            "Normal action budget reached": event.get("action_budget_exhausted"),
            "Post-limit experimental trial": event.get("post_epoch_exploration"),
            "Continuation reason": event.get("continuation_reason"),
            "Next continuation reason": event.get("next_continuation_reason"),
            "Rationale": event.get("action_rationale"),
            "Duration s": event.get("duration_seconds"),
            "Error": event.get("error_message"),
        })
    return pd.DataFrame(rows)


def session_run_counts(session_dir: str | Path, events: list[dict[str, Any]],
                       state: dict[str, Any]) -> dict[str, int]:
    """Count unique run attempts from per-run records, with event-log fallback."""
    folder = Path(session_dir)
    normal_limit = int(state.get("max_epochs", 0))
    records: dict[str, dict[str, Any]] = {}
    for run_config_path in sorted((folder / "epochs").glob("*/run_config.json")):
        try:
            record = json.loads(run_config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        key = str(record.get("run_id") or record.get("epoch") or run_config_path.parent.name)
        records[key] = record
    for event in events:
        key = str(event.get("run_id") or event.get("epoch") or "")
        if key and key not in records:
            records[key] = event

    counts = {
        "normal_attempted": 0, "normal_completed": 0, "normal_failed": 0,
        "experimental_attempted": 0, "experimental_completed": 0, "experimental_failed": 0,
        "validation_attempted": 0, "validation_completed": 0, "validation_failed": 0,
    }
    active_statuses = {"STARTING", "RUNNING", "PARSING_RESULTS", "RESERVED"}
    for record in records.values():
        epoch = int(record.get("epoch", 0) or 0)
        if record.get("is_validation_epoch"):
            prefix = "validation"
        else:
            is_experimental = record.get("post_epoch_exploration")
            if is_experimental is None:
                is_experimental = epoch > normal_limit
            prefix = "experimental" if is_experimental else "normal"
        counts[f"{prefix}_attempted"] += 1
        status = str(record.get("run_status", "UNKNOWN")).upper()
        if status == "COMPLETED":
            counts[f"{prefix}_completed"] += 1
        elif status not in active_statuses:
            counts[f"{prefix}_failed"] += 1
    return counts


def _run_count_caption(counts: dict[str, int], state: dict[str, Any]) -> str:
    normal_limit = int(state.get("max_epochs", 0))
    extra_limit = (int(state.get("max_additional_exploratory_epochs", 0))
                   if state.get("run_remaining_experimental_candidates") else 0)
    validation_limit = int(state.get("stability_validation_epochs", 5))
    return (
        f"Normal attempts: {counts['normal_attempted']}/{normal_limit} "
        f"(completed {counts['normal_completed']}, failed/interrupted {counts['normal_failed']}) · "
        f"additional experimental attempts: {counts['experimental_attempted']}/{extra_limit} "
        f"(completed {counts['experimental_completed']}, failed/interrupted {counts['experimental_failed']}) · "
        f"additional stability-validation runs: {counts['validation_attempted']}/{validation_limit} "
        f"(completed {counts['validation_completed']}, failed/interrupted {counts['validation_failed']})."
    )


def activate_session(session_state: Any, session_id: str, session_dir: str | Path) -> None:
    """Activate a session and defer widget selection until the next script run."""
    session_state["live_active_session_id"] = session_id
    session_state["live_session_id"] = session_id
    session_state["live_session_dir"] = str(session_dir)
    session_state["live_pending_session_review_id"] = session_id


def prepare_session_review_state(session_state: Any, session_ids: list[str]) -> None:
    """Apply a pending selection before Streamlit instantiates its keyed widget."""
    pending = session_state.get("live_pending_session_review_id")
    if pending in session_ids:
        session_state["live_session_review"] = pending
    if pending is not None:
        session_state.pop("live_pending_session_review_id", None)


def session_status_label(state: dict[str, Any]) -> str:
    status = state.get("status", "UNKNOWN")
    if status in {"QUEUED", "RUNNING"} and state.get("stop_requested"):
        return "Stopping"
    labels = {
        "IDLE": "Idle",
        "COMPLETED": "Completed",
        "STOPPING": "Stopping",
        "STOP_REQUESTED": "Stopping",
        "TARGET_REACHED": "Target reached",
        "SUCCESS_STABLE": "Target stable across validation runs",
        "VALIDATION_FAILED": "Target stability validation failed",
        "EPOCH_LIMIT_REACHED": "Epoch limit reached",
        "EXPLORATORY_EPOCH_LIMIT_REACHED": "Additional experimental epoch limit reached",
        "NO_EXPERIMENTAL_CANDIDATES": "No eligible experimental candidates remain",
        "STOP_NO_EXPERIMENTAL_CANDIDATES": "No eligible experimental candidates remain",
        "STOP_NO_SAFE_CANDIDATES": "No safe candidates under current risk policy",
        "NO_VALIDATED_CANDIDATES": "No validated candidates",
        "NO_MEANINGFUL_IMPROVEMENT": "No meaningful improvement",
        "NO_FEASIBLE_CANDIDATES": "No validated candidates (legacy stop label)",
        "NO_IMPROVEMENT_AFTER_AVAILABLE_TRIALS": "No improvement after available trials",
        "SIMULATION_FAILURE": "Simulation failure",
        "SIMULATION_TIMEOUT": "Simulation timeout",
        "USER_CANCELLED": "User cancelled",
        "FAILED": "Simulation failure",
        "TIMED_OUT": "Simulation timeout",
        "CANCELLED": "User cancelled",
    }
    if status in {"TARGET_REACHED", "EPOCH_LIMIT_REACHED", "EXPLORATORY_EPOCH_LIMIT_REACHED",
                  "NO_EXPERIMENTAL_CANDIDATES", "STOP_NO_EXPERIMENTAL_CANDIDATES",
                  "STOP_NO_SAFE_CANDIDATES", "NO_VALIDATED_CANDIDATES",
                  "NO_MEANINGFUL_IMPROVEMENT", "NO_FEASIBLE_CANDIDATES",
                  "NO_IMPROVEMENT_AFTER_AVAILABLE_TRIALS", "ACTION_LIMIT_REACHED",
                  "SUCCESS_STABLE", "VALIDATION_FAILED"}:
        qoe = state.get("latest_measured_qoe_proxy")
        target = state.get("target_qoe")
        if qoe is not None and target is not None and float(qoe) < float(target):
            return f"Target unmet · {labels.get(status, status)}"
    return labels.get(status, status.replace("_", " ").title())


def session_status_explanation(state: dict[str, Any]) -> str:
    """Return a brief plain-language explanation for the saved session status."""
    status = state.get("status", "UNKNOWN")
    if status in {"QUEUED", "RUNNING"} and state.get("stop_requested"):
        return "A safe stop was requested. The owned ns-3 process is being cancelled."
    explanations = {
        "IDLE": "No simulation process is running.",
        "COMPLETED": "The session completed its configured work.",
        "STOPPING": "A safe stop was requested for this session.",
        "STOP_REQUESTED": "A safe stop was requested for this session.",
        "QUEUED": "The session is queued and will start its first ns-3 process shortly.",
        "RUNNING": "A simulation run is active. Status and validated measurements refresh automatically.",
        "TARGET_REACHED": "Validated simulation measurements met the configured target criterion.",
        "EPOCH_LIMIT_REACHED": "The configured number of normal simulation runs has been used.",
        "EXPLORATORY_EPOCH_LIMIT_REACHED": "The separate additional experimental run limit has been used.",
        "USER_CANCELLED": "The user requested a safe stop for this session.",
        "CANCELLED": "The user requested a safe stop for this session.",
        "SIMULATION_FAILURE": "A simulator run failed. Review its status and logs before retrying.",
        "SIMULATION_TIMEOUT": "A simulator run exceeded its timeout and was stopped.",
        "FAILED": "The session failed. Review its process output and error details.",
        "TIMED_OUT": "A simulator run exceeded its timeout and was stopped.",
    }
    return explanations.get(status, state.get("final_outcome") or "Review the session details for its latest controller outcome.")


def legacy_candidate_scope_note(state: dict[str, Any]) -> str | None:
    if state.get("status") == "NO_FEASIBLE_CANDIDATES":
        return ("Legacy stop policy: this referred only to the old campaign candidate registry. "
                "The two-repetition results do not rule out other ns-3-supported settings; "
                "new sessions can opt in to those exploratory candidates.")
    return None


def candidate_and_outcome_summary(events: list[dict[str, Any]], state: dict[str, Any],
                                  session_config: dict[str, Any]) -> dict[str, Any]:
    """Summarize measured best, attempted configs and remaining opt-in choices."""
    valid = [e for e in events if e.get("run_status") == "COMPLETED"
             and e.get("measured_qoe_proxy") is not None]
    best = max(valid, key=lambda e: float(e["measured_qoe_proxy"])) if valid else None
    attempted = {
        (int(e["config"]["ue_count"]), float(e["config"]["bandwidth_band1_mhz"]),
         float(e["config"]["total_tx_power_dbm"]))
        for e in events if e.get("config")
    }
    attempted_fingerprints = {
        configuration_fingerprint(e["config"]) for e in events if e.get("config")
    }


    ue = int((state.get("current_config") or state.get("initial_config") or {}).get("ue_count", -1))
    performance = {tuple([int(r["key"][0]), float(r["key"][1]), float(r["key"][2])]): r
                   for r in session_config.get("candidate_performance", [])}
    pilot_performance = {tuple([int(r["key"][0]), float(r["key"][1]), float(r["key"][2])]): r
                        for r in session_config.get("pilot_candidate_performance", [])}
    live_performance = {tuple([int(r["key"][0]), float(r["key"][1]), float(r["key"][2])]): r
                        for r in session_config.get("live_candidate_performance", [])}
    validated = [c for c in session_config.get("feasible_configurations", []) if int(c["ue_count"]) == ue]
    remaining_validated_raw = [c for c in validated if (
        int(c["ue_count"]), float(c["bandwidth_band1_mhz"]), float(c["total_tx_power_dbm"])
    ) not in attempted]
    validated_keys = {(int(c["ue_count"]), float(c["bandwidth_band1_mhz"]), float(c["total_tx_power_dbm"]))
                      for c in validated}
    exploratory_configs = session_config.get("exploratory_configurations")
    if exploratory_configs is None:
        template = state.get("initial_config") or state.get("current_config") or {}
        exploratory_configs = supported_radio_grid(template, validated)
    exploratory = []
    exploratory_raw = []
    reference = state.get("current_config") or state.get("initial_config") or {}
    topology_remaining = []
    for candidate in exploratory_configs:
        key = (int(candidate["ue_count"]), float(candidate["bandwidth_band1_mhz"]),
               float(candidate["total_tx_power_dbm"]))
        is_topology = bool(candidate.get("topology_candidate", False))
        if int(candidate["ue_count"]) != ue:
            continue
        if is_topology:
            if configuration_fingerprint(candidate) in attempted_fingerprints:
                continue
        elif key in attempted or key in validated_keys:
            continue
        try:
            validate_exploratory_candidate(candidate, reference)
        except (ValueError, TypeError, KeyError):
            continue
        (topology_remaining if is_topology else exploratory_raw).append(candidate)
    allow_exploratory = bool(state.get("allow_exploratory_candidates", False))
    allow_risk_override = bool(state.get("allow_high_risk_candidates", False))
    risk_excluded = []
    risk_review = []
    remaining_validated = []
    for candidate in remaining_validated_raw:
        risk = assess_candidate_risk((int(candidate["ue_count"]), float(candidate["bandwidth_band1_mhz"]),
                                      float(candidate["total_tx_power_dbm"])), campaign=performance,
                                    pilot=pilot_performance, live=live_performance)
        if risk["excluded_by_default"] and not allow_risk_override:
            risk_excluded.append({"config_id": candidate.get("config_id"), "reason": risk["reason"]})
            risk_review.append({"config_id": candidate.get("config_id"), "reason": risk["reason"], "decision": "excluded"})
        else:
            if risk["excluded_by_default"]:
                risk_review.append({"config_id": candidate.get("config_id"), "reason": risk["reason"],
                                    "decision": "permitted by advanced override"})
            remaining_validated.append(candidate)
    for candidate in exploratory_raw:
        key = (int(candidate["ue_count"]), float(candidate["bandwidth_band1_mhz"]),
               float(candidate["total_tx_power_dbm"]))
        risk = assess_candidate_risk(key, campaign=performance, pilot=pilot_performance, live=live_performance)
        if risk["excluded_by_default"] and not allow_risk_override:
            risk_excluded.append({"config_id": candidate.get("config_id"), "reason": risk["reason"]})
            risk_review.append({"config_id": candidate.get("config_id"), "reason": risk["reason"], "decision": "excluded"})
        else:
            if risk["excluded_by_default"]:
                risk_review.append({"config_id": candidate.get("config_id"), "reason": risk["reason"],
                                    "decision": "permitted by advanced override"})
            exploratory.append(candidate)
    for candidate in topology_remaining:
        risk_review.append({
            "config_id": candidate.get("config_id"),
            "reason": "Supported multi-gNB scenario construction; no matching topology outcome exists in the one-gNB campaign.",
            "decision": "unvalidated; requires simulation",
        })
    post_epoch_enabled = bool(state.get("run_remaining_experimental_candidates", False))
    return {
        "target_gap": max(0.0, float(state.get("target_qoe", 0.0)) - float(best["measured_qoe_proxy"])) if best else None,
        "best_qoe": float(best["measured_qoe_proxy"]) if best else None,
        "best_config": (best.get("config") or {}).get("config_id") if best else None,
        "attempted_configs": list(dict.fromkeys((e.get("config") or {}).get("config_id", "unknown") for e in events)),
        "remaining_validated_count": len(remaining_validated),
        "remaining_exploratory_count": len(exploratory),
        "remaining_topology_count": len(topology_remaining) if state.get("allow_topology_exploration") else 0,
        "remaining_candidate_count": (len(remaining_validated)
                                      + (len(exploratory) if allow_exploratory or post_epoch_enabled else 0)
                                      + (len(topology_remaining) if state.get("allow_topology_exploration") else 0)),
        "remaining_eligible_count": (len(remaining_validated)
                                     + (len(exploratory) if allow_exploratory or post_epoch_enabled else 0)
                                     + (len(topology_remaining) if state.get("allow_topology_exploration") else 0)),
        "remaining_risk_excluded_count": len(risk_excluded),
        "risk_excluded_candidates": risk_excluded,
        "risk_review_candidates": risk_review,
        "high_risk_override": allow_risk_override,
        "repeat_trials_enabled": bool(state.get("allow_repeat_trials", False)),
        "exploratory_enabled": allow_exploratory,
        "post_epoch_exploration_enabled": post_epoch_enabled,
        "post_epoch_exploration_count": sum(bool(e.get("post_epoch_exploration")) for e in events),
        "post_epoch_exploration_limit": int(state.get("max_additional_exploratory_epochs", 0)),
    }


def recovery_summary(events: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    """Summarize a target miss using validated measurements and the real tolerance."""
    valid = [event for event in events if event.get("run_status") == "COMPLETED"
             and event.get("measured_qoe_proxy") is not None]
    target = float(state.get("target_qoe", 0.8))
    tolerance = float(state.get("target_tolerance", 0.005))
    target_band = target - tolerance
    latest = valid[-1] if valid else None
    best = max(valid, key=lambda event: float(event["measured_qoe_proxy"])) if valid else None
    counts = session_run_counts(state.get("session_dir", "."), events, state)
    return {
        "target_qoe": target,
        "target_tolerance": tolerance,
        "target_band": target_band,
        "latest_qoe": float(latest["measured_qoe_proxy"]) if latest else None,
        "best_qoe": float(best["measured_qoe_proxy"]) if best else None,
        "shortfall": max(0.0, target_band - float(best["measured_qoe_proxy"])) if best else None,
        "target_met": bool(best and float(best["measured_qoe_proxy"]) >= target_band),
        "epochs_used": counts["normal_attempted"] + counts["experimental_attempted"],
        "stop_reason": state.get("final_outcome") or state.get("status", "Unknown"),
        "best_config": dict(best.get("config") or {}) if best else {},
        "original_config": dict(state.get("initial_config") or {}),
        "latest_event": latest,
        "best_event": best,
    }


def recovery_session_parameters(
    parent_session_id: str,
    summary: dict[str, Any],
    parent_state: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    run_budget: int,
    start_from_best: bool,
    allow_exploratory: bool = False,
    allow_high_risk: bool = False,
) -> dict[str, Any]:
    """Build a bounded, separate recovery session using only validated prior measurements."""
    budget = int(run_budget)
    if not 1 <= budget <= 6:
        raise ValueError("Recovery run budget must be between 1 and 6.")
    initial = summary["best_config"] if start_from_best else summary["original_config"]
    if not initial:
        raise ValueError("Recovery requires a previously recorded starting configuration.")
    last_rng_run = max(
        (int(item.get("rng_run", 0) or 0) for item in events),
        default=int(parent_state.get("rng_run_start", 1)) - 1,
    )
    return {
        "initial_config": dict(initial),
        "target_qoe": float(summary["target_qoe"]),
        "max_epochs": budget,
        "mode": "adaptive",
        "preset_name": f"Recovery from {parent_session_id}",
        "rng_seed": int(parent_state.get("rng_seed", 20261026)),
        "rng_run_start": last_rng_run + 1,
        "timeout_seconds": float(parent_state.get("timeout_seconds", 120)),
        "max_actions": int(parent_state.get("max_actions", 3)),
        "allow_exploratory_candidates": bool(allow_exploratory),
        "allow_high_risk_candidates": bool(allow_high_risk),
        "allow_repeat_trials": False,
        "run_remaining_experimental_candidates": False,
        "target_tolerance": float(summary["target_tolerance"]),
        "stable_epochs_required": int(parent_state.get("stable_epochs_required", 2)),
        "maintenance_margin": float(parent_state.get("maintenance_margin", 0.02)),
        "maintenance_hysteresis": float(parent_state.get("maintenance_hysteresis", 0.02)),
        "recovery_context": {
            "parent_session_id": parent_session_id,
            "parent_stop_reason": summary["stop_reason"],
            "parent_target_qoe": float(summary["target_qoe"]),
            "parent_target_tolerance": float(summary["target_tolerance"]),
            "parent_best_measured_qoe": summary["best_qoe"],
            "parent_best_configuration": dict(summary["best_config"]),
            "recovery_run_budget": budget,
        },
    }


def render_live_ns3_control(st: Any) -> None:
    page_header(
        "Live Simulator",
        "Configure an ns-3 scenario, run it, and inspect validated network measurements.",
        "Simulation only · each run starts a separate ns-3 process. No in-process or real-hardware control is performed.",
    )
    st.caption("The simulated network quality score (0–1) is a KPI proxy, not subjective user QoE.")
    model_audit = production_feature_contract_audit()
    with st.expander("Prediction and model scope", expanded=False):
        st.write(model_audit["reason"])
        st.caption("The production ML model remains disconnected. The live page uses an exact-condition empirical lookup with limited run-level support.")
        st.caption("The separate expanded-data Random Forest reported grouped leave-one-configuration-out MAE 0.02423 and RMSE 0.05181 (unitless QoE proxy, 11 configurations/22 runs); this is within-study evidence only, and that all-condition artifact is not loaded for control.")

    try:
        validated_configs = load_verified_configurations()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Validated expansion configurations are unavailable: {exc}")
        return

    # Select the session before presenting setup controls so the user can orient
    # themselves and review a historical run without changing or restarting it.
    st.subheader("Current session")
    sessions = list_live_sessions()
    session_dir: Path | None = None
    session_state: dict[str, Any] = {}
    selected_id: str | None = None
    if sessions:
        session_ids = [path.name for path in sessions]
        current_id = st.session_state.get("live_active_session_id")
        # Streamlit only permits assignment to a widget key before that widget
        # is instantiated in the current run. Activation stores a pending ID.
        prepare_session_review_state(st.session_state, session_ids)
        selected_default = session_ids.index(current_id) if current_id in session_ids else 0
        session_statuses = {}
        for path in sessions:
            saved_state = read_session_state(path)
            saved_state["stop_requested"] = (path / "STOP_REQUESTED").exists()
            session_statuses[path.name] = session_status_label(saved_state)

        def _session_option(session_id: str) -> str:
            active_marker = " · active" if session_id == current_id else ""
            return f"{session_id} · {session_statuses[session_id]}{active_marker}"

        selected_id = st.selectbox(
            "Session history · select a run to review",
            session_ids,
            index=selected_default,
            format_func=_session_option,
            key="live_session_review",
            help="Selecting an older session only changes the displayed history; it does not restart it.",
        )
        st.session_state["live_selected_session_id"] = selected_id
        session_dir = next(path for path in sessions if path.name == selected_id)
        session_state = read_session_state(session_dir)
        session_state["stop_requested"] = (session_dir / "STOP_REQUESTED").exists()
        overview_events = read_session_events(session_dir)
        overview_valid = [
            event for event in overview_events
            if event.get("run_status") == "COMPLETED" and event.get("measured_qoe_proxy") is not None
        ]
        overview_latest = overview_valid[-1] if overview_valid else None
        overview_counts = session_run_counts(session_dir, overview_events, session_state)
        overview_cols = st.columns(5)
        overview_cols[0].metric("Selected session", session_status_label(session_state))
        overview_cols[1].metric("Normal run attempts", f"{overview_counts['normal_attempted']} / {session_state.get('max_epochs', 0)}")
        extra_limit = (int(session_state.get("max_additional_exploratory_epochs", 0))
                       if session_state.get("run_remaining_experimental_candidates") else 0)
        overview_cols[2].metric("Extra experimental attempts", f"{overview_counts['experimental_attempted']} / {extra_limit}")
        overview_cols[3].metric("Target score (0–1)", f"{float(session_state.get('target_qoe', 0.8)):.3f}")
        overview_cols[4].metric(
            "Latest measured score",
            f"{float(overview_latest['measured_qoe_proxy']):.4f}" if overview_latest else "No valid measurement",
        )
        st.caption(_run_count_caption(overview_counts, session_state))
        latest_phase = session_state.get("controller_phase", "TARGET_SEEKING")
        st.caption(
            f"Session `{selected_id}` · controller phase: {latest_phase} · "
            f"current run state: {session_state.get('epoch_state', session_state.get('phase', session_state.get('status', 'unknown')))}. "
            f"{session_status_explanation(session_state)} The score is calculated only from validated ns-3 output."
        )
        recovery_terminal_statuses = {
            "EPOCH_LIMIT_REACHED", "EXPLORATORY_EPOCH_LIMIT_REACHED",
            "NO_VALIDATED_CANDIDATES", "NO_EXPERIMENTAL_CANDIDATES",
            "STOP_NO_EXPERIMENTAL_CANDIDATES", "STOP_NO_SAFE_CANDIDATES",
            "NO_MEANINGFUL_IMPROVEMENT", "NO_IMPROVEMENT_AFTER_AVAILABLE_TRIALS",
            "NO_FEASIBLE_CANDIDATES",
        }
        recovery = recovery_summary(overview_events, session_state)
        if session_state.get("status") in recovery_terminal_statuses and not recovery["target_met"] and recovery["best_event"]:
            with st.expander("Explore alternative configurations", expanded=False):
                st.markdown("**More evaluation needed** — the bounded candidate evidence does not establish that the target is impossible.")
                recovery_cols = st.columns(4)
                recovery_cols[0].metric("Requested target", f"{recovery['target_qoe']:.4f}")
                recovery_cols[1].metric("Latest measured score", f"{recovery['latest_qoe']:.4f}")
                recovery_cols[2].metric("Best measured score", f"{recovery['best_qoe']:.4f}")
                recovery_cols[3].metric("Shortfall to target band", f"{recovery['shortfall']:.4f}")
                st.caption(
                    f"Target comparison uses the configured tolerance ({recovery['target_tolerance']:.4f}); "
                    f"target band starts at {recovery['target_band']:.4f}. "
                    f"{recovery['epochs_used']} run(s) used. Stop reason: {recovery['stop_reason']}"
                )
                st.write("**Best measured configuration:**", _config_caption(recovery["best_config"]))
                try:
                    recovery_config = json.loads((session_dir / "session_config.json").read_text(encoding="utf-8"))
                    candidate_status = candidate_and_outcome_summary(
                        overview_events, session_state, recovery_config,
                    )
                    st.caption(
                        f"Current evidence pool: {candidate_status['remaining_validated_count']} untried validated candidate(s); "
                        f"{candidate_status['remaining_exploratory_count']} supported exploratory candidate(s); "
                        f"{candidate_status['remaining_topology_count']} opt-in, unvalidated topology candidate(s); "
                        f"{candidate_status['remaining_risk_excluded_count']} risk-excluded under the current policy. "
                        "This is bounded evidence, not proof of global feasibility."
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    st.caption(f"Candidate evidence details unavailable: {exc}")
                st.caption(
                    "Recovery starts a new session and preserves this history. It evaluates candidates using separate ns-3 processes; "
                    "each additional run is a real simulation, not a prediction. Maximum cost is the selected run budget. "
                    "The empirical estimate may rank candidates only inside its exact evidence domain and does not replace simulation."
                )
                with st.form(f"live_recovery_form_{selected_id}"):
                    start_from = st.radio(
                        "Recovery starting point",
                        ["Best measured configuration", "Original starting configuration"],
                        key=f"recovery_start_from_{selected_id}",
                        help="This selects the initial configuration for a new adaptive session; prior results remain unchanged.",
                    )
                    recovery_budget = st.number_input(
                        "Additional simulation-run budget", min_value=1, max_value=6,
                        value=min(3, max(1, int(session_state.get("max_epochs", 3)))), step=1,
                        key=f"recovery_budget_{selected_id}",
                        help="This is a new bounded session with at most this many normal runs.",
                    )
                    recovery_exploration = st.checkbox(
                        "Allow supported exploratory configurations in recovery", value=False,
                        key=f"recovery_exploration_{selected_id}",
                        help="Off by default. If enabled, candidates outside validated campaign conditions may be tried and may perform worse or outage.",
                    )
                    recovery_high_risk = st.checkbox(
                        "Advanced override: permit configurations with prior outage evidence", value=False,
                        key=f"recovery_high_risk_{selected_id}",
                        help="Allows candidates normally excluded by the conservative risk policy. A complete outage may consume a run.",
                    )
                    recovery_submit = st.form_submit_button("Start bounded recovery session", type="primary")
                if recovery_submit:
                    try:
                        new_id, new_dir, _worker = create_live_session(
                            **recovery_session_parameters(
                                selected_id, recovery, session_state, overview_events,
                                run_budget=int(recovery_budget),
                                start_from_best=start_from == "Best measured configuration",
                                allow_exploratory=bool(recovery_exploration),
                                allow_high_risk=bool(recovery_high_risk),
                            ),
                        )
                        activate_session(st.session_state, new_id, new_dir)
                        st.success(f"Started bounded recovery session {new_id}. The earlier session remains available in history.")
                        st.rerun()
                    except (OSError, ValueError, RuntimeError) as exc:
                        st.error(f"Could not start recovery: {exc}")
    else:
        st.info("No sessions yet. Configure a scenario below to start the first simulation.")

    st.subheader("Scenario configuration")
    config_mode = st.radio("Scenario configuration", ["Preset scenario", "Custom configuration"], horizontal=True,
                           key="live_config_mode")
    with st.form("live_ns3_start_form"):
        st.markdown("**Network and traffic settings**")
        if config_mode == "Preset scenario":
            preset_name = st.selectbox("Preset test scenario", list(PRESET_DEFINITIONS), key="live_preset_name")
            preset = PRESET_DEFINITIONS[preset_name]
            initial_config = next(c for c in validated_configs if c["config_id"] == preset["config_id"])
            st.caption(preset["purpose"])
            st.info("Effective configuration: " + _config_caption(initial_config) +
                    f"; BE={initial_config['lambda_be_pps']} pps × {initial_config['packet_size_be_bytes']} B; "
                    f"ULL={initial_config['lambda_ull_pps']} pps × {initial_config['packet_size_ull_bytes']} B; "
                    f"simulation={initial_config['sim_time']}; area={initial_config['scenario_size_m']} m.")
        else:
            preset_name = "Custom"
            st.caption("Custom settings use supported scenario inputs. Combinations outside the expansion are exploratory.")
            st.caption("Band 2 is shown separately from Band 1. Campaign evidence matches only when Band 2, traffic, payloads, area and duration also match.")
            custom_cols = st.columns(3)
            ue_count = custom_cols[0].selectbox("UE count", [2, 6, 10], index=0, key="custom_ue_count")
            bw1 = custom_cols[1].selectbox("Band 1 bandwidth (MHz)", [20, 50, 100], index=1, key="custom_bw1")
            bw2 = custom_cols[2].selectbox("Band 2 bandwidth (MHz)", [20, 50, 100], index=1, key="custom_bw2")
            gnb_count = custom_cols[0].selectbox(
                "Number of gNB cells", [1, 2, 3], index=0, key="custom_gnb_count",
                help="Each gNB is created by ns-3. The UE population stays fixed; association is selected below.",
            )
            ue_association_policy = custom_cols[1].selectbox(
                "UE-to-cell assignment", ["closest", "round_robin", "by_traffic_class"], index=0,
                key="custom_ue_association_policy",
                help=("Closest uses ns-3 radio-distance association. Round-robin distributes existing UEs by ID; "
                      "by_traffic_class attaches low-latency and voice-QoS groups to separate cells. "
                      "Both require at least two cells; neither adds UEs or implements handover."),
            )
            if ue_association_policy != "closest" and int(gnb_count) < 2:
                st.warning("The selected UE assignment requires at least two gNB cells.")
            tx_power = custom_cols[0].selectbox("Requested total TX power (dBm)", [30, 35, 40], index=1, key="custom_power")
            lambda_be = custom_cols[1].number_input("Best-effort rate (packets/s)", 1000, 20000, 20000, 1000, key="custom_lambda_be")
            lambda_ull = custom_cols[2].number_input("Low-latency rate (packets/s)", 1000, 10000, 10000, 1000, key="custom_lambda_ull")
            packet_be = custom_cols[0].number_input("Best-effort UDP payload (bytes)", 64, 1472, 1252, 64, key="custom_packet_be")
            packet_ull = custom_cols[1].number_input("Low-latency UDP payload (bytes)", 64, 1472, 100, 64, key="custom_packet_ull")
            duration = custom_cols[2].number_input("Simulation duration (seconds)", 1.0, 3.0, 1.0, 0.5, key="custom_duration")
            scenario_size = custom_cols[0].number_input("Scenario placement area (metres)", 30.0, 200.0, 30.0, 10.0, key="custom_scenario_size")
            try:
                initial_config = validate_custom_config({
                    "config_id": "CUSTOM", "ue_count": ue_count,
                    "gnb_count": gnb_count,
                    "ue_association_policy": ue_association_policy,
                    "bandwidth_band1_mhz": bw1, "bandwidth_band2_mhz": bw2,
                    "total_tx_power_dbm": tx_power, "lambda_be_pps": lambda_be,
                    "lambda_ull_pps": lambda_ull, "packet_size_be_bytes": packet_be,
                    "packet_size_ull_bytes": packet_ull, "sim_time_seconds": duration,
                    "scenario_size_m": scenario_size,
                })
                st.caption("Effective custom configuration: " + _config_caption(initial_config) +
                           f"; BE={lambda_be} pps × {packet_be} B; ULL={lambda_ull} pps × {packet_ull} B; "
                           f"simulation={initial_config['sim_time']}; area={scenario_size:g} m.")
            except ValueError as exc:
                st.error(str(exc))
                initial_config = None
        st.markdown("**Controller and target settings**")
        mode = st.radio("Controller mode", ["adaptive", "fixed"], horizontal=True,
                        format_func=lambda value: "Adaptive (measurement-driven)" if value == "adaptive" else "Fixed baseline",
                        key="live_controller_mode",
                        help="Adaptive mode selects later configurations from measured results and supported empirical estimates. Fixed mode keeps the selected configuration for each run.")
        target_qoe = st.slider("Target simulated network quality score (0–1)", 0.0, 1.0, 0.80, 0.01,
                               key="live_target_qoe", help="This is a synthetic ns-3 score, not a user survey score.")
        target_tolerance = st.slider("Target tolerance (QoE proxy)", 0.0, 0.05, 0.005, 0.005,
                                     key="live_target_tolerance",
                                     help="A validated epoch counts toward stable attainment when measured QoE is at least target minus this tolerance.")
        stable_epochs_required = st.number_input("Consecutive epochs required for stable target", 1, 5, 2, 1,
                                                 key="live_stable_epochs_required")
        stability_validation_epochs = st.number_input(
            "Additional stability-validation runs", 1, 10, 5, 1,
            key="live_stability_validation_epochs",
            help=("After the score first enters the absolute target range, freeze the exact configuration and "
                  "repeat it in separate ns-3 processes this many times. These validation runs are additional "
                  "to the normal epoch budget."),
        )
        maintenance_margin = st.slider("Maintenance prediction margin", 0.0, 0.10, 0.02, 0.005,
                                       key="live_maintenance_margin")
        maintenance_hysteresis = st.slider("Maintenance switch hysteresis", 0.0, 0.10, 0.02, 0.005,
                                           key="live_maintenance_hysteresis")
        st.caption("Two consecutive validated measurements at target minus tolerance establish TARGET_MAINTENANCE. Maintenance evaluates each run separately, with an ns-3 process restart per run.")
        st.markdown("**Run limits and safety options**")
        limit_cols = st.columns(2)
        max_epochs = limit_cols[0].number_input(
            "Maximum number of normal simulation runs", 1, 6, 3, 1, key="live_max_epochs",
            help="The normal epoch limit is the maximum number of normal simulation runs.",
        )
        max_actions = limit_cols[1].number_input(
            "Normal adaptive action budget", 0, 3, 3, 1, key="live_max_actions",
            help="How many normal configuration changes the adaptive controller may make. This is not the session stop limit.",
        )
        st.caption("After the action budget, adaptive sessions may use eligible exploratory trials if enabled. The epoch limit remains the normal run bound.")
        if mode == "adaptive":
            allow_exploratory_candidates = st.checkbox(
                "Allow exploratory configurations",
                value=False, key="live_allow_exploratory_candidates",
                help=("Opt in to trying supported Band 1/power combinations outside the validated campaign. "
                      "These are exploratory, may fail or reduce QoE, and have no predicted benefit."),
            )
            if allow_exploratory_candidates:
                st.caption("Exploratory configurations can be considered before the normal epoch limit after the action budget; Band 2 and traffic stay fixed.")
            allow_high_risk_candidates = st.checkbox(
                "Advanced override: permit high-risk configurations", value=False,
                key="live_allow_high_risk_candidates",
                help="Permits exact prior-outage settings and settings matching a repeated outage pattern. Such runs can produce complete outages.",
            )
            if allow_high_risk_candidates:
                st.warning("High-risk override enabled: prior evidence includes complete outages. Results are uncertain and simulation time may be spent on an outage.")
            allow_repeat_trials = st.checkbox(
                "Allow repeat trials of configurations already attempted", value=False,
                key="live_allow_repeat_trials",
                help="Useful for repeatability checks; repeats consume epochs and may reproduce an outage.",
            )
            run_remaining_experimental = st.checkbox(
                "Run remaining experimental candidates after epoch limit",
                value=False, key="live_post_epoch_exploration",
                help=("After the normal epoch limit, optionally run additional untried supported experimental configurations. "
                      "This uses a separate extra-epoch budget and does not predict improvement."),
            )
            if run_remaining_experimental:
                max_additional_exploratory_epochs = st.number_input(
                    "Additional experimental epoch limit", 1, 9, 3, 1,
                    key="live_max_post_epoch_exploration",
                )
                st.caption("The normal epoch limit remains unchanged; these extra process restarts are tracked separately.")
            else:
                max_additional_exploratory_epochs = 3
            allow_topology_exploration = st.checkbox(
                "Allow bounded additional-gNB topology candidates", value=False,
                key="live_allow_topology_exploration",
                help=("After same-topology candidates are exhausted, test up to two real multi-cell scenario rebuilds. "
                      "The structural trial keeps UE count, workload and radio settings fixed; existing UEs are "
                      "explicitly attached round-robin across the new cells. "
                      "If a cell-count candidate runs, later same-topology radio tuning is also permitted while budgets remain. "
                      "These alternatives have no matching campaign performance evidence."),
            )
            if allow_topology_exploration:
                topology_candidate_budget = st.number_input(
                "Maximum additional topology candidates", 1, 3, 1, 1,
                    key="live_topology_candidate_budget",
                )
                st.caption("Cell-count candidates are unvalidated until their own ns-3 runs complete. They use the same EPC and IP network; no subnet isolation or handover is modeled.")
            else:
                topology_candidate_budget = 1
        else:
            allow_exploratory_candidates = False
            allow_high_risk_candidates = False
            allow_repeat_trials = False
            run_remaining_experimental = False
            max_additional_exploratory_epochs = 3
            allow_topology_exploration = False
            topology_candidate_budget = 1
            st.caption("Fixed mode repeats the selected configuration; exploratory action selection is unavailable.")
        st.markdown("**Reproducibility and process safety**")
        run_cols = st.columns(3)
        rng_seed = run_cols[0].number_input(
            "RNG seed", 1, 4_294_967_295, 20261026, 1, key="live_rng_seed",
            help="The seed and run number set ns-3's reproducible random streams.",
        )
        rng_run_start = run_cols[1].number_input(
            "First RNG run number", 1, 4_294_967_295, 1, 1, key="live_rng_run",
            help="Use different run numbers for distinct random realizations under the selected seed.",
        )
        timeout_recommendation, estimated_packets = recommended_timeout_seconds(initial_config or {})
        timeout_config_key = configuration_fingerprint(initial_config or {})[:12]
        timeout_seconds = run_cols[2].number_input(
            "Per-run timeout (seconds)", 10.0, 300.0, float(timeout_recommendation), 10.0,
            key=f"live_epoch_timeout_{timeout_config_key}",
            help=("Pre-filled using configured packet workload and a measured high-load runtime, with 25% headroom. "
                  "This is a bounded estimate, not guaranteed progress; the selected timeout is enforced exactly."),
        )
        st.caption(
            f"Timeout suggestion: {timeout_recommendation} s for approximately {estimated_packets:,} configured "
            "application packets, based on one completed 6-UE, 3-second high-load run; capped at 300 s. "
            "Lower or raise the value explicitly if needed. The workload and simulation duration are not altered."
        )
        st.subheader("Simulation controls")
        start = st.form_submit_button("Start ns-3 simulation/controller", type="primary")
    if start:
        try:
            if initial_config is None:
                raise ValueError("Correct the custom configuration values before starting.")
            session_id, session_dir, _worker = create_live_session(
                initial_config=initial_config,
                target_qoe=float(target_qoe),
                max_epochs=int(max_epochs),
                mode=mode,
                preset_name=preset_name,
                rng_seed=int(rng_seed), rng_run_start=int(rng_run_start),
                timeout_seconds=float(timeout_seconds),
                max_actions=int(max_actions),
                allow_exploratory_candidates=bool(allow_exploratory_candidates),
                allow_high_risk_candidates=bool(allow_high_risk_candidates),
                allow_repeat_trials=bool(allow_repeat_trials),
                allow_topology_exploration=bool(allow_topology_exploration),
                topology_candidate_budget=int(topology_candidate_budget),
                run_remaining_experimental_candidates=bool(run_remaining_experimental),
                max_additional_exploratory_epochs=int(max_additional_exploratory_epochs),
                target_tolerance=float(target_tolerance),
                stable_epochs_required=int(stable_epochs_required),
                stability_validation_epochs=int(stability_validation_epochs),
                maintenance_margin=float(maintenance_margin),
                maintenance_hysteresis=float(maintenance_hysteresis),
            )
            # Keep one authoritative active ID and explicitly reset the history widget;
            # its prior widget state otherwise overrides the newly computed default.
            activate_session(st.session_state, session_id, session_dir)
            st.success(f"Started session {session_id}. Each epoch restarts ns-3 in a separate process.")
            st.rerun()
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Could not start the simulation: {exc}")

    if session_dir is not None and session_state.get("status") in {"QUEUED", "RUNNING"}:
        if st.button("Stop selected session safely", key=f"live_stop_{selected_id}"):
            request_session_stop(session_dir)
            st.warning("Stop requested. The owned ns-3 process will be stopped safely; no point is added for an invalid or cancelled run.")

    if not sessions:
        st.info("No live-control sessions have been recorded yet. Starting above will create a new isolated session.")
        st.plotly_chart(build_qoe_figure([], 0.80), width="stretch",
                        key="live_qoe_empty_history")
        return

    @st.fragment(run_every=1.0)
    def live_monitor(path_string: str) -> None:
        folder = Path(path_string)
        try:
            state = mark_worker_lost(folder)
            state["stop_requested"] = (folder / "STOP_REQUESTED").exists()
            events = read_session_events(folder)
            session_config = json.loads((folder / "session_config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Could not read session log: {exc}")
            return
        run_counts = session_run_counts(folder, events, state)
        st.subheader("Live performance")
        status_cols = st.columns(4)
        status_cols[0].metric("Session status", session_status_label(state))
        epoch_label = f"Normal {run_counts['normal_attempted']} / {state.get('max_epochs', 0)}"
        if state.get("is_post_epoch_exploration"):
            epoch_label = (f"Extra {run_counts['experimental_attempted']} / "
                           f"{state.get('max_additional_exploratory_epochs', 0)}")
        status_cols[1].metric("Run-limit progress", epoch_label)
        status_cols[2].metric("Target score (0–1)", f"{state.get('target_qoe', 0.8):.3f}")
        status_cols[3].metric("Elapsed this epoch", f"{float(state.get('current_elapsed_seconds', 0)):.1f} s")
        st.caption(_run_count_caption(run_counts, state))
        phase = state.get("controller_phase", "TARGET_SEEKING")
        target = float(state.get("target_qoe", 0.8))
        tolerance = float(state.get("target_tolerance", 0.005))
        validation_done = int(state.get("validation_epochs_completed", 0))
        validation_required = int(state.get("validation_epochs_required", state.get("stability_validation_epochs", 5)))
        st.info(f"Controller phase: **{phase}** · target range [{target - tolerance:.3f}, {target + tolerance:.3f}] "
                f"(absolute tolerance). Stability validation: {validation_done}/{validation_required} additional "
                "same-configuration runs; no serving configuration is changed during validation.")
        if state.get("queue_reason"):
            st.info(state["queue_reason"])
        if state.get("phase") == "WORKER_EXITED":
            st.error("Controller worker exited unexpectedly. The session is marked failed; inspect worker.log and worker_exception.log.")
        st.caption(f"Process policy: {state.get('epoch_process_policy', 'new process per epoch')} · "
                   f"preset/config mode: {state.get('preset_name', 'Custom')} · seed={state.get('rng_seed')} · "
                   f"run={state.get('rng_run', 'pending')}")
        st.caption(f"Candidate evidence: {state.get('validated_candidate_scope', 'legacy session policy')}; "
                   f"exploratory supported settings {'enabled' if state.get('allow_exploratory_candidates') else 'disabled'}.")
        st.caption(f"Post-limit experimental runs {'enabled' if state.get('run_remaining_experimental_candidates') else 'disabled'}; "
                   f"completed={state.get('post_epoch_exploration_count', 0)} / "
                   f"extra limit={state.get('max_additional_exploratory_epochs', 0)}.")
        current = state.get("current_config") or {}
        st.write("**Current configuration:**", _config_caption(current))

        valid_events = [
            event for event in events
            if event.get("run_status") == "COMPLETED" and event.get("measured_qoe_proxy") is not None
        ]
        latest = valid_events[-1] if valid_events else None
        if latest is not None:
            metrics = latest.get("metrics", {})
            previous = valid_events[-2] if len(valid_events) > 1 else None
            delta = (
                float(latest["measured_qoe_proxy"]) - float(previous["measured_qoe_proxy"])
                if previous else None
            )
            columns = st.columns(5)
            columns[0].metric("Measured score (0–1)", f"{latest['measured_qoe_proxy']:.4f}",
                              delta=f"{delta:+.4f} vs prior valid epoch" if delta is not None else None)
            columns[1].metric("Throughput", f"{metrics['mean_throughput_mbps']:.3f} Mbps")
            columns[2].metric("Latency", f"{metrics['mean_delay_ms']:.3f} ms")
            columns[3].metric("Jitter", f"{metrics['mean_jitter_ms']:.4f} ms")
            columns[4].metric("Packet loss", f"{metrics['mean_packet_loss']*100:.2f}%")
            st.caption(
                f"Latest run duration {latest.get('duration_seconds', 0):.2f}s; "
                f"outage={bool(metrics.get('is_outage'))}; pre-run empirical estimate="
                f"{latest.get('predicted_qoe') if latest.get('predicted_qoe') is not None else 'Prediction unavailable'}; "
                f"prediction error={latest.get('prediction_error') if latest.get('prediction_error') is not None else '—'}."
            )
            if latest.get("predicted_qoe") is not None:
                st.caption(f"Pre-run support: {latest.get('prediction_domain')} · n={latest.get('prediction_run_count', 0)} · "
                           f"variation band=[{latest.get('predicted_qoe_lower') if latest.get('predicted_qoe_lower') is not None else '—'}, "
                           f"{latest.get('predicted_qoe_upper') if latest.get('predicted_qoe_upper') is not None else '—'}]; "
                           f"{latest.get('prediction_confidence', 'uncertainty unavailable')}.")
        elif events:
            latest_event = events[-1]
            st.warning(
                f"Latest epoch has no valid measurement ({latest_event.get('run_status')}). "
                "Failed or invalid epochs are not plotted as zero QoE."
            )
            if latest_event.get("error_message"):
                st.error(f"{latest_event.get('error_type', 'Epoch error')}: {latest_event['error_message']}")
        else:
            st.info(f"No completed measurements yet. Epoch state: {state.get('epoch_state', state.get('phase', 'QUEUED'))}. "
                    "The ns-3 scenario may take tens of seconds to initialize and simulate; elapsed time and logs update below.")

        topology_col, config_col = st.columns([1.35, 1])
        if latest and latest.get("config"):
            topology_config = latest["config"]
        else:
            topology_config = current or state.get("initial_config")
        topology_config = topology_config or {}
        ue_measurements = list(latest.get("ue_measurements") or []) if latest else []
        ue_measurement_status = latest.get("ue_measurements_status", "UNAVAILABLE") if latest else "UNAVAILABLE"
        from dashboard.operations_views import topology_figure, ue_measurement_table

        with st.expander("Filter and inspect simulated UE profiles", expanded=False):
            st.caption(
                "The scenario alternates UEs between low-latency UDP and voice QoS UDP workloads. "
                "These are simulated traffic profiles, not full gaming or calling applications. "
                "The scenario defines a global mean QoE target, not individual UE latency, loss, jitter, "
                "throughput, or QoE service-level targets."
            )
            profiles = ue_measurement_table(
                ue_measurements, session_target=float(state.get("target_qoe", 0.8)),
                ue_count=int(topology_config.get("ue_count", 0)),
            )
            filter_cols = st.columns(3)
            traffic_filter = filter_cols[0].selectbox(
                "Traffic class", ["All", "low_latency", "voice_qos"],
                key=f"live_ue_class_{selected_id}",
            )
            status_filter = filter_cols[1].selectbox(
                "Per-UE observation", ["All"] + sorted(profiles["Comparison"].unique().tolist()),
                key=f"live_ue_status_{selected_id}",
            )
            ue_choices = ["All UEs"] + profiles["UE"].tolist()
            highlighted_value = filter_cols[2].selectbox(
                "Highlight device", ue_choices, key=f"live_ue_highlight_{selected_id}",
            )
        highlighted_ue = int(highlighted_value.split()[-1]) if highlighted_value != "All UEs" else None

        topology_col.plotly_chart(
            topology_figure(
                topology_config, ue_measurements=ue_measurements,
                highlighted_ue=highlighted_ue,
                session_target=float(state.get("target_qoe", 0.8)),
            ), width="stretch",
            key=f"live_topology_{selected_id}",
        )
        if ue_measurement_status == "VALIDATED" and any(
            row.get("serving_cell_id") is not None for row in ue_measurements
        ):
            topology_col.caption("Schematic only, not to scale. Cell IDs and UE associations come from ns-3 RRC results; lines do not show utilization or link quality. A red marker indicates a measured zero-reception outage.")
        else:
            topology_col.caption("Schematic only, not to scale. gNB count reflects scenario configuration; measured UE-to-cell associations are unavailable for this run. Lines do not imply radio quality.")
        config_col.subheader("Active scenario")
        config_col.metric("User devices", f"{topology_config.get('ue_count', '—')} UEs")
        config_col.metric("Configured radio cells", f"{topology_config.get('gnb_count', 1)} gNBs")
        config_col.metric("UE-to-cell assignment", topology_config.get("ue_association_policy", "closest"))
        config_col.metric("Band 1 / Band 2", f"{topology_config.get('bandwidth_band1_mhz', '—')} / {topology_config.get('bandwidth_band2_mhz', '—')} MHz")
        config_col.metric("Requested total transmit power", f"{topology_config.get('total_tx_power_dbm', '—')} dBm")
        config_col.caption(
            f"Traffic: best-effort {topology_config.get('lambda_be_pps', '—')} packets/s · "
            f"low-latency {topology_config.get('lambda_ull_pps', '—')} packets/s · "
            f"run duration {topology_config.get('sim_time', '—')}."
        )

        if traffic_filter != "All":
            profiles = profiles[profiles["Traffic class"] == traffic_filter]
        if status_filter != "All":
            profiles = profiles[profiles["Comparison"] == status_filter]
        st.subheader("UE workloads and per-flow measurements")
        if ue_measurement_status == "VALIDATED":
            st.caption(f"Per-UE rows were mapped by destination IP and validated against the configured UE count. Latest run ended {latest.get('end_utc', 'timestamp unavailable')}. QoE comparisons below use the session target only as a reference; they are not per-UE SLA compliance.")
        elif ue_measurement_status == "INVALID":
            st.warning(f"Per-UE sidecar data failed validation and is withheld: {latest.get('ue_measurements_error', 'validation error unavailable')}. Aggregate session metrics remain separate.")
        else:
            st.info("Per-UE measured KPIs are unavailable for this run (the simulator output did not include the new identity-mapped sidecar). UE workload classes are shown from scenario configuration only.")
        st.dataframe(profiles, hide_index=True, width="stretch", key=f"live_ue_profile_table_{selected_id}")
        if ue_measurement_status == "VALIDATED" and latest.get("ue_measurements_path"):
            ue_csv_path = Path(latest["ue_measurements_path"])
            if ue_csv_path.is_file():
                st.download_button(
                    "Download latest validated per-UE CSV",
                    data=ue_csv_path.read_bytes(),
                    file_name=f"{selected_id}_epoch_{latest.get('epoch')}_ue_measurements.csv",
                    mime="text/csv", key=f"live_ue_csv_{selected_id}_{latest.get('epoch')}",
                )

        if state.get("current_stdout_path") or state.get("current_stderr_path"):
            with st.expander("Current process output", expanded=False):
                for label, key in (("stdout", "current_stdout_path"), ("stderr", "current_stderr_path")):
                    path = Path(state[key]) if state.get(key) else None
                    if path and path.is_file():
                        try:
                            tail = path.read_text(encoding="utf-8", errors="replace")[-4000:]
                        except OSError as exc:
                            tail = f"Could not read {label}: {exc}"
                        st.text(f"{label}:\n{tail or '(no output emitted yet)'}")

        graph_col, action_col = st.columns([3, 2])
        graph_col.caption("Blue points are validated measured outcomes. Purple diamonds are pre-run empirical estimates. A gap indicates a failed or invalid run; a validated outage remains a measured zero.")
        graph_col.plotly_chart(
            build_qoe_figure(events, float(state.get("target_qoe", 0.8))),
            width="stretch",
            key=f"live_qoe_chart_{selected_id}",
        )
        graph_col.caption("Measured values come from validated completed runs. Failed or invalid runs appear as gaps; estimates are plotted separately and never replace measurements.")
        action_col.subheader("5 · Controller decisions")
        action_col.caption("The explanation describes why the controller kept or selected a configuration. Estimates are not measured results.")
        if state.get("latest_action"):
            action_col.write(f"**Latest action:** {state['latest_action']}")
            action_col.write(state.get("latest_rationale", ""))
        if events:
            from dashboard.config_explanations import render_configuration_change

            last_event = events[-1]
            prior_config = (events[-2].get("config") or {}) if len(events) > 1 else {}
            latest_config = last_event.get("config") or {}
            run_status = last_event.get("run_status", "UNKNOWN")
            latest_metrics = last_event.get("metrics") or {}
            if run_status == "COMPLETED" and last_event.get("measured_qoe_proxy") is not None:
                measured_change = "No previous validated score is available for comparison."
                if last_event.get("previous_valid_qoe_delta") is not None:
                    measured_change = (
                        f"Measured QoE proxy change {float(last_event['previous_valid_qoe_delta']):+.4f} "
                        f"({last_event.get('action_outcome_from_previous', 'outcome not classified')})."
                    )
                throughput_value = latest_metrics.get("mean_throughput_mbps")
                latency_value = latest_metrics.get("mean_delay_ms")
                loss_value = latest_metrics.get("mean_packet_loss")
                throughput_text = f"{float(throughput_value):.3f} Mbps" if throughput_value is not None else "unavailable"
                latency_text = f"{float(latency_value):.3f} ms" if latency_value is not None else "unavailable"
                loss_text = f"{float(loss_value) * 100:.2f}%" if loss_value is not None else "unavailable"
                evaluation_result = (
                    f"Measured QoE proxy {float(last_event['measured_qoe_proxy']):.4f}; "
                    f"throughput {throughput_text}; latency {latency_text}; "
                    f"packet loss {loss_text}; "
                    f"outage={bool(latest_metrics.get('is_outage'))}. {measured_change}"
                )
            else:
                evaluation_result = f"No valid measurement was recorded ({run_status})."
            with action_col.expander("Explain the latest configuration", expanded=False) as explain_box:
                decision_record = last_event.get("decision_record")
                if decision_record:
                    explain_box.markdown("**Recorded controller decision**")
                    explain_box.write(decision_record.get("selected_reason") or decision_record.get("reason")
                                      or "No rationale was recorded for this event.")
                    if decision_record.get("expected_effect"):
                        explain_box.caption(f"Expected effect: {decision_record['expected_effect']}")
                    if decision_record.get("validation_progress"):
                        explain_box.info(
                            f"Stability validation: {decision_record['validation_progress']}. "
                            "The recorded configuration is frozen during these independent process runs."
                        )
                    with explain_box.expander("Technical decision record", expanded=False):
                        explain_box.json(decision_record)
                else:
                    explain_box.info(
                        "No structured decision is recorded for this epoch yet. "
                        "A decision appears after the simulator result has been validated."
                    )
                render_configuration_change(
                    explain_box, prior_config, latest_config,
                    reason=last_event.get("candidate_selection_rationale"),
                    status=("Evaluated · measured ns-3 run" if run_status == "COMPLETED"
                            else f"Attempted · {run_status}; no valid measurement implied"),
                    evaluation_result=evaluation_result,
                    key=f"live_applied_config_diff_{selected_id}",
                )
                proposed = last_event.get("next_config")
                if proposed:
                    explain_box.markdown("**Next configuration, if the session continues**")
                    next_prediction = last_event.get("next_prediction") or {}
                    effect = None
                    if next_prediction.get("qoe") is not None:
                        effect = (
                            f"Empirical estimate: QoE proxy {float(next_prediction['qoe']):.4f}, "
                            f"based on {next_prediction.get('run_count', 0)} exact-condition run(s). "
                            "This is a pre-run estimate, not a measured outcome."
                        )
                    render_configuration_change(
                        explain_box, latest_config, proposed,
                        reason=last_event.get("action_rationale"),
                        status="Proposed for a separate ns-3 process",
                        expected_effect=effect,
                        evaluation_result="Not yet evaluated; the next process has not produced measurements.",
                        key=f"live_proposed_config_diff_{selected_id}",
                    )
        if events:
            action_col.dataframe(
                _history_table(events)[["Epoch", "Config", "Status", "Exploration status",
                                        "Prior QoE proxy", "Current QoE proxy", "Outage", "Action",
                                        "Candidate evidence", "UE detail status", "UE detail rows",
                                        "Selected config risk", "Selected config risk evidence",
                                        "Next config risk", "Risk level", "Risk evidence", "High-risk override",
                                        "Adaptive actions used", "Exploratory trials used",
                                        "Normal action budget reached", "Post-limit experimental trial",
                                        "Continuation reason", "Rationale"]],
                hide_index=True,
                width="stretch",
            )
        else:
            action_col.caption("Actions will appear after a validated measurement.")
        upcoming = state.get("next_prediction") or {}
        if upcoming.get("qoe") is not None:
            action_col.metric("Predicted next-epoch QoE proxy", f"{upcoming['qoe']:.4f}")
            action_col.caption(f"{upcoming.get('method', 'empirical estimate')} · {upcoming.get('domain')} · "
                               f"n={upcoming.get('run_count', 0)}; {upcoming.get('confidence', 'limited support')}; "
                               f"outage fraction={upcoming.get('outage_fraction') if upcoming.get('outage_fraction') is not None else 'unavailable'}.")
        else:
            action_col.caption("Next-epoch prediction unavailable outside the exact-condition evidence domain.")

        if events:
            from dashboard.operations_views import kpi_trend_figure

            st.subheader("Network monitoring · measured outcomes by run")
            st.caption("The scenario stores per-run aggregates, not within-run timestamps. These trends use the run index; gaps indicate failed or invalid measurements.")
            trend_cols = st.columns(2)
            trend_cols[0].plotly_chart(kpi_trend_figure(events, "Throughput"), width="stretch",
                                       key=f"live_throughput_{selected_id}")
            trend_cols[1].plotly_chart(kpi_trend_figure(events, "Latency"), width="stretch",
                                       key=f"live_latency_{selected_id}")

        summary = candidate_and_outcome_summary(events, state, session_config)
        recovery_context = state.get("recovery_context")
        if recovery_context:
            st.subheader("Recovery attempt vs. original session")
            parent_qoe = recovery_context.get("parent_best_measured_qoe")
            recovery_best = summary.get("best_qoe")
            compare_cols = st.columns(4)
            compare_cols[0].metric("Original best measured score",
                                   f"{float(parent_qoe):.4f}" if parent_qoe is not None else "—")
            compare_cols[1].metric("Recovery best measured score",
                                   f"{float(recovery_best):.4f}" if recovery_best is not None else "No valid result")
            compare_cols[2].metric(
                "Change from original",
                f"{float(recovery_best) - float(parent_qoe):+.4f}"
                if recovery_best is not None and parent_qoe is not None else "—",
            )
            compare_cols[3].metric("Target reached in recovery",
                                   "Yes" if recovery_best is not None and float(recovery_best) >=
                                   float(state.get("target_qoe", 0.8)) - float(state.get("target_tolerance", 0.005))
                                   else "Not yet")
            st.caption(
                f"New bounded session from `{recovery_context.get('parent_session_id', 'unknown')}`; "
                f"budget={recovery_context.get('recovery_run_budget', '—')} runs. "
                "Comparison uses measured QoE proxies only; an incomplete recovery has no final conclusion."
            )
        summary_cols = st.columns(5)
        summary_cols[0].metric("Final target gap", f"{summary['target_gap']:.4f}" if summary["target_gap"] is not None else "—")
        summary_cols[1].metric("Best measured QoE", f"{summary['best_qoe']:.4f}" if summary["best_qoe"] is not None else "—")
        summary_cols[2].metric("Best configuration", summary["best_config"] or "—")
        summary_cols[3].metric("Configurations attempted", str(len(summary["attempted_configs"])))
        summary_cols[4].metric("Remaining candidates", str(summary["remaining_candidate_count"]))
        st.caption(f"Risk policy: conservative; high-risk override={'enabled' if summary['high_risk_override'] else 'disabled'}; "
                   f"repeat trials={'enabled' if summary['repeat_trials_enabled'] else 'disabled'}. "
                   f"Eligible remaining={summary['remaining_eligible_count']}; "
                   f"excluded for risk={summary['remaining_risk_excluded_count']}.")
        if summary["risk_review_candidates"]:
            with st.expander("Candidate risk evidence and filtering decisions"):
                for item in summary["risk_review_candidates"]:
                    st.write(f"**{item['config_id']}** — {item['decision']}: {item['reason']}")
        if summary["attempted_configs"]:
            st.caption("Attempted: " + ", ".join(summary["attempted_configs"]) +
                       f". Remaining validated={summary['remaining_validated_count']}; "
                       f"remaining exploratory={summary['remaining_exploratory_count']} "
                       f"(pre-limit={'enabled' if summary['exploratory_enabled'] else 'disabled'}, "
                       f"post-limit={'enabled' if summary['post_epoch_exploration_enabled'] else 'disabled'}; "
                       f"extra epochs={summary['post_epoch_exploration_count']}/"
                       f"{summary['post_epoch_exploration_limit']}).")

        if events:
            st.subheader("6 · Epoch measurements and session history")
            st.dataframe(
                _history_table(events), hide_index=True, width="stretch",
                column_config={
                    "Measured QoE proxy": st.column_config.ProgressColumn(
                        "Measured quality score (0–1)", min_value=0.0, max_value=1.0, format="%.4f",
                    ),
                    "Throughput Mbps": st.column_config.NumberColumn("Throughput (Mbps)", format="%.3f"),
                    "Latency ms": st.column_config.NumberColumn("Latency (ms)", format="%.3f"),
                    "Jitter ms": st.column_config.NumberColumn("Jitter (ms)", format="%.4f"),
                    "Packet loss %": st.column_config.NumberColumn("Packet loss (%)", format="%.2f"),
                    "Outage": st.column_config.CheckboxColumn("Outage"),
                },
            )
            validation_errors = [float(event["prediction_error"]) for event in events
                                 if event.get("run_status") == "COMPLETED"
                                 and event.get("prediction_error") is not None]
            if validation_errors:
                mae = sum(abs(error) for error in validation_errors) / len(validation_errors)
                rmse = (sum(error * error for error in validation_errors) / len(validation_errors)) ** 0.5
                st.caption(f"Pre-run estimate validation for this session only: n={len(validation_errors)}, "
                           f"MAE={mae:.4f}, RMSE={rmse:.4f} QoE proxy units. This small, same-session result is descriptive, not a generalization estimate.")
        csv_path = folder / "epoch_history.csv"
        if csv_path.is_file():
            st.download_button(
                "Download epoch history CSV",
                data=csv_path.read_bytes(),
                file_name=f"{state['session_id']}_epoch_history.csv",
                mime="text/csv",
                key=f"live_download_{selected_id}",
            )
        st.caption(f"Session logs: {folder}")
        with st.expander("Technical status and log paths", expanded=False):
            st.json({
                "session_id": state.get("session_id"),
                "session_status": state.get("status"),
                "epoch_state": state.get("epoch_state", state.get("phase")),
                "controller_phase": state.get("controller_phase"),
                "latest_action_code": state.get("latest_action"),
                "latest_rationale": state.get("latest_rationale"),
                "session_directory": str(folder),
                "controller_log": str(folder / "controller.log"),
                "worker_log": str(folder / "worker.log"),
                "epoch_history_csv": str(folder / "epoch_history.csv"),
            })
        if state.get("final_outcome"):
            label = session_status_label(state)
            if label.startswith("Target unmet"):
                st.warning(f"{label}. {state['final_outcome']}")
            else:
                st.info(f"{label}: {state['final_outcome']}")
        legacy_note = legacy_candidate_scope_note(state)
        if legacy_note:
            st.info(legacy_note)

    live_monitor(str(session_dir))
