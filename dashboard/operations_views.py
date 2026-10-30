"""Read-only overview, experiment, and analytics views for live ns-3 sessions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import DASHBOARD_PALETTE, page_header, style_figure
from dashboard.live_ns3_control import (
    build_qoe_figure,
    session_run_counts,
    session_status_explanation,
    session_status_label,
)
from simulation.live_control import LIVE_RESULTS, list_live_sessions, read_session_events, read_session_state


METRIC_SPECS = {
    "QoE proxy": ("measured_qoe_proxy", "Simulated network quality score (0–1)", "score"),
    "Throughput": ("mean_throughput_mbps", "Throughput", "Mbps"),
    "Latency": ("mean_delay_ms", "One-way latency", "ms"),
    "Jitter": ("mean_jitter_ms", "Jitter", "ms"),
    "Packet loss": ("mean_packet_loss", "Packet loss", "%"),
}


def load_live_records(output_root: str | Path = LIVE_RESULTS) -> list[dict[str, Any]]:
    """Read saved session and event records without changing worker state."""
    records = []
    for folder in list_live_sessions(output_root):
        try:
            state = read_session_state(folder)
            events = read_session_events(folder)
        except (OSError, ValueError, KeyError) as exc:
            records.append({"path": folder, "id": folder.name, "error": str(exc)})
            continue
        records.append({"path": folder, "id": folder.name, "state": state, "events": events})
    return records


def valid_measurements(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only validated completed run records contribute measured values."""
    return [
        event for event in events
        if event.get("run_status") == "COMPLETED"
        and event.get("measured_qoe_proxy") is not None
        and isinstance(event.get("metrics"), dict)
    ]


def session_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a concise session index; selected-session pages expose full records."""
    rows = []
    for record in records:
        if record.get("error"):
            rows.append({"Session": record["id"], "Status": "Unreadable", "Details": record["error"]})
            continue
        state, events, folder = record["state"], record["events"], record["path"]
        counts = session_run_counts(folder, events, state)
        valid = valid_measurements(events)
        best = max((float(event["measured_qoe_proxy"]) for event in valid), default=None)
        latest = valid[-1] if valid else None
        current = (latest or {}).get("config") or state.get("current_config") or {}
        completed_duration = sum(float(event.get("duration_seconds") or 0) for event in events)
        started = state.get("created_utc", "—")
        try:
            started = pd.Timestamp(started).strftime("%Y-%m-%d %H:%M")
        except (TypeError, ValueError):
            pass
        rows.append({
            "Session": record["id"],
            "Started": started,
            "Status": session_status_label({
                **state, "stop_requested": (folder / "STOP_REQUESTED").exists(),
            }),
            "Mode": state.get("controller_mode", "—"),
            "Scenario": state.get("preset_name", "Custom"),
            "Attempts": f"{counts['normal_attempted']}/{state.get('max_epochs', 0)}",
            "Extra attempts": f"{counts['experimental_attempted']}/"
            f"{state.get('max_additional_exploratory_epochs', 0) if state.get('run_remaining_experimental_candidates') else 0}",
            "Valid runs": len(valid),
            "Latest score": float(latest["measured_qoe_proxy"]) if latest else None,
            "Best score": best,
            "Throughput (Mbps)": (latest or {}).get("metrics", {}).get("mean_throughput_mbps"),
            "Latency (ms)": (latest or {}).get("metrics", {}).get("mean_delay_ms"),
            "Packet loss (%)": (float(latest["metrics"]["mean_packet_loss"]) * 100
                                if latest and latest.get("metrics", {}).get("mean_packet_loss") is not None else None),
            "UEs": current.get("ue_count"),
            "gNBs": current.get("gnb_count", 1),
            "Band 1 (MHz)": current.get("bandwidth_band1_mhz"),
            "Band 2 (MHz)": current.get("bandwidth_band2_mhz"),
            "Power (dBm)": current.get("total_tx_power_dbm"),
            "Duration (s)": round(completed_duration, 2),
            "Termination": state.get("final_outcome", "—"),
        })
    table = pd.DataFrame(rows)
    summary_columns = [
        "Session", "Started", "Status", "Attempts", "Best score",
    ]
    return table[[column for column in summary_columns if column in table.columns]]


def topology_figure(
    config: dict[str, Any] | None,
    *,
    ue_measurements: list[dict[str, Any]] | None = None,
    highlighted_ue: int | None = None,
    session_target: float | None = None,
) -> go.Figure:
    """Draw configured gNBs and measured serving-cell associations as a schematic."""
    figure = go.Figure()
    if not config:
        return style_figure(figure)
    ue_count = max(0, int(config.get("ue_count", 0)))
    gnb_count = max(1, min(3, int(config.get("gnb_count", 1))))
    measured_cells = {int(row["serving_cell_id"]) for row in (ue_measurements or [])
                      if row.get("serving_cell_id") is not None}
    # Cell IDs are assigned by the scenario's gNB NetDevices; include configured
    # cells even when no UE happened to associate with a cell in the measurements.
    cell_ids = sorted(set(range(1, gnb_count + 1)) | measured_cells)
    cell_x = {cell_id: (0.0 if len(cell_ids) == 1 else -0.55 + 1.1 * index / (len(cell_ids) - 1))
              for index, cell_id in enumerate(cell_ids)}
    cell_y = -0.05
    ue_x = [0.0] if ue_count == 1 else [(-1.0 + 2.0 * index / max(1, ue_count - 1)) for index in range(ue_count)]
    ue_y = [0.75 + (0.08 if index % 2 else 0.0) for index in range(ue_count)]
    measured_by_id = {int(row["ue_id"]): row for row in (ue_measurements or [])}
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for index, (x, y) in enumerate(zip(ue_x, ue_y)):
        item = measured_by_id.get(index + 1, {})
        cell_id = item.get("serving_cell_id")
        if cell_id in cell_x:
            edge_x.extend([cell_x[cell_id], x, None])
            edge_y.extend([cell_y, y, None])
    if edge_x:
        figure.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines", name="Measured UE association",
            line={"color": "#50647e", "width": 1.5}, hoverinfo="skip",
        ))
    for even, name, color in ((True, "Low-latency UDP workload", "#087f8c"), (False, "Voice QoS UDP workload", "#4267b2")):
        positions = [index for index in range(ue_count) if (index % 2 == 0) == even]
        ids = [index + 1 for index in positions]
        customdata = []
        marker_colors = []
        marker_sizes = []
        for ue_id in ids:
            item = measured_by_id.get(ue_id, {})
            customdata.append([
                item.get("application_profile", name), item.get("qoe_proxy"),
                item.get("throughput_mbps"), item.get("delay_ms"),
                item.get("jitter_ms"),
                (float(item["packet_loss_fraction"]) * 100 if item.get("packet_loss_fraction") is not None else None),
                "Measured outage" if item.get("is_outage") else (
                    "Measurements available" if item else "Per-UE measurements unavailable"
                ),
                item.get("serving_cell_id"),
            ])
            marker_colors.append("#c24450" if item.get("is_outage") else color)
            marker_sizes.append(25 if highlighted_ue == ue_id else 20)
        figure.add_trace(go.Scatter(
            x=[ue_x[index] for index in positions], y=[ue_y[index] for index in positions],
            mode="markers+text", name=name,
            text=[f"UE {ue_id}" for ue_id in ids], textposition="top center",
            customdata=customdata,
            marker={"size": marker_sizes, "color": marker_colors, "line": {"color": "#e8f1fb", "width": 1}},
            hovertemplate=("%{text}<br>Simulated profile: %{customdata[0]}"
                           "<br>Serving cell: %{customdata[7]}"
                           "<br>Per-flow QoE proxy: %{customdata[1]}"
                           "<br>Throughput: %{customdata[2]} Mbps"
                           "<br>Latency: %{customdata[3]} ms"
                           "<br>Jitter: %{customdata[4]} ms"
                           "<br>Packet loss: %{customdata[5]}%"
                           "<br>%{customdata[6]}<extra></extra>"),
        ))
    figure.add_trace(go.Scatter(
        x=[cell_x[cell_id] for cell_id in cell_ids], y=[cell_y] * len(cell_ids),
        mode="markers+text", name="gNB / cell",
        text=[f"Cell {cell_id}" for cell_id in cell_ids], textposition="bottom center",
        customdata=[[cell_id, config.get("bandwidth_band1_mhz", "—"),
                     config.get("bandwidth_band2_mhz", "—")] for cell_id in cell_ids],
        marker={"size": 28, "symbol": "diamond", "color": "#23835b", "line": {"color": "#ffffff", "width": 1}},
        hovertemplate=("Cell %{customdata[0]} (gNB)<br>Band 1: %{customdata[1]} MHz"
                       "<br>Band 2: %{customdata[2]} MHz<extra></extra>"),
    ))
    figure.update_layout(
        title=f"{len(cell_ids)}-cell scenario · schematic, not to scale",
        height=350,
        margin={"l": 10, "r": 10, "t": 55, "b": 20},
        legend={"orientation": "h", "y": -0.12},
    )
    figure.update_xaxes(visible=False, range=[-1.25, 1.25])
    figure.update_yaxes(visible=False, range=[-0.35, 1.2], scaleanchor="x", scaleratio=1)
    if session_target is not None:
        figure.add_annotation(
            text=f"Session QoE target reference: {session_target:.3f} (not a per-UE SLA)",
            xref="paper", yref="paper", x=0.5, y=1.14, showarrow=False,
            font={"size": 11, "color": "#52657d"},
        )
    return style_figure(figure)


def ue_measurement_table(
    rows: list[dict[str, Any]], *, session_target: float, ue_count: int | None = None,
) -> pd.DataFrame:
    """Present per-flow measurements without inventing per-UE service targets."""
    display = []
    measured = {int(row["ue_id"]): row for row in rows}
    count = int(ue_count) if ue_count is not None else len(measured)
    for ue_id in range(1, count + 1):
        row = measured.get(ue_id)
        traffic_class = "low_latency" if (ue_id - 1) % 2 == 0 else "voice_qos"
        profile = "Low-latency UDP workload" if traffic_class == "low_latency" else "Voice QoS UDP workload"
        score = float(row["qoe_proxy"]) if row else None
        if not row:
            comparison = "Insufficient per-UE measurements"
        elif row.get("is_outage"):
            comparison = "Measured outage"
        elif score is not None and score < float(session_target):
            comparison = "Below session target reference"
        else:
            comparison = "At/above session target reference"
        display.append({
            "UE": f"UE {ue_id}",
            "Serving cell": row.get("serving_cell_id") if row else None,
            "Traffic class": row["traffic_class"] if row else traffic_class,
            "Simulated workload profile": row["application_profile"] if row else profile,
            "Throughput (Mbps)": row["throughput_mbps"] if row else None,
            "Latency (ms)": row["delay_ms"] if row and row["rx_packets"] else None,
            "Jitter (ms)": row["jitter_ms"] if row and row["rx_packets"] else None,
            "Packet loss (%)": float(row["packet_loss_fraction"]) * 100 if row else None,
            "Per-flow QoE proxy": score,
            "Comparison": comparison,
            "Individual service SLA": "Not defined by this scenario",
        })
    return pd.DataFrame(display)


def kpi_trend_figure(events: list[dict[str, Any]], metric: str) -> go.Figure:
    """Plot a saved per-run metric; failed/invalid runs create visible gaps."""
    field, title, unit = METRIC_SPECS[metric]
    x_values, y_values = [], []
    for event in events:
        x_values.append(int(event.get("epoch", len(x_values) + 1)))
        metrics = event.get("metrics") or {}
        value = event.get("measured_qoe_proxy") if metric == "QoE proxy" else metrics.get(field)
        if event.get("run_status") != "COMPLETED" or value is None:
            y_values.append(None)
        else:
            y_values.append(float(value) * 100 if metric == "Packet loss" else float(value))
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=x_values, y=y_values, mode="lines+markers", name=title,
        connectgaps=False, line={"width": 2.5, "color": DASHBOARD_PALETTE[0]},
        marker={"size": 8},
        hovertemplate=f"Run %{{x}}<br>{title}: %{{y:.4f}} {unit}<extra></extra>",
    ))
    figure.update_layout(
        title=title, height=300, xaxis_title="Simulation run", yaxis_title=unit,
        margin={"l": 40, "r": 20, "t": 55, "b": 45}, showlegend=False,
    )
    if metric == "QoE proxy":
        figure.update_yaxes(range=[0, 1.02])
    return style_figure(figure)


def render_overview(st_module: Any) -> None:
    """Give recent ns-3 evidence a clear, quality-first summary."""
    page_header("Network overview", "Latest validated simulation evidence and saved session history.")
    records = load_live_records()
    if not records:
        st_module.info("No simulator sessions are saved yet. Open Run simulation to configure and start an ns-3 run.")
        st_module.caption("This overview will populate from persisted session records after a run is started.")
        return
    record = records[0]
    if record.get("error"):
        st_module.error(f"Could not read the latest session record: {record['error']}")
        return
    folder, state, events = record["path"], record["state"], record["events"]
    valid = valid_measurements(events)
    latest = valid[-1] if valid else None
    config = (state.get("current_config") or state.get("initial_config") or {})
    if latest and latest.get("config"):
        config = latest["config"]
    metrics = latest.get("metrics", {}) if latest else {}
    status = session_status_label({**state, "stop_requested": (folder / "STOP_REQUESTED").exists()})
    display_state = {**state, "stop_requested": (folder / "STOP_REQUESTED").exists()}
    st.caption(f"Latest session · `{record['id']}` · {session_status_explanation(display_state)}")
    st.subheader("Latest validated run")
    if latest:
        def measured(value: Any, unit: str, scale: float = 1.0) -> str:
            return f"{float(value) * scale:.2f} {unit}" if value is not None else "Not available"

        score_col, context_col = st.columns([1, 2], gap="large")
        with score_col:
            st.metric("Measured QoE proxy", f"{float(latest['measured_qoe_proxy']):.4f}")
            st.caption("Dimensionless simulated score · 0–1 · not a subjective user rating")
            st.caption(f"Session status: {status}")
        with context_col:
            with st.container(border=True):
                st.markdown("**Run configuration**")
                st.markdown(
                    f"**{config.get('ue_count', '—')} UEs** · **{config.get('gnb_count', 1)} gNB cells**"
                )
                st.caption(
                    f"Band 1 {config.get('bandwidth_band1_mhz', 'Not available')} MHz · "
                    f"Band 2 {config.get('bandwidth_band2_mhz', 'Not available')} MHz · "
                    f"Requested total power {config.get('total_tx_power_dbm', 'Not available')} dBm"
                )
        st.markdown("**Measured network KPIs**")
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Throughput", measured(metrics.get("mean_throughput_mbps"), "Mbps"))
        kpi_cols[1].metric("One-way latency", measured(metrics.get("mean_delay_ms"), "ms"))
        kpi_cols[2].metric("Jitter", measured(metrics.get("mean_jitter_ms"), "ms"))
        kpi_cols[3].metric("Packet loss", measured(metrics.get("mean_packet_loss"), "%", 100))
    else:
        with st.container(border=True):
            st.metric("Measured QoE proxy", "Not available")
            st.caption("This session has no validated completed measurement. Failed or invalid runs are not displayed as measured zero values.")
            st.caption(f"Session status: {status} · configuration ID: {config.get('config_id', '—')}")

    if valid:
        st.subheader("QoE proxy across simulation runs")
        st.caption("Each point is a validated run; the dashed reference is this session’s configured target.")
        st.plotly_chart(build_qoe_figure(events, float(state.get("target_qoe", 0.8))), width="stretch", key="overview_recent_qoe")

    st.subheader("Recent sessions")
    with st.container(border=True):
        st.dataframe(session_table(records[:8]), hide_index=True, width="stretch")


def render_experiments(st_module: Any) -> None:
    """List saved sessions and expose their existing detail/history artifacts."""
    page_header("Experiments", "Review ns-3 sessions, run limits, outcomes, and saved epoch records.")
    records = load_live_records()
    if not records:
        st_module.info("No experiment sessions found. Starting a simulation is always an explicit action on Live Simulator.")
        return
    summary = session_table(records)
    status_values = sorted(summary["Status"].dropna().unique().tolist()) if "Status" in summary else []
    with st_module.expander("Filter sessions", expanded=False):
        selected_statuses = st_module.multiselect(
            "Status", status_values, default=status_values, key="experiments_status_filter",
        )
    def record_status(record: dict[str, Any]) -> str:
        if record.get("error"):
            return "Unreadable"
        folder = record["path"]
        state = {**record["state"], "stop_requested": (folder / "STOP_REQUESTED").exists()}
        return session_status_label(state)

    filtered_records = [
        record for record in records
        if record_status(record) in selected_statuses
    ]
    with st_module.container(border=True):
        st_module.caption("Start times are UTC. Best score is the measured QoE proxy (0–1).")
        st_module.dataframe(
            session_table(filtered_records), hide_index=True, width="stretch",
            column_config={
                "Session": st_module.column_config.TextColumn(width="medium"),
                "Started": st_module.column_config.TextColumn(width="small"),
                "Status": st_module.column_config.TextColumn(width="large"),
                "Attempts": st_module.column_config.TextColumn(width="small"),
                "Best score": st_module.column_config.NumberColumn(format="%.4f", width="small"),
            },
        )
    usable = [record for record in filtered_records if not record.get("error")]
    if not usable:
        st_module.info("No readable sessions match the selected status filter.")
        return
    label_to_id = {f"{record['id']} · {record_status(record)}": record["id"] for record in usable}
    selected_label = st_module.selectbox("Inspect session", list(label_to_id), key="experiments_detail_session")
    selected = next(record for record in usable if record["id"] == label_to_id[selected_label])
    state, events, folder = selected["state"], selected["events"], selected["path"]
    counts = session_run_counts(folder, events, state)
    valid = valid_measurements(events)
    st_module.subheader("Selected session")
    st_module.markdown(f"`{selected['id']}` · **{record_status(selected)}**")
    st_module.caption(state.get("final_outcome") or state.get("phase") or "Session is in progress.")
    summary_cols = st_module.columns(5)
    extra_limit = (int(state.get("max_additional_exploratory_epochs", 0))
                   if state.get("run_remaining_experimental_candidates") else 0)
    summary_cols[0].metric("Validated runs", len(valid))
    summary_cols[1].metric("Normal attempts", f"{counts['normal_attempted']} / {state.get('max_epochs', 0)}")
    summary_cols[2].metric("Extra attempts", f"{counts['experimental_attempted']} / {extra_limit}")
    best_score = max((float(event["measured_qoe_proxy"]) for event in valid), default=None)
    summary_cols[3].metric("Best measured QoE proxy", f"{best_score:.4f}" if best_score is not None else "Not available")
    summary_cols[4].metric("Controller mode", state.get("controller_mode", "Not recorded"))
    detail_tab, chart_tab, files_tab = st_module.tabs(["Configuration and measurements", "Run timeline", "Saved files"])
    with detail_tab:
        st_module.json({
            "initial_configuration": state.get("initial_config"),
            "current_configuration": state.get("current_config"),
            "controller_mode": state.get("controller_mode"),
            "target_score": state.get("target_qoe"),
            "normal_run_limit": state.get("max_epochs"),
            "additional_experiment_limit": (state.get("max_additional_exploratory_epochs")
                                             if state.get("run_remaining_experimental_candidates") else 0),
        })
        if events:
            from dashboard.live_ns3_control import _history_table
            st_module.dataframe(_history_table(events), hide_index=True, width="stretch")
        else:
            st_module.info("No run records have been saved for this session yet.")
    with chart_tab:
        if valid_measurements(events):
            st_module.plotly_chart(build_qoe_figure(events, float(state.get("target_qoe", 0.8))), width="stretch",
                                   key=f"experiment_timeline_{selected['id']}")
        else:
            st_module.info("A timeline will appear after a validated simulation result is saved.")
    with files_tab:
        st_module.caption(f"Session directory: {folder}")
        for filename in ("session.json", "session_config.json", "events.jsonl", "controller.log", "worker.log", "epoch_history.csv"):
            path = folder / filename
            if path.is_file():
                st_module.write(f"**{filename}** · {path.stat().st_size:,} bytes")
        csv_path = folder / "epoch_history.csv"
        if csv_path.is_file():
            st_module.download_button(
                "Download epoch history CSV", data=csv_path.read_bytes(),
                file_name=f"{selected['id']}_epoch_history.csv", mime="text/csv",
                key=f"experiments_csv_{selected['id']}",
            )


def render_analytics(st_module: Any) -> None:
    """Compare measured session outcomes and inspect per-run KPI trends."""
    page_header("Analytics", "Compare measured ns-3 outcomes and inspect KPI changes across simulation runs.")
    records = [record for record in load_live_records() if not record.get("error") and valid_measurements(record["events"])]
    if not records:
        st_module.info("No validated run measurements are available for analysis yet.")
        return
    ids = [record["id"] for record in records]
    default_ids = ids[:min(5, len(ids))]
    selected_ids = st_module.multiselect("Sessions to compare", ids, default=default_ids, key="analytics_compare_sessions")
    selected_records = [record for record in records if record["id"] in selected_ids]
    metric = st_module.selectbox("Comparison metric", list(METRIC_SPECS), key="analytics_metric")
    field, title, unit = METRIC_SPECS[metric]
    values = []
    for record in selected_records:
        measurements = valid_measurements(record["events"])
        samples = [
            float(event["measured_qoe_proxy"] if metric == "QoE proxy" else event["metrics"].get(field))
            for event in measurements
            if (event.get("measured_qoe_proxy") if metric == "QoE proxy" else event["metrics"].get(field)) is not None
        ]
        if samples:
            value = (sum(samples) / len(samples)) * 100 if metric == "Packet loss" else sum(samples) / len(samples)
            configs = [event.get("config") or {} for event in measurements]
            configs = [config for config in configs if config]
            distinct_configurations = sorted({
                (config.get("ue_count"), config.get("bandwidth_band1_mhz"),
                 config.get("bandwidth_band2_mhz"), config.get("total_tx_power_dbm"))
                for config in configs
            })
            config = configs[-1] if configs else (
                record["state"].get("current_config") or record["state"].get("initial_config") or {}
            )
            config_summary = (f"{len(distinct_configurations)} configurations; latest: " if len(distinct_configurations) > 1 else "")
            values.append({
                "Session": record["id"], "Mean measured value": value,
                "Configuration": (f"{config_summary}{config.get('ue_count')} UEs; B1/B2 "
                                  f"{config.get('bandwidth_band1_mhz')}/{config.get('bandwidth_band2_mhz')} MHz; "
                                  f"{config.get('total_tx_power_dbm')} dBm"),
                "Valid runs": len(samples),
            })
    if values:
        comparison = pd.DataFrame(values)
        chart = go.Figure(go.Bar(
            x=comparison["Session"], y=comparison["Mean measured value"],
            marker_color=DASHBOARD_PALETTE[0],
            customdata=comparison[["Configuration", "Valid runs"]].values,
            hovertemplate="%{x}<br>Mean %{y:.4f} " + unit + "<br>%{customdata[0]}<br>Valid runs: %{customdata[1]}<extra></extra>",
        ))
        chart.update_layout(title=f"Mean measured {title} by session", xaxis_title="Session", yaxis_title=unit,
                            height=360, margin={"l": 45, "r": 15, "t": 55, "b": 80})
        st_module.plotly_chart(style_figure(chart), width="stretch", key="analytics_session_comparison")
        st_module.caption("These are per-session means over validated runs. Sessions can use different configurations, traffic, durations, and random seeds; comparisons are descriptive, not controlled causal estimates.")
        st_module.dataframe(comparison, hide_index=True, width="stretch")
    if not selected_records:
        st_module.info("Select at least one session to show a comparison.")
        return
    label_to_record = {f"{record['id']} · {session_status_label(record['state'])}": record for record in selected_records}
    detail_label = st_module.selectbox("Inspect run-by-run trends", list(label_to_record), key="analytics_detail_session")
    detail = label_to_record[detail_label]
    events = detail["events"]
    st_module.subheader(f"Run-by-run measurements · {detail['id']}")
    st_module.caption("Run index is used because the scenario stores per-run aggregates, not within-run metric timestamps. Failed or invalid runs remain gaps.")
    chart_cols = st_module.columns(2)
    chart_cols[0].plotly_chart(kpi_trend_figure(events, "Throughput"), width="stretch", key=f"analytics_throughput_{detail['id']}")
    chart_cols[1].plotly_chart(kpi_trend_figure(events, "Latency"), width="stretch", key=f"analytics_latency_{detail['id']}")
    chart_cols[0].plotly_chart(kpi_trend_figure(events, "Jitter"), width="stretch", key=f"analytics_jitter_{detail['id']}")
    chart_cols[1].plotly_chart(kpi_trend_figure(events, "Packet loss"), width="stretch", key=f"analytics_loss_{detail['id']}")
