"""
dashboard/components.py
=======================
Reusable Streamlit / Plotly components for the Phase 4 dashboard.

All functions are pure "render" helpers that accept data and render it;
none of them read files or call the ML / simulation APIs directly.
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOUR_PRIMARY   = "#1f77b4"   # muted blue
COLOUR_SUCCESS   = "#2ca02c"   # green
COLOUR_WARNING   = "#ff7f0e"   # orange
COLOUR_DANGER    = "#d62728"   # red
COLOUR_OUTAGE    = "#9467bd"   # purple

DASHBOARD_PALETTE = ["#087f8c", "#4267b2", "#23835b", "#ad6d00", "#c24450", "#7651a8"]


def render_dashboard_theme() -> None:
    """Apply a calm, high-contrast workspace theme across all dashboard pages."""
    st.markdown(
        """
        <style>
        :root {
          --app-bg: #0e1117;
          --app-panel: #161b22;
          --app-panel-raised: #1f2630;
          --app-border: #303844;
          --app-text: #e6edf3;
          --app-muted: #a5b2c3;
          --app-accent: #35cbb2;
        }
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
          background: var(--app-bg);
          color: var(--app-text);
        }
        [data-testid="stSidebar"] {
          background: #11161d;
          border-right: 1px solid var(--app-border);
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
          color: var(--app-text) !important;
        }
        [data-testid="stMainBlockContainer"] {
          max-width: 1420px;
          padding: 1.5rem 2rem 3rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
          border-radius: 8px;
          padding: 0.28rem 0.55rem;
          transition: background 120ms ease;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
          background: #173b37;
          border-left: 3px solid var(--app-accent);
          font-weight: 650;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
          background: #202a35;
        }
        [data-testid="stMetric"] {
          background: var(--app-panel);
          border: 1px solid var(--app-border);
          border-radius: 12px;
          padding: 0.95rem 1.05rem;
          min-height: 92px;
          box-shadow: 0 2px 6px rgba(0, 0, 0, .24);
        }
        [data-testid="stMetricLabel"] p {
          color: var(--app-muted);
        }
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
          color: #a5b2c3 !important;
          opacity: 1 !important;
          font-size: 0.84rem;
        }
        [data-testid="stMetricValue"] { color: var(--app-text); }
        [data-testid="stVerticalBlockBorderWrapper"] {
          border-color: var(--app-border);
          border-radius: 12px;
          background: var(--app-panel);
        }
        [data-testid="stForm"] {
          border: 1px solid var(--app-border);
          border-radius: 12px;
          padding: 1.15rem 1.25rem 0.8rem;
          background: var(--app-panel);
          box-shadow: 0 2px 8px rgba(0, 0, 0, .24);
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
          border: 1px solid var(--app-border);
          border-radius: 8px;
          overflow: hidden;
        }
        h1, h2, h3 { letter-spacing: -0.025em; color: var(--app-text); }
        h1 { font-size: 2.05rem; line-height: 1.2; margin-bottom: 0.15rem; }
        h2 { font-size: 1.35rem; margin-top: 1.45rem; }
        h3 { font-size: 1.08rem; }
        p, label, li { color: var(--app-text); }
        a { color: var(--app-accent) !important; }
        button[kind="primary"] {
          border: 1px solid #35cbb2;
          background: #167f70;
          border-radius: 8px;
          font-weight: 650;
        }
        button[kind="primary"]:hover { background: #1b9b88; border-color: #1b9b88; }
        input:focus, textarea:focus, [data-baseweb="select"] > div:focus-within {
          border-color: var(--app-accent) !important;
          box-shadow: 0 0 0 2px rgba(53, 203, 178, .24) !important;
        }
        div[data-testid="stAlert"] { border-radius: 10px; }
        [data-testid="stTabs"] [role="tab"] { color: var(--app-muted); }
        [data-testid="stTabs"] [aria-selected="true"] { color: var(--app-accent); }
        @media (max-width: 760px) {
          [data-testid="stMainBlockContainer"] { padding: 1rem 0.85rem 2rem; }
          [data-testid="stMetric"] { min-height: 84px; padding: 0.65rem; }
          [data-testid="stForm"] { padding: 0.75rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, description: str, context: str | None = None) -> None:
    """Render a consistent concise title and supporting description."""
    st.title(title)
    st.caption(description)
    if context:
        st.info(context)


def style_figure(figure: go.Figure) -> go.Figure:
    """Style Plotly figures to stay readable on the dashboard's dark surfaces."""
    figure.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font={"color": "#e6edf3", "family": "Inter, Segoe UI, sans-serif", "size": 12},
        colorway=DASHBOARD_PALETTE,
        hoverlabel={"bgcolor": "#303844", "font": {"color": "#ffffff"}},
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    figure.update_xaxes(showgrid=False, linecolor="#748094", zerolinecolor="#748094")
    figure.update_yaxes(showgrid=True, gridcolor="#303844", linecolor="#748094", zerolinecolor="#748094")
    return figure

STATUS_COLOURS = {
    "VERIFIED_SUCCESS":   COLOUR_SUCCESS,
    "SIMULATION_FAILED":  COLOUR_DANGER,
    "OPTIMISER_INFEASIBLE": COLOUR_WARNING,
    "UNKNOWN":            "#7f7f7f",
}


# ---------------------------------------------------------------------------
# KPI card
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, delta: str | None = None,
             delta_colour: str = "normal") -> None:
    """Render a single KPI metric using st.metric."""
    st.metric(label=label, value=value, delta=delta,
              delta_color=delta_colour)  # type: ignore[arg-type]


def kpi_row(metrics: list[dict]) -> None:
    """Render a horizontal row of KPI cards.

    Each dict in *metrics* may have keys: label, value, delta, delta_colour.
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            kpi_card(
                label=m["label"],
                value=str(m["value"]),
                delta=m.get("delta"),
                delta_colour=m.get("delta_colour", "normal"),
            )


# ---------------------------------------------------------------------------
# QoE gauge
# ---------------------------------------------------------------------------

def qoe_gauge(qoe: float, title: str = "Predicted QoE") -> go.Figure:
    """Return a Plotly gauge chart for a QoE value in [0, 1]."""
    colour = (
        COLOUR_SUCCESS if qoe >= 0.75
        else COLOUR_WARNING if qoe >= 0.50
        else COLOUR_DANGER
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(qoe, 4),
        title={"text": title, "font": {"size": 16}},
        number={"font": {"size": 32}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": colour},
            "steps": [
                {"range": [0.00, 0.50], "color": "#ffe0e0"},
                {"range": [0.50, 0.75], "color": "#fff3cd"},
                {"range": [0.75, 1.00], "color": "#d4edda"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": qoe,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=20, l=20, r=20))
    return style_figure(fig)


# ---------------------------------------------------------------------------
# QoE histogram
# ---------------------------------------------------------------------------

def qoe_histogram(df: pd.DataFrame, nbins: int = 30) -> go.Figure:
    """Return a Plotly histogram of qoe_proxy values."""
    fig = px.histogram(
        df,
        x="qoe_proxy",
        nbins=nbins,
        title="QoE Distribution",
        labels={"qoe_proxy": "QoE proxy (dimensionless)"},
        color_discrete_sequence=[COLOUR_PRIMARY],
    )
    fig.update_layout(
        bargap=0.05,
        xaxis_range=[0, 1],
        height=350,
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return style_figure(fig)


# ---------------------------------------------------------------------------
# QoE vs UE count box plot
# ---------------------------------------------------------------------------

def qoe_vs_ue_box(df: pd.DataFrame) -> go.Figure:
    """Box plot of QoE by UE count, coloured by bandwidth."""
    fig = px.box(
        df.sort_values("ue_count"),
        x="ue_count",
        y="qoe_proxy",
        color="bandwidth_mhz",
        title="QoE vs UE Count (by Bandwidth)",
        labels={
            "ue_count": "UE Count",
            "qoe_proxy": "QoE proxy",
            "bandwidth_mhz": "Bandwidth (MHz)",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=380, margin=dict(t=40, b=40, l=40, r=20))
    return style_figure(fig)


# ---------------------------------------------------------------------------
# Throughput vs Delay scatter
# ---------------------------------------------------------------------------

def throughput_vs_delay_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter: throughput_mbps vs delay_ms, coloured by flow_type."""
    plot_df = df.copy()
    # Mark outage rows clearly
    plot_df["marker"] = plot_df["is_outage"].apply(
        lambda x: "Outage" if x else "Delivered"
    )
    fig = px.scatter(
        plot_df,
        x="delay_ms",
        y="throughput_mbps",
        color="flow_type",
        symbol="marker",
        title="Throughput vs Delay",
        labels={
            "delay_ms": "Delay (ms)",
            "throughput_mbps": "Throughput (Mbps)",
            "flow_type": "Flow Type",
            "marker": "Status",
        },
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_layout(height=400, margin=dict(t=40, b=40, l=40, r=20))
    return style_figure(fig)


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def model_comparison_bar(model_results_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing model MAE, RMSE, R² across benchmarked models."""
    df = model_results_df.copy()
    fig = go.Figure()
    metrics = [
        ("test_mae",  "Test MAE",  COLOUR_DANGER),
        ("test_rmse", "Test RMSE", COLOUR_WARNING),
    ]
    for col, label, colour in metrics:
        if col in df.columns:
            fig.add_trace(go.Bar(
                name=label,
                x=df["model_name"],
                y=df[col],
                marker_color=colour,
            ))
    fig.update_layout(
        barmode="group",
        title="Model Benchmark — MAE & RMSE (lower is better)",
        xaxis_title="Model",
        yaxis_title="Error",
        height=360,
        margin=dict(t=50, b=60, l=40, r=20),
    )
    return style_figure(fig)


def model_r2_bar(model_results_df: pd.DataFrame) -> go.Figure:
    """Bar chart of test R² for each benchmarked model."""
    df = model_results_df.copy()
    if "test_r2" not in df.columns:
        return style_figure(go.Figure())
    fig = px.bar(
        df,
        x="model_name",
        y="test_r2",
        title="Model Benchmark — Test R² (higher is better)",
        labels={"model_name": "Model", "test_r2": "Test R²"},
        color="test_r2",
        color_continuous_scale="Blues",
        range_y=[0, 1.05],
    )
    fig.update_layout(height=340, margin=dict(t=50, b=60, l=40, r=20))
    return style_figure(fig)


# ---------------------------------------------------------------------------
# Candidate ranking table for optimiser page
# ---------------------------------------------------------------------------

def render_candidate_table(candidates: list[dict]) -> None:
    """Render the optimiser candidate table with colour highlighting."""
    if not candidates:
        st.info("No candidates evaluated.")
        return
    df = pd.DataFrame(candidates)
    cols_to_show = [c for c in [
        "rank", "bandwidth_mhz", "total_tx_power_dbm",
        "predicted_qoe", "feasible", "violations",
    ] if c in df.columns]
    df_show = df[cols_to_show].copy()
    if "predicted_qoe" in df_show.columns:
        df_show["predicted_qoe"] = df_show["predicted_qoe"].round(4)
    st.dataframe(
        df_show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="%d"),
            "bandwidth_mhz": st.column_config.NumberColumn("BW (MHz)", format="%.0f"),
            "total_tx_power_dbm": st.column_config.NumberColumn("Power (dBm)", format="%.0f"),
            "predicted_qoe": st.column_config.ProgressColumn(
                "Predicted QoE", min_value=0, max_value=1, format="%.4f"
            ),
            "feasible": st.column_config.CheckboxColumn("Feasible?"),
        },
    )


# ---------------------------------------------------------------------------
# Phase 3 results table
# ---------------------------------------------------------------------------

def render_phase3_table(df: pd.DataFrame) -> None:
    """Render the Phase-3 verification results table with status colour coding."""
    if df.empty:
        st.info("No verification runs recorded yet.")
        return

    display_cols = [c for c in [
        "run_id", "timestamp", "ue_count", "flow_type", "lambda_val",
        "optimised_bandwidth_mhz", "predicted_qoe", "measured_qoe",
        "absolute_prediction_error", "verification_status", "sim_success",
    ] if c in df.columns]

    display_df = df[display_cols].copy().sort_values(
        "timestamp", ascending=False
    ) if "timestamp" in display_cols else df[display_cols].copy()

    # Truncate run_id for readability
    if "run_id" in display_df.columns:
        display_df["run_id"] = display_df["run_id"].astype(str)

    if "timestamp" in display_df.columns:
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")

    col_config: dict = {}
    if "predicted_qoe" in display_df.columns:
        col_config["predicted_qoe"] = st.column_config.ProgressColumn(
            "Predicted QoE", min_value=0, max_value=1, format="%.4f"
        )
    if "measured_qoe" in display_df.columns:
        col_config["measured_qoe"] = st.column_config.ProgressColumn(
            "Measured QoE", min_value=0, max_value=1, format="%.4f"
        )
    if "absolute_prediction_error" in display_df.columns:
        col_config["absolute_prediction_error"] = st.column_config.NumberColumn(
            "Abs Error", format="%.4f"
        )
    if "sim_success" in display_df.columns:
        col_config["sim_success"] = st.column_config.CheckboxColumn("ns-3 OK?")

    st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=col_config)


# ---------------------------------------------------------------------------
# Disclaimer banner
# ---------------------------------------------------------------------------

def research_disclaimer() -> None:
    """Render a small disclaimer that this is a research dashboard."""
    st.caption(
        "⚠️ **Research dashboard** — not a live network controller. "
        "All ML predictions are estimates; verification requires ns-3 simulation."
    )

