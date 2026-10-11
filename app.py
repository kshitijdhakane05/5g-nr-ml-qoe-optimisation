"""
app.py  —  Phase 4: 5G NR QoE Optimisation Dashboard
======================================================
Entry point: `streamlit run app.py`

Pages
-----
  1. Overview
  2. Simulation Explorer
  3. QoE Predictor
  4. Network Optimiser
  5. Closed-Loop Verification

IMPORTANT
---------
* This dashboard is a research/demo tool — NOT a live network controller.
* Simulations are launched only on explicit button clicks, never on page load.
* All predictions are ML estimates; verification requires ns-3 runs.
* The raw dataset (`data/raw/qoe_dataset.csv`) is never written to.
"""

from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path

# Make sure the project root is on sys.path so that ml/, optimisation/,
# simulation/ can be imported when Streamlit changes cwd.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="5G NR QoE Optimisation Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports of project modules (after sys.path fix)
# ---------------------------------------------------------------------------
from dashboard.data_loader import (          # noqa: E402
    load_processed_dataset,
    load_model_results,
    load_predictions,
    load_phase3_results,
    dataset_summary,
    PHASE3_RESULTS,
)
from dashboard.components import (           # noqa: E402
    kpi_row,
    qoe_gauge,
    qoe_histogram,
    qoe_vs_ue_box,
    throughput_vs_delay_scatter,
    model_comparison_bar,
    model_r2_bar,
    render_candidate_table,
    render_phase3_table,
    research_disclaimer,
    STATUS_COLOURS,
    COLOUR_SUCCESS,
    COLOUR_DANGER,
    COLOUR_WARNING,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "📊 Overview",
    "🔍 Simulation Explorer",
    "🤖 QoE Predictor",
    "⚙️ Network Optimiser",
    "🔁 Closed-Loop Verification",
]

with st.sidebar:
    st.title("📡 5G NR Dashboard")
    st.caption("Phase 4 — ML-based QoE Optimisation")
    st.divider()
    page = st.radio("Navigate to", PAGES, label_visibility="collapsed")
    st.divider()
    research_disclaimer()


# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
if page == PAGES[0]:
    st.title("📊 Project Overview")
    st.markdown(
        "End-to-end ML-based 5G NR network optimisation: "
        "ns-3 simulation → QoE prediction → configuration search → ns-3 verification."
    )
    research_disclaimer()

    df = load_processed_dataset()
    summary = dataset_summary(df)
    model_df = load_model_results()
    phase3_df = load_phase3_results()

    # --- KPI row 1: dataset stats ---
    st.subheader("Dataset")
    if summary:
        kpi_row([
            {"label": "Total Flow Records",   "value": f"{summary['rows']:,}"},
            {"label": "Unique Scenarios",     "value": summary["unique_scenarios"]},
            {"label": "Unique Flows",         "value": summary["unique_flows"]},
            {"label": "UE-Count Range",       "value": f"{summary['ue_count_range'][0]}–{summary['ue_count_range'][1]}"},
            {"label": "Outage Flows",         "value": summary["outage_count"]},
        ])
    else:
        st.warning("Processed dataset not found.")

    # --- KPI row 2: best model ---
    st.subheader("ML Model (Best: Random Forest)")
    if not model_df.empty:
        best = model_df.sort_values("test_r2", ascending=False).iloc[0]
        kpi_row([
            {"label": "Model",     "value": best.get("model_name", "—")},
            {"label": "Test MAE",  "value": f"{best.get('test_mae',0):.4f}"},
            {"label": "Test RMSE", "value": f"{best.get('test_rmse',0):.4f}"},
            {"label": "Test R²",   "value": f"{best.get('test_r2',0):.4f}"},
        ])
    else:
        st.warning("`results/ml/model_results.csv` not found.")

    # --- KPI row 3: phase 3 verification ---
    st.subheader("Phase 3: Closed-Loop Verification Runs")
    if not phase3_df.empty and "verification_status" in phase3_df.columns:
        n_total   = len(phase3_df)
        n_success = int((phase3_df["verification_status"] == "VERIFIED_SUCCESS").sum())
        n_failed  = int((phase3_df["verification_status"] == "SIMULATION_FAILED").sum())
        avg_err   = phase3_df["absolute_prediction_error"].dropna().mean() if "absolute_prediction_error" in phase3_df.columns else None
        kpi_row([
            {"label": "Total Runs",    "value": n_total},
            {"label": "✅ Successful", "value": n_success},
            {"label": "❌ Failed",     "value": n_failed},
            {"label": "Mean |Error|",  "value": f"{avg_err:.4f}" if avg_err is not None else "—"},
        ])
        latest = phase3_df.sort_values("timestamp", ascending=False).iloc[0] if "timestamp" in phase3_df.columns else phase3_df.iloc[0]
        with st.expander("Latest verification run"):
            st.json({k: str(v) for k, v in latest.items()})
    else:
        st.info("No Phase-3 verification runs recorded yet.")

    # --- Workflow diagram ---
    st.subheader("Workflow")
    st.markdown("""
```
Network Context (UE count, flow type, λ, scenario size)
    │
    ▼
Phase 2: Offline Optimiser  →  selects bandwidth configuration
    │
    ▼
Phase 3: ns-3 Simulation  →  measures real QoE metrics
    │
    ▼
QoE Comparison  →  Predicted vs Measured (verification record)
```
*Phase 1: ML model trained on ns-3 data — provides the QoE surrogate for Phase 2.*
""")

    # --- QoE distribution overview chart ---
    if not df.empty and "qoe_proxy" in df.columns:
        st.subheader("QoE Distribution (full dataset)")
        st.plotly_chart(qoe_histogram(df), use_container_width=True)


# ===========================================================================
# PAGE 2 — SIMULATION EXPLORER
# ===========================================================================
elif page == PAGES[1]:
    st.title("🔍 Simulation Data Explorer")
    research_disclaimer()

    df = load_processed_dataset()
    if df.empty:
        st.error("Processed dataset not found — cannot show explorer.")
        st.stop()

    st.caption(f"Dataset: `data/processed/qoe_dataset.csv`  |  {len(df):,} rows")

    # --- Filters ---
    with st.expander("🔧 Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        all_ue   = sorted(df["ue_count"].dropna().unique().tolist())
        all_bw   = sorted(df["bandwidth_mhz"].dropna().unique().tolist())
        all_ft   = sorted(df["flow_type"].dropna().unique().tolist()) if "flow_type" in df.columns else []
        all_lam  = sorted(df["flow_lambda"].dropna().unique().tolist()) if "flow_lambda" in df.columns else []

        sel_ue  = col1.multiselect("UE Count",      all_ue,  default=all_ue)
        sel_bw  = col2.multiselect("Bandwidth (MHz)", all_bw, default=all_bw)
        sel_ft  = col3.multiselect("Flow Type",     all_ft,  default=all_ft)
        sel_lam = col4.multiselect("λ (pkt/s)",     all_lam, default=all_lam)

        qoe_min_val = float(df["qoe_proxy"].min()) if "qoe_proxy" in df.columns else 0.0
        qoe_max_val = float(df["qoe_proxy"].max()) if "qoe_proxy" in df.columns else 1.0
        qoe_range = st.slider(
            "QoE proxy range",
            min_value=0.0, max_value=1.0,
            value=(qoe_min_val, qoe_max_val),
            step=0.01,
        )
        show_outage = st.checkbox("Include outage rows", value=True)

    # Apply filters
    mask = (
        df["ue_count"].isin(sel_ue)
        & df["bandwidth_mhz"].isin(sel_bw)
    )
    if sel_ft and "flow_type" in df.columns:
        mask &= df["flow_type"].isin(sel_ft)
    if sel_lam and "flow_lambda" in df.columns:
        mask &= df["flow_lambda"].isin(sel_lam)
    if "qoe_proxy" in df.columns:
        mask &= df["qoe_proxy"].between(qoe_range[0], qoe_range[1])
    if not show_outage and "is_outage" in df.columns:
        mask &= df["is_outage"] != True  # noqa: E712

    filtered = df[mask].copy()
    st.caption(f"Showing **{len(filtered):,}** rows after filters.")

    # --- Data table ---
    show_cols = [c for c in [
        "scenario_id", "ue_count", "flow_type", "flow_lambda",
        "bandwidth_mhz", "total_tx_power_dbm",
        "throughput_mbps", "delay_ms", "jitter_ms",
        "packet_loss", "qoe_proxy", "delivery_status", "is_outage",
    ] if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True,
                 column_config={
                     "qoe_proxy": st.column_config.ProgressColumn(
                         "QoE proxy", min_value=0, max_value=1, format="%.4f"
                     ),
                     "is_outage": st.column_config.CheckboxColumn("Outage?"),
                 })

    # --- Charts ---
    tab1, tab2, tab3 = st.tabs(["QoE Histogram", "QoE vs UE Count", "Throughput vs Delay"])
    with tab1:
        if not filtered.empty and "qoe_proxy" in filtered.columns:
            st.plotly_chart(qoe_histogram(filtered), use_container_width=True)
    with tab2:
        if not filtered.empty:
            st.plotly_chart(qoe_vs_ue_box(filtered), use_container_width=True)
    with tab3:
        if not filtered.empty:
            st.plotly_chart(throughput_vs_delay_scatter(filtered), use_container_width=True)


# ===========================================================================
# PAGE 3 — QoE PREDICTOR
# ===========================================================================
elif page == PAGES[2]:
    st.title("🤖 QoE Predictor")
    st.markdown(
        "Enter a network configuration below and click **Predict** to get "
        "an ML-estimated QoE score.  "
        "⚠️ *This is a surrogate model estimate, not a measured value.*"
    )
    research_disclaimer()

    # --- Import ML API (lazy) ---
    try:
        from ml.predict import format_input_features, load_model, predict_qoe
        from optimisation.search_space import SUPPORTED_SEARCH_SPACE
        _ml_ok = True
    except ImportError as e:
        st.error(f"Failed to import ML modules: {e}")
        _ml_ok = False

    if _ml_ok:
        MODEL_PATH = _ROOT / "models" / "qoe_model.joblib"

        with st.form("predictor_form"):
            col1, col2 = st.columns(2)

            ue_count = col1.selectbox(
                "UE Count",
                SUPPORTED_SEARCH_SPACE["ue_count"],
                index=0,
            )
            bandwidth_mhz = col1.selectbox(
                "Bandwidth (MHz)",
                SUPPORTED_SEARCH_SPACE["bandwidth_mhz"],
                index=1,   # default 50 MHz
                help="⚠️ 20 MHz at 28 GHz FR2 may produce 100% outage in the simulator.",
            )
            scenario_size_m = col1.selectbox(
                "Scenario Size (m)",
                SUPPORTED_SEARCH_SPACE["scenario_size_m"],
                index=0,
            )
            flow_type = col2.radio(
                "Flow Type",
                SUPPORTED_SEARCH_SPACE["flow_type"],
                index=1,
            )
            lambda_val = col2.selectbox(
                "Arrival Rate λ (pkt/s)",
                SUPPORTED_SEARCH_SPACE["flow_lambda"],
                index=0,
            )
            total_tx_power_dbm = col2.selectbox(
                "Tx Power (dBm) — fixed in dataset",
                SUPPORTED_SEARCH_SPACE["total_tx_power_dbm"],
                disabled=True,
            )

            submitted = st.form_submit_button("🔮 Predict QoE", type="primary")

        if submitted:
            # Warn about 20 MHz
            if bandwidth_mhz == 20.0:
                st.warning(
                    "⚠️ 20 MHz bandwidth at 28 GHz FR2 produces 100% outage "
                    "in the ns-3 simulation. The model may have learned this "
                    "outage pattern — predicted QoE may be near 0."
                )

            try:
                bundle = load_model(str(MODEL_PATH))
                features_df = format_input_features(
                    ue_count=int(ue_count),
                    bandwidth_mhz=float(bandwidth_mhz),
                    total_tx_power_dbm=float(total_tx_power_dbm),
                    scenario_size_m=float(scenario_size_m),
                    flow_type=str(flow_type),
                    lambda_val=float(lambda_val),
                )
                preds = predict_qoe(bundle, features_df)
                qoe_val = float(preds[0])

                col_g, col_d = st.columns([1, 1])
                with col_g:
                    st.plotly_chart(qoe_gauge(qoe_val, "Predicted QoE"), use_container_width=True)
                with col_d:
                    st.subheader("Input features")
                    st.dataframe(features_df.T.rename(columns={0: "Value"}), use_container_width=True)
                    _metrics = bundle.get("test_metrics", {})
                    # The model bundle stores R² under key 'R2', not 'test_r2'.
                    _r2 = _metrics.get("R2") or _metrics.get("test_r2")
                    _r2_str = f"{float(_r2):.4f}" if _r2 is not None else "—"
                    st.caption(
                        f"Model: **{bundle.get('model_name', 'RandomForestRegressor')}** | "
                        f"Test R²: **{_r2_str}**"
                    )

                st.info(
                    "📌 **Predicted QoE is a dimensionless score** (0 = worst, 1 = best).  "
                    "It is *not* a latency or throughput value.  "
                    "To verify this prediction with ns-3, use the **Closed-Loop Verification** page."
                )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        # --- Model performance summary ---
        with st.expander("📈 Model performance summary"):
            model_df = load_model_results()
            if not model_df.empty:
                st.plotly_chart(model_comparison_bar(model_df), use_container_width=True)
                st.plotly_chart(model_r2_bar(model_df), use_container_width=True)
            else:
                st.info("`results/ml/model_results.csv` not found.")


# ===========================================================================
# PAGE 4 — NETWORK OPTIMISER
# ===========================================================================
elif page == PAGES[3]:
    st.title("⚙️ Network Optimiser")
    st.markdown(
        "Given a network context, the optimiser searches the supported "
        "configuration space and selects the configuration with the highest "
        "**predicted** QoE.  "
        "⚠️ *Predicted improvement is pending ns-3 verification.*"
    )
    research_disclaimer()

    try:
        from optimisation.optimizer import NetworkOptimiser
        from optimisation.search_space import SUPPORTED_SEARCH_SPACE
        _opt_ok = True
    except ImportError as e:
        st.error(f"Failed to import optimisation modules: {e}")
        _opt_ok = False

    if _opt_ok:
        MODEL_PATH = str(_ROOT / "models" / "qoe_model.joblib")

        with st.form("optimiser_form"):
            st.subheader("Network Context")
            col1, col2 = st.columns(2)

            ue_count = col1.selectbox(
                "UE Count",
                SUPPORTED_SEARCH_SPACE["ue_count"],
                index=2,
            )
            flow_type = col1.radio(
                "Flow Type",
                SUPPORTED_SEARCH_SPACE["flow_type"],
                index=1,
            )
            lambda_val = col1.selectbox(
                "Arrival Rate λ (pkt/s)",
                SUPPORTED_SEARCH_SPACE["flow_lambda"],
                index=1,
            )
            scenario_size_m = col2.selectbox(
                "Scenario Size (m)",
                SUPPORTED_SEARCH_SPACE["scenario_size_m"],
                index=0,
            )
            baseline_bw = col2.selectbox(
                "Baseline Bandwidth (MHz)",
                SUPPORTED_SEARCH_SPACE["bandwidth_mhz"],
                index=0,
                help="Bandwidth currently in use — used to calculate predicted improvement.",
            )
            baseline_power = col2.selectbox(
                "Baseline Tx Power (dBm)",
                SUPPORTED_SEARCH_SPACE["total_tx_power_dbm"],
                disabled=True,
            )

            st.subheader("Constraints (optional)")
            c1, c2 = st.columns(2)
            min_qoe = c1.slider(
                "Minimum acceptable QoE",
                min_value=0.0, max_value=0.99, value=0.0, step=0.01,
                help="Set > 0 to filter out configurations below this threshold.",
            )
            use_min_qoe = c1.checkbox("Apply minimum QoE constraint", value=False)

            run_opt = st.form_submit_button("🚀 Run Optimiser", type="primary")

        if run_opt:
            constraints = {}
            if use_min_qoe and min_qoe > 0:
                constraints["min_predicted_qoe"] = float(min_qoe)

            with st.spinner("Searching configuration space…"):
                try:
                    optimiser = NetworkOptimiser(MODEL_PATH)
                    result = optimiser.optimise(
                        ue_count=int(ue_count),
                        scenario_size_m=float(scenario_size_m),
                        flow_type=str(flow_type),
                        lambda_val=float(lambda_val),
                        baseline_bandwidth_mhz=float(baseline_bw),
                        baseline_power_dbm=float(baseline_power),
                        constraints=constraints if constraints else None,
                        candidate_bandwidths=SUPPORTED_SEARCH_SPACE["bandwidth_mhz"],
                    )
                except Exception as exc:
                    st.error(f"Optimiser failed: {exc}")
                    result = None

            if result:
                status = result.get("selection_status", "UNKNOWN")
                best   = result.get("best_configuration", {})
                base   = result.get("baseline", {})

                # Status banner
                if status == "FEASIBLE_OPTIMAL":
                    st.success(f"✅ **{status}** — feasible optimal configuration found.")
                elif "INFEASIBLE" in status:
                    st.error(f"❌ **{status}** — no configuration met the constraints.")
                else:
                    st.warning(f"⚠️ **{status}**")

                if best:
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Selected Bandwidth",
                                 f"{best.get('bandwidth_mhz', '—')} MHz",
                                 help="Configuration chosen by optimiser")
                    col_b.metric("Predicted QoE",
                                 f"{best.get('predicted_qoe', 0):.4f}",
                                 delta=result.get("predicted_improvement_label", ""),
                                 delta_color="normal")
                    col_c.metric("Candidates Evaluated",
                                 result.get("num_candidates_evaluated", "—"))

                    # Gauge
                    st.plotly_chart(
                        qoe_gauge(float(best.get("predicted_qoe", 0)), "Best Config — Predicted QoE"),
                        use_container_width=True,
                    )

                # Baseline comparison
                if base:
                    with st.expander("📊 Baseline comparison"):
                        b_col1, b_col2 = st.columns(2)
                        b_col1.metric("Baseline Bandwidth", f"{result['context']['baseline_bandwidth_mhz']} MHz")
                        b_col1.metric("Baseline Predicted QoE", f"{base.get('predicted_qoe', 0):.4f}")
                        b_col2.metric("Optimised Bandwidth", f"{best.get('bandwidth_mhz', '—')} MHz")
                        b_col2.metric("Optimised Predicted QoE", f"{best.get('predicted_qoe', 0):.4f}")

                # Candidate table
                with st.expander("🗂️ All candidates evaluated"):
                    render_candidate_table(result.get("all_candidates", []))

                st.warning(
                    "📌 **Predicted improvement is pending ns-3 verification.** "
                    "Run a closed-loop verification on the **Closed-Loop Verification** page "
                    "to confirm with real simulation data."
                )


# ===========================================================================
# PAGE 5 — CLOSED-LOOP VERIFICATION
# ===========================================================================
elif page == PAGES[4]:
    st.title("🔁 Closed-Loop Verification")
    st.markdown(
        "Run the Phase 3 pipeline: Optimiser → ns-3 simulation → QoE comparison.  "
        "Simulations are launched **only on button click**, not automatically."
    )
    research_disclaimer()

    # --- Import simulation API ---
    try:
        from simulation.closed_loop import ClosedLoopVerifier
        from optimisation.search_space import SUPPORTED_SEARCH_SPACE
        _sim_ok = True
    except ImportError as e:
        st.error(f"Failed to import simulation modules: {e}")
        _sim_ok = False

    # --- Existing results table (always shown) ---
    st.subheader("Previous Verification Runs")
    phase3_df = load_phase3_results()
    render_phase3_table(phase3_df)

    # --- Per-run details ---
    if not phase3_df.empty and "run_id" in phase3_df.columns:
        selected_run = st.selectbox(
            "Expand a run for details",
            options=["— select —"] + phase3_df["run_id"].tolist(),
        )
        if selected_run != "— select —":
            row = phase3_df[phase3_df["run_id"] == selected_run].iloc[0]
            with st.expander(f"Details: {selected_run}", expanded=True):
                detail_cols = st.columns(3)
                detail_cols[0].metric("Verification Status", row.get("verification_status", "—"))
                detail_cols[1].metric("Predicted QoE",  f"{row.get('predicted_qoe', 0):.4f}"  if "predicted_qoe"  in row.index else "—")
                detail_cols[2].metric("Measured QoE",   f"{row.get('measured_qoe', 0):.4f}"   if "measured_qoe"   in row.index else "—")

                if "absolute_prediction_error" in row.index:
                    st.metric("Absolute Prediction Error", f"{row['absolute_prediction_error']:.4f}")

                artifact_paths = {k: row[k] for k in ["run_dir", "csv_path", "log_path"] if k in row.index and str(row[k]) != "nan"}
                if artifact_paths:
                    st.subheader("Artifact paths")
                    for k, v in artifact_paths.items():
                        st.code(str(v))

                if "error_message" in row.index and str(row["error_message"]) not in ("", "nan"):
                    st.error(f"Error: {row['error_message']}")

    st.divider()

    # --- New verification run ---
    if not _sim_ok:
        st.stop()

    st.subheader("▶️ Run New Verification")
    st.caption(
        "This launches a real ns-3 simulation (~5 s). "
        "Only click if you have the ns-3 binary built."
    )

    with st.form("new_verification_form"):
        col1, col2 = st.columns(2)

        ue_count = col1.selectbox(
            "UE Count",
            SUPPORTED_SEARCH_SPACE["ue_count"],
            index=0,
        )
        flow_type = col1.radio(
            "Flow Type",
            SUPPORTED_SEARCH_SPACE["flow_type"],
            index=1,
        )
        lambda_val = col1.selectbox(
            "Arrival Rate λ (pkt/s)",
            SUPPORTED_SEARCH_SPACE["flow_lambda"],
            index=0,
        )
        scenario_size_m = col2.selectbox(
            "Scenario Size (m)",
            SUPPORTED_SEARCH_SPACE["scenario_size_m"],
            index=0,
        )
        baseline_bw = col2.selectbox(
            "Baseline Bandwidth (MHz)",
            SUPPORTED_SEARCH_SPACE["bandwidth_mhz"],
            index=0,
        )
        run_baseline_sim = col2.checkbox(
            "Also run baseline simulation",
            value=False,
            help="Runs a second ns-3 simulation with the baseline bandwidth for direct comparison.",
        )

        sim_time = col2.selectbox(
            "Simulation time",
            ["1.5s", "3s", "5s"],
            index=0,
            help="Longer times give more stable averages but take more wall-clock time.",
        )

        st.markdown("##### 🔄 QoE Recovery Mode (Optional)")
        enable_recovery = st.checkbox(
            "Enable QoE Recovery Mode",
            value=False,
            help="If measured QoE is below target, performs at most one corrective ns-3 simulation with upgraded bandwidth.",
        )
        target_qoe = st.number_input(
            "Target QoE Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.80,
            step=0.05,
            disabled=not enable_recovery,
            help="Trigger corrective action if initial measured QoE is below this threshold (default: 0.80).",
        )

        confirmed = st.checkbox(
            "✅ I confirm: this will launch a real ns-3 simulation process (~5 s).",
            value=False,
        )
        run_btn = st.form_submit_button("🚀 Run Verification", type="primary")

    if run_btn:
        if not confirmed:
            st.session_state.pop("phase3_last_run", None)
            st.error("Please tick the confirmation checkbox before running.")
        else:
            MODEL_PATH  = str(_ROOT / "models" / "qoe_model.joblib")
            OUTPUT_DIR  = str(_ROOT / "results" / "phase3" / "runs")
            RESULTS_CSV = str(PHASE3_RESULTS)
            RECOVERY_CSV = str(_ROOT / "results" / "phase3" / "recovery_results.csv")

            with st.spinner(f"Running ns-3 simulation… (estimated {sim_time})"):
                try:
                    verifier = ClosedLoopVerifier(
                        model_path=MODEL_PATH,
                        output_base_dir=OUTPUT_DIR,
                        results_csv_path=RESULTS_CSV,
                    )
                    if enable_recovery:
                        recovery_record = verifier.run_verification_with_recovery(
                            ue_count=int(ue_count),
                            initial_bandwidth_mhz=float(baseline_bw),
                            total_tx_power_dbm=35.0,
                            scenario_size_m=float(scenario_size_m),
                            flow_type=str(flow_type),
                            lambda_val=float(lambda_val),
                            target_qoe=float(target_qoe),
                            sim_time_str=str(sim_time),
                            recovery_results_csv_path=RECOVERY_CSV,
                        )
                        record = recovery_record.corrective_record or recovery_record.initial_record
                    else:
                        recovery_record = None
                        record = verifier.run_closed_loop_verification(
                            ue_count=int(ue_count),
                            scenario_size_m=float(scenario_size_m),
                            flow_type=str(flow_type),
                            lambda_val=float(lambda_val),
                            baseline_bandwidth_mhz=float(baseline_bw),
                            baseline_power_dbm=35.0,
                            constraints=None,
                            run_baseline_sim=bool(run_baseline_sim),
                            sim_time_str=str(sim_time),
                        )
                    run_ok = True
                    exc_msg = None
                except Exception as exc:
                    record = None
                    recovery_record = None
                    run_ok = False
                    exc_msg = str(exc)

            # Persist outcome in session state so it survives st.rerun()
            st.session_state["phase3_last_run"] = {
                "run_ok": run_ok,
                "record": record,
                "recovery_record": recovery_record,
                "error_message": exc_msg,
            }

            # Clear cache and trigger rerun so the table and details reload immediately
            load_phase3_results.clear()
            if hasattr(st, "rerun"):
                st.rerun()

    # --- Display outcome of latest run (persists across st.rerun) ---
    if "phase3_last_run" in st.session_state:
        last_run = st.session_state["phase3_last_run"]
        run_ok = last_run.get("run_ok", False)
        record = last_run.get("record")
        recovery_record = last_run.get("recovery_record")
        exc_msg = last_run.get("error_message")

        if not run_ok or (record is None and recovery_record is None):
            st.error(f"Verification failed with exception: {exc_msg}")
        else:
            if recovery_record is not None:
                # Render QoE Recovery details
                st.markdown("#### 🔄 QoE Recovery Mode Report")
                rec_status = getattr(recovery_record, "status", "UNKNOWN")
                target_val = getattr(recovery_record, "target_qoe", 0.80)
                init_bw = getattr(recovery_record, "previous_bandwidth_mhz", 20.0)
                corr_bw = getattr(recovery_record, "corrected_bandwidth_mhz", None)
                init_metrics = getattr(recovery_record, "initial_metrics", {})
                corr_metrics = getattr(recovery_record, "corrected_metrics", None)
                rationale = getattr(recovery_record, "rationale", "")

                if rec_status == "SUCCESS":
                    st.success(f"✅ **RECOVERY SUCCESS**: Corrective simulation met target QoE ({target_val:.2f})!")
                elif rec_status == "NOT_NEEDED":
                    st.info(f"ℹ️ **RECOVERY NOT NEEDED**: Initial measured QoE ({init_metrics.get('measured_qoe', 0.0):.4f}) already meets target ({target_val:.2f}).")
                elif rec_status == "NO_ALTERNATIVE_CONFIGURATION":
                    st.warning(f"⚠️ **NO ALTERNATIVE CONFIGURATION**: Network is already at maximum supported bandwidth ({init_bw:.1f} MHz).")
                elif rec_status == "TARGET_NOT_MET":
                    st.warning(f"⚠️ **TARGET NOT MET**: Corrective run completed but measured QoE ({corr_metrics.get('measured_qoe', 0.0) if corr_metrics else 0.0:.4f}) is below target ({target_val:.2f}).")
                elif rec_status == "SIMULATION_FAILED":
                    st.error(f"❌ **SIMULATION FAILED**: {recovery_record.error_message or 'Simulation crashed during execution.'}")

                st.caption(f"**Rationale:** {rationale}")

                # Change log and parameters
                with st.expander("📋 Recovery Change Log & Parameters", expanded=True):
                    fixed_params = getattr(recovery_record, "fixed_parameters", {})
                    cl_col1, cl_col2 = st.columns(2)
                    with cl_col1:
                        st.write(f"**Previous Bandwidth:** {init_bw:.1f} MHz")
                        st.write(f"**Corrected Bandwidth:** {f'{corr_bw:.1f} MHz' if corr_bw is not None else 'None'}")
                        st.write(f"**Target QoE:** {target_val:.2f}")
                        st.write(f"**Target Achieved:** {'Yes' if getattr(recovery_record, 'target_achieved', False) else 'No'}")
                    with cl_col2:
                        st.write("**Fixed Parameters:**")
                        st.write(f"- Tx Power: {fixed_params.get('total_tx_power_dbm', 35.0)} dBm")
                        st.write(f"- UE Count: {fixed_params.get('ue_count', '—')}")
                        st.write(f"- Flow Type: {fixed_params.get('flow_type', '—')}")
                        st.write(f"- Arrival Rate: {fixed_params.get('lambda_val', '—')} pkt/s")
                        st.write(f"- Scenario Size: {fixed_params.get('scenario_size_m', '—')} m")

                # Metrics comparison table
                cmp_data = [
                    {
                        "KPI": "Measured QoE",
                        "Initial (Simulated)": f"{init_metrics.get('measured_qoe', 0.0):.4f}",
                        "Corrected (Simulated)": f"{corr_metrics.get('measured_qoe', 0.0):.4f}" if corr_metrics else "—",
                        "Target": f"{target_val:.4f}",
                    },
                    {
                        "KPI": "Throughput (Mbps)",
                        "Initial (Simulated)": f"{init_metrics.get('measured_throughput_mbps', 0.0):.2f}",
                        "Corrected (Simulated)": f"{corr_metrics.get('measured_throughput_mbps', 0.0):.2f}" if corr_metrics else "—",
                        "Target": "—",
                    },
                    {
                        "KPI": "Mean Delay (ms)",
                        "Initial (Simulated)": f"{init_metrics.get('measured_delay_ms', 0.0):.2f}",
                        "Corrected (Simulated)": f"{corr_metrics.get('measured_delay_ms', 0.0):.2f}" if corr_metrics else "—",
                        "Target": "—",
                    },
                    {
                        "KPI": "Mean Jitter (ms)",
                        "Initial (Simulated)": f"{init_metrics.get('measured_jitter_ms', 0.0):.2f}",
                        "Corrected (Simulated)": f"{corr_metrics.get('measured_jitter_ms', 0.0):.2f}" if corr_metrics else "—",
                        "Target": "—",
                    },
                    {
                        "KPI": "Packet Loss",
                        "Initial (Simulated)": f"{init_metrics.get('measured_packet_loss', 0.0):.4f}",
                        "Corrected (Simulated)": f"{corr_metrics.get('measured_packet_loss', 0.0):.4f}" if corr_metrics else "—",
                        "Target": "—",
                    },
                ]
                st.table(pd.DataFrame(cmp_data))

            else:
                vs = getattr(record, "verification_status", "UNKNOWN") if hasattr(record, "verification_status") else record.get("verification_status", "UNKNOWN")
                if vs == "VERIFIED_SUCCESS":
                    st.success(f"✅ **{vs}** — simulation completed and QoE measured.")
                elif vs == "SIMULATION_FAILED":
                    err_detail = getattr(record, "error_message", None) if hasattr(record, "error_message") else record.get("error_message")
                    if err_detail:
                        st.error(f"❌ **{vs}** — ns-3 simulation did not succeed: {err_detail}")
                    else:
                        st.error(f"❌ **{vs}** — ns-3 simulation did not succeed. Check the log file.")
                else:
                    st.warning(f"⚠️ {vs}")

                if vs != "SIMULATION_FAILED":
                    r_cols = st.columns(3)
                    pred_val = getattr(record, "predicted_qoe", 0) if hasattr(record, "predicted_qoe") else record.get("predicted_qoe", 0)
                    meas_val = getattr(record, "measured_qoe", 0) if hasattr(record, "measured_qoe") else record.get("measured_qoe", 0)
                    abs_val = getattr(record, "absolute_prediction_error", None) if hasattr(record, "absolute_prediction_error") else record.get("absolute_prediction_error")

                    r_cols[0].metric("Predicted QoE", f"{pred_val:.4f}")
                    r_cols[1].metric("Measured QoE", f"{meas_val:.4f}")
                    r_cols[2].metric("Abs Error", f"{abs_val:.4f}" if abs_val is not None else "—")


