"""
dashboard/data_loader.py
========================
Read-only, cached data-loading helpers for the Phase 4 Streamlit dashboard.

All functions use @st.cache_data so that repeated page navigations do not
re-read files from disk.  The raw dataset and model artifacts are NEVER
written to by this module.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Project root resolution (works regardless of cwd)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW       = _PROJECT_ROOT / "data" / "raw"  / "qoe_dataset.csv"
DATA_PROCESSED = _PROJECT_ROOT / "data" / "processed" / "qoe_dataset.csv"
MODEL_RESULTS  = _PROJECT_ROOT / "results" / "ml" / "model_results.csv"
PREDICTIONS    = _PROJECT_ROOT / "results" / "ml" / "predictions.csv"
OPT_RESULTS    = _PROJECT_ROOT / "results" / "optimisation" / "optimisation_results.csv"
PHASE3_RESULTS = _PROJECT_ROOT / "results" / "phase3" / "verification_results.csv"


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=None)
def load_processed_dataset() -> pd.DataFrame:
    """Load the Phase-1 processed dataset (read-only)."""
    if not DATA_PROCESSED.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PROCESSED)
    return df


@st.cache_data(ttl=None)
def load_raw_dataset() -> pd.DataFrame:
    """Load the original raw dataset (read-only, for reference)."""
    if not DATA_RAW.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_RAW)
    return df


# ---------------------------------------------------------------------------
# ML results loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=None)
def load_model_results() -> pd.DataFrame:
    """Load benchmarked model comparison CSV."""
    if not MODEL_RESULTS.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_RESULTS)


@st.cache_data(ttl=None)
def load_predictions() -> pd.DataFrame:
    """Load hold-out set predictions CSV."""
    if not PREDICTIONS.exists():
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS)


# ---------------------------------------------------------------------------
# Optimisation results loader
# ---------------------------------------------------------------------------

@st.cache_data(ttl=None)
def load_optimisation_results() -> pd.DataFrame:
    """Load Phase-2 optimisation results CSV."""
    if not OPT_RESULTS.exists():
        return pd.DataFrame()
    return pd.read_csv(OPT_RESULTS)


# ---------------------------------------------------------------------------
# Phase 3 results loader  (short TTL so new runs appear quickly)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def load_phase3_results() -> pd.DataFrame:
    """Load Phase-3 closed-loop verification results CSV.

    TTL=5 s so results from a new simulation run appear promptly after
    the run button is pressed and the page rerenders.
    """
    if not PHASE3_RESULTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(PHASE3_RESULTS)
    # Parse timestamp to datetime for sorting / display
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a dict of high-level statistics for the processed dataset."""
    if df.empty:
        return {}
    summary: dict = {
        "rows": len(df),
        "unique_scenarios": int(df["scenario_id"].nunique()) if "scenario_id" in df.columns else 0,
        "unique_flows": int(df["flow_id"].nunique()) if "flow_id" in df.columns else 0,
        "ue_count_range": (int(df["ue_count"].min()), int(df["ue_count"].max())) if "ue_count" in df.columns else (0, 0),
        "bandwidth_values_mhz": sorted(df["bandwidth_mhz"].dropna().unique().tolist()) if "bandwidth_mhz" in df.columns else [],
        "power_values_dbm": sorted(df["total_tx_power_dbm"].dropna().unique().tolist()) if "total_tx_power_dbm" in df.columns else [],
        "qoe_min": float(df["qoe_proxy"].min()) if "qoe_proxy" in df.columns else 0.0,
        "qoe_max": float(df["qoe_proxy"].max()) if "qoe_proxy" in df.columns else 1.0,
        "qoe_mean": float(df["qoe_proxy"].mean()) if "qoe_proxy" in df.columns else 0.0,
        "outage_count": int((df["is_outage"] == True).sum()) if "is_outage" in df.columns else 0,  # noqa: E712
        "flow_types": sorted(df["flow_type"].dropna().unique().tolist()) if "flow_type" in df.columns else [],
    }
    return summary

