"""Offline view of the saved TU Wien KPI comparison experiment.

This module deliberately exposes no arbitrary-input prediction API. It reads
saved out-of-fold results only and does not import production QoE or simulation
code.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results/experiments/tuwien_kpi_comparison_dashboard_20261026"
METRICS_PATH = RESULTS_DIR / "comparison_metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "comparison_predictions.csv"

TARGETS = {
    "throughput": {"label": "Throughput", "unit": "Mbps"},
    "latency": {"label": "One-way latency", "unit": "ms"},
    "jitter": {"label": "Jitter", "unit": "ms"},
    "packet_loss": {"label": "Packet loss", "unit": "% (percentage points)"},
}
REQUIRED_PREDICTION_COLUMNS = {
    "target", "unit", "qIdx", "configured_rate_Mbps", "trial", "group_id",
    "fold", "observed", "rate_only_prediction", "ridge_prediction_raw",
    "ridge_physical_status", "global_mean_prediction",
}
# These are selectors over completed experimental results, not model inputs.
OFFLINE_FILTER_FIELDS = ("target", "configured_rate_Mbps", "qIdx")


def load_saved_results(
    metrics_path: Path = METRICS_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Load and validate saved comparison artifacts without fitting a model."""
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Experimental KPI metrics artifact not found: {metrics_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Experimental KPI predictions artifact not found: {predictions_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    targets = metrics.get("targets")
    if not isinstance(targets, dict) or set(targets) != set(TARGETS):
        raise ValueError("Metrics artifact must contain exactly the four supported KPI targets")
    for name, spec in TARGETS.items():
        if targets[name].get("unit") != ("percentage points" if name == "packet_loss" else spec["unit"]):
            raise ValueError(f"Unexpected unit for {name}: {targets[name].get('unit')!r}")
        if targets[name].get("primary_benchmark") != "rate_only":
            raise ValueError(f"Rate-only benchmark is not marked primary for {name}")
    validation = metrics.get("validation", {})
    if validation.get("no_condition_overlap") is not True:
        raise ValueError("Saved validation does not certify condition-group separation")

    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_PREDICTION_COLUMNS.issubset(set(reader.fieldnames or ())):
            raise ValueError("Predictions artifact is missing required offline result columns")
        rows = list(reader)
    for row in rows:
        target = row["target"]
        if target not in TARGETS:
            raise ValueError(f"Unexpected target in predictions: {target!r}")
        expected_unit = "percentage points" if target == "packet_loss" else TARGETS[target]["unit"]
        if row["unit"] != expected_unit:
            raise ValueError(f"Unexpected prediction unit for {target}: {row['unit']!r}")
        if row["ridge_physical_status"] == "INVALID_NEGATIVE" and float(row["ridge_prediction_raw"]) >= 0:
            raise ValueError("Negative ridge status conflicts with its raw value")
    return metrics, rows


def render_experimental_kpi_results(st: Any) -> None:
    """Render a compact, offline-only dashboard page using a Streamlit-like API."""
    st.title("🧪 Experimental KPI results (offline)")
    st.caption(
        "Single TU Wien testbed study. These results are experimental and are not "
        "validated predictions for arbitrary ns-3 scenarios or live networks."
    )
    st.info(
        "This page shows saved grouped out-of-fold comparisons; it does not make "
        "interactive predictions. The experiment used configured rate and a fixed-site "
        "qIdx label, and no validated mapping exists from the app's ns-3 controls to "
        "those testbed features. The existing QoE model predicts QoE only."
    )
    try:
        metrics, rows = load_saved_results()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        st.warning(f"Offline KPI results are unavailable: {exc}")
        return

    st.subheader("Saved validation comparison")
    st.caption(
        f"{metrics['validation']['name']} · seed {metrics['validation']['seed']} · "
        f"{metrics['validation']['condition_groups']} condition groups; all repetitions "
        "of qIdx × configured-rate conditions stayed together. Rate-only is the primary benchmark."
    )
    summary = []
    for target, spec in TARGETS.items():
        item = metrics["targets"][target]
        summary.append({
            "KPI": f"{spec['label']} ({spec['unit']})",
            "N": item["n_rows"],
            "Rate-only MAE": round(item["rate_only"]["mae"], 4),
            "Rate-only RMSE": round(item["rate_only"]["rmse"], 4),
            "Ridge MAE": round(item["ridge"]["mae"], 4),
            "Ridge RMSE": round(item["ridge"]["rmse"], 4),
        })
    st.table(summary)
    st.caption("MAE and RMSE use each target's displayed unit. Global-mean results are context only.")

    st.warning(
        "Ridge produced 35 negative jitter predictions and 10 negative packet-loss "
        "predictions. Raw values are retained and marked invalid; they are not clipped "
        "or presented as physically valid."
    )

    target = st.selectbox("Offline target", list(TARGETS), format_func=lambda key: TARGETS[key]["label"])
    target_rows = [row for row in rows if row["target"] == target]
    st.caption(f"Showing saved observations and out-of-fold predictions for {TARGETS[target]['label']}.")
    st.dataframe(target_rows, use_container_width=True, hide_index=True)
