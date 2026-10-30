"""Explainable one-cycle QoE trend forecast over validated ns-3 observations."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from simulation.result_parser import calculate_qoe_proxy


def telemetry_freshness(event: dict[str, Any] | None, *, max_age_seconds: float = 60.0,
                        now: datetime | None = None) -> dict[str, Any]:
    """Classify the latest measurement without treating missing timestamps as live."""
    if not event or event.get("run_status") != "COMPLETED" or not event.get("end_utc"):
        return {"status": "UNAVAILABLE", "age_seconds": None, "is_fresh": False}
    try:
        stamp = datetime.fromisoformat(str(event["end_utc"]).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        age = max(0.0, (current - stamp).total_seconds())
    except (TypeError, ValueError):
        return {"status": "INVALID_TIMESTAMP", "age_seconds": None, "is_fresh": False}
    fresh = age <= float(max_age_seconds)
    return {"status": "FRESH" if fresh else "STALE", "age_seconds": age,
            "is_fresh": fresh, "max_age_seconds": float(max_age_seconds)}


def evaluate_next_forecast(previous_event: dict[str, Any], current_event: dict[str, Any]) -> dict[str, Any] | None:
    """Compare a persisted one-epoch estimate with the next real simulator result."""
    forecast = previous_event.get("forecast_for_next")
    if not forecast or current_event.get("run_status") != "COMPLETED":
        return None
    expected_epoch = int(forecast.get("target_epoch", -1))
    if int(current_event.get("epoch", -2)) != expected_epoch:
        return None
    if previous_event.get("configuration_fingerprint") != current_event.get("configuration_fingerprint"):
        return {"target_epoch": expected_epoch, "status": "INVALIDATED_CONFIGURATION_CHANGE",
                "reason": "Configuration changed before the forecast horizon; comparison would be confounded."}
    actual = current_event.get("measured_qoe_proxy")
    predicted = forecast.get("predicted_qoe")
    if actual is None or predicted is None:
        return None
    return {"target_epoch": expected_epoch, "status": "EVALUATED",
            "predicted_qoe": float(predicted),
            "observed_qoe": float(actual), "absolute_error": abs(float(actual) - float(predicted)),
            "source": "subsequent validated ns-3 measurement"}


def _linear_forecast(values: list[float], horizon: int = 1) -> dict[str, float] | None:
    if len(values) < 3 or not all(math.isfinite(float(v)) for v in values):
        return None
    n = len(values)
    xs = list(range(n))
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom
    intercept = y_mean - slope * x_mean
    fitted = [intercept + slope * x for x in xs]
    residual = [y - y_hat for y, y_hat in zip(values, fitted)]
    sigma = math.sqrt(sum(r * r for r in residual) / max(1, n - 2))
    point = intercept + slope * (n - 1 + int(horizon))
    return {"value": point, "lower": point - 1.96 * sigma, "upper": point + 1.96 * sigma,
            "slope_per_cycle": slope, "residual_sigma": sigma}


def forecast_qoe_degradation(
    events: list[dict[str, Any]], *, target_qoe: float, horizon_epochs: int = 1,
    min_history: int = 3, max_history: int = 5,
) -> dict[str, Any]:
    """Forecast per-UE and aggregate QoE from same-configuration completed runs.

    The residual band is a simple descriptive trend-error band, not a calibrated
    confidence interval. Configuration changes reset the usable history.
    """
    now = datetime.now(timezone.utc).isoformat()
    valid = [e for e in events if e.get("run_status") == "COMPLETED"
             and e.get("measured_qoe_proxy") is not None and e.get("config")]
    if not valid:
        return {"status": "INSUFFICIENT_HISTORY", "reason": "No completed measured epochs.",
                "horizon_epochs": horizon_epochs, "predicted_at_utc": now,
                "data_through_utc": None, "users": [], "predicted_qoe": None,
                "lower": None, "upper": None}
    latest_fingerprint = valid[-1].get("configuration_fingerprint")
    same_config = [e for e in valid if e.get("configuration_fingerprint") == latest_fingerprint]
    same_config = same_config[-max_history:]
    data_through = same_config[-1].get("end_utc")
    if len(same_config) < min_history:
        return {"status": "INSUFFICIENT_HISTORY",
                "reason": f"Need {min_history} same-configuration completed epochs; found {len(same_config)}.",
                "horizon_epochs": horizon_epochs, "predicted_at_utc": now,
                "data_through_utc": data_through, "users": [], "predicted_qoe": None,
                "lower": None, "upper": None}
    user_ids = set(int(row["ue_id"]) for row in (same_config[-1].get("ue_measurements") or []))
    forecasts = []
    for ue_id in sorted(user_ids):
        rows = []
        for event in same_config:
            match = next((row for row in (event.get("ue_measurements") or [])
                          if int(row.get("ue_id", -1)) == ue_id), None)
            if match is None:
                rows = []
                break
            rows.append(match)
        if len(rows) != len(same_config):
            continue
        metric_series = {
            "throughput_mbps": [float(r["throughput_mbps"]) for r in rows],
            "delay_ms": [float(r["delay_ms"]) for r in rows],
            "jitter_ms": [float(r["jitter_ms"]) for r in rows],
            "packet_loss_fraction": [float(r["packet_loss_fraction"]) for r in rows],
        }
        metric_forecasts = {name: _linear_forecast(values, horizon_epochs)
                            for name, values in metric_series.items()}
        if any(value is None for value in metric_forecasts.values()):
            continue
        point = {name: max(0.0, value["value"]) for name, value in metric_forecasts.items()}
        point["packet_loss_fraction"] = min(1.0, point["packet_loss_fraction"])
        predicted_qoe = calculate_qoe_proxy(
            point["throughput_mbps"], point["delay_ms"], point["jitter_ms"],
            point["packet_loss_fraction"], rx_packets=1,
            traffic_class=rows[-1].get("traffic_class"),
            formula_version=rows[-1].get("qoe_formula_version", "legacy_v1"),
        )
        qoe_series = [float(r["qoe_proxy"]) for r in rows]
        qoe_prediction = _linear_forecast(qoe_series, horizon_epochs)
        if qoe_prediction is None:
            continue
        lower = max(0.0, min(1.0, qoe_prediction["lower"]))
        upper = max(0.0, min(1.0, qoe_prediction["upper"]))
        falling = []
        if metric_forecasts["throughput_mbps"]["slope_per_cycle"] < 0:
            falling.append("throughput trending down")
        if metric_forecasts["delay_ms"]["slope_per_cycle"] > 0:
            falling.append("delay trending up")
        if metric_forecasts["jitter_ms"]["slope_per_cycle"] > 0:
            falling.append("jitter trending up")
        if metric_forecasts["packet_loss_fraction"]["slope_per_cycle"] > 0:
            falling.append("packet loss trending up")
        forecasts.append({
            "ue_id": ue_id, "current_qoe": float(rows[-1]["qoe_proxy"]),
            "predicted_qoe": predicted_qoe, "trend_qoe": max(0.0, min(1.0, qoe_prediction["value"])),
            "lower": lower, "upper": upper,
            "at_risk": predicted_qoe < float(target_qoe) and float(rows[-1]["qoe_proxy"]) >= float(target_qoe),
            "already_below_target": float(rows[-1]["qoe_proxy"]) < float(target_qoe),
            "reason": ", ".join(falling) if falling else "no adverse KPI trend detected",
            "predicted_metrics": point,
        })
    if not forecasts:
        return {"status": "INSUFFICIENT_HISTORY", "reason": "Per-UE measurement rows are incomplete or incompatible.",
                "horizon_epochs": horizon_epochs, "predicted_at_utc": now,
                "data_through_utc": data_through, "users": [], "predicted_qoe": None,
                "lower": None, "upper": None}
    overall = sum(row["predicted_qoe"] for row in forecasts) / len(forecasts)
    lower = sum(row["lower"] for row in forecasts) / len(forecasts)
    upper = sum(row["upper"] for row in forecasts) / len(forecasts)
    at_risk = [row for row in forecasts if row["at_risk"]]
    return {
        "status": "RISK_PREDICTED" if at_risk else "NO_RISK_PREDICTED",
        "method": "per-UE linear trend on measured throughput/delay/jitter/loss; QoE recomputed with ns-3 formula",
        "horizon_epochs": int(horizon_epochs), "predicted_at_utc": now,
        "data_through_utc": data_through, "history_epochs": len(same_config),
        "predicted_qoe": overall, "lower": lower, "upper": upper,
        "uncertainty": "descriptive 1.96 × regression residual sigma; not calibrated",
        "at_risk_ue_ids": [row["ue_id"] for row in at_risk],
        "currently_below_target_ue_ids": [row["ue_id"] for row in forecasts if row["already_below_target"]],
        "users": forecasts,
        "reason": (f"{len(at_risk)} UE(s) forecast below QoE target {target_qoe:.3f} next epoch."
                   if at_risk else "No UE is forecast to cross the target next epoch."),
    }
