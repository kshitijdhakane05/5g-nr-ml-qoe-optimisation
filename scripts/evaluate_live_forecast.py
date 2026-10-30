#!/usr/bin/env python3
"""Session-grouped, no-action evaluation of the live QoE trend forecast.

Uses only Monitor sessions and censors forecast windows crossed by an action,
restart, impairment, or segment boundary. Writes a report; never changes the
live controller or its configured thresholds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results/live_sessions"
DEFAULT_OUT = ROOT / "results/live_forecast_calibration"
INTERVENTIONS = {"action_applied", "structural_action_applied", "structural_restart_started",
                 "test_impairment_applied", "demand_changed", "configuration_changed"}


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _events(folder: Path) -> list[dict[str, Any]]:
    result = []
    try:
        for line in (folder / "events.jsonl").read_text().splitlines():
            try:
                result.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return result


def _telemetry(folder: Path) -> list[dict[str, Any]]:
    result = []
    paths = sorted((folder / "segments").glob("segment_*/telemetry.jsonl"))
    if not paths and (folder / "telemetry.jsonl").exists():
        paths = [folder / "telemetry.jsonl"]
    for path in paths:
        segment_id = path.parent.name if path.parent.name.startswith("segment_") else "segment_001"
        try:
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                row["_segment_id"] = segment_id
                result.append(row)
        except OSError:
            continue
    return sorted(result, key=lambda r: (r["_segment_id"], float(r.get("sim_time_s", 0))))


def _trend(history: list[dict[str, Any]], horizon: float, lookback: int) -> float | None:
    if len(history) < 4:
        return None
    rows = history[-lookback:]
    xs = [float(r["sim_time_s"]) for r in rows]
    ys = [float(r["qoe"]) for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return max(0.0, min(1.0, ys[-1] + slope * horizon))


def _score_sessions(root: Path, horizon: int, lookback: int) -> tuple[list[dict[str, Any]], list[str]]:
    scored: list[dict[str, Any]] = []
    skipped: list[str] = []
    for folder in sorted(root.iterdir() if root.exists() else []):
        meta = _load(folder / "session.json")
        if not meta or meta.get("mode") != "Monitor":
            continue
        points = _telemetry(folder)
        if not points:
            continue
        events = _events(folder)
        interventions = [e for e in events if e.get("type") in INTERVENTIONS]
        # A session-level split prevents near-identical windows from leaking across sets.
        for i, point in enumerate(points):
            if len(point.get("ues", [])) == 0:
                continue
            seg = point["_segment_id"]
            t = float(point["sim_time_s"])
            future = next((p for p in points[i + 1:] if p["_segment_id"] == seg
                           and float(p["sim_time_s"]) >= t + horizon), None)
            if future is None or float(future["sim_time_s"]) - (t + horizon) > 1.5:
                continue
            if any(e.get("sim_time_s") is not None and t < float(e["sim_time_s"]) <= float(future["sim_time_s"])
                   for e in interventions):
                continue
            # Keep only UEs present at both endpoints; service class and ID must match.
            future_ues = {int(u["ue_id"]): u for u in future.get("ues", []) if "ue_id" in u}
            histories: dict[tuple[int, str], list[dict[str, Any]]] = {}
            for old in points[:i + 1]:
                if old["_segment_id"] != seg:
                    continue
                for u in old.get("ues", []):
                    if "ue_id" in u and "qoe" in u:
                        key = (int(u["ue_id"]), str(u.get("traffic_class", "unknown")))
                        histories.setdefault(key, []).append({"sim_time_s": old["sim_time_s"], "qoe": u["qoe"]})
            for u in point.get("ues", []):
                if "ue_id" not in u or "qoe" not in u:
                    continue
                uid = int(u["ue_id"])
                actual = future_ues.get(uid)
                if actual is None or "qoe" not in actual:
                    continue
                profile = str(u.get("traffic_class", "unknown"))
                history = histories.get((uid, profile), [])
                prediction = _trend(history, horizon, lookback)
                if prediction is None:
                    continue
                scored.append({"session_id": meta.get("session_id", folder.name),
                               "created_at": meta.get("created_at", ""), "segment_id": seg,
                               "mode": "Monitor", "ue_id": uid, "traffic_class": profile,
                               "sim_time_s": t, "horizon_s": horizon, "lookback_windows": lookback,
                               "predicted_qoe": prediction, "measured_qoe": float(actual["qoe"]),
                               "persistence_qoe": float(u["qoe"]),
                               "safety_floor": float(meta.get("safety_floors", {}).get(profile, 0.0)),
                               "absolute_error": abs(prediction - float(actual["qoe"]))})
    return scored, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=5)
    args = parser.parse_args()
    rows, _ = _score_sessions(args.sessions, args.horizon, args.lookback)
    sessions = sorted({r["session_id"] for r in rows},
                      key=lambda sid: min(r["created_at"] for r in rows if r["session_id"] == sid))
    # The most recent 25% of eligible whole sessions are held out chronologically.
    split_at = max(1, math.ceil(len(sessions) * .75)) if sessions else 0
    train_ids, test_ids = set(sessions[:split_at]), set(sessions[split_at:])
    train = [r for r in rows if r["session_id"] in train_ids]
    test = [r for r in rows if r["session_id"] in test_ids]

    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        errors = [r["predicted_qoe"] - r["measured_qoe"] for r in group]
        persistence_errors = [r["persistence_qoe"] - r["measured_qoe"] for r in group]
        predicted_risk = [r["predicted_qoe"] < r["safety_floor"] for r in group]
        actual_risk = [r["measured_qoe"] < r["safety_floor"] for r in group]
        tp = sum(p and a for p, a in zip(predicted_risk, actual_risk))
        fp = sum(p and not a for p, a in zip(predicted_risk, actual_risk))
        fn = sum(not p and a for p, a in zip(predicted_risk, actual_risk))
        mae = lambda values: sum(abs(e) for e in values) / len(values) if values else None
        return {"samples": len(group), "trend_mae": mae(errors),
                "trend_rmse": math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None,
                "persistence_mae": mae(persistence_errors),
                "trend_bias_prediction_minus_measurement": sum(errors) / len(errors) if errors else None,
                "safety_floor_crossing": {"positive_cases": sum(actual_risk), "true_positive": tp,
                    "false_positive": fp, "missed_crossing": fn,
                    "precision": tp / (tp + fp) if tp + fp else None,
                    "recall": tp / (tp + fn) if tp + fn else None}}

    status = "READY_FOR_REVIEW" if len(test_ids) >= 3 and len(test) >= 100 else "INSUFFICIENT_INDEPENDENT_VALIDATION_DATA"
    report = {"status": status, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol": "Monitor-only sessions; per-UE chronological forecasts; same segment; intervention-censored; session-level chronological 75/25 split.",
              "target": "measured per-UE service QoE (0-1)", "method": f"OLS linear trend, last {args.lookback} windows, {args.horizon}s horizon",
              "eligible_sessions": len(sessions), "train_sessions": len(train_ids), "held_out_sessions": len(test_ids),
              "train": metrics(train), "held_out": metrics(test),
              "limitations": ["Overlapping forecast windows are correlated; sample count is not independent user-run count.",
                              "Only Monitor data is included; intervention and impairment windows are censored.",
                              "Trend must outperform persistence on held-out sessions before it is presented as useful forecasting.",
                              "This report evaluates accuracy; it does not tune or silently change live thresholds.",
                              "Do not claim calibrated probabilities from this point forecast."]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "held_out_predictions.csv").open("w", newline="") as f:
        fields = list(test[0]) if test else ["session_id", "ue_id", "sim_time_s", "predicted_qoe", "measured_qoe"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(test)
    print(json.dumps(report, indent=2))
    return 0 if status == "READY_FOR_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
