"""Guarded KPI-model advice for compatible live ns-3 legacy-profile runs.

The model predicts measured KPIs from the project's generated ns-3 dataset.
It is never used for NGMN traffic or unsupported configuration contexts. The
live trend forecast remains independent and the model can only veto or explain
a heuristic candidate; simulator measurements remain the outcome authority.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from simulation.result_parser import calculate_qoe_proxy

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "models/live_hybrid_ns3_20261030"
MODEL_PATH = ARTIFACT_DIR / "kpi_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
FEATURES = ["ue_count", "bandwidth_mhz", "total_tx_power_dbm", "scenario_size_m",
            "flow_type_id", "flow_lambda", "packet_size_bytes", "offered_throughput_mbps"]
TARGETS = ["throughput_mbps", "delay_ms", "jitter_ms", "packet_loss"]
CONTEXT = ["ue_count", "bandwidth_mhz", "total_tx_power_dbm", "scenario_size_m",
           "flow_type_id", "flow_lambda"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bundle() -> dict[str, Any] | None:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        return None
    try:
        bundle = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if bundle.get("feature_columns") != FEATURES or bundle.get("target_columns") != TARGETS:
            return None
        if "receive_model" not in bundle or file_sha256(MODEL_PATH) != metadata.get("model_sha256"):
            return None
        if metadata.get("artifact_version") != "live-ns3-kpi-v1":
            return None
        return {**bundle, "metadata": metadata}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def predict_ue(bundle: dict[str, Any], *, config: dict[str, Any], traffic_class: str,
               lambda_be: int | None = None) -> dict[str, Any]:
    """Predict QoE and uncertainty for one UE, or return why it is unsupported."""
    if config.get("traffic_mix", "legacy") != "legacy":
        return {"supported": False, "reason": "NGMN gaming/video profiles are outside this dataset's domain."}
    if int(config.get("gnb_count", 1)) != 1:
        return {"supported": False, "reason": "Training scenarios use one gNB; multi-cell predictions are unvalidated."}
    if abs(float(config.get("total_tx_power_dbm", 35)) - 35.0) > 1e-9:
        return {"supported": False, "reason": "Training scenarios use 35 dBm; other transmit powers are unsupported."}
    if traffic_class not in {"low_latency", "voice_qos"}:
        return {"supported": False, "reason": f"Traffic class {traffic_class!r} is not represented in training."}
    flow_id = 0 if traffic_class == "low_latency" else 1
    flow_lambda = int(config.get("lambda_ull", 2500)) if flow_id == 0 else int(
        lambda_be if lambda_be is not None else config.get("lambda_be", 10000))
    packet_size = 100 if flow_id == 0 else 1252
    offered = flow_lambda * packet_size * 8.0 / 1_000_000.0
    values = {
        "ue_count": int(config.get("ue_count", 2)),
        "bandwidth_mhz": float(config.get("bandwidth_mhz", 50)),
        "total_tx_power_dbm": float(config.get("total_tx_power_dbm", 35)),
        "scenario_size_m": float(config.get("scenario_size_m", 30)),
        "flow_type_id": flow_id, "flow_lambda": flow_lambda,
        "packet_size_bytes": packet_size, "offered_throughput_mbps": offered,
    }
    context = tuple(float(values[name]) for name in CONTEXT)
    supported = {tuple(float(v) for v in row) for row in bundle["supported_contexts"]}
    if context not in supported:
        return {"supported": False,
                "reason": "This exact UE-count/radio/workload context is absent from the training scenarios."}
    frame = pd.DataFrame([{name: values[name] for name in FEATURES}], columns=FEATURES)
    feature_array = frame.to_numpy(dtype=float)
    outputs = np.asarray([tree.predict(feature_array)[0] for tree in bundle["model"].estimators_], dtype=float)
    receive_probability = float(bundle["receive_model"].predict_proba(frame)[0, list(bundle["receive_model"].classes_).index(1)])
    receive_samples = np.asarray([tree.predict_proba(feature_array)[0, list(tree.classes_).index(1)]
                                  for tree in bundle["receive_model"].estimators_], dtype=float)
    means = outputs.mean(axis=0)
    stds = outputs.std(axis=0)
    means[0] = max(0.0, float(means[0]))
    means[1:3] = np.maximum(0.0, means[1:3])
    means[3] = min(1.0, max(0.0, float(means[3])))
    # QoE is computed from predicted measured KPIs using the current live formula.
    # Since packet reception is not a separately predicted class, a predicted
    # complete-loss outcome is conservatively treated as zero reception.
    rx = 0 if means[3] >= 0.999999 else 1
    qoe_samples = []
    for idx, row in enumerate(outputs):
        loss = min(1.0, max(0.0, float(row[3])))
        qoe_samples.append(calculate_qoe_proxy(max(0.0, float(row[0])), max(0.0, float(row[1])),
            max(0.0, float(row[2])), loss, rx_packets=0 if loss >= 0.999999 else 1,
            traffic_class=traffic_class, formula_version="service_v2") * receive_samples[idx])
    return {
        "supported": True,
        "predicted_qoe": float(np.mean(qoe_samples)),
        "uncertainty_qoe_std": float(np.std(qoe_samples)),
        "predicted_receive_probability": receive_probability,
        "_qoe_tree_predictions": list(map(float, qoe_samples)),
        "predicted_metrics": dict(zip(TARGETS, map(float, means))),
        "training_metric_tree_std": dict(zip(TARGETS, map(float, stds))),
        "model_version": bundle["metadata"]["artifact_version"],
        "training_source": bundle["metadata"]["dataset_path"],
    }


def compare_rate_candidate(meta: dict[str, Any], proposed_rate: int) -> dict[str, Any]:
    """Compare measured-state config vs a packet-rate candidate, if fully supported."""
    bundle = load_bundle()
    if bundle is None:
        return {"available": False, "reason": "The versioned live KPI model is unavailable."}
    config = dict(meta.get("config", {}))
    current_rate = int(meta.get("active_parameters", {}).get("lambda_be_pps",
                         config.get("lambda_be", 10000)))
    latest = meta.get("latest") or {}
    classes = [str(ue.get("traffic_class")) for ue in latest.get("ues", [])]
    if not classes:
        # The simulator assigns floor(N/2) low-latency UEs; the remainder are voice.
        ue_count = int(config.get("ue_count", 2))
        classes = (["low_latency"] * (ue_count // 2)
                   + ["voice_qos"] * (ue_count - ue_count // 2))
    baseline, candidate = [], []
    for traffic_class in classes:
        old = predict_ue(bundle, config=config, traffic_class=traffic_class, lambda_be=current_rate)
        new = predict_ue(bundle, config=config, traffic_class=traffic_class, lambda_be=int(proposed_rate))
        if not old.get("supported") or not new.get("supported"):
            return {"available": False,
                    "reason": old.get("reason") if not old.get("supported") else new.get("reason"),
                    "model_version": bundle["metadata"]["artifact_version"]}
        baseline.append(old)
        candidate.append(new)
    before = [item["predicted_qoe"] for item in baseline]
    after = [item["predicted_qoe"] for item in candidate]
    mean_delta = float(np.mean(after) - np.mean(before))
    worst_delta = float(min(after) - min(before))
    tree_deltas = np.mean(np.asarray([new["_qoe_tree_predictions"] for new in candidate]) -
                          np.asarray([old["_qoe_tree_predictions"] for old in baseline]), axis=0)
    uncertainty = float(np.std(tree_deltas))
    supported_benefit = mean_delta > max(0.005, uncertainty * 0.5) and worst_delta >= -0.01
    live_ues = latest.get("ues", [])
    measured_qoe = [float(ue["qoe"]) for ue in live_ues if ue.get("qoe") is not None]
    hybrid_after = ([max(0.0, min(1.0, observed + (new["predicted_qoe"] - old["predicted_qoe"])))
                    for observed, old, new in zip(measured_qoe, baseline, candidate)]
                   if len(measured_qoe) == len(classes) else [])
    return {
        "available": True, "model_version": bundle["metadata"]["artifact_version"],
        "candidate_rate_pps": int(proposed_rate), "baseline_rate_pps": current_rate,
        "predicted_mean_qoe_before": float(np.mean(before)),
        "predicted_mean_qoe_after": float(np.mean(after)), "predicted_mean_delta": mean_delta,
        "predicted_worst_qoe_before": min(before), "predicted_worst_qoe_after": min(after),
        "predicted_worst_delta": worst_delta, "uncertainty_qoe_delta_std": uncertainty,
        "measured_mean_qoe_before": float(np.mean(measured_qoe)) if measured_qoe else None,
        "hybrid_estimated_mean_qoe_after": float(np.mean(hybrid_after)) if hybrid_after else None,
        "hybrid_estimated_worst_qoe_after": min(hybrid_after) if hybrid_after else None,
        "candidate_supported": bool(supported_benefit),
        "reason": ("Predicted mean QoE improves beyond the model's uncertainty allowance and worst-user QoE is protected."
                   if supported_benefit else "Predicted gain is too small for the model uncertainty or risks the worst user."),
        "source": "model estimates from generated ns-3 KPI data; not a live measurement",
    }
