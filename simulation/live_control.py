"""Epoch-based, measurement-driven controller for the ns-3 simulation.

Each epoch is a fresh ns-3 process. The code does not claim in-process
reconfiguration or control of deployed 5G equipment.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
import sys
import time
import uuid
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.ns3_qoe_expansion_20261026_r1.preprocessing import derive_effective_band_powers_dbm
from simulation.ns3_runner import DEFAULT_NS3_BINARY, Ns3Runner
from simulation.result_parser import (
    parse_simulation_csv, parse_ue_measurements, validate_simulation_csv,
)
from simulation.qoe_forecast import evaluate_next_forecast, forecast_qoe_degradation

ROOT = Path(__file__).resolve().parents[1]
EXPANSION_RESULTS = ROOT / "results/experiments/ns3_qoe_expansion_20261026_r1_expansion"
LIVE_RESULTS = ROOT / "results/live_control"
RNG_SEED = 20261026
RUN_TIMEOUT_SECONDS = 120
MAX_RUN_TIMEOUT_SECONDS = 300
# One completed custom run on this machine: 6 UEs, 3 s, 234,000 configured
# application packets, 115.42 s wall time. Used only as a conservative UI
# recommendation, not as a runtime guarantee or a reason to alter the workload.
_TIMEOUT_REFERENCE_PACKETS = 234_000
_TIMEOUT_REFERENCE_SECONDS = 115.42
_TIMEOUT_RECOMMENDATION_MARGIN = 1.25
MAX_EPOCHS = 6
MAX_ACTIONS = 3
MAX_POST_EPOCH_EXPLORATORY_EPOCHS = 9
NO_IMPROVEMENT_TOLERANCE = 0.005
OPERATING_MODES = ("Monitor", "Recommend", "Automatic")
SUPPORTED_ACTIONS = {
    "bandwidth_band1_mhz": {
        "name": "Band 1 bandwidth",
        "allowed_values_mhz": [20.0, 50.0, 100.0],
        "application": "ns-3 process restart through --bandwidthBand1",
        "verification": "completed run_config.json fingerprint and validated qoe_dataset.csv",
        "cooldown_epochs": 1,
    },
}
PRESET_DEFINITIONS = {
    "Baseline": {"config_id": "ANCHOR-UE2", "purpose": "Verified 2-UE, 50/50 MHz, 35 dBm reference."},
    "Narrow bandwidth": {"config_id": "L9-01", "purpose": "20 MHz Band 1 with 2 UEs; Band 2 remains 50 MHz."},
    "Wide bandwidth": {"config_id": "L9-08", "purpose": "100 MHz Band 1 with 2 UEs; Band 2 remains 50 MHz."},
    "Low transmit power": {"config_id": "L9-04", "purpose": "30 dBm requested power; 6-UE tested condition."},
    "High transmit power": {"config_id": "L9-06", "purpose": "40 dBm requested power; 2-UE tested condition."},
    "Higher UE load": {"config_id": "ANCHOR-UE6", "purpose": "6-UE, 50/50 MHz, 35 dBm reference condition."},
    "Combined stress": {"config_id": "L9-07", "purpose": "10 UEs, 100 MHz Band 1 and 30 dBm; tested stress point."},
}
HISTORY_COLUMNS = [
    "session_id", "epoch", "run_id", "scenario_id", "config_id", "controller_mode",
    "ue_count", "gnb_count", "ue_association_policy", "bandwidth_band1_mhz", "bandwidth_band2_mhz", "total_tx_power_dbm",
    "effective_band1_tx_power_dbm", "effective_band2_tx_power_dbm", "lambda_be_pps",
    "lambda_ull_pps", "packet_size_be_bytes", "packet_size_ull_bytes", "scenario_size_m",
    "sim_time", "rng_seed", "rng_run", "command_json", "start_utc", "end_utc",
    "duration_seconds", "exit_code", "run_status", "raw_csv_path", "stdout_path",
    "stderr_path", "summary_path", "mean_throughput_mbps", "mean_delay_ms",
    "mean_jitter_ms", "mean_packet_loss_fraction", "mean_packet_loss_percent",
    "measured_qoe_proxy", "target_qoe", "predicted_qoe", "previous_valid_qoe_delta",
    "action_outcome_from_previous", "controller_action", "action_rationale",
    "prior_qoe_proxy", "current_qoe_proxy", "is_outage", "next_decision",
    "next_configuration_json", "candidate_scope", "is_exploratory", "candidate_selection_rationale",
    "candidate_evidence", "candidate_validation", "adaptive_action_count",
    "candidate_risk_level", "candidate_risk_reason", "high_risk_override",
    "selected_candidate_risk_level", "selected_candidate_risk_reason",
    "next_candidate_risk_level", "next_candidate_risk_reason",
    "repeat_trials_enabled", "risk_excluded_count", "duplicate_excluded_count",
    "controller_phase", "predicted_qoe_lower", "predicted_qoe_upper",
    "prediction_method", "prediction_domain", "prediction_run_count",
    "prediction_confidence", "predicted_target_fraction", "predicted_outage_fraction", "prediction_error",
    "configuration_fingerprint", "recovery_target_key", "is_recovery_trial",
    "next_predicted_qoe", "next_predicted_qoe_lower", "next_predicted_qoe_upper",
    "next_prediction_domain", "next_prediction_run_count",
    "topology_candidate_id", "topology_version", "topology_change", "topology_change_reason",
    "exploratory_trial_count", "is_validation_epoch", "validation_sequence", "validation_pass",
    "validation_progress", "target_range_pass", "qos_constraint_violations", "decision_record",
    "adaptive_action_budget", "action_budget_exhausted", "post_epoch_exploration",
    "post_epoch_exploration_count", "post_epoch_exploration_limit", "continuation_reason",
    "next_continuation_reason",
    "operating_mode", "forecast_status", "forecast_horizon_epochs", "forecast_predicted_qoe",
    "forecast_lower", "forecast_upper", "forecast_risk_ue_ids", "forecast_reason",
    "forecast_evaluation_json", "action_application_status", "action_fingerprint",
    "application_error", "verification_epoch", "predicted_qoe_before", "measured_qoe_before",
    "observed_qoe_after",
    "error_message",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_key(config: dict[str, Any]) -> tuple[int, float, float]:
    return int(config["ue_count"]), float(config["bandwidth_band1_mhz"]), float(config["total_tx_power_dbm"])


def run_slot_category(
    epoch: int, *, normal_limit: int, extra_enabled: bool, extra_limit: int,
) -> str | None:
    """Return the permitted slot type for a one-based run index, if any."""
    if int(epoch) < 1:
        return None
    if int(epoch) <= int(normal_limit):
        return "normal"
    if extra_enabled and int(epoch) <= int(normal_limit) + int(extra_limit):
        return "experimental"
    return None


def configuration_fingerprint(config: dict[str, Any]) -> str:
    """Stable identity over all pre-run candidate feature dimensions."""
    fields = (
        "ue_count", "gnb_count", "ue_association_policy", "bandwidth_band1_mhz", "bandwidth_band2_mhz", "total_tx_power_dbm",
        "lambda_be_pps", "lambda_ull_pps", "packet_size_be_bytes", "packet_size_ull_bytes",
        "scenario_size_m", "sim_time",
    )
    return json.dumps({field: config.get(field) for field in fields}, sort_keys=True, separators=(",", ":"))


def known_all_outage_evidence(
    key: tuple[int, float, float],
    campaign: dict[tuple[int, float, float], dict[str, Any]] | None = None,
    pilot: dict[tuple[int, float, float], dict[str, Any]] | None = None,
    live: dict[tuple[int, float, float], dict[str, Any]] | None = None,
) -> str | None:
    campaign_record = (campaign or {}).get(key, {})
    if campaign_record.get("all_rows_outage"):
        return f"campaign all-outage evidence across {campaign_record.get('repetition_count', '?')} repetitions"
    live_record = (live or {}).get(key, {})
    if live_record.get("all_runs_outage"):
        return f"all {live_record.get('run_count', '?')} prior matching-semantics live run(s) were outages"
    pilot_record = (pilot or {}).get(key, {})
    if pilot_record.get("all_runs_outage"):
        return (f"all {pilot_record.get('run_count', '?')} lower-load pilot run(s) were outages "
                f"at {pilot_record.get('lambda_be_pps', '?')} BE pps")
    return None


def assess_candidate_risk(
    key: tuple[int, float, float], *,
    campaign: dict[tuple[int, float, float], dict[str, Any]] | None = None,
    pilot: dict[tuple[int, float, float], dict[str, Any]] | None = None,
    live: dict[tuple[int, float, float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify exact and repeated same-band outage evidence without generalizing from one neighbor."""
    campaign, pilot, live = campaign or {}, pilot or {}, live or {}
    campaign_exact = campaign.get(key, {})
    live_exact = live.get(key, {})
    exact_campaign_outages = int(campaign_exact.get(
        "outage_repetition_count",
        campaign_exact.get("repetition_count", 0) if campaign_exact.get("all_rows_outage") else 0,
    ))
    exact_live_outages = int(live_exact.get(
        "outage_run_count", live_exact.get("run_count", 0) if live_exact.get("all_runs_outage") else 0,
    ))
    exact_outage_count = exact_campaign_outages + exact_live_outages
    pilot_exact = pilot.get(key, {})

    # A broader Band 1 risk signal requires at least two distinct other radio
    # tuples with complete outages under the matching campaign/live workload,
    # and at least three such repetitions in total. A lone failed neighbor is
    # disclosed but never promoted to a family exclusion.
    related_outage_configs: dict[tuple[int, float, float], int] = {}
    for other_key, record in campaign.items():
        if other_key[0] == key[0] and other_key[1] == key[1] and other_key != key:
            count = int(record.get("outage_repetition_count", 0))
            if count:
                related_outage_configs[other_key] = related_outage_configs.get(other_key, 0) + count
    for other_key, record in live.items():
        if other_key[0] == key[0] and other_key[1] == key[1] and other_key != key:
            count = int(record.get("outage_run_count", 0))
            if count:
                related_outage_configs[other_key] = related_outage_configs.get(other_key, 0) + count
    related_run_count = sum(related_outage_configs.values())
    repeated_band_pattern = len(related_outage_configs) >= 2 and related_run_count >= 3

    reasons: list[str] = []
    if exact_outage_count:
        reasons.append(f"exact configuration had {exact_outage_count} complete outage run(s) "
                       f"({exact_campaign_outages} campaign repetition(s), {exact_live_outages} matching-semantics live run(s))")
    if pilot_exact.get("all_runs_outage"):
        reasons.append(f"same radio tuple had {pilot_exact.get('run_count', 0)} lower-load pilot outage run(s) "
                       f"at {pilot_exact.get('lambda_be_pps', '?')} BE pps; workload differs")
    if related_outage_configs:
        related_summary = ", ".join(
            f"UE{other_key[0]}/B1{other_key[1]:g}/P{other_key[2]:g}dBm: {count} outage run(s)"
            for other_key, count in sorted(related_outage_configs.items())
        )
        reasons.append(f"same UE/Band 1 related evidence: {related_summary}")

    if exact_outage_count >= 2:
        level = "HIGH_RISK_EXACT_REPEATED"
    elif exact_outage_count == 1:
        level = "PREVIOUSLY_FAILED_EXACT"
    elif repeated_band_pattern:
        level = "HIGH_RISK_REPEATED_BAND_PATTERN"
    elif pilot_exact.get("all_runs_outage"):
        level = "PRIOR_PILOT_OUTAGE_DIFFERENT_LOAD"
    elif related_outage_configs:
        level = "RELATED_OUTAGE_SIGNAL"
    elif campaign_exact or live_exact:
        level = "OBSERVED_NON_OUTAGE"
    else:
        level = "NO_EXACT_OUTCOME"

    # Conservative mode excludes a previously failed exact setting and a
    # repeated, multi-configuration outage pattern. A single related failure
    # only changes ranking and is visible in the rationale.
    excluded_by_default = bool(exact_outage_count or repeated_band_pattern or pilot_exact.get("all_runs_outage"))
    return {
        "level": level,
        "excluded_by_default": excluded_by_default,
        "exact_outage_run_count": exact_outage_count,
        "related_outage_config_count": len(related_outage_configs),
        "related_outage_run_count": related_run_count,
        "repeated_band_pattern": repeated_band_pattern,
        "reason": "; ".join(reasons) if reasons else "No prior exact or related complete-outage evidence.",
    }


def classify_action_outcome(delta: float, tolerance: float = NO_IMPROVEMENT_TOLERANCE) -> str:
    """Classify measured QoE change without conflating it with session stop reason."""
    if not math.isfinite(float(delta)):
        return "INVALID_COMPARISON"
    if float(delta) > float(tolerance):
        return "IMPROVED"
    if float(delta) < -float(tolerance):
        return "WORSENED"
    return "NO_MEANINGFUL_CHANGE"


def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_id": str(raw["config_id"]), "ue_count": int(raw["ue_count"]),
        "gnb_count": int(raw.get("gnb_count", 1)),
        "ue_association_policy": str(raw.get("ue_association_policy", "closest")),
        "bandwidth_band1_mhz": float(raw["bandwidth_band1_mhz"]),
        "bandwidth_band2_mhz": float(raw["bandwidth_band2_mhz"]),
        "total_tx_power_dbm": float(raw["total_tx_power_dbm"]),
        "lambda_be_pps": int(raw["lambda_be_pps"]), "lambda_ull_pps": int(raw["lambda_ull_pps"]),
        "packet_size_be_bytes": int(raw["packet_size_be_bytes"]),
        "packet_size_ull_bytes": int(raw["packet_size_ull_bytes"]),
        "scenario_size_m": float(raw["scenario_size_m"]), "sim_time": str(raw["sim_time"]),
        "app_start_seconds": float(raw["app_start_seconds"]),
    }


def load_verified_configurations(expansion_dir: str | Path = EXPANSION_RESULTS) -> list[dict[str, Any]]:
    """Load candidate actions only from the passing expanded-experiment metadata."""
    folder = Path(expansion_dir)
    validation_path, campaign_path = folder / "campaign_validation.json", folder / "campaign_config.json"
    if not validation_path.is_file() or not campaign_path.is_file():
        raise FileNotFoundError("Validated expanded experiment metadata is unavailable")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("success_count") != 22:
        raise ValueError("Expanded experiment is not fully validated; refusing candidate actions")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    configs: dict[tuple[int, float, float], dict[str, Any]] = {}
    for raw in campaign.get("configurations", []):
        config = _normalize_config(raw)
        configs.setdefault(config_key(config), config)
    if len(configs) < 3:
        raise ValueError("Expanded experiment contains too few unique configurations")
    for config in configs.values():
        if config["bandwidth_band2_mhz"] != 50.0:
            raise ValueError("Live demo only supports the verified Band 2=50 MHz setup")
        if config["lambda_be_pps"] != 20_000 or config["lambda_ull_pps"] != 10_000:
            raise ValueError("Traffic configuration differs from the validated expansion")
    return list(configs.values())


def load_candidate_performance(expansion_dir: str | Path = EXPANSION_RESULTS) -> dict[tuple[int, float, float], dict[str, Any]]:
    """Summarize observed campaign QoE/outages so known-outage candidates are not retried blindly."""
    path = Path(expansion_dir) / "expanded_flow_dataset_reconciled.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Validated per-flow campaign data is unavailable: {path}")
    grouped: dict[tuple[int, float, float], list[dict[str, float]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["ue_count"]), float(row["bandwidth_mhz"]), float(row["total_tx_power_dbm"]))
            grouped.setdefault(key, []).append({
                "qoe": float(row["qoe_proxy"]), "rx_packets": float(row["rx_packets"]),
                "repetition_id": float(row["repetition_id"]),
                "rng_seed": int(row.get("rng_seed", 0)), "rng_run": int(row.get("rng_run", 0)),
            })
    stats: dict[tuple[int, float, float], dict[str, Any]] = {}
    for key, rows in grouped.items():
        by_repetition: dict[float, list[dict[str, float]]] = {}
        for row in rows:
            by_repetition.setdefault(row["repetition_id"], []).append(row)
        outage_repetitions = sum(all(row["rx_packets"] == 0 for row in repetition_rows)
                                 for repetition_rows in by_repetition.values())
        stats[key] = {
            "mean_qoe_proxy": sum(r["qoe"] for r in rows) / len(rows),
            "row_count": len(rows),
            "repetition_count": len({r["repetition_id"] for r in rows}),
            "outage_row_count": sum(r["rx_packets"] == 0 for r in rows),
            "all_rows_outage": all(r["rx_packets"] == 0 for r in rows),
            "outage_repetition_count": int(outage_repetitions),
            "run_qoe_values": [sum(float(r["qoe"]) for r in values) / len(values)
                               for _, values in sorted(by_repetition.items())],
            "run_outage_values": [all(r["rx_packets"] == 0 for r in values)
                                  for _, values in sorted(by_repetition.items())],
            "run_keys": [(int(values[0]["rng_seed"]), int(values[0]["rng_run"]))
                         for _, values in sorted(by_repetition.items())],
        }
    return stats


def supported_radio_grid(template: dict[str, Any], validated_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the CLI-supported 3x3 Band-1/power grid for explicit opt-in exploration."""
    validated = {config_key(c): c for c in validated_configs
                 if int(c["ue_count"]) == int(template["ue_count"])}
    grid = []
    for bandwidth in (20.0, 50.0, 100.0):
        for power in (30.0, 35.0, 40.0):
            key = (int(template["ue_count"]), bandwidth, power)
            candidate = dict(validated.get(key, template))
            candidate.update({
                "ue_count": int(template["ue_count"]),
                "gnb_count": int(template.get("gnb_count", 1)),
                "bandwidth_band1_mhz": bandwidth,
                "bandwidth_band2_mhz": float(template.get("bandwidth_band2_mhz", 50.0)),
                "total_tx_power_dbm": power,
                "config_id": validated.get(key, {}).get(
                    "config_id", f"EXPLORATORY-B1{int(bandwidth)}-P{int(power)}-UE{int(template['ue_count'])}"
                ),
            })
            grid.append(candidate)
    return grid


def supported_topology_candidates(
    template: dict[str, Any], validated_configs: list[dict[str, Any]], *,
    candidate_budget: int = 1,
    candidate_performance: dict[tuple[int, float, float], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build bounded fixed-population alternatives using the scenario's multi-gNB API.

    The existing campaign contains only one-gNB results, so these candidates are
    deliberately marked unvalidated and receive no campaign-based prediction.
    """
    if not 1 <= int(candidate_budget) <= 3:
        raise ValueError("Topology candidate budget must be one, two or three.")
    ue_count = int(template["ue_count"])
    # Keep the user's selected radio/workload values fixed so this candidate
    # isolates topology and association changes. Campaign rows are single-gNB
    # evidence and cannot validate a multi-cell result.
    seed_config = dict(template)
    variants = [
        (2, "round_robin", "distribute the existing UE IDs round-robin across cells"),
        (2, "by_traffic_class", "group low-latency and voice-QoS UEs onto separate cells"),
        (3, "round_robin", "distribute the existing UE IDs round-robin across three cells"),
        (3, "by_traffic_class", "group low-latency and voice-QoS UEs onto separate cells"),
    ]
    start = max(1, int(template.get("gnb_count", 1)) + 1)
    variants = [variant for variant in variants if variant[0] >= start][:int(candidate_budget)]
    candidates = []
    for count, association, description in variants:
        candidate = dict(seed_config)
        candidate.update({
            "config_id": f"TOPO-GNB{count}-{association.upper()}-UE{ue_count}",
            "gnb_count": count,
            "ue_association_policy": association,
            "topology_candidate": True,
            "topology_change_reason": (
                f"Rebuild the same {ue_count}-UE workload with {count} gNBs and {description} "
                "using NrHelper::AttachToGnb. "
                "The campaign contains one-gNB measurements only, so outcome is unknown until simulated."
            ),
        })
        candidates.append(candidate)
    return candidates


def load_pilot_candidate_evidence(
    pilot_dir: str | Path = ROOT / "results/experiments/ns3_qoe_expansion_20261026_r1_run2",
) -> dict[tuple[int, float, float], dict[str, Any]]:
    """Read only passing pilot measurements; these used a lower BE offered load."""
    folder = Path(pilot_dir)
    validation_path, manifest_path = folder / "pilot_validation.json", folder / "run_manifest.csv"
    if not validation_path.is_file() or not manifest_path.is_file():
        return {}
    if json.loads(validation_path.read_text(encoding="utf-8")).get("status") != "PASS":
        return {}
    groups: dict[tuple[int, float, float], list[dict[str, Any]]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "PASS" or int(row.get("process_returncode", -1)) != 0:
                continue
            metrics = json.loads(row.get("metrics_json") or "{}")
            key = (int(row["ue_count"]), float(row["bandwidth_band1_mhz"]),
                   float(row["total_tx_power_dbm"]))
            groups.setdefault(key, []).append({
                "outage_rows": int(metrics.get("outage_rows", 0)),
                "row_count": int(metrics.get("row_count", 0)),
                "lambda_be_pps": int(row["lambda_be_pps"]),
            })
    return {key: {
        "run_count": len(rows), "all_runs_nonoutage": all(r["outage_rows"] == 0 for r in rows),
        "all_runs_outage": all(r["row_count"] > 0 and r["outage_rows"] == r["row_count"] for r in rows),
        "lambda_be_pps": rows[0]["lambda_be_pps"],
    } for key, rows in groups.items()}


def load_live_candidate_evidence(
    reference: dict[str, Any], live_results: str | Path = LIVE_RESULTS,
) -> dict[tuple[int, float, float], dict[str, Any]]:
    """Summarize prior exact-semantics live runs for exploration ordering and outage avoidance."""
    comparable_fields = (
        "ue_count", "gnb_count", "bandwidth_band2_mhz", "lambda_be_pps", "lambda_ull_pps",
        "packet_size_be_bytes", "packet_size_ull_bytes", "scenario_size_m", "sim_time",
    )
    grouped: dict[tuple[int, float, float], list[dict[str, Any]]] = {}
    root = Path(live_results)
    if not root.is_dir():
        return {}
    for folder in root.iterdir():
        state_path, events_path = folder / "session.json", folder / "events.jsonl"
        if not folder.is_dir() or not state_path.is_file() or not events_path.is_file():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            initial = state.get("initial_config", {})
            if any(str(initial.get(field, 1 if field == "gnb_count" else None))
                   != str(reference.get(field, 1 if field == "gnb_count" else None))
                   for field in comparable_fields):
                continue
            with events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    event = json.loads(line)
                    config, metrics = event.get("config", {}), event.get("metrics") or {}
                    if event.get("run_status") != "COMPLETED" or event.get("measured_qoe_proxy") is None:
                        continue
                    if any(str(config.get(field, 1 if field == "gnb_count" else None))
                           != str(reference.get(field, 1 if field == "gnb_count" else None))
                           for field in comparable_fields):
                        continue
                    key = config_key(config)
                    grouped.setdefault(key, []).append({
                        "qoe": float(event["measured_qoe_proxy"]),
                        "is_outage": bool(metrics.get("is_outage", event.get("is_outage", False))),
                        "rng_seed": event.get("rng_seed"), "rng_run": event.get("rng_run"),
                    })
        except (OSError, ValueError, TypeError, KeyError):
            # An incomplete historical session is not evidence for candidate ranking.
            continue
    return {key: {
        "run_count": len(rows), "mean_qoe_proxy": sum(r["qoe"] for r in rows) / len(rows),
        "all_runs_outage": all(r["is_outage"] for r in rows),
        "any_run_outage": any(r["is_outage"] for r in rows),
        "outage_run_count": sum(r["is_outage"] for r in rows),
        "run_qoe_values": [r["qoe"] for r in rows],
        "run_outage_values": [r["is_outage"] for r in rows],
        "run_keys": [(int(r["rng_seed"]), int(r["rng_run"]))
                     for r in rows if r.get("rng_seed") is not None and r.get("rng_run") is not None],
    } for key, rows in grouped.items()}


PREDICTION_FEATURES = (
    "ue_count", "bandwidth_band1_mhz", "bandwidth_band2_mhz", "total_tx_power_dbm",
    "lambda_be_pps", "lambda_ull_pps", "packet_size_be_bytes", "packet_size_ull_bytes",
    "scenario_size_m", "sim_time",
)


def empirical_qoe_prediction(
    config: dict[str, Any], *, target_qoe: float,
    campaign: dict[tuple[int, float, float], dict[str, Any]] | None = None,
    live: dict[tuple[int, float, float], dict[str, Any]] | None = None,
    session_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Exact-condition empirical estimate from independent run summaries only.

    It is unavailable outside the recorded exact radio tuple. The two-repetition
    campaign and matching-semantics live runs provide descriptive run variation,
    not a statistically calibrated confidence interval.
    """
    key = config_key(config)
    values: list[float] = []
    outages: list[bool] = []
    seen_runs: set[tuple[int, int]] = set()

    def add_source(record: dict[str, Any], tag: str) -> None:
        qoes = record.get("run_qoe_values", [])
        outage_values = record.get("run_outage_values", [])
        run_keys = record.get("run_keys", [])
        for index, qoe_value in enumerate(qoes):
            run_key = run_keys[index] if index < len(run_keys) else None
            if run_key is not None:
                normalized_key = (int(run_key[0]), int(run_key[1]))
                if normalized_key in seen_runs:
                    continue
                seen_runs.add(normalized_key)
            values.append(float(qoe_value))
            outages.append(bool(outage_values[index]) if index < len(outage_values) else False)

    if int(config.get("gnb_count", 1)) == 1:
        record = (campaign or {}).get(key, {})
        add_source(record, "campaign")
        record = (live or {}).get(key, {})
        add_source(record, "live")
    for event in session_history or []:
        if (event.get("run_status") == "COMPLETED" and event.get("measured_qoe_proxy") is not None
                and event.get("config")
                and configuration_fingerprint(event["config"]) == configuration_fingerprint(config)):
            if event.get("rng_seed") is not None and event.get("rng_run") is not None:
                run_key = (int(event["rng_seed"]), int(event["rng_run"]))
                if run_key in seen_runs:
                    continue
                seen_runs.add(run_key)
            values.append(float(event["measured_qoe_proxy"]))
            outages.append(bool((event.get("metrics") or {}).get("is_outage", event.get("is_outage", False))))
    if not values:
        return {
            "available": False, "method": "exact-condition empirical run means",
            "domain": "OUT_OF_DOMAIN_OR_UNOBSERVED", "run_count": 0,
            "reason": "No prior QoE measurement exists for this exact configuration and compatible scenario semantics.",
            "qoe": None, "lower": None, "upper": None, "target_fraction": None,
            "outage_fraction": None, "confidence": "unavailable",
        }
    mean = sum(values) / len(values)
    if len(values) >= 2:
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        # A descriptive range based on run-to-run spread, not a calibrated CI.
        lower, upper = max(0.0, mean - 1.96 * sd), min(1.0, mean + 1.96 * sd)
    else:
        sd = None
        lower = upper = None
    return {
        "available": True, "method": "exact-condition empirical run means",
        "domain": "VALIDATED_EXACT_CONDITION" if len(values) >= 2 else "LIMITED_SINGLE_RUN",
        "run_count": len(values), "qoe": mean, "run_sd": sd,
        "lower": lower, "upper": upper,
        "target_fraction": sum(v >= float(target_qoe) for v in values) / len(values),
        "outage_fraction": sum(outages) / len(outages) if outages else None,
        "confidence": "limited; descriptive run-variation band, not calibrated probability" if len(values) >= 2
            else "very limited; one observed run, no uncertainty band",
        "reason": f"Exact pre-run configuration match; {len(values)} prior run-level observation(s).",
    }


def controller_phase(history: list[dict[str, Any]], *, target_qoe: float,
                     target_tolerance: float = 0.0, stable_epochs_required: int = 2) -> str:
    """TARGET_MAINTENANCE begins only after same-config results stay in target range."""
    streak = 0
    frozen_fingerprint: str | None = None
    for event in reversed(history):
        if event.get("run_status", "COMPLETED") != "COMPLETED" or event.get("measured_qoe_proxy") is None:
            break
        config = event.get("config") or {}
        fingerprint = configuration_fingerprint(config) if config else None
        score = float(event["measured_qoe_proxy"])
        if abs(score - float(target_qoe)) <= float(target_tolerance) and (
            frozen_fingerprint is None or fingerprint == frozen_fingerprint
        ):
            streak += 1
            frozen_fingerprint = fingerprint
        else:
            break
    return "TARGET_MAINTENANCE" if streak >= int(stable_epochs_required) else "TARGET_SEEKING"


def target_range_status(measured_qoe: float, target_qoe: float, tolerance: float) -> bool:
    """Absolute QoE target-range criterion used by target attainment and validation."""
    values = (float(measured_qoe), float(target_qoe), float(tolerance))
    if not all(math.isfinite(value) for value in values) or values[2] < 0:
        return False
    return abs(values[0] - values[1]) <= values[2]


def recommended_timeout_seconds(config: dict[str, Any]) -> tuple[int, int]:
    """Return (bounded timeout suggestion, estimated configured packet count).

    Packet count uses the configured per-UE application rates and active window;
    it is a workload-size proxy, not simulator progress. The suggested wall limit
    is calibrated from one measured high-load run, includes 25% headroom, retains
    the established 120-second floor, and is capped by the existing 300-second UI
    maximum. Users can still set a different explicit per-run timeout.
    """
    ue_count = int(config.get("ue_count", 0))
    sim_seconds = float(config.get("sim_time_seconds", str(config.get("sim_time", "1s")).rstrip("s")))
    app_start = float(config.get("app_start_seconds", 0.4))
    active_seconds = max(0.0, sim_seconds - app_start)
    low_latency_count = ue_count // 2
    voice_count = ue_count - low_latency_count
    configured_packets = int(math.ceil(active_seconds * (
        low_latency_count * int(config.get("lambda_ull_pps", 0))
        + voice_count * int(config.get("lambda_be_pps", 0))
    )))
    estimate = math.ceil(
        _TIMEOUT_REFERENCE_SECONDS * configured_packets / _TIMEOUT_REFERENCE_PACKETS
        * _TIMEOUT_RECOMMENDATION_MARGIN
    )
    return min(MAX_RUN_TIMEOUT_SECONDS, max(RUN_TIMEOUT_SECONDS, estimate)), configured_packets


def apply_target_attainment_record(
    event: dict[str, Any], *, config: dict[str, Any], previous_config: dict[str, Any] | None,
    target_qoe: float, tolerance: float, validation_epochs: int, epoch: int,
) -> dict[str, Any]:
    """Persist target-entry decision robustly, including older/minimal run records."""
    measured = float(event["measured_qoe_proxy"])
    rationale = event.get("action_rationale") or (
        f"Measured QoE {measured:.4f} is within the absolute target range "
        f"[{float(target_qoe) - float(tolerance):.4f}, {float(target_qoe) + float(tolerance):.4f}]. "
        f"Freeze this exact configuration for {int(validation_epochs)} additional independent ns-3 process runs; "
        "no configuration change is made."
    )
    fingerprint = configuration_fingerprint(config)
    event.update({
        "controller_phase": "STABILITY_VALIDATION",
        "controller_action": "TARGET_REACHED_START_VALIDATION",
        "action_rationale": rationale,
        "next_decision": "ENTER_STABILITY_VALIDATION",
        "decision_record": {
            "epoch": int(epoch), "timestamp_utc": event.get("end_utc"),
            "decision": "ENTER_STABILITY_VALIDATION", "phase": "STABILITY_VALIDATION",
            "config_id": config.get("config_id"), "configuration_fingerprint": fingerprint,
            "previous_config": previous_config, "resulting_config": config,
            "expected_effect": "Not estimated.", "measured_qoe_proxy": measured,
            "target_qoe": float(target_qoe), "tolerance": float(tolerance),
            "validation_required": int(validation_epochs), "reason": rationale,
        },
    })
    return event


def stability_epoch_passes(event: dict[str, Any], *, target_qoe: float, tolerance: float) -> tuple[bool, str]:
    """Validate a repeated run without treating missing per-UE data as an SLA pass."""
    if event.get("run_status") != "COMPLETED" or event.get("measured_qoe_proxy") is None:
        return False, "Simulation did not produce a validated measurement."
    if not target_range_status(float(event["measured_qoe_proxy"]), target_qoe, tolerance):
        return False, f"Measured QoE {float(event['measured_qoe_proxy']):.4f} is outside the target range."
    ue_rows = event.get("ue_measurements") or []
    if any(bool(row.get("is_outage")) for row in ue_rows):
        return False, "At least one UE had a measured complete outage."
    return True, "Measured QoE is in range and no per-UE complete outage was observed."


def verified_ue_counts(expansion_dir: str | Path = EXPANSION_RESULTS) -> list[int]:
    return sorted({c["ue_count"] for c in load_verified_configurations(expansion_dir)})


def verified_initial_configs(ue_count: int, expansion_dir: str | Path = EXPANSION_RESULTS) -> list[dict[str, Any]]:
    return [c for c in load_verified_configurations(expansion_dir) if c["ue_count"] == int(ue_count)]


def validate_custom_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate values accepted by the current scenario CLI; custom combinations remain exploratory."""
    result = dict(config)
    gnb_count = int(result.get("gnb_count", 1))
    if gnb_count not in {1, 2, 3}:
        raise ValueError("gNB count must be one of 1, 2 or 3 for this bounded scenario.")
    result["gnb_count"] = gnb_count
    association_policy = str(result.get("ue_association_policy", "closest"))
    if association_policy not in {"closest", "round_robin", "by_traffic_class"}:
        raise ValueError("UE association policy must be closest, round_robin or by_traffic_class.")
    if association_policy != "closest" and gnb_count < 2:
        raise ValueError("Non-closest UE association requires at least two gNB cells.")
    result["ue_association_policy"] = association_policy
    allowed_bandwidth = {20.0, 50.0, 100.0}
    allowed_power = {30.0, 35.0, 40.0}
    if int(result.get("ue_count", 0)) not in {2, 6, 10}:
        raise ValueError("UE count must be 2, 6 or 10 (verified scenario sizes).")
    for field in ("bandwidth_band1_mhz", "bandwidth_band2_mhz"):
        if float(result.get(field, 0)) not in allowed_bandwidth:
            raise ValueError(f"{field} must be one of 20, 50 or 100 MHz.")
        result[field] = float(result[field])
    if float(result.get("total_tx_power_dbm", 0)) not in allowed_power:
        raise ValueError("Total requested power must be 30, 35 or 40 dBm.")
    result["total_tx_power_dbm"] = float(result["total_tx_power_dbm"])
    for field, low, high in (("lambda_be_pps", 1000, 20000), ("lambda_ull_pps", 1000, 10000)):
        value = int(result.get(field, 0))
        if value < low or value > high:
            raise ValueError(f"{field} must be between {low} and {high} packets/s.")
        result[field] = value
    for field in ("packet_size_be_bytes", "packet_size_ull_bytes"):
        value = int(result.get(field, 0))
        if not 64 <= value <= 1472:
            raise ValueError(f"{field} must be between 64 and 1472 bytes.")
        result[field] = value
    sim_seconds = float(result.get("sim_time_seconds", str(result.get("sim_time", "1s")).rstrip("s")))
    if not 1.0 <= sim_seconds <= 3.0:
        raise ValueError("Simulation duration must be between 1 and 3 seconds.")
    result["sim_time"] = f"{sim_seconds:g}s"
    result["sim_time_seconds"] = sim_seconds
    scenario_size = float(result.get("scenario_size_m", 30.0))
    if not 30.0 <= scenario_size <= 200.0:
        raise ValueError("Scenario placement area must be between 30 and 200 metres.")
    result["scenario_size_m"] = scenario_size
    return result


def validate_exploratory_candidate(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Validate an exploratory radio tuple while freezing the rest of the scenario."""
    validated = validate_custom_config(candidate)
    if int(validated["ue_count"]) != int(reference["ue_count"]):
        raise ValueError("Exploration may not change UE count within a session.")
    frozen_fields = (
        "bandwidth_band2_mhz", "lambda_be_pps", "lambda_ull_pps", "packet_size_be_bytes",
        "packet_size_ull_bytes", "scenario_size_m", "sim_time",
    )
    for field in frozen_fields:
        if str(validated.get(field)) != str(reference.get(field)):
            raise ValueError(f"Exploration may not change session parameter {field}.")
    p1, p2 = derive_effective_band_powers_dbm(
        validated["total_tx_power_dbm"], validated["bandwidth_band1_mhz"], validated["bandwidth_band2_mhz"]
    )
    if not all(math.isfinite(float(power)) for power in (p1, p2)):
        raise ValueError("Exploratory power allocation is non-finite.")
    return validated


def production_feature_contract_audit(
    preprocessing_path: str | Path = ROOT / "ml/preprocessing.py",
    inference_path: str | Path = ROOT / "ml/predict.py",
) -> dict[str, Any]:
    """Detect the production training/inference offered-load definition mismatch."""
    preprocessing = Path(preprocessing_path).read_text(encoding="utf-8")
    inference = Path(inference_path).read_text(encoding="utf-8")
    training_uses_postrun_tx = 'df_proc["tx_packets"] * df_proc["packet_size_bytes"]' in preprocessing
    inference_uses_configured_load = (
        "tx_packets = flow_lambda * 0.6" in inference
        and "offered_throughput_mbps = (tx_packets * packet_size * 8.0)" in inference
    )
    return {
        "compatible": not training_uses_postrun_tx and inference_uses_configured_load,
        "training_uses_postrun_tx_packets": training_uses_postrun_tx,
        "inference_uses_configured_rate_and_payload": inference_uses_configured_load,
        "prediction_enabled": False,
        "reason": (
            "Production predictions are disabled: training offered_throughput_mbps uses post-run tx_packets, "
            "while inference uses configured rate × payload. Matching names do not imply matching semantics."
        ),
    }


def verify_ns3_runtime(binary: str | Path = DEFAULT_NS3_BINARY) -> tuple[bool, str]:
    """Confirm the built program exposes the exact CLI settings used by the epoch runner."""
    path = Path(binary).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        return False, f"ns-3 executable is absent or not executable: {path}"
    try:
        help_text = subprocess.run([str(path), "--PrintHelp"], check=True, capture_output=True,
                                   text=True, timeout=10).stdout
        globals_text = subprocess.run([str(path), "--PrintGlobals"], check=True, capture_output=True,
                                      text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Could not inspect ns-3 command-line support: {exc}"
    required = (
        "--ueNumPergNb", "--ueCount", "--gNbNum", "--ueAssociationPolicy", "--bandwidthBand1", "--bandwidthBand2", "--totalTxPower",
        "--lambdaBe", "--lambdaUll", "--packetSizeBe", "--packetSizeUll",
        "--scenarioId", "--outputDir", "--simTime",
    )
    missing = [name for name in required if name not in help_text]
    if missing:
        return False, f"Compiled simulator lacks required settings: {missing}"
    if "--RngSeed=[1]" not in globals_text or "--RngRun=[1]" not in globals_text:
        return False, "Compiled simulator does not expose RngSeed and RngRun globals"
    source = ROOT / "ns-3.48/scratch/qoe-dataset.cc"
    if not source.is_file():
        return False, f"Scenario source is absent: {source}"
    code = source.read_text(encoding="utf-8")
    for marker in (
        "gridScenario.SetUtNumber(effectiveUeCount)",
        "for (uint32_t gnbIndex = 0; gnbIndex < gnbNetDev.GetN(); ++gnbIndex)",
        "ueDevice->GetRrc()->GetCellId()",
        "AttachToGnb(ueDevice",
        'ueAssociationPolicy == "by_traffic_class"',
        "Simulator::Run()",
    ):
        if marker not in code:
            return False, f"Scenario source no longer matches verified setup: {marker}"
    return True, "CLI, RNG globals, fixed UE population, supported association policies, band power allocation and process-run lifecycle verified."


def decide_next_epoch(
    history: list[dict[str, Any]], feasible_configs: list[dict[str, Any]], *,
    target_qoe: float, max_epochs: int, mode: str = "adaptive",
    max_actions: int = MAX_ACTIONS, no_improvement_tolerance: float = NO_IMPROVEMENT_TOLERANCE,
    candidate_performance: dict[tuple[int, float, float], dict[str, Any]] | None = None,
    exploratory_configs: list[dict[str, Any]] | None = None,
    allow_exploratory_candidates: bool = False,
    pilot_performance: dict[tuple[int, float, float], dict[str, Any]] | None = None,
    live_performance: dict[tuple[int, float, float], dict[str, Any]] | None = None,
    allow_high_risk_candidates: bool = False,
    allow_topology_exploration: bool = False,
    allow_repeat_trials: bool = False,
    target_tolerance: float = 0.0,
    stable_epochs_required: int = 2,
    maintenance_margin: float = 0.02,
    maintenance_hysteresis: float = 0.02,
    predictive_forecast: dict[str, Any] | None = None,
    minimum_expected_benefit: float = 0.0,
) -> dict[str, Any]:
    """Choose a next epoch using exact-condition empirical evidence and measurements."""
    if not 0.0 <= float(target_qoe) <= 1.0:
        raise ValueError("target_qoe must be in [0,1]")
    if not 1 <= int(max_epochs) <= MAX_EPOCHS + MAX_POST_EPOCH_EXPLORATORY_EPOCHS:
        raise ValueError(f"max_epochs must be in 1..{MAX_EPOCHS + MAX_POST_EPOCH_EXPLORATORY_EPOCHS}")
    if mode not in {"adaptive", "fixed"}:
        raise ValueError("mode must be adaptive or fixed")
    if mode == "fixed" and (allow_exploratory_candidates or allow_topology_exploration):
        raise ValueError("Exploratory candidate selection is only available in adaptive mode.")
    if not history:
        raise ValueError("A valid measured epoch is required")
    if not 0.0 <= float(target_tolerance) <= 0.1:
        raise ValueError("target_tolerance must be in [0,0.1]")
    if not 1 <= int(stable_epochs_required) <= 5:
        raise ValueError("stable_epochs_required must be in 1..5")
    actions = sum(h.get("controller_action") == "CHANGE_CONFIGURATION"
                  and not h.get("is_exploratory", False) for h in history)
    exploratory_trials = sum(bool(h.get("is_exploratory", False)) for h in history)
    action_budget_exhausted = actions >= max_actions
    budget_fields = {
        "adaptive_action_count": actions,
        "exploratory_trial_count": exploratory_trials,
        "adaptive_action_budget": int(max_actions),
        "action_budget_exhausted": bool(action_budget_exhausted),
    }
    last = history[-1]
    qoe = last.get("measured_qoe_proxy")
    if qoe is None or not math.isfinite(float(qoe)) or not 0.0 <= float(qoe) <= 1.0:
        return {"action": "STOP_INVALID_MEASUREMENT", "rationale": "No valid measured QoE is available.", **budget_fields}
    phase = controller_phase(history, target_qoe=target_qoe, target_tolerance=target_tolerance,
                             stable_epochs_required=stable_epochs_required)
    forecast = predictive_forecast or {}
    early_warning = (forecast.get("status") == "RISK_PREDICTED"
                     and bool(forecast.get("at_risk_ue_ids")))
    target_band = (float(target_qoe) - float(target_tolerance),
                   float(target_qoe) + float(target_tolerance))
    if len(history) >= max_epochs:
        if phase == "TARGET_MAINTENANCE":
            return {"action": "STOP_TARGET_STABLE", "controller_phase": phase,
                    "rationale": f"Target remained in absolute range [{target_band[0]:.4f}, {target_band[1]:.4f}] "
                                 f"for {stable_epochs_required} consecutive validated epochs; normal epoch budget is complete.",
                    **budget_fields}
        return {"action": "STOP_EPOCH_LIMIT", "controller_phase": phase,
                "rationale": f"Reached {max_epochs} epochs; target stability criterion was not met.", **budget_fields}
    if mode == "fixed":
        return {"action": "CONTINUE_FIXED", "next_config": last["config"],
                "rationale": "Fixed mode repeats the same configuration as a baseline.", **budget_fields}
    current = last["config"]
    performance = candidate_performance or {}
    pilot_history = pilot_performance or {}
    live_history = live_performance or {}
    current_prediction = empirical_qoe_prediction(
        current, target_qoe=target_qoe, campaign=performance,
        live=live_history, session_history=history,
    )
    recent_streak = 0
    for item in reversed(history):
        if item.get("measured_qoe_proxy") is not None and target_range_status(
            float(item["measured_qoe_proxy"]), float(target_qoe), float(target_tolerance)
        ):
            recent_streak += 1
        else:
            break
    if qoe is not None and target_range_status(float(qoe), float(target_qoe), float(target_tolerance)) \
            and phase == "TARGET_SEEKING" and not early_warning:
        return {
            "action": "CONTINUE_STABILITY", "next_config": current, "controller_phase": phase,
            "prediction": current_prediction,
            "candidate_scope": last.get("candidate_scope", "current measured configuration"),
            "is_exploratory": bool(last.get("is_exploratory", False)),
            "rationale": f"Target range reached provisionally ({qoe:.4f} in [{target_band[0]:.4f}, {target_band[1]:.4f}]); "
                         f"need {stable_epochs_required - recent_streak} more consecutive validated epoch(s) "
                         "on this configuration before declaring stable attainment.", **budget_fields,
        }
    if phase == "TARGET_MAINTENANCE" and not early_warning:
        safe_margin = float(target_qoe) + float(maintenance_margin)
        if not current_prediction["available"]:
            return {"action": "CONTINUE_MAINTENANCE", "next_config": current,
                    "controller_phase": phase, "prediction": current_prediction,
                    "candidate_scope": last.get("candidate_scope", "current measured configuration"),
                    "is_exploratory": bool(last.get("is_exploratory", False)),
                    "rationale": "Target stability criterion is met. Prediction unavailable for the current exact condition; "
                                 "fallback retains the last measured-safe configuration without claiming a forecast.",
                    **budget_fields}
        current_lcb = current_prediction.get("lower")
        if current_lcb is None or current_lcb >= safe_margin:
            return {"action": "CONTINUE_MAINTENANCE", "next_config": current,
                    "controller_phase": phase, "prediction": current_prediction,
                    "candidate_scope": last.get("candidate_scope", "current measured configuration"),
                    "is_exploratory": bool(last.get("is_exploratory", False)),
                    "rationale": f"Target stability criterion is met and empirical lower run-variation bound "
                                 f"{current_lcb if current_lcb is not None else current_prediction['qoe']:.4f} "
                                 f"meets maintenance threshold {safe_margin:.4f}; retain configuration with hysteresis.",
                    **budget_fields}

    # On a measured outage or meaningful regression, first offer a single
    # rollback to the best previously measured non-outage configuration.
    latest_metrics = last.get("metrics") or {}
    meaningful_regression = last.get("action_outcome_from_previous") == "WORSENED"
    if bool(latest_metrics.get("is_outage", last.get("is_outage", False))) or meaningful_regression:
        safe_events = [item for item in history if item.get("config")
                       and item.get("run_status", "COMPLETED") == "COMPLETED"
                       and item.get("measured_qoe_proxy") is not None
                       and not bool((item.get("metrics") or {}).get("is_outage", item.get("is_outage", False)))]
        best_safe = max(safe_events, key=lambda item: float(item["measured_qoe_proxy"])) if safe_events else None
        if best_safe and config_key(best_safe["config"]) != config_key(current):
            fallback_key = config_key(best_safe["config"])
            already_recovered = any(item.get("recovery_target_key") == list(fallback_key) for item in history)
            risk = assess_candidate_risk(fallback_key, campaign=performance, pilot=pilot_history, live=live_history)
            directly_measured_safe = any(item.get("config") and config_key(item["config"]) == fallback_key
                                          and item.get("run_status", "COMPLETED") == "COMPLETED"
                                          and item.get("measured_qoe_proxy") is not None
                                          and not bool((item.get("metrics") or {}).get("is_outage", item.get("is_outage", False)))
                                          for item in history)
            if not already_recovered and (directly_measured_safe or allow_high_risk_candidates
                                          or not risk["excluded_by_default"]):
                prediction = empirical_qoe_prediction(best_safe["config"], target_qoe=target_qoe,
                                                      campaign=performance, live=live_history,
                                                      session_history=history)
                return {
                    "action": "CHANGE_CONFIGURATION", "next_config": best_safe["config"],
                    "controller_phase": "TARGET_SEEKING", "is_recovery": True,
                    "recovery_target_key": list(fallback_key),
                    "candidate_scope": best_safe.get("candidate_scope", "previously measured safe fallback"),
                    "is_exploratory": bool(best_safe.get("is_exploratory", False)),
                    "candidate_evidence": f"previously measured QoE {float(best_safe['measured_qoe_proxy']):.4f}; non-outage",
                    "candidate_risk_level": risk["level"], "candidate_risk_reason": risk["reason"],
                    "prediction": prediction, "predicted_qoe": prediction.get("qoe"),
                    "predicted_qoe_lower": prediction.get("lower"), "predicted_qoe_upper": prediction.get("upper"),
                    "prediction_domain": prediction.get("domain"), "prediction_run_count": prediction.get("run_count", 0),
                    "prediction_method": prediction.get("method"),
                    "rationale": f"Measured {'outage' if latest_metrics.get('is_outage') else 'meaningful regression'}; "
                                 f"return once to best prior non-outage configuration {best_safe['config'].get('config_id')} "
                                 f"(measured QoE {float(best_safe['measured_qoe_proxy']):.4f}). This is a recovery trial, not a guarantee.",
                    **budget_fields,
                }
    tried_fingerprints = {configuration_fingerprint(h["config"]) for h in history if h.get("config")}
    tried = {config_key(h["config"]) for h in history if h.get("config")}
    candidates_for_ue = [
        c for c in feasible_configs
        if int(c["ue_count"]) == int(current["ue_count"])
        and int(c.get("gnb_count", 1)) == int(current.get("gnb_count", 1))
    ]
    duplicate_candidates = [c for c in candidates_for_ue
                            if configuration_fingerprint(c) in tried_fingerprints]
    available = [c for c in candidates_for_ue
                 if configuration_fingerprint(c) not in tried_fingerprints]
    risk_assessments = {config_key(c): assess_candidate_risk(
        config_key(c),
        campaign=performance if int(c.get("gnb_count", 1)) == 1 else {},
        pilot=pilot_history if int(c.get("gnb_count", 1)) == 1 else {},
        live=live_history if int(c.get("gnb_count", 1)) == 1 else {},
    ) for c in available}
    risky = [c for c in available if risk_assessments[config_key(c)]["excluded_by_default"]]
    viable = [c for c in available if allow_high_risk_candidates or c not in risky]
    if allow_repeat_trials:
        viable.extend(c for c in duplicate_candidates
                      if allow_high_risk_candidates or not assess_candidate_risk(
                          config_key(c), campaign=performance, pilot=pilot_history, live=live_history,
                      )["excluded_by_default"])
    historically_bad = risky
    risk_exclusion_details = [
        f"{candidate.get('config_id', config_key(candidate))}: {risk_assessments[config_key(candidate)]['reason']}"
        for candidate in risky
    ]
    used_exploratory = False
    exploratory_validation_failures: list[str] = []
    exploratory_outage_exclusions: list[str] = []
    # max_actions marks the end of the normal adaptive-selection phase; it is
    # not terminal. Opt-in sessions move to exploratory trials at that point.
    explore_before_validated = allow_exploratory_candidates and action_budget_exhausted
    allow_same_topology_exploration = allow_exploratory_candidates or (
        allow_topology_exploration and int(current.get("gnb_count", 1)) > 1
    )
    if (allow_same_topology_exploration or allow_topology_exploration) and (
        not viable or explore_before_validated
    ):
        validated_exploratory = []
        validated_topologies = []
        campaign_keys = {config_key(v) for v in candidates_for_ue}
        seen_exploratory: set[str] = set()
        for candidate in exploratory_configs or []:
            try:
                key = config_key(candidate)
                fingerprint = configuration_fingerprint(candidate)
            except (KeyError, TypeError, ValueError) as exc:
                exploratory_validation_failures.append(f"malformed candidate: {exc}")
                continue
            is_topology_candidate = bool(candidate.get("topology_candidate", False))
            if is_topology_candidate and (
                not allow_topology_exploration
                or int(candidate.get("gnb_count", 1)) <= int(current.get("gnb_count", 1))
            ):
                continue
            if not is_topology_candidate and not allow_same_topology_exploration:
                continue
            candidate = dict(candidate)
            if not is_topology_candidate:
                candidate["gnb_count"] = int(current.get("gnb_count", 1))
                candidate["ue_association_policy"] = str(current.get("ue_association_policy", "closest"))
                if candidate["gnb_count"] > 1:
                    candidate["config_id"] = f"{candidate['config_id']}-GNB{candidate['gnb_count']}"
                fingerprint = configuration_fingerprint(candidate)
            if ((fingerprint in tried_fingerprints and not allow_repeat_trials)
                    or (not is_topology_candidate and key in campaign_keys
                        and int(current.get("gnb_count", 1)) == 1)
                    or fingerprint in seen_exploratory):
                continue
            seen_exploratory.add(fingerprint)
            try:
                checked = validate_exploratory_candidate(candidate, current)
            except (ValueError, TypeError, KeyError) as exc:
                exploratory_validation_failures.append(f"{candidate.get('config_id', key)}: {exc}")
                continue
            same_scope = (int(candidate.get("gnb_count", 1)) == int(history[0]["config"].get("gnb_count", 1)))
            comparable_campaign = performance if same_scope and int(candidate.get("gnb_count", 1)) == 1 else {}
            comparable_pilot = pilot_history if same_scope and int(candidate.get("gnb_count", 1)) == 1 else {}
            comparable_live = live_history if same_scope else {}
            risk = (assess_candidate_risk(key, campaign=comparable_campaign,
                                          pilot=comparable_pilot, live=comparable_live)
                    if not is_topology_candidate else {
                        "level": "UNTESTED_TOPOLOGY", "reason": "No matching multi-gNB campaign outcome exists.",
                        "excluded_by_default": False,
                    })
            if risk["excluded_by_default"] and not allow_high_risk_candidates:
                exploratory_outage_exclusions.append(f"{checked.get('config_id', key)} ({risk['reason']})")
                risk_exclusion_details.append(f"{checked.get('config_id', key)}: {risk['reason']}")
                continue
            if is_topology_candidate:
                validated_topologies.append(checked)
            else:
                validated_exploratory.append(checked)
        if validated_exploratory:
            viable = validated_exploratory
            used_exploratory = True
        elif validated_topologies:
            viable = validated_topologies
            used_exploratory = True
    if not viable:
        if phase == "TARGET_MAINTENANCE" and target_range_status(
            float(qoe), float(target_qoe), float(target_tolerance)
        ):
            return {"action": "CONTINUE_MAINTENANCE", "next_config": current,
                    "controller_phase": phase, "prediction": current_prediction,
                    "candidate_scope": last.get("candidate_scope", "current measured configuration"),
                    "is_exploratory": bool(last.get("is_exploratory", False)),
                    "risk_excluded_count": len(risky) + len(exploratory_outage_exclusions),
                    "duplicate_excluded_count": len(duplicate_candidates),
                    "rationale": "No eligible alternative remains under the configured risk and duplicate policy; "
                                 "retain the last measured-safe configuration and verify it in the next bounded epoch.",
                    **budget_fields}
        attempted = len(tried & {config_key(c) for c in candidates_for_ue})
        remaining = len(available)
        initial_qoe = next((float(h["measured_qoe_proxy"]) for h in history
                            if h.get("measured_qoe_proxy") is not None), float(qoe))
        best_qoe = max(float(h["measured_qoe_proxy"]) for h in history
                       if h.get("measured_qoe_proxy") is not None)
        if allow_exploratory_candidates and not (exploratory_configs or []):
            return {"action": "STOP_NO_VALIDATED_CANDIDATES",
                    "rationale": "No supported exploratory candidate list was supplied; no automatic action was taken.",
                    **budget_fields}
        if (not risky and not exploratory_outage_exclusions and len(history) > 1
                and best_qoe - initial_qoe <= no_improvement_tolerance):
            return {"action": "STOP_NO_MEANINGFUL_IMPROVEMENT",
                    "rationale": f"All eligible supported candidates were attempted or excluded; best measured QoE improved by only "
                                 f"{best_qoe - initial_qoe:.6f}, within the {no_improvement_tolerance:.6f} tolerance.",
                    **budget_fields}
        if allow_exploratory_candidates:
            outage_details = (" Conservative risk exclusions: "
                              + "; ".join(risk_exclusion_details) + ".") if risk_exclusion_details else ""
            return {"action": "STOP_NO_SAFE_CANDIDATES" if risky or exploratory_outage_exclusions else "STOP_NO_EXPERIMENTAL_CANDIDATES",
                    "rationale": f"No eligible untried candidate remains in the validated or opt-in supported pool "
                                 f"({attempted} of {len(candidates_for_ue)} validated conditions attempted; "
                                 f"{len(historically_bad) + len(exploratory_outage_exclusions)} known-outage condition(s) excluded; "
                                 f"{len(exploratory_validation_failures)} exploratory candidate(s) failed validation)."
                    f"{outage_details}",
                    "risk_excluded_count": len(risky) + len(exploratory_outage_exclusions),
                    "duplicate_excluded_count": len(duplicate_candidates), **budget_fields}
        return {"action": "STOP_NO_SAFE_CANDIDATES" if risky else "STOP_NO_VALIDATED_CANDIDATES",
                "rationale": f"No untried candidate remains in the validated campaign registry under the current risk policy "
                             f"({attempted} of {len(candidates_for_ue)} conditions attempted; {remaining} untried, "
                             f"{len(historically_bad)} excluded by conservative risk evidence). "
                             "This does not establish that other supported ns-3 settings are impossible; enable "
                             "the explicit exploratory-candidate option before the session to try other supported settings."
                             + ((" Risk exclusions: " + "; ".join(risk_exclusion_details) + ".")
                                if risk_exclusion_details else ""),
                "risk_excluded_count": len(risky) + len(exploratory_outage_exclusions),
                "duplicate_excluded_count": len(duplicate_candidates), **budget_fields}

    current_value = current_prediction.get("qoe") if current_prediction.get("available") else None

    def rank(c: dict[str, Any]) -> tuple[Any, ...]:
        is_new_topology = bool(c.get("topology_candidate", False))
        observed = (performance.get(config_key(c))
                    if not is_new_topology and int(c.get("gnb_count", 1)) == 1 else None)
        # Prefer verified candidates with better observed campaign QoE. Distance
        # breaks ties and avoids selecting a poor point just because it is close.
        mean_qoe = float(observed["mean_qoe_proxy"]) if observed else -1.0
        distance = (
            abs(c["bandwidth_band1_mhz"] - current["bandwidth_band1_mhz"]) / 50.0
            + abs(c["total_tx_power_dbm"] - current["total_tx_power_dbm"]) / 5.0
        )
        prediction = empirical_qoe_prediction(c, target_qoe=target_qoe,
                                              campaign=performance, live=live_history,
                                              session_history=history)
        if prediction["available"]:
            predicted_target = float(prediction["target_fraction"] or 0.0)
            outage_risk = float(prediction["outage_fraction"]) if prediction.get("outage_fraction") is not None else 1.0
            conservative_value = prediction["lower"] if prediction["lower"] is not None else prediction["qoe"]
            improvement = float(prediction["qoe"]) - float(current_value if current_value is not None else qoe)
            # Target probability is empirical frequency, not calibrated probability.
            return (0, -predicted_target, outage_risk, -float(conservative_value), -improvement,
                    distance, c["bandwidth_band1_mhz"], c["total_tx_power_dbm"])
        # Validated but unobserved/exploratory conditions have no fabricated forecast.
        if is_new_topology:
            return (3, int(c.get("gnb_count", 1)), c["bandwidth_band1_mhz"], c["total_tx_power_dbm"])
        if used_exploratory:
            key = config_key(c)
            comparable_scope = int(c.get("gnb_count", 1)) == int(history[0]["config"].get("gnb_count", 1))
            live = live_history.get(key, {}) if comparable_scope else {}
            pilot = pilot_history.get(key, {}) if comparable_scope and int(c.get("gnb_count", 1)) == 1 else {}
            # Prefer exact-semantics live evidence, then lower-load pilot
            # feasibility evidence, then unobserved combinations.
            evidence_rank = 0 if live else (1 if pilot.get("all_runs_nonoutage") else 2)
            prior_qoe = float(live.get("mean_qoe_proxy", -1.0))
            return (1, evidence_rank, -prior_qoe if evidence_rank == 0 else 0.0,
                    distance, c["bandwidth_band1_mhz"], c["total_tx_power_dbm"])
        return (2, -mean_qoe, distance, c["bandwidth_band1_mhz"], c["total_tx_power_dbm"])

    next_config = min(viable, key=rank)
    is_topology_candidate = bool(next_config.get("topology_candidate", False))
    same_scope = int(next_config.get("gnb_count", 1)) == int(history[0]["config"].get("gnb_count", 1))
    next_risk = (assess_candidate_risk(
                     config_key(next_config),
                     campaign=performance if same_scope and int(next_config.get("gnb_count", 1)) == 1 else {},
                     pilot=pilot_history if same_scope and int(next_config.get("gnb_count", 1)) == 1 else {},
                     live=live_history if same_scope else {},
                 )
                 if not is_topology_candidate else {
                     "level": "UNTESTED_TOPOLOGY", "reason": "No matching multi-gNB campaign outcome exists.",
                     "excluded_by_default": False,
                 })
    next_prediction = empirical_qoe_prediction(next_config, target_qoe=target_qoe,
                                                campaign=performance, live=live_history,
                                                session_history=history)
    if minimum_expected_benefit > 0:
        if not next_prediction.get("available"):
            return {"action": "NO_ACTION_UNSUPPORTED_BENEFIT", "controller_phase": phase,
                    "candidate_rejected": next_config,
                    "rationale": "Candidate has no exact-condition empirical outcome; minimum-benefit policy rejected it.",
                    "predictive_risk": forecast if early_warning else None, **budget_fields}
        expected_benefit = float(next_prediction["qoe"]) - float(qoe)
        if expected_benefit < float(minimum_expected_benefit):
            return {"action": "NO_ACTION_MINIMUM_BENEFIT", "controller_phase": phase,
                    "candidate_rejected": next_config,
                    "expected_qoe_change": expected_benefit,
                    "rationale": f"Candidate's exact-condition estimate changes QoE by {expected_benefit:+.4f}, "
                                 f"below the required minimum {float(minimum_expected_benefit):.4f}; no action taken.",
                    "predictive_risk": forecast if early_warning else None, **budget_fields}
    if phase == "TARGET_MAINTENANCE" and next_prediction["available"]:
        candidate_bound = next_prediction.get("lower")
        current_bound = current_prediction.get("lower")
        candidate_score = candidate_bound if candidate_bound is not None else next_prediction["qoe"]
        current_score = current_bound if current_bound is not None else current_prediction["qoe"]
        target_threatened = current_prediction["qoe"] < float(target_qoe) + float(maintenance_margin)
        if not target_threatened or candidate_score < current_score + float(maintenance_hysteresis):
            return {"action": "CONTINUE_MAINTENANCE", "next_config": current,
                    "controller_phase": phase, "prediction": current_prediction,
                    "candidate_scope": last.get("candidate_scope", "current measured configuration"),
                    "is_exploratory": bool(last.get("is_exploratory", False)),
                    "rationale": "No alternative has at least the configured hysteresis improvement; retain the measured-safe configuration.",
                    **budget_fields}
    observed = performance.get(config_key(next_config)) if same_scope and int(next_config.get("gnb_count", 1)) == 1 else None
    pilot_observed = (pilot_history.get(config_key(next_config), {})
                      if same_scope and int(next_config.get("gnb_count", 1)) == 1 else {})
    live_observed = live_history.get(config_key(next_config), {}) if same_scope else {}
    if is_topology_candidate:
        empirical = "no matching multi-gNB result; candidate requires actual ns-3 evaluation"
    elif observed:
        empirical = f"campaign mean QoE {observed['mean_qoe_proxy']:.4f} over {observed['repetition_count']} repetitions"
    elif live_observed:
        empirical = (f"previous matching-semantics live measurement(s): {live_observed['run_count']} run(s), "
                     f"mean QoE {live_observed['mean_qoe_proxy']:.4f}, no all-run outage")
    elif pilot_observed.get("all_runs_nonoutage"):
        empirical = (f"prior low-load pilot completed without outage at {pilot_observed['lambda_be_pps']} BE pps; "
                     "not equivalent to this session load")
    elif used_exploratory:
        empirical = "accepted by the ns-3 CLI but not run in the campaign"
        if pilot_observed.get("all_runs_outage"):
            empirical += f"; low-load pilot also had outage evidence at {pilot_observed['lambda_be_pps']} BE pps"
    else:
        empirical = "no campaign outcome summary available"
    if historically_bad:
        bad_details = []
        for candidate in historically_bad:
            evidence = performance.get(config_key(candidate), {})
            bad_details.append(f"{candidate['config_id']} ({evidence.get('repetition_count', '?')} campaign repetitions; all rows outage)")
        excluded = " Excluded from automatic selection after matching-workload all-outage evidence: " + "; ".join(bad_details) + "."
    else:
        excluded = ""
    if is_topology_candidate:
        basis = "fixed UE population; supported multi-gNB GridScenarioHelper construction and closest-gNB association; no performance prediction"
    elif used_exploratory:
        if live_observed:
            basis = "prior matching-semantics live measurement and distance; this is an observation, not a prediction"
        elif pilot_observed.get("all_runs_nonoutage"):
            basis = "prior lower-load pilot feasibility evidence and distance; QoE under this session load is unknown"
        else:
            basis = "validated parameter values and distance only; this combination's outcome is unknown"
    else:
        basis = "campaign evidence and distance"
    if action_budget_exhausted:
        basis += "; normal adaptive action budget reached, continuing within the epoch budget"
    if risk_exclusion_details:
        excluded += " Risk policy skipped: " + "; ".join(risk_exclusion_details) + "."
    prediction_summary = "Prediction unavailable outside the exact-condition evidence domain."
    if next_prediction.get("available"):
        prediction_summary = (
            f"empirical QoE={next_prediction['qoe']:.4f}; target-run fraction="
            f"{next_prediction['target_fraction']:.3f}; "
            f"observed outage fraction={next_prediction['outage_fraction']:.3f}; "
            f"support n={next_prediction['run_count']} ({next_prediction['confidence']})"
        )
    rejected_alternatives = [{
        "candidate_id": candidate.get("config_id"),
        "reason": risk_assessments[config_key(candidate)]["reason"],
    } for candidate in risky]
    rejected_alternatives.extend({
        "candidate_id": candidate.get("config_id"),
        "reason": "Lower-ranked than the selected option by observed outcome evidence and configuration distance.",
    } for candidate in viable if configuration_fingerprint(candidate) != configuration_fingerprint(next_config))
    if early_warning:
        no_action_reason = ("Rejected because the one-epoch forecast identifies UE(s) "
                            + ", ".join(map(str, forecast.get("at_risk_ue_ids", [])))
                            + " below the configured target.")
    elif float(qoe) < float(target_qoe):
        no_action_reason = (f"Rejected because measured QoE {float(qoe):.4f} is below target "
                            f"{float(target_qoe):.4f} and the selected candidate has a supported estimate.")
    else:
        no_action_reason = "Rejected because the controller ranked a supported alternative above retaining the current setting."
    rejected_alternatives.append({"candidate_id": None, "kind": "no_action", "reason": no_action_reason})
    return {
        "action": "CHANGE_CONFIGURATION", "next_config": next_config,
        "controller_phase": phase,
        "candidate_scope": ("structural topology alternative · unvalidated" if is_topology_candidate else
                            "exploratory supported configuration" if used_exploratory else
                            "validated campaign configuration"),
        "topology_change": is_topology_candidate,
        "is_exploratory": bool(used_exploratory),
        "candidate_evidence": empirical,
        "candidate_validation": "PASS: discrete CLI values, preserved UE/Band 2/traffic, finite effective band powers",
        "candidate_risk_level": next_risk["level"], "candidate_risk_reason": next_risk["reason"],
        "prediction": next_prediction,
        "predicted_qoe": next_prediction.get("qoe"),
        "predicted_qoe_lower": next_prediction.get("lower"),
        "predicted_qoe_upper": next_prediction.get("upper"),
        "predicted_target_fraction": next_prediction.get("target_fraction"),
        "predicted_outage_fraction": next_prediction.get("outage_fraction"),
        "prediction_method": next_prediction.get("method"),
        "prediction_domain": next_prediction.get("domain"),
        "prediction_run_count": next_prediction.get("run_count", 0),
        "expected_qoe_change": (float(next_prediction["qoe"]) - float(qoe)
                                 if next_prediction.get("available") else None),
        "high_risk_override": bool(allow_high_risk_candidates and next_risk["excluded_by_default"]),
        "repeat_trials_enabled": bool(allow_repeat_trials),
        "risk_excluded_count": len(risky) + len(exploratory_outage_exclusions),
        "duplicate_excluded_count": len(duplicate_candidates),
        "rejected_alternatives": rejected_alternatives,
        **budget_fields,
        "is_recovery": False,
        "rationale": f"{phase}: Measured QoE {qoe:.4f}; "
                     f"{'forecast risk for UE(s) ' + ', '.join(map(str, forecast.get('at_risk_ue_ids', []))) + '; ' if early_warning else ''}select "
                     f"{'exploratory' if used_exploratory else 'validated'} config "
                     f"{next_config['config_id']} ({empirical}); prediction: {prediction_summary}; "
                     f"ranking basis: {basis}.{excluded}",
    }


def decide_post_epoch_exploration(
    history: list[dict[str, Any]], feasible_configs: list[dict[str, Any]],
    exploratory_configs: list[dict[str, Any]], *, target_qoe: float,
    max_additional_epochs: int, normal_action_budget: int = MAX_ACTIONS,
    candidate_performance: dict[tuple[int, float, float], dict[str, Any]] | None = None,
    pilot_performance: dict[tuple[int, float, float], dict[str, Any]] | None = None,
    live_performance: dict[tuple[int, float, float], dict[str, Any]] | None = None,
    allow_high_risk_candidates: bool = False,
    allow_topology_exploration: bool = False,
    allow_repeat_trials: bool = False,
    target_tolerance: float = 0.0,
    stable_epochs_required: int = 2,
    maintenance_margin: float = 0.02,
    maintenance_hysteresis: float = 0.02,
) -> dict[str, Any]:
    """Select optional post-limit experimental trials under a separate epoch budget."""
    if not 1 <= int(max_additional_epochs) <= MAX_POST_EPOCH_EXPLORATORY_EPOCHS:
        raise ValueError(f"Additional exploratory epochs must be in 1..{MAX_POST_EPOCH_EXPLORATORY_EPOCHS}.")
    completed_extra = sum(bool(event.get("post_epoch_exploration")) for event in history)
    exploratory_trials = sum(bool(event.get("is_exploratory")) for event in history)
    adaptive_actions = sum(event.get("controller_action") == "CHANGE_CONFIGURATION"
                           and not event.get("is_exploratory", False) for event in history)
    budget_metadata = {
        "adaptive_action_count": adaptive_actions,
        "exploratory_trial_count": exploratory_trials,
        "adaptive_action_budget": int(normal_action_budget),
        "action_budget_exhausted": adaptive_actions >= int(normal_action_budget),
    }
    if completed_extra >= int(max_additional_epochs):
        return {
            "action": "STOP_POST_EPOCH_LIMIT",
            "rationale": f"Reached the separate limit of {max_additional_epochs} additional experimental epochs; normal epoch limit was already reached.",
            "post_epoch_exploration_count": completed_extra,
            **budget_metadata,
        }

    # Only candidates outside the validated campaign set belong to this extra
    # phase. Candidate validation, duplicate prevention and outage exclusions
    # remain centralized in decide_next_epoch.
    validated_keys = {config_key(config) for config in feasible_configs}
    experimental_only = [candidate for candidate in exploratory_configs
                         if bool(candidate.get("topology_candidate", False))
                         or config_key(candidate) not in validated_keys]
    choice = decide_next_epoch(
        history, [], target_qoe=target_qoe, max_epochs=len(history) + 1,
        mode="adaptive", max_actions=normal_action_budget, exploratory_configs=experimental_only,
        allow_exploratory_candidates=True, candidate_performance=candidate_performance,
        pilot_performance=pilot_performance, live_performance=live_performance,
        allow_high_risk_candidates=allow_high_risk_candidates,
        allow_topology_exploration=allow_topology_exploration,
        allow_repeat_trials=allow_repeat_trials,
        target_tolerance=target_tolerance,
        stable_epochs_required=stable_epochs_required,
        maintenance_margin=maintenance_margin,
        maintenance_hysteresis=maintenance_hysteresis,
    )
    if choice.get("action") not in {"CHANGE_CONFIGURATION", "CONTINUE_STABILITY", "CONTINUE_MAINTENANCE"}:
        return {
            "action": choice.get("action") if choice.get("action") in {
                "STOP_NO_SAFE_CANDIDATES", "STOP_NO_EXPERIMENTAL_CANDIDATES"
            } else "STOP_NO_EXPERIMENTAL_CANDIDATES",
            "rationale": f"No eligible untried experimental configuration remains after the normal epoch limit. {choice.get('rationale', '')}",
            "post_epoch_exploration_count": completed_extra,
            **budget_metadata,
        }
    rationale = (f"Normal epoch limit reached; continue with experimental trial {completed_extra + 1} "
                 f"of at most {max_additional_epochs}. {choice['rationale']}")
    return {
        **choice,
        **budget_metadata,
        "action": "CONTINUE_POST_EPOCH_EXPLORATION",
        "post_epoch_exploration_count": completed_extra,
        "post_epoch_exploration_limit": int(max_additional_epochs),
        "post_epoch_exploration": True,
        "continuation_reason": rationale,
        "rationale": rationale,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _log(session_dir: Path, message: str) -> None:
    with (session_dir / "controller.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {message}\n")


def _history_row(event: dict[str, Any]) -> dict[str, Any]:
    config = event.get("config", {})
    p1, p2 = derive_effective_band_powers_dbm(
        config.get("total_tx_power_dbm", 35), config.get("bandwidth_band1_mhz", 50),
        config.get("bandwidth_band2_mhz", 50),
    )
    metrics = event.get("metrics") or {}
    row = {key: "" for key in HISTORY_COLUMNS}
    row.update({
        "session_id": event.get("session_id"), "epoch": event.get("epoch"),
        "run_id": event.get("run_id"), "scenario_id": event.get("scenario_id"),
        "configuration_fingerprint": configuration_fingerprint(config) if config else "",
        "config_id": config.get("config_id"), "controller_mode": event.get("controller_mode"),
        "ue_count": config.get("ue_count"), "gnb_count": config.get("gnb_count", 1),
        "ue_association_policy": config.get("ue_association_policy", "closest"),
        "topology_candidate_id": config.get("config_id") if config.get("topology_candidate") else "",
        "topology_version": f"gNB-{config.get('gnb_count', 1)}",
        "topology_change": bool(event.get("topology_change", config.get("topology_candidate", False))),
        "topology_change_reason": config.get("topology_change_reason", ""),
        "is_validation_epoch": bool(event.get("is_validation_epoch", False)),
        "validation_sequence": event.get("validation_sequence", ""),
        "validation_pass": event.get("validation_pass", ""),
        "validation_progress": event.get("validation_progress", ""),
        "target_range_pass": event.get("target_range_pass", ""),
        "qos_constraint_violations": json.dumps(event.get("qos_constraint_violations", [])),
        "decision_record": json.dumps(event.get("decision_record"), separators=(",", ":"))
            if event.get("decision_record") else "",
        "bandwidth_band1_mhz": config.get("bandwidth_band1_mhz"),
        "bandwidth_band2_mhz": config.get("bandwidth_band2_mhz"),
        "total_tx_power_dbm": config.get("total_tx_power_dbm"),
        "effective_band1_tx_power_dbm": p1, "effective_band2_tx_power_dbm": p2,
        "lambda_be_pps": config.get("lambda_be_pps"), "lambda_ull_pps": config.get("lambda_ull_pps"),
        "packet_size_be_bytes": config.get("packet_size_be_bytes"),
        "packet_size_ull_bytes": config.get("packet_size_ull_bytes"),
        "scenario_size_m": config.get("scenario_size_m"), "sim_time": config.get("sim_time"),
        "rng_seed": event.get("rng_seed"), "rng_run": event.get("rng_run"),
        "command_json": json.dumps(event.get("command", []), separators=(",", ":")),
        "start_utc": event.get("start_utc"), "end_utc": event.get("end_utc"),
        "duration_seconds": event.get("duration_seconds"), "exit_code": event.get("exit_code"),
        "run_status": event.get("run_status"), "raw_csv_path": event.get("raw_csv_path"),
        "stdout_path": event.get("stdout_path"), "stderr_path": event.get("stderr_path"),
        "summary_path": event.get("summary_path"),
        "mean_throughput_mbps": metrics.get("mean_throughput_mbps", ""),
        "mean_delay_ms": metrics.get("mean_delay_ms", ""),
        "mean_jitter_ms": metrics.get("mean_jitter_ms", ""),
        "mean_packet_loss_fraction": metrics.get("mean_packet_loss", ""),
        "mean_packet_loss_percent": float(metrics["mean_packet_loss"]) * 100 if "mean_packet_loss" in metrics else "",
        "measured_qoe_proxy": event.get("measured_qoe_proxy") if event.get("measured_qoe_proxy") is not None else "",
        "target_qoe": event.get("target_qoe"),
        "predicted_qoe": event.get("predicted_qoe") if event.get("predicted_qoe") is not None else "",
        "controller_phase": event.get("controller_phase", ""),
        "predicted_qoe_lower": event.get("predicted_qoe_lower") if event.get("predicted_qoe_lower") is not None else "",
        "predicted_qoe_upper": event.get("predicted_qoe_upper") if event.get("predicted_qoe_upper") is not None else "",
        "prediction_method": event.get("prediction_method", ""),
        "prediction_domain": event.get("prediction_domain", ""),
        "prediction_run_count": event.get("prediction_run_count", ""),
        "prediction_confidence": event.get("prediction_confidence", ""),
        "predicted_target_fraction": event.get("predicted_target_fraction") if event.get("predicted_target_fraction") is not None else "",
        "predicted_outage_fraction": event.get("predicted_outage_fraction") if event.get("predicted_outage_fraction") is not None else "",
        "prediction_error": event.get("prediction_error") if event.get("prediction_error") is not None else "",
        "recovery_target_key": json.dumps(event.get("recovery_target_key")) if event.get("recovery_target_key") else "",
        "is_recovery_trial": event.get("is_recovery_trial", ""),
        "next_predicted_qoe": event.get("next_predicted_qoe") if event.get("next_predicted_qoe") is not None else "",
        "next_predicted_qoe_lower": event.get("next_predicted_qoe_lower") if event.get("next_predicted_qoe_lower") is not None else "",
        "next_predicted_qoe_upper": event.get("next_predicted_qoe_upper") if event.get("next_predicted_qoe_upper") is not None else "",
        "next_prediction_domain": event.get("next_prediction_domain", ""),
        "next_prediction_run_count": event.get("next_prediction_run_count", ""),
        "previous_valid_qoe_delta": event.get("previous_valid_qoe_delta", ""),
        "action_outcome_from_previous": event.get("action_outcome_from_previous", ""),
        "controller_action": event.get("controller_action"),
        "action_rationale": event.get("action_rationale"),
        "operating_mode": event.get("operating_mode", ""),
        "forecast_status": (event.get("forecast") or {}).get("status", ""),
        "forecast_horizon_epochs": (event.get("forecast") or {}).get("horizon_epochs", ""),
        "forecast_predicted_qoe": (event.get("forecast") or {}).get("predicted_qoe", ""),
        "forecast_lower": (event.get("forecast") or {}).get("lower", ""),
        "forecast_upper": (event.get("forecast") or {}).get("upper", ""),
        "forecast_risk_ue_ids": json.dumps((event.get("forecast") or {}).get("at_risk_ue_ids", [])),
        "forecast_reason": (event.get("forecast") or {}).get("reason", ""),
        "forecast_evaluation_json": json.dumps(event.get("forecast_evaluation")) if event.get("forecast_evaluation") else "",
        "action_application_status": event.get("action_application_status", ""),
        "action_fingerprint": event.get("action_fingerprint", ""),
        "application_error": event.get("application_error", ""),
        "verification_epoch": event.get("verification_epoch", ""),
        "predicted_qoe_before": event.get("predicted_qoe_before", ""),
        "measured_qoe_before": event.get("measured_qoe_before", ""),
        "observed_qoe_after": event.get("observed_qoe_after", ""),
        "prior_qoe_proxy": event.get("prior_qoe_proxy") if event.get("prior_qoe_proxy") is not None else "",
        "current_qoe_proxy": event.get("measured_qoe_proxy") if event.get("measured_qoe_proxy") is not None else "",
        "is_outage": (event.get("metrics") or {}).get("is_outage", ""),
        "next_decision": event.get("next_decision", event.get("controller_action")),
        "next_configuration_json": json.dumps(event["next_config"], separators=(",", ":")) if event.get("next_config") else "",
        "candidate_scope": event.get("candidate_scope", ""),
        "is_exploratory": event.get("is_exploratory", ""),
        "candidate_selection_rationale": event.get("candidate_selection_rationale", ""),
        "candidate_evidence": event.get("candidate_evidence", ""),
        "candidate_validation": event.get("candidate_validation", ""),
        "candidate_risk_level": event.get("candidate_risk_level", ""),
        "candidate_risk_reason": event.get("candidate_risk_reason", ""),
        "selected_candidate_risk_level": event.get("selected_candidate_risk_level", ""),
        "selected_candidate_risk_reason": event.get("selected_candidate_risk_reason", ""),
        "next_candidate_risk_level": event.get("next_candidate_risk_level", ""),
        "next_candidate_risk_reason": event.get("next_candidate_risk_reason", ""),
        "high_risk_override": event.get("high_risk_override", ""),
        "repeat_trials_enabled": event.get("repeat_trials_enabled", ""),
        "risk_excluded_count": event.get("risk_excluded_count", ""),
        "duplicate_excluded_count": event.get("duplicate_excluded_count", ""),
        "adaptive_action_count": event.get("adaptive_action_count", ""),
        "exploratory_trial_count": event.get("exploratory_trial_count", ""),
        "adaptive_action_budget": event.get("adaptive_action_budget", ""),
        "action_budget_exhausted": event.get("action_budget_exhausted", ""),
        "post_epoch_exploration": event.get("post_epoch_exploration", ""),
        "post_epoch_exploration_count": event.get("post_epoch_exploration_count", ""),
        "post_epoch_exploration_limit": event.get("post_epoch_exploration_limit", ""),
        "continuation_reason": event.get("continuation_reason", ""),
        "next_continuation_reason": event.get("next_continuation_reason", ""),
        "error_message": event.get("error_message", ""),
    })
    return row


def _append_history_csv(session_dir: Path, event: dict[str, Any]) -> None:
    with (session_dir / "epoch_history.csv").open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS).writerow(_history_row(event))


def _persist_updated_event(session_dir: Path, event: dict[str, Any]) -> None:
    """Replace a previously written epoch snapshot after approval/application verification."""
    key = (str(event.get("session_id", "")), str(event.get("epoch", "")))
    events_path = session_dir / "events.jsonl"
    if events_path.is_file():
        records = []
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (str(record.get("session_id", "")), str(record.get("epoch", ""))) == key:
                record = event
            records.append(record)
        temporary = events_path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
                                                 for item in records), encoding="utf-8")
        temporary.replace(events_path)

    csv_path = session_dir / "epoch_history.csv"
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if (str(row.get("session_id", "")), str(row.get("epoch", ""))) == key:
                row.update(_history_row(event))
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(csv_path)


def _terminate_owned_process(process: subprocess.Popen[Any]) -> str:
    """Terminate only the child handle created by this epoch and reap it safely."""
    if process.poll() is not None:
        return f"process had already exited ({process.returncode})"
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
        return f"terminated with SIGTERM (exit code {process.returncode})"
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
            return f"killed after SIGTERM grace period (exit code {process.returncode})"
        except subprocess.TimeoutExpired:
            return "SIGKILL sent but child did not reap within the additional 5-second wait"


def create_live_session(
    *, initial_config: dict[str, Any], target_qoe: float = 0.80, max_epochs: int = 3,
    mode: str = "adaptive", output_root: str | Path = LIVE_RESULTS,
    operating_mode: str = "Automatic",
    expansion_dir: str | Path = EXPANSION_RESULTS, binary: str | Path = DEFAULT_NS3_BINARY,
    preset_name: str = "Custom", rng_seed: int = RNG_SEED, rng_run_start: int = 1,
    timeout_seconds: float = RUN_TIMEOUT_SECONDS, max_actions: int = MAX_ACTIONS,
    allow_exploratory_candidates: bool = False,
    run_remaining_experimental_candidates: bool = False,
    max_additional_exploratory_epochs: int = 3,
    allow_high_risk_candidates: bool = False,
    allow_repeat_trials: bool = False,
    allow_topology_exploration: bool = False,
    topology_candidate_budget: int = 1,
    recovery_context: dict[str, Any] | None = None,
    target_tolerance: float = 0.005,
    stable_epochs_required: int = 2,
    maintenance_margin: float = 0.02,
    maintenance_hysteresis: float = 0.02,
    minimum_expected_benefit: float = 0.02,
    stability_validation_epochs: int = 5,
) -> tuple[str, Path, subprocess.Popen[Any]]:
    if operating_mode not in OPERATING_MODES:
        raise ValueError(f"Operating mode must be one of {OPERATING_MODES}.")
    if operating_mode == "Monitor":
        mode = "fixed"
    if not 0.0 <= float(target_qoe) <= 1.0:
        raise ValueError("Target QoE must be in [0,1]")
    if not 1 <= int(max_epochs) <= MAX_EPOCHS:
        raise ValueError(f"Maximum epochs must be in 1..{MAX_EPOCHS}")
    if mode not in {"adaptive", "fixed"}:
        raise ValueError("Mode must be adaptive or fixed")
    if mode == "fixed" and (allow_exploratory_candidates or run_remaining_experimental_candidates
                             or allow_topology_exploration):
        raise ValueError("Exploratory configurations can only be enabled in adaptive mode.")
    if not 0 <= int(max_actions) <= MAX_ACTIONS:
        raise ValueError(f"Maximum configuration actions must be in 0..{MAX_ACTIONS}.")
    if not 1 <= int(max_additional_exploratory_epochs) <= MAX_POST_EPOCH_EXPLORATORY_EPOCHS:
        raise ValueError(f"Additional experimental epoch limit must be in 1..{MAX_POST_EPOCH_EXPLORATORY_EPOCHS}.")
    if not 0.0 <= float(target_tolerance) <= 0.1 or not 1 <= int(stable_epochs_required) <= 5:
        raise ValueError("Target tolerance must be 0..0.1 and stable epoch count must be 1..5.")
    if not 0.0 <= float(maintenance_margin) <= 0.2 or not 0.0 <= float(maintenance_hysteresis) <= 0.2:
        raise ValueError("Maintenance margin and hysteresis must be in 0..0.2.")
    if not 0.0 <= float(minimum_expected_benefit) <= 0.5:
        raise ValueError("Minimum expected benefit must be in 0..0.5.")
    if not 1 <= int(stability_validation_epochs) <= 10:
        raise ValueError("Stability validation epochs must be in 1..10.")
    feasible = load_verified_configurations(expansion_dir)
    if not 1 <= int(rng_seed) <= 4_294_967_295 or not 1 <= int(rng_run_start) <= 4_294_967_295:
        raise ValueError("RNG seed and run number must be positive 32-bit integers.")
    if not 10.0 <= float(timeout_seconds) <= 300.0:
        raise ValueError("Per-epoch timeout must be between 10 and 300 seconds.")
    exact = next((c for c in feasible if config_key(c) == config_key(initial_config)
                  and float(c["bandwidth_band2_mhz"]) == float(initial_config.get("bandwidth_band2_mhz", 50))), None)
    historically_tested = bool(exact) and all(
        str(initial_config.get(key, exact.get(key))) == str(exact.get(key))
        for key in ("lambda_be_pps", "lambda_ull_pps", "packet_size_be_bytes", "packet_size_ull_bytes",
                    "scenario_size_m", "sim_time", "gnb_count", "ue_association_policy")
    )
    if exact is not None:
        normalized = dict(exact)
        normalized.update({k: v for k, v in initial_config.items() if k in {
            "lambda_be_pps", "lambda_ull_pps", "packet_size_be_bytes", "packet_size_ull_bytes",
            "scenario_size_m", "sim_time", "sim_time_seconds", "gnb_count",
            "ue_association_policy",
        }})
        normalized = validate_custom_config(normalized)
        normalized.setdefault("sim_time", exact["sim_time"])
    else:
        normalized = validate_custom_config(initial_config)
        normalized["config_id"] = "CUSTOM"
        if mode == "adaptive" and not any(c["ue_count"] == normalized["ue_count"] for c in feasible):
            raise ValueError("Adaptive mode requires a UE count with validated candidate actions.")
    if allow_topology_exploration and int(normalized.get("gnb_count", 1)) >= 3:
        raise ValueError("No higher supported gNB count remains for topology exploration from this configuration.")
    if preset_name == "Custom" and not historically_tested:
        normalized["config_id"] = "CUSTOM"
    # Adaptive candidates preserve custom traffic and scenario settings. The radio
    # tuples themselves come from the validated campaign; customized combinations
    # are explicitly recorded as exploratory in the session metadata.
    action_configs = []
    for candidate in feasible:
        if int(candidate["ue_count"]) == int(normalized["ue_count"]):
            adjusted = dict(candidate)
            for key in ("bandwidth_band2_mhz", "lambda_be_pps", "lambda_ull_pps", "packet_size_be_bytes",
                        "packet_size_ull_bytes", "scenario_size_m", "sim_time", "sim_time_seconds"):
                if key in normalized:
                    adjusted[key] = normalized[key]
            action_configs.append(adjusted)
    # The first controller release exposes one verified action only: Band 1
    # bandwidth. It is applied to the next ns-3 process, never hot-swapped.
    action_configs = [candidate for candidate in action_configs
                      if float(candidate.get("total_tx_power_dbm", 0))
                      == float(normalized.get("total_tx_power_dbm", 35))
                      and float(candidate.get("bandwidth_band1_mhz", -1))
                      in SUPPORTED_ACTIONS["bandwidth_band1_mhz"]["allowed_values_mhz"]
                      and int(candidate.get("gnb_count", 1)) == int(normalized.get("gnb_count", 1))]
    campaign_semantics_match = int(normalized.get("gnb_count", 1)) == 1 and all([
        float(normalized.get("bandwidth_band2_mhz", 50)) == 50.0,
        int(normalized.get("lambda_be_pps", 20_000)) == 20_000,
        int(normalized.get("lambda_ull_pps", 10_000)) == 10_000,
        int(normalized.get("packet_size_be_bytes", 1252)) == 1252,
        int(normalized.get("packet_size_ull_bytes", 100)) == 100,
        float(normalized.get("scenario_size_m", 30)) == 30.0,
        str(normalized.get("sim_time", "1s")) == "1s",
    ])
    candidate_performance = load_candidate_performance(expansion_dir) if campaign_semantics_match else {}
    live_candidate_performance = load_live_candidate_evidence(normalized)
    validated_action_configs = action_configs if campaign_semantics_match else []
    exploratory_configs = supported_radio_grid(normalized, feasible)
    if allow_topology_exploration and operating_mode != "Monitor":
        exploratory_configs.extend(supported_topology_candidates(
            normalized, feasible, candidate_budget=topology_candidate_budget,
            candidate_performance=candidate_performance if campaign_semantics_match else {},
        ))
    pilot_performance = load_pilot_candidate_evidence() if int(normalized.get("gnb_count", 1)) == 1 else {}
    binary_path = Path(binary).resolve()
    binary_ok, binary_message = verify_ns3_runtime(binary_path)
    if not binary_ok:
        raise RuntimeError(binary_message)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    session_dir = root / session_id
    session_dir.mkdir(exist_ok=False)
    (session_dir / "epochs").mkdir()
    with (session_dir / "epoch_history.csv").open("x", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS).writeheader()
    state = {
        "session_id": session_id, "session_dir": str(session_dir), "created_utc": utc_now(),
        "status": "QUEUED", "phase": "QUEUED", "heartbeat_utc": utc_now(),
        "preset_name": preset_name, "target_qoe": float(target_qoe), "max_epochs": int(max_epochs),
        "controller_mode": mode, "initial_config": normalized, "current_config": normalized,
        "operating_mode": operating_mode,
        "supported_action_registry": SUPPORTED_ACTIONS,
        "current_epoch": 0, "max_actions": int(max_actions),
        "allow_exploratory_candidates": bool(allow_exploratory_candidates),
        "allow_high_risk_candidates": bool(allow_high_risk_candidates),
        "allow_repeat_trials": bool(allow_repeat_trials),
        "allow_topology_exploration": bool(allow_topology_exploration),
        "topology_candidate_budget": int(topology_candidate_budget),
        "run_remaining_experimental_candidates": bool(run_remaining_experimental_candidates),
        "max_additional_exploratory_epochs": int(max_additional_exploratory_epochs),
        "target_tolerance": float(target_tolerance),
        "stable_epochs_required": int(stable_epochs_required),
        "stability_validation_epochs": int(stability_validation_epochs),
        "validation_retry_policy": "No automatic retry; start a bounded recovery session after validation failure.",
        "validation_phase": "TARGET_SEEKING",
        "validation_epochs_completed": 0,
        "validation_epochs_required": int(stability_validation_epochs),
        "maintenance_margin": float(maintenance_margin),
        "maintenance_hysteresis": float(maintenance_hysteresis),
        "minimum_expected_benefit": float(minimum_expected_benefit),
        "controller_phase": "TARGET_SEEKING",
        "post_epoch_exploration_count": 0,
        "no_improvement_tolerance": NO_IMPROVEMENT_TOLERANCE, "rng_seed": int(rng_seed),
        "rng_run_start": int(rng_run_start), "timeout_seconds": float(timeout_seconds),
        "configuration_scope": "historically tested preset" if historically_tested else "custom CLI-supported exploratory combination",
        "validated_candidate_scope": ("campaign workload/traffic semantics match" if campaign_semantics_match
                                      else "none: custom workload/scenario semantics differ from campaign"),
        "scenario_id_base": 100_000 + int(uuid.uuid4().hex[:8], 16) % 800_000_000,
        "epoch_process_policy": "one new ns-3 process per epoch; no in-process reconfiguration",
        "prediction_policy": {
            "method": "exact-condition empirical run-mean lookup; no production or experimental ML artifact loaded",
            "features": list(PREDICTION_FEATURES),
            "target": "mean flow-level synthetic qoe_proxy per run, unitless [0,1]",
            "target_formula": "ns-3 validated throughput, delay, jitter and packet-loss proxy; not subjective user QoE",
            "training_data": "22 successful ns-3 runs / 11 configurations / two repetitions in expanded experiment",
            "validation": "grouped leave-one-configuration-out RF reported MAE 0.02423 and RMSE 0.05181, but that RF is not used for control; empirical lookup has no calibrated held-out prediction guarantee",
            "uncertainty": "descriptive run-to-run variation band when at least two exact matching runs exist; uncalibrated",
            "production_model_compatible": False,
            "production_feature_contract": production_feature_contract_audit(),
            "offered_load_definition": "configured lambda × payload bytes × 8 / 1e6; no post-run TX counter is used as a predictor",
            "rng_policy": "RNG seed/run are logged and used to distinguish repetitions, but are not predictor inputs or guarantees of identical event sequences",
        },
        "expanded_experiment_evidence": str(Path(expansion_dir).resolve()),
        "binary": str(binary_path),
        "recovery_context": recovery_context,
    }
    _write_json(session_dir / "session.json", state)
    _write_json(session_dir / "session_config.json", {
        "target_qoe": target_qoe, "max_epochs": max_epochs, "max_actions": int(max_actions),
        "mode": mode, "operating_mode": operating_mode, "preset_name": preset_name,
        "initial_config": normalized, "feasible_configurations": validated_action_configs,
        "allow_exploratory_candidates": bool(allow_exploratory_candidates),
        "allow_high_risk_candidates": bool(allow_high_risk_candidates),
        "allow_repeat_trials": bool(allow_repeat_trials),
        "allow_topology_exploration": bool(allow_topology_exploration),
        "topology_candidate_budget": int(topology_candidate_budget),
        "run_remaining_experimental_candidates": bool(run_remaining_experimental_candidates),
        "max_additional_exploratory_epochs": int(max_additional_exploratory_epochs),
        "target_tolerance": float(target_tolerance),
        "stable_epochs_required": int(stable_epochs_required),
        "stability_validation_epochs": int(stability_validation_epochs),
        "maintenance_margin": float(maintenance_margin),
        "maintenance_hysteresis": float(maintenance_hysteresis),
        "minimum_expected_benefit": float(minimum_expected_benefit),
        "exploratory_configurations": exploratory_configs,
        "candidate_performance": [{"key": list(k), **v} for k, v in candidate_performance.items()],
        "pilot_candidate_performance": [{"key": list(k), **v} for k, v in pilot_performance.items()],
        "live_candidate_performance": [{"key": list(k), **v} for k, v in live_candidate_performance.items()],
        "candidate_performance_source": str(Path(expansion_dir) / "expanded_flow_dataset_reconciled.csv")
            if campaign_semantics_match else None,
        "rng_seed": int(rng_seed), "rng_run_start": int(rng_run_start),
        "rng_run_policy": "RngRun increments from explicitly selected start number per epoch",
        "timeout_seconds": float(timeout_seconds),
        "recovery_context": recovery_context,
    })
    _log(session_dir, f"Session created: preset={preset_name}; mode={mode}; target={target_qoe:.4f}; "
         f"normal_epochs={max_epochs}; post_epoch_exploration={bool(run_remaining_experimental_candidates)}; "
         f"additional_epoch_limit={max_additional_exploratory_epochs}.")
    log = (session_dir / "worker.log").open("x", encoding="utf-8")
    try:
        worker = subprocess.Popen(
            [sys.executable, "-m", "simulation.live_worker", "--session-dir", str(session_dir)],
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        (session_dir / "worker.pid").write_text(str(worker.pid) + "\n", encoding="utf-8")
    except Exception as exc:
        if "worker" in locals() and worker.poll() is None:
            _terminate_owned_process(worker)
        state.update({"status": "FAILED", "phase": "WORKER_START_FAILED", "ended_utc": utc_now(),
                      "error_type": type(exc).__name__, "error_message": str(exc),
                      "final_outcome": f"Could not start controller worker: {type(exc).__name__}: {exc}"})
        _write_json(session_dir / "session.json", state)
        _log(session_dir, f"Worker launch failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        log.close()
    return session_id, session_dir, worker


def request_session_stop(session_dir: str | Path) -> None:
    folder = Path(session_dir)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    (folder / "STOP_REQUESTED").write_text(utc_now() + "\n", encoding="utf-8")
    _log(folder, "User requested stop; active ns-3 child will be terminated if running.")


def request_action_approval(session_dir: str | Path, *, approve: bool) -> dict[str, Any]:
    """Approve or reject only the currently persisted recommendation fingerprint."""
    folder = Path(session_dir)
    state = read_session_state(folder)
    pending = state.get("pending_action") or {}
    if state.get("status") != "WAITING_FOR_APPROVAL" or not pending.get("fingerprint"):
        raise ValueError("There is no pending configuration action to approve or reject.")
    marker = folder / f"action_decision_epoch_{int(pending['epoch'])}.json"
    if marker.exists():
        raise ValueError("A decision has already been recorded for this recommendation.")
    decision = {"approve": bool(approve), "fingerprint": pending["fingerprint"],
                "timestamp_utc": utc_now(), "reviewed_by": "dashboard user"}
    _write_json(marker, decision)
    _append_jsonl(folder / "interventions.jsonl", {
        "timestamp_utc": decision["timestamp_utc"], "session_id": state["session_id"],
        "epoch": pending["epoch"], "status": "APPROVED" if approve else "REJECTED",
        "fingerprint": pending["fingerprint"], "action": pending["action"],
        "previous_config": pending["previous_config"], "proposed_config": pending["proposed_config"],
        "rationale": pending["rationale"],
    })
    return decision


def read_intervention_events(session_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(session_dir) / "interventions.jsonl"
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def _wait_for_recommendation(folder: Path, state: dict[str, Any], event: dict[str, Any],
                             decision: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Pause Recommend mode until the UI approves this exact configuration."""
    candidate = decision.get("next_config")
    if not candidate:
        return False, decision
    proposal = {
        "epoch": int(event["epoch"]), "action": "Band 1 bandwidth restart",
        "previous_config": event.get("config"), "proposed_config": candidate,
        "fingerprint": configuration_fingerprint(candidate),
        "rationale": decision.get("rationale", ""),
        "predicted_qoe": decision.get("predicted_qoe"),
        "forecast": event.get("forecast"),
    }
    state.update({"status": "WAITING_FOR_APPROVAL", "phase": "WAITING_FOR_APPROVAL",
                  "pending_action": proposal, "heartbeat_utc": utc_now()})
    _write_json(folder / "session.json", state)
    marker = folder / f"action_decision_epoch_{int(event['epoch'])}.json"
    _log(folder, f"Recommendation pending user approval: {proposal['fingerprint']}")
    while True:
        if (folder / "STOP_REQUESTED").exists():
            _append_jsonl(folder / "interventions.jsonl", {
                "timestamp_utc": utc_now(), "session_id": state["session_id"],
                "epoch": event["epoch"], "status": "CANCELLED_BEFORE_APPLICATION",
                "fingerprint": proposal["fingerprint"],
                "reason": "User stopped session while approval was pending.",
            })
            return False, decision
        if marker.is_file():
            try:
                answer = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                answer = {}
            if answer.get("fingerprint") != proposal["fingerprint"]:
                raise ValueError("Approval fingerprint does not match the persisted recommendation.")
            if answer.get("approve") is True:
                _log(folder, "Recommendation approved; next epoch will apply it through a new ns-3 process.")
                event["action_application_status"] = "APPROVED_RESTART"
                state.update({"status": "RUNNING", "phase": "ACTION_APPROVED",
                              "pending_action": None, "heartbeat_utc": utc_now()})
                _write_json(folder / "session.json", state)
                return True, decision
            _log(folder, "Recommendation rejected; next epoch retains the current configuration.")
            event["action_application_status"] = "REJECTED"
            decision = dict(decision)
            decision["next_config"] = event["config"]
            decision["action"] = "CONTINUE_FIXED"
            decision["rationale"] = "User rejected the recommendation; retain the current configuration."
            state.update({"status": "RUNNING", "phase": "ACTION_REJECTED",
                          "pending_action": None, "heartbeat_utc": utc_now()})
            _write_json(folder / "session.json", state)
            return True, decision
        state["heartbeat_utc"] = utc_now()
        _write_json(folder / "session.json", state)
        time.sleep(0.5)


def read_session_state(session_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(session_dir) / "session.json").read_text(encoding="utf-8"))


def read_session_events(session_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(session_dir) / "events.jsonl"
    if not path.is_file():
        return []
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                if line.strip():
                    events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _update_session_progress(folder: Path, update: dict[str, Any]) -> None:
    state = read_session_state(folder)
    state.update(update)
    state["heartbeat_utc"] = utc_now()
    _write_json(folder / "session.json", state)


def _mark_worker_failure(folder: Path, exc: BaseException) -> None:
    try:
        state = read_session_state(folder)
        state.update({"status": "FAILED", "phase": "FAILED", "ended_utc": utc_now(),
                      "error_type": type(exc).__name__, "error_message": str(exc),
                      "final_outcome": f"Controller worker failed: {type(exc).__name__}: {exc}",
                      "heartbeat_utc": utc_now()})
        _write_json(folder / "session.json", state)
        _log(folder, f"Worker failed: {type(exc).__name__}: {exc}")
        (folder / "worker_exception.log").write_text(traceback.format_exc(), encoding="utf-8")
    except Exception:
        # The worker's redirected worker.log still retains the original traceback.
        pass


def mark_worker_lost(session_dir: str | Path) -> dict[str, Any]:
    """Persist a visible failure if the recorded worker has exited unexpectedly."""
    folder = Path(session_dir)
    state = read_session_state(folder)
    if state.get("status") not in {"RUNNING", "QUEUED", "WAITING_FOR_APPROVAL"}:
        return state
    pid = state.get("worker_pid")
    if not pid and (folder / "worker.pid").is_file():
        try:
            pid = int((folder / "worker.pid").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = None
    if not pid:
        return state
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        state.update({"status": "FAILED", "phase": "WORKER_EXITED", "ended_utc": utc_now(),
                      "error_type": "WorkerExited", "error_message": "Controller worker exited before finalizing the session.",
                      "final_outcome": "Worker exited unexpectedly; inspect worker.log and worker_exception.log.",
                      "heartbeat_utc": utc_now()})
        _write_json(folder / "session.json", state)
        _log(folder, "Dashboard detected that the controller worker exited while the session was active.")
    except PermissionError:
        state["worker_liveness"] = "unknown: permission denied while checking worker PID"
    return state


def list_live_sessions(output_root: str | Path = LIVE_RESULTS) -> list[Path]:
    root = Path(output_root)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and (p / "session.json").is_file()), reverse=True)


def build_epoch_command(
    config: dict[str, Any], *, binary: str | Path, run_dir: str | Path,
    scenario_id: int, rng_run: int, rng_seed: int = RNG_SEED,
) -> list[str]:
    """Use the existing runner's argument builder, adding explicit supported globals/settings."""
    base_config = {
        "ue_count": config["ue_count"], "bandwidth_mhz": config["bandwidth_band1_mhz"],
        "gnb_count": int(config.get("gnb_count", 1)),
        "ue_association_policy": str(config.get("ue_association_policy", "closest")),
        "total_tx_power_dbm": config["total_tx_power_dbm"], "scenario_size_m": config["scenario_size_m"],
        "lambda_be": config["lambda_be_pps"], "scenario_id": scenario_id, "sim_time": config["sim_time"],
    }
    runner = Ns3Runner(executable_path=binary, default_output_base=Path(run_dir).parent)
    args = runner.build_cli_arguments(base_config, Path(run_dir), sim_tag="sim_summary.txt")
    args.extend([
        "--doubleOperationalBand=true",
        f"--bandwidthBand2={int(config['bandwidth_band2_mhz'] * 1_000_000)}",
        f"--lambdaUll={config['lambda_ull_pps']}",
        f"--packetSizeUll={config['packet_size_ull_bytes']}",
        f"--packetSizeBe={config['packet_size_be_bytes']}",
        "--numerologyBwp1=4", "--numerologyBwp2=2",
        "--centralFrequencyBand1=28000000000", "--centralFrequencyBand2=28200000000",
        f"--RngSeed={rng_seed}", f"--RngRun={rng_run}",
    ])
    return args


def _run_epoch(
    session_dir: Path, config: dict[str, Any], *, epoch: int, scenario_id: int,
    rng_seed: int, rng_run: int, timeout_seconds: float, binary: str,
    progress_callback: Any = None, preset_name: str = "Custom", configuration_scope: str = "",
    post_epoch_exploration: bool = False, validation_epoch: bool = False,
) -> dict[str, Any]:
    run_id = f"epoch_{epoch:02d}_{config['config_id'].lower().replace('-', '_')}_rng{rng_run}"
    run_dir = session_dir / "epochs" / run_id
    run_dir.mkdir(exist_ok=False)
    command = build_epoch_command(config, binary=binary, run_dir=run_dir, scenario_id=scenario_id,
                                  rng_run=rng_run, rng_seed=rng_seed)
    p1, p2 = derive_effective_band_powers_dbm(
        config["total_tx_power_dbm"], config["bandwidth_band1_mhz"], config["bandwidth_band2_mhz"]
    )
    record: dict[str, Any] = {
        "session_id": session_dir.name, "epoch": epoch, "run_id": run_id,
        "scenario_id": scenario_id, "config_id": config["config_id"], "config": config,
        "preset_name": preset_name, "configuration_scope": configuration_scope,
        "post_epoch_exploration": bool(post_epoch_exploration),
        "is_validation_epoch": bool(validation_epoch),
        "rng_seed": rng_seed, "rng_run": rng_run, "command": command, "start_utc": utc_now(),
        "run_status": "STARTING", "phase": "STARTING", "exit_code": None,
        "raw_csv_path": str(run_dir / "qoe_dataset.csv"),
        "ue_measurements_path": str(run_dir / "ue_measurements.csv"),
        "ue_measurements_status": "UNAVAILABLE",
        "stdout_path": str(run_dir / "stdout.log"),
        "stderr_path": str(run_dir / "stderr.log"), "summary_path": str(run_dir / "sim_summary.txt"),
        "effective_band1_tx_power_dbm": p1, "effective_band2_tx_power_dbm": p2, "predicted_qoe": None,
    }
    _write_json(run_dir / "run_config.json", {**record, "timeout_seconds": timeout_seconds})
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    try:
        if progress_callback:
            progress_callback({"phase": "STARTING", "current_run_id": run_id,
                               "current_config": config, "current_epoch_started_utc": record["start_utc"],
                               "current_stdout_path": record["stdout_path"], "current_stderr_path": record["stderr_path"],
                               "current_output_dir": str(run_dir), "current_elapsed_seconds": 0.0})
        with Path(record["stdout_path"]).open("x", encoding="utf-8") as stdout, Path(record["stderr_path"]).open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=run_dir, stdout=stdout, stderr=stderr, text=True)
            record.update({"run_status": "RUNNING", "pid": getattr(process, "pid", None)})
            if progress_callback:
                progress_callback({"phase": "RUNNING", "simulator_pid": process.pid})
            deadline = time.monotonic() + timeout_seconds
            last_progress = 0.0
            while process.poll() is None:
                if (session_dir / "STOP_REQUESTED").exists():
                    termination_result = _terminate_owned_process(process)
                    record.update({"run_status": "STOPPED", "exit_code": process.returncode,
                                   "termination_result": termination_result,
                                   "process_terminated": process.poll() is not None,
                                   "error_message": "User requested stop; active ns-3 process terminated."})
                    break
                if time.monotonic() >= deadline:
                    termination_result = _terminate_owned_process(process)
                    record.update({"run_status": "TIMEOUT", "exit_code": process.returncode,
                                   "timed_out": True, "timeout_stage": "ns-3 process execution",
                                   "termination_result": termination_result,
                                   "process_terminated": process.poll() is not None,
                                   "error_message": f"ns-3 process exceeded the {timeout_seconds:.1f}s execution limit."})
                    break
                elapsed = time.monotonic() - started
                if progress_callback and elapsed - last_progress >= 1.0:
                    progress_callback({"phase": "RUNNING", "current_elapsed_seconds": elapsed})
                    last_progress = elapsed
                time.sleep(0.2)
            else:
                record["exit_code"] = process.returncode
            if record["run_status"] == "RUNNING" and process.returncode == 0:
                record["run_status"] = "PARSING_RESULTS"
                if progress_callback:
                    progress_callback({"phase": "PARSING_RESULTS", "current_elapsed_seconds": time.monotonic() - started})
                csv_path = Path(record["raw_csv_path"])
                if not Path(record["summary_path"]).is_file():
                    record["run_status"] = "MISSING_ARTIFACT"
                    raise RuntimeError("ns-3 exited zero but sim_summary.txt is missing")
                if not csv_path.is_file():
                    record["run_status"] = "MISSING_ARTIFACT"
                    raise FileNotFoundError(f"ns-3 exited zero but qoe_dataset.csv is missing: {csv_path}")
                valid, error = validate_simulation_csv(
                    csv_path,
                    expected_config={
                        "scenario_id": scenario_id, "ue_count": config["ue_count"],
                        "bandwidth_mhz": config["bandwidth_band1_mhz"],
                        "total_tx_power_dbm": config["total_tx_power_dbm"],
                        "scenario_size_m": config["scenario_size_m"], "lambda_be": config["lambda_be_pps"],
                    },
                    expected_flow_count=config["ue_count"],
                )
                if not valid:
                    record["run_status"] = "INVALID_OUTPUT"
                    raise ValueError(error or "CSV validation failed")
                try:
                    metrics = parse_simulation_csv(csv_path, target_flow_type="all")
                except Exception as exc:
                    record.update({"run_status": "PARSER_ERROR", "error_type": type(exc).__name__})
                    raise
                if not metrics.success or metrics.target_flow_count != config["ue_count"]:
                    record["run_status"] = "PARSER_ERROR"
                    raise ValueError(metrics.error_message or "Parser returned invalid measurements")
                result_metrics = {
                    "mean_throughput_mbps": metrics.mean_throughput_mbps,
                    "mean_delay_ms": metrics.mean_delay_ms, "mean_jitter_ms": metrics.mean_jitter_ms,
                    "mean_packet_loss": metrics.mean_packet_loss, "mean_qoe_proxy": metrics.mean_qoe_proxy,
                    "is_outage": metrics.is_outage, "target_flow_count": metrics.target_flow_count,
                    "total_tx_packets": metrics.total_tx_packets, "total_rx_packets": metrics.total_rx_packets,
                }
                ue_csv = Path(record["ue_measurements_path"])
                if ue_csv.is_file():
                    try:
                        per_ue = parse_ue_measurements(
                            ue_csv, expected_ue_count=int(config["ue_count"]),
                            expected_gnb_count=int(config.get("gnb_count", 1)),
                        )
                        record.update({"ue_measurements": per_ue, "ue_measurements_status": "VALIDATED"})
                    except (OSError, ValueError) as exc:
                        record.update({"ue_measurements": [], "ue_measurements_status": "INVALID",
                                       "ue_measurements_error": f"{type(exc).__name__}: {exc}"})
                else:
                    record.update({"ue_measurements": [], "ue_measurements_status": "UNAVAILABLE_OLD_BINARY"})
                if not all(math.isfinite(float(result_metrics[k])) for k in (
                    "mean_throughput_mbps", "mean_delay_ms", "mean_jitter_ms", "mean_packet_loss", "mean_qoe_proxy"
                )):
                    raise ValueError("Parser returned non-finite metrics")
                record.update({"run_status": "COMPLETED", "metrics": result_metrics,
                               "measured_qoe_proxy": metrics.mean_qoe_proxy})
                if progress_callback:
                    progress_callback({"phase": "COMPLETED", "current_elapsed_seconds": time.monotonic() - started})
            elif record["run_status"] == "RUNNING":
                record.update({"run_status": "FAILED", "exit_code": process.returncode,
                               "error_type": "ProcessExit",
                               "error_message": f"ns-3 process exited with status {process.returncode}."})
    except Exception as exc:
        if process is not None and process.poll() is None:
            record["termination_result"] = _terminate_owned_process(process)
            record["process_terminated"] = process.poll() is not None
        if record.get("run_status") not in {"MISSING_ARTIFACT", "INVALID_OUTPUT", "PARSER_ERROR"}:
            record["run_status"] = "EXECUTION_ERROR"
        record.update({"error_type": type(exc).__name__, "error_message": f"{type(exc).__name__}: {exc}"})
    elapsed_seconds = time.monotonic() - started
    if record.get("run_status") == "TIMEOUT":
        def tail(path: str) -> str:
            try:
                value = Path(path).read_text(encoding="utf-8", errors="replace").strip()
            except OSError as exc:
                return f"unavailable ({type(exc).__name__}: {exc})"
            return value[-600:] if value else "empty"

        record["error_message"] = (
            f"ns-3 process execution timed out after {elapsed_seconds:.1f}s "
            f"(configured limit {timeout_seconds:.1f}s). Owned child cleanup: "
            f"{record.get('termination_result', 'unknown')}; process reaped="
            f"{bool(record.get('process_terminated'))}. stdout: {record['stdout_path']} "
            f"(tail: {tail(record['stdout_path'])}); stderr: {record['stderr_path']} "
            f"(tail: {tail(record['stderr_path'])}). Partial outputs are retained but are not measurements."
        )
    record["duration_seconds"] = elapsed_seconds
    record["end_utc"] = utc_now()
    _write_json(run_dir / "run_config.json", {**record, "timeout_seconds": timeout_seconds})
    return record


def run_live_session(session_dir: str | Path, timeout_seconds: float = RUN_TIMEOUT_SECONDS) -> None:
    """Run one session worker, refusing duplicate scheduling for the same session."""
    folder = Path(session_dir).resolve()
    lock_path = folder / ".controller_worker.lock"
    with lock_path.open("a+", encoding="utf-8") as worker_lock:
        try:
            fcntl.flock(worker_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _log(folder, "Duplicate controller worker rejected; this session already owns its worker lock.")
            return
        try:
            state = read_session_state(folder)
            if state.get("status") != "QUEUED":
                _log(folder, f"Duplicate or stale worker rejected for session status {state.get('status')}.")
                return
            _run_live_session_locked(folder, timeout_seconds)
        finally:
            fcntl.flock(worker_lock.fileno(), fcntl.LOCK_UN)


def _run_live_session_locked(session_dir: str | Path, timeout_seconds: float = RUN_TIMEOUT_SECONDS) -> None:
    folder = Path(session_dir).resolve()
    state = read_session_state(folder)
    saved_config = json.loads((folder / "session_config.json").read_text(encoding="utf-8"))
    feasible = saved_config["feasible_configurations"]
    candidate_performance = {
        tuple([int(row["key"][0]), float(row["key"][1]), float(row["key"][2])]): row
        for row in saved_config.get("candidate_performance", [])
    }
    pilot_performance = {
        (int(row["key"][0]), float(row["key"][1]), float(row["key"][2])): row
        for row in saved_config.get("pilot_candidate_performance", [])
    }
    live_performance = {
        (int(row["key"][0]), float(row["key"][1]), float(row["key"][2])): row
        for row in saved_config.get("live_candidate_performance", [])
    }
    exploratory_configs = saved_config.get("exploratory_configurations", [])
    run_remaining_experimental = bool(saved_config.get("run_remaining_experimental_candidates", False))
    extra_epoch_limit = int(saved_config.get("max_additional_exploratory_epochs", 3))
    normal_epoch_limit = int(state["max_epochs"])
    validation_epoch_limit = int(state.get("stability_validation_epochs", 5))
    total_epoch_limit = (normal_epoch_limit
                         + (extra_epoch_limit if run_remaining_experimental else 0)
                         + validation_epoch_limit)
    history: list[dict[str, Any]] = []
    current = state["initial_config"]
    initial_risk = assess_candidate_risk(config_key(current), campaign=candidate_performance,
                                         pilot=pilot_performance, live=live_performance)
    initial_prediction = empirical_qoe_prediction(current, target_qoe=float(state["target_qoe"]),
                                                    campaign=candidate_performance,
                                                    live=live_performance)
    validated_keys = {config_key(config) for config in feasible}
    current_is_validated = config_key(current) in validated_keys and bool(state.get("validated_candidate_scope", "").startswith("campaign"))
    current_selection = {
        "candidate_scope": "validated campaign configuration" if current_is_validated else "exploratory/user-selected configuration",
        "is_exploratory": not current_is_validated,
        "candidate_selection_rationale": "Initial configuration selected by the user; no controller action is inferred.",
        "candidate_evidence": "campaign condition" if current_is_validated else "not validated under the campaign condition",
        "candidate_validation": "PASS: custom configuration whitelist and scenario invariants checked at session start",
        "candidate_risk_level": initial_risk["level"],
        "candidate_risk_reason": initial_risk["reason"],
        "selected_candidate_risk_level": initial_risk["level"],
        "selected_candidate_risk_reason": initial_risk["reason"],
        "controller_phase": "TARGET_SEEKING",
        "predicted_qoe": initial_prediction.get("qoe"),
        "predicted_qoe_lower": initial_prediction.get("lower"),
        "predicted_qoe_upper": initial_prediction.get("upper"),
        "prediction_method": initial_prediction.get("method"),
        "prediction_domain": initial_prediction.get("domain"),
        "prediction_run_count": initial_prediction.get("run_count", 0),
        "predicted_target_fraction": initial_prediction.get("target_fraction"),
        "predicted_outage_fraction": initial_prediction.get("outage_fraction"),
        "prediction_confidence": initial_prediction.get("confidence"),
        "post_epoch_exploration": False,
        "continuation_reason": "",
    }
    timeout_seconds = float(state.get("timeout_seconds", timeout_seconds))
    state.update({"status": "QUEUED", "phase": "QUEUED", "worker_pid": os.getpid(),
                  "heartbeat_utc": utc_now(), "started_utc": utc_now(), "current_epoch": 0,
                  "current_config": current, "next_epoch_restarts_ns3": True})
    _write_json(folder / "session.json", state)
    _log(folder, "Controller worker started; waiting for the serialized ns-3 execution slot.")
    lock_path = folder.parent / ".ns3_live_execution.lock"
    with lock_path.open("a+", encoding="utf-8") as execution_lock:
        while True:
            if (folder / "STOP_REQUESTED").exists():
                state.update({"status": "USER_CANCELLED", "phase": "CANCELLED", "ended_utc": utc_now(),
                              "final_outcome": "Session cancelled while queued; no ns-3 process was launched."})
                _write_json(folder / "session.json", state)
                _log(folder, "Session cancelled while waiting for the execution slot.")
                return
            try:
                fcntl.flock(execution_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                state.update({"status": "QUEUED", "phase": "QUEUED", "queue_reason": "Another session owns the ns-3 execution slot.",
                              "heartbeat_utc": utc_now()})
                _write_json(folder / "session.json", state)
                time.sleep(0.5)
        try:
            state.update({"status": "RUNNING", "phase": "PREPARING_EPOCH", "queue_reason": None,
                          "heartbeat_utc": utc_now()})
            _write_json(folder / "session.json", state)
            _log(folder, "Controller acquired the ns-3 execution slot; each epoch is a separate process.")
            for epoch in range(1, total_epoch_limit + 1):
                validation_active = bool(state.get("validation_active", False))
                slot_category = ("validation" if validation_active else run_slot_category(
                    epoch, normal_limit=normal_epoch_limit,
                    extra_enabled=run_remaining_experimental,
                    extra_limit=extra_epoch_limit,
                ))
                if slot_category is None:
                    final_status = "EXPLORATORY_EPOCH_LIMIT_REACHED" if run_remaining_experimental else "EPOCH_LIMIT_REACHED"
                    reason = (f"No run slot remains: {normal_epoch_limit} normal and "
                              f"{extra_epoch_limit if run_remaining_experimental else 0} additional experimental run(s) allowed.")
                    state.update({"status": final_status, "phase": "COMPLETED", "controller_phase": "STOPPED",
                                  "ended_utc": utc_now(), "final_outcome": reason})
                    _write_json(folder / "session.json", state)
                    _log(folder, f"Launch rejected before ns-3 process start: {reason}")
                    return
                if (folder / "STOP_REQUESTED").exists():
                    state.update({"status": "CANCELLED", "phase": "CANCELLED", "ended_utc": utc_now(),
                                  "final_outcome": "User cancelled the session."})
                    _write_json(folder / "session.json", state)
                    return
                is_post_epoch_trial = slot_category == "experimental"
                is_validation_epoch = slot_category == "validation"
                if is_validation_epoch:
                    current = state["validation_config"]
                state.update({"status": "RUNNING",
                              "phase": ("STABILITY_VALIDATION" if is_validation_epoch else
                                        "POST_EPOCH_EXPLORATION" if is_post_epoch_trial else "STARTING"),
                              "epoch_state": "STARTING",
                              "current_epoch": epoch, "current_config": current,
                              "is_post_epoch_exploration": is_post_epoch_trial,
                              "rng_run": int(state.get("rng_run_start", 1)) + epoch - 1,
                              "heartbeat_utc": utc_now()})
                _write_json(folder / "session.json", state)
                event = _run_epoch(
                    folder, current, epoch=epoch, scenario_id=int(state["scenario_id_base"]) + epoch - 1,
                    rng_seed=int(state["rng_seed"]), rng_run=int(state.get("rng_run_start", 1)) + epoch - 1,
                    timeout_seconds=timeout_seconds, binary=state["binary"],
                    preset_name=state.get("preset_name", "Custom"),
                    configuration_scope=current_selection["candidate_scope"],
                    post_epoch_exploration=is_post_epoch_trial,
                    validation_epoch=is_validation_epoch,
                    progress_callback=lambda progress: _update_session_progress(folder, {
                        "status": "RUNNING", "epoch_state": progress.get("phase", state.get("epoch_state")), **progress,
                    }),
                )
                event.update(current_selection)
                event.update({"controller_mode": state["controller_mode"],
                              "operating_mode": state.get("operating_mode", "Automatic"),
                              "target_qoe": float(state["target_qoe"]),
                              "configuration_fingerprint": configuration_fingerprint(current),
                              "controller_phase": state.get("controller_phase", "TARGET_SEEKING"),
                              "post_epoch_exploration": is_post_epoch_trial,
                              "is_validation_epoch": is_validation_epoch,
                              "validation_sequence": state.get("validation_sequence", 0) if is_validation_epoch else "",
                              "target_range_pass": (target_range_status(
                                  float(event["measured_qoe_proxy"]), float(state["target_qoe"]),
                                  float(state.get("target_tolerance", 0.005)))
                                  if event.get("measured_qoe_proxy") is not None else False)})
                if history:
                    prior_action = history[-1]
                    application_status = prior_action.get("action_application_status")
                    expected_config = prior_action.get("next_config")
                    if (application_status in {"SCHEDULED_RESTART", "APPROVED_RESTART"}
                            and expected_config
                            and configuration_fingerprint(expected_config) == configuration_fingerprint(current)):
                        verified = (event.get("run_status") == "COMPLETED"
                                    and event.get("configuration_fingerprint") == configuration_fingerprint(expected_config))
                        event["action_application_status"] = "APPLIED_AND_VERIFIED" if verified else "APPLICATION_FAILED"
                        prior_action["action_application_status"] = (
                            "APPLIED_AND_VERIFIED" if verified else "APPLICATION_FAILED"
                        )
                        prior_action["application_error"] = None if verified else event.get("error_message")
                        prior_action["verification_epoch"] = epoch
                        prior_action["observed_qoe_after"] = (
                            event.get("measured_qoe_proxy") if verified else None
                        )
                        predicted_after = prior_action.get("next_predicted_qoe")
                        observed_after = event.get("measured_qoe_proxy") if verified else None
                        _append_jsonl(folder / "interventions.jsonl", {
                            "timestamp_utc": utc_now(), "session_id": state["session_id"],
                            "epoch": prior_action.get("epoch"),
                            "status": "VERIFIED" if verified else "APPLICATION_FAILED",
                            "action": "Band 1 bandwidth restart",
                            "previous_config": prior_action.get("config"),
                            "new_config": current,
                            "configuration_fingerprint": event.get("configuration_fingerprint"),
                            "predicted_qoe_after": predicted_after,
                            "observed_qoe_after": observed_after,
                            "prediction_error": (abs(float(observed_after) - float(predicted_after))
                                                 if observed_after is not None and predicted_after is not None else None),
                            "measured_metrics": event.get("metrics") if verified else None,
                            "verification": "successful process, matching configuration fingerprint, validated simulator CSV"
                                if verified else event.get("error_message"),
                        })
                        _persist_updated_event(folder, prior_action)
                event_history = history + [event]
                event.update({
                    "adaptive_action_count": sum(
                        item.get("controller_action") == "CHANGE_CONFIGURATION"
                        and not item.get("is_exploratory", False) for item in history
                    ),
                    "exploratory_trial_count": sum(bool(item.get("is_exploratory")) for item in event_history),
                    "adaptive_action_budget": int(state["max_actions"]),
                    "action_budget_exhausted": sum(
                        item.get("controller_action") == "CHANGE_CONFIGURATION"
                        and not item.get("is_exploratory", False) for item in history
                    ) >= int(state["max_actions"]),
                    "post_epoch_exploration_count": sum(
                        bool(item.get("post_epoch_exploration")) for item in event_history
                    ),
                    "post_epoch_exploration_limit": extra_epoch_limit if run_remaining_experimental else "",
                })
                if history and event.get("measured_qoe_proxy") is not None and history[-1].get("measured_qoe_proxy") is not None:
                    delta = float(event["measured_qoe_proxy"]) - float(history[-1]["measured_qoe_proxy"])
                    event["previous_valid_qoe_delta"] = delta
                    if history[-1].get("controller_action") in {
                        "CHANGE_CONFIGURATION", "CONTINUE_POST_EPOCH_EXPLORATION",
                        "CONTINUE_STABILITY", "CONTINUE_MAINTENANCE", "CONTINUE_FIXED",
                    }:
                        tol = float(state["no_improvement_tolerance"])
                        event["action_outcome_from_previous"] = classify_action_outcome(delta, tol)
                prior_qoe = history[-1].get("measured_qoe_proxy") if history else None
                event.update({
                    "prior_qoe_proxy": prior_qoe,
                    "current_qoe_proxy": event.get("measured_qoe_proxy"),
                    "is_outage": bool((event.get("metrics") or {}).get("is_outage", False)),
                })
                if history and event.get("run_status") == "COMPLETED":
                    evaluation = evaluate_next_forecast(history[-1], event)
                    if evaluation:
                        event["forecast_evaluation"] = evaluation
                forecast = forecast_qoe_degradation(
                    event_history, target_qoe=float(state["target_qoe"]), horizon_epochs=1,
                ) if event.get("run_status") == "COMPLETED" else {
                    "status": "UNAVAILABLE", "reason": "No valid measurement from this epoch.",
                    "predicted_qoe": None, "at_risk_ue_ids": [],
                }
                event["forecast"] = forecast
                event["forecast_for_next"] = ({**forecast, "target_epoch": epoch + 1}
                                                if forecast.get("predicted_qoe") is not None else None)
                event["action_application_status"] = "NO_ACTION"
                state.update({"latest_forecast": forecast,
                              "latest_forecast_error": event.get("forecast_evaluation"),
                              "heartbeat_utc": utc_now()})
                predicted = event.get("predicted_qoe")
                if predicted is not None and event.get("run_status") == "COMPLETED" \
                        and event.get("measured_qoe_proxy") is not None:
                    event["prediction_error"] = float(event["measured_qoe_proxy"]) - float(predicted)
                if event["run_status"] == "STOPPED":
                    event.update({"run_status": "CANCELLED", "worker_state": "CANCELLED", "controller_action": "USER_CANCELLED",
                                  "next_decision": "USER_CANCELLED", "action_rationale": event.get("error_message")})
                    _append_jsonl(folder / "events.jsonl", event)
                    _append_history_csv(folder, event)
                    state.update({"status": "USER_CANCELLED", "phase": "CANCELLED", "ended_utc": utc_now(),
                                  "final_outcome": "User cancelled active ns-3 epoch; no measurement was added.",
                                  "post_epoch_exploration_count": event.get("post_epoch_exploration_count", 0)})
                    _write_json(folder / "session.json", state)
                    _log(folder, f"Epoch {epoch} cancelled by user.")
                    return
                if event["run_status"] != "COMPLETED":
                    failure_decision = "SIMULATION_TIMEOUT" if event["run_status"] == "TIMEOUT" else "SIMULATION_FAILURE"
                    event.update({"worker_state": event["run_status"], "controller_action": "STOP_FAILED_EPOCH",
                                  "next_decision": failure_decision, "action_rationale": event.get("error_message")})
                    _append_jsonl(folder / "events.jsonl", event)
                    _append_history_csv(folder, event)
                    state.update({"status": "SIMULATION_FAILURE" if event["run_status"] != "TIMEOUT" else "SIMULATION_TIMEOUT",
                                  "phase": event["run_status"], "epoch_state": event["run_status"],
                                  "ended_utc": utc_now(), "final_outcome": event.get("error_message"),
                                  "latest_measured_qoe_proxy": None, "error_type": event.get("error_type"),
                                  "post_epoch_exploration_count": event.get("post_epoch_exploration_count", 0)})
                    _write_json(folder / "session.json", state)
                    _log(folder, f"Epoch {epoch} {event['run_status']}: {event.get('error_message')}")
                    return
                if is_validation_epoch:
                    sequence = int(state.get("validation_sequence", 0))
                    fingerprint = configuration_fingerprint(current)
                    passed, validation_reason = stability_epoch_passes(
                        event, target_qoe=float(state["target_qoe"]),
                        tolerance=float(state["target_tolerance"]),
                    )
                    if fingerprint != state.get("validation_fingerprint"):
                        passed = False
                        validation_reason = "Frozen validation configuration fingerprint changed."
                    completed = int(state.get("validation_epochs_completed", 0)) + 1
                    event.update({
                        "controller_phase": "STABILITY_VALIDATION",
                        "validation_pass": passed,
                        "validation_progress": f"{completed}/{validation_epoch_limit}",
                        "controller_action": "VALIDATION_EPOCH_PASSED" if passed else "VALIDATION_EPOCH_FAILED",
                        "action_rationale": validation_reason,
                        "next_decision": "CONTINUE_STABILITY_VALIDATION" if passed else "VALIDATION_FAILED",
                        "decision_record": {
                            "epoch": epoch, "timestamp_utc": event.get("end_utc"),
                            "decision": "CONTINUE_STABILITY_VALIDATION" if passed else "VALIDATION_FAILED",
                            "phase": "STABILITY_VALIDATION", "config_id": current.get("config_id"),
                            "configuration_fingerprint": fingerprint,
                            "expected_effect": "Not estimated.",
                            "measured_qoe_proxy": event.get("measured_qoe_proxy"),
                            "target_qoe": state["target_qoe"], "tolerance": state["target_tolerance"],
                            "validation_progress": f"{completed}/{validation_epoch_limit}",
                            "reason": validation_reason,
                        },
                    })
                    _append_jsonl(folder / "events.jsonl", event)
                    _append_history_csv(folder, event)
                    history.append(event)
                    state.update({"validation_epochs_completed": completed if passed else 0,
                                  "validation_last_epoch": epoch,
                                  "latest_measured_qoe_proxy": event.get("measured_qoe_proxy"),
                                  "latest_action": event["controller_action"],
                                  "latest_rationale": validation_reason,
                                  "heartbeat_utc": utc_now()})
                    if not passed:
                        state.update({"validation_active": False, "validation_phase": "VALIDATION_FAILED",
                                      "status": "VALIDATION_FAILED", "phase": "COMPLETED",
                                      "controller_phase": "STOPPED", "ended_utc": utc_now(),
                                      "final_outcome": (f"Stability validation failed at epoch {epoch}: "
                                                        f"{validation_reason} Automatic retry budget is zero; "
                                                        "start a bounded recovery session to try another candidate.")})
                        _write_json(folder / "session.json", state)
                        _log(folder, state["final_outcome"])
                        return
                    if completed >= validation_epoch_limit:
                        state.update({"validation_active": False, "validation_phase": "SUCCESS_STABLE",
                                      "status": "SUCCESS_STABLE", "phase": "COMPLETED",
                                      "controller_phase": "STOPPED", "ended_utc": utc_now(),
                                      "best_acceptable_config": state.get("validation_config"),
                                      "final_selected_config": state.get("validation_config"),
                                      "final_selected_config_applied": True,
                                      "final_outcome": (f"Target range passed for {validation_epoch_limit} "
                                                        "additional same-configuration validation epoch(s).")})
                        _write_json(folder / "session.json", state)
                        _log(folder, state["final_outcome"])
                        return
                    state.update({"validation_phase": "STABILITY_VALIDATION", "phase": "STABILITY_VALIDATION"})
                    _write_json(folder / "session.json", state)
                    continue

                target_hit = target_range_status(
                    float(event["measured_qoe_proxy"]), float(state["target_qoe"]),
                    float(state.get("target_tolerance", 0.005)),
                )
                if target_hit:
                    fingerprint = configuration_fingerprint(current)
                    apply_target_attainment_record(
                        event, config=current,
                        previous_config=history[-1].get("config") if history else None,
                        target_qoe=float(state["target_qoe"]),
                        tolerance=float(state.get("target_tolerance", 0.005)),
                        validation_epochs=validation_epoch_limit, epoch=epoch,
                    )
                    _append_jsonl(folder / "events.jsonl", event)
                    _append_history_csv(folder, event)
                    history.append(event)
                    state.update({"validation_active": True, "validation_phase": "STABILITY_VALIDATION",
                                  "validation_config": current, "validation_fingerprint": fingerprint,
                                  "validation_sequence": int(state.get("validation_sequence", 0)) + 1,
                                  "validation_epochs_completed": 0,
                                  "latest_measured_qoe_proxy": event["measured_qoe_proxy"],
                                  "latest_action": event["controller_action"],
                                  "latest_rationale": event["action_rationale"],
                                  "controller_phase": "STABILITY_VALIDATION",
                                  "phase": "STABILITY_VALIDATION", "heartbeat_utc": utc_now()})
                    _write_json(folder / "session.json", state)
                    _log(folder, event["action_rationale"])
                    continue

                decision_history = history + [event]
                if is_post_epoch_trial:
                    decision = decide_post_epoch_exploration(
                        decision_history, feasible, exploratory_configs,
                        target_qoe=float(state["target_qoe"]),
                        max_additional_epochs=extra_epoch_limit,
                        normal_action_budget=int(state["max_actions"]),
                        candidate_performance=candidate_performance,
                        pilot_performance=pilot_performance,
                        live_performance=live_performance,
                        allow_high_risk_candidates=bool(state.get("allow_high_risk_candidates", False)),
                        allow_topology_exploration=bool(state.get("allow_topology_exploration", False)),
                        allow_repeat_trials=bool(state.get("allow_repeat_trials", False)),
                        target_tolerance=float(state.get("target_tolerance", 0.005)),
                        stable_epochs_required=int(state.get("stable_epochs_required", 2)),
                        maintenance_margin=float(state.get("maintenance_margin", 0.02)),
                        maintenance_hysteresis=float(state.get("maintenance_hysteresis", 0.02)),
                    )
                else:
                    decision = decide_next_epoch(
                        decision_history, feasible, target_qoe=float(state["target_qoe"]),
                        max_epochs=normal_epoch_limit, mode=state["controller_mode"],
                        max_actions=int(state["max_actions"]),
                        no_improvement_tolerance=float(state["no_improvement_tolerance"]),
                        candidate_performance=candidate_performance,
                        exploratory_configs=exploratory_configs,
                        allow_exploratory_candidates=bool(state.get("allow_exploratory_candidates", False)),
                        pilot_performance=pilot_performance,
                        live_performance=live_performance,
                        allow_high_risk_candidates=bool(state.get("allow_high_risk_candidates", False)),
                        allow_topology_exploration=bool(state.get("allow_topology_exploration", False)),
                        allow_repeat_trials=bool(state.get("allow_repeat_trials", False)),
                        target_tolerance=float(state.get("target_tolerance", 0.005)),
                        stable_epochs_required=int(state.get("stable_epochs_required", 2)),
                        maintenance_margin=float(state.get("maintenance_margin", 0.02)),
                        maintenance_hysteresis=float(state.get("maintenance_hysteresis", 0.02)),
                        predictive_forecast=forecast,
                        minimum_expected_benefit=float(state.get("minimum_expected_benefit", 0.02)),
                    )
                    if decision.get("action") in {"STOP_EPOCH_LIMIT", "STOP_TARGET_STABLE"} and run_remaining_experimental:
                        decision = decide_post_epoch_exploration(
                            decision_history, feasible, exploratory_configs,
                            target_qoe=float(state["target_qoe"]),
                            max_additional_epochs=extra_epoch_limit,
                            normal_action_budget=int(state["max_actions"]),
                            candidate_performance=candidate_performance,
                            pilot_performance=pilot_performance,
                            live_performance=live_performance,
                            allow_high_risk_candidates=bool(state.get("allow_high_risk_candidates", False)),
                            allow_topology_exploration=bool(state.get("allow_topology_exploration", False)),
                            allow_repeat_trials=bool(state.get("allow_repeat_trials", False)),
                            target_tolerance=float(state.get("target_tolerance", 0.005)),
                            stable_epochs_required=int(state.get("stable_epochs_required", 2)),
                            maintenance_margin=float(state.get("maintenance_margin", 0.02)),
                            maintenance_hysteresis=float(state.get("maintenance_hysteresis", 0.02)),
                        )
                completed_extra_epochs = sum(bool(item.get("post_epoch_exploration"))
                                             for item in decision_history)
                proposed_config = decision.get("next_config")
                action_changes_config = (decision.get("action") == "CHANGE_CONFIGURATION"
                                         and proposed_config is not None
                                         and configuration_fingerprint(proposed_config)
                                             != configuration_fingerprint(current))
                if action_changes_config:
                    event["action_application_status"] = (
                        "PENDING_APPROVAL" if state.get("operating_mode") == "Recommend"
                        else "SCHEDULED_RESTART"
                    )
                    event["action_fingerprint"] = configuration_fingerprint(proposed_config)
                    event["predicted_qoe_before"] = forecast.get("predicted_qoe")
                    event["measured_qoe_before"] = event.get("measured_qoe_proxy")
                    intervention = {
                        "timestamp_utc": utc_now(), "session_id": state["session_id"],
                        "epoch": epoch, "status": "PROPOSED",
                        "action": "Band 1 bandwidth restart",
                        "parameter": "bandwidth_band1_mhz",
                        "previous_value_mhz": current.get("bandwidth_band1_mhz"),
                        "new_value_mhz": proposed_config.get("bandwidth_band1_mhz"),
                        "previous_config": current, "new_config": proposed_config,
                        "configuration_fingerprint": event["action_fingerprint"],
                        "predicted_qoe_before": forecast.get("predicted_qoe"),
                        "measured_qoe_before": event.get("measured_qoe_proxy"),
                        "predicted_qoe_after": decision.get("predicted_qoe"),
                        "forecast": forecast, "affected_ue_ids": forecast.get("at_risk_ue_ids", []),
                        "rationale": decision.get("rationale"),
                        "application": SUPPORTED_ACTIONS["bandwidth_band1_mhz"]["application"],
                        "verification_method": SUPPORTED_ACTIONS["bandwidth_band1_mhz"]["verification"],
                        "rejected_alternatives": decision.get("rejected_alternatives", []),
                    }
                    _append_jsonl(folder / "interventions.jsonl", intervention)
                    if state.get("operating_mode") == "Automatic":
                        _append_jsonl(folder / "interventions.jsonl", {
                            **intervention, "timestamp_utc": utc_now(), "status": "SCHEDULED_RESTART",
                        })
                event.update({"controller_action": decision["action"], "action_rationale": decision["rationale"],
                              "next_config": decision.get("next_config"), "worker_state": "COMPLETED",
                              "next_topology_candidate_id": (decision.get("next_config") or {}).get("config_id")
                                  if decision.get("topology_change") else None,
                              "next_decision": decision["action"],
                              "adaptive_action_count": decision.get("adaptive_action_count", ""),
                              "exploratory_trial_count": decision.get("exploratory_trial_count", ""),
                              "adaptive_action_budget": decision.get("adaptive_action_budget", ""),
                              "action_budget_exhausted": decision.get("action_budget_exhausted", False),
                              "post_epoch_exploration_count": completed_extra_epochs,
                              "post_epoch_exploration_limit": extra_epoch_limit if run_remaining_experimental else "",
                              "next_continuation_reason": decision.get("continuation_reason", ""),
                              "candidate_risk_level": decision.get("candidate_risk_level", event.get("candidate_risk_level", "")),
                              "candidate_risk_reason": decision.get("candidate_risk_reason", event.get("candidate_risk_reason", "")),
                              "next_candidate_risk_level": decision.get("candidate_risk_level", ""),
                              "next_candidate_risk_reason": decision.get("candidate_risk_reason", ""),
                              "next_predicted_qoe": decision.get("predicted_qoe", (decision.get("prediction") or {}).get("qoe")),
                              "next_predicted_qoe_lower": decision.get("predicted_qoe_lower", (decision.get("prediction") or {}).get("lower")),
                              "next_predicted_qoe_upper": decision.get("predicted_qoe_upper", (decision.get("prediction") or {}).get("upper")),
                              "next_prediction_domain": decision.get("prediction_domain", (decision.get("prediction") or {}).get("domain", "")),
                              "next_prediction_run_count": decision.get("prediction_run_count", (decision.get("prediction") or {}).get("run_count", 0)),
                              "high_risk_override": decision.get("high_risk_override", event.get("high_risk_override", False)),
                              "repeat_trials_enabled": bool(state.get("allow_repeat_trials", False)),
                              "risk_excluded_count": decision.get("risk_excluded_count", 0),
                              "duplicate_excluded_count": decision.get("duplicate_excluded_count", 0)})
                event["is_recovery_trial"] = bool(decision.get("is_recovery", False))
                event["recovery_target_key"] = decision.get("recovery_target_key")
                event["controller_phase"] = decision.get("controller_phase", controller_phase(
                    decision_history, target_qoe=float(state["target_qoe"]),
                    target_tolerance=float(state.get("target_tolerance", 0.005)),
                    stable_epochs_required=int(state.get("stable_epochs_required", 2))))
                prediction = decision.get("prediction") or {
                    "available": decision.get("predicted_qoe") is not None,
                    "qoe": decision.get("predicted_qoe"),
                    "lower": decision.get("predicted_qoe_lower"),
                    "upper": decision.get("predicted_qoe_upper"),
                    "target_fraction": decision.get("predicted_target_fraction"),
                    "outage_fraction": decision.get("predicted_outage_fraction"),
                    "method": decision.get("prediction_method"),
                    "domain": decision.get("prediction_domain"),
                    "run_count": decision.get("prediction_run_count", 0),
                    "confidence": "see empirical exact-condition support",
                }
                event["next_predicted_qoe"] = prediction.get("qoe")
                event["next_predicted_qoe_lower"] = prediction.get("lower")
                event["next_predicted_qoe_upper"] = prediction.get("upper")
                event["next_prediction_domain"] = prediction.get("domain")
                event["next_prediction_run_count"] = prediction.get("run_count", 0)
                event["decision_record"] = {
                    "epoch": epoch, "timestamp_utc": event.get("end_utc"),
                    "decision": decision["action"], "phase": event.get("controller_phase"),
                    "candidate_id": (decision.get("next_config") or {}).get("config_id"),
                    "configuration_fingerprint": event.get("configuration_fingerprint"),
                    "previous_config": history[-1].get("config") if history else None,
                    "resulting_config": decision.get("next_config"),
                    "selected_reason": decision.get("rationale"),
                    "objective": "Enter the configured absolute QoE target range; preserve outage validity.",
                    "expected_effect": (f"Empirical QoE estimate {float(prediction['qoe']):.4f}"
                                        if prediction.get("available") and prediction.get("qoe") is not None
                                        else "Not estimated."),
                    "rejected_alternatives": decision.get("rejected_alternatives", []),
                    "measured_qoe_proxy": event.get("measured_qoe_proxy"),
                    "measured_metrics": event.get("metrics"),
                    "target_range_pass": event.get("target_range_pass"),
                    "validation_state": state.get("validation_phase", "TARGET_SEEKING"),
                    "reason_for_transition": decision.get("rationale"),
                }
                _append_jsonl(folder / "events.jsonl", event)
                _append_history_csv(folder, event)
                history.append(event)
                _log(folder, f"Epoch {epoch}: measured QoE={event['measured_qoe_proxy']:.6f}; "
                     f"action={event['controller_action']}; {event['action_rationale']}")
                if decision["action"] in {"CHANGE_CONFIGURATION", "CONTINUE_FIXED", "CONTINUE_POST_EPOCH_EXPLORATION",
                                           "CONTINUE_STABILITY", "CONTINUE_MAINTENANCE"}:
                    if action_changes_config and state.get("operating_mode") == "Recommend":
                        proceed, decision = _wait_for_recommendation(folder, state, event, decision)
                        _persist_updated_event(folder, event)
                        if not proceed:
                            state.update({"status": "USER_CANCELLED", "phase": "CANCELLED",
                                          "ended_utc": utc_now(), "pending_action": None,
                                          "final_outcome": "User stopped while a proposed action was awaiting approval."})
                            _write_json(folder / "session.json", state)
                            return
                        if event.get("action_application_status") == "REJECTED":
                            event["controller_action"] = "ACTION_REJECTED"
                            event["action_rationale"] = decision["rationale"]
                    current = decision["next_config"]
                    current_selection = {
                        "candidate_scope": decision.get("candidate_scope", "fixed baseline configuration"),
                        "topology_change": bool(decision.get("topology_change", False)),
                        "is_exploratory": bool(decision.get("is_exploratory", False)),
                        "candidate_selection_rationale": decision.get("rationale", "Fixed mode repeats the selected configuration."),
                        "candidate_evidence": decision.get("candidate_evidence", "fixed mode; no candidate ranking"),
                        "candidate_validation": decision.get("candidate_validation", "PASS: fixed configuration retained unchanged"),
                        "candidate_risk_level": decision.get("candidate_risk_level", ""),
                        "candidate_risk_reason": decision.get("candidate_risk_reason", ""),
                        "selected_candidate_risk_level": decision.get("candidate_risk_level", ""),
                        "selected_candidate_risk_reason": decision.get("candidate_risk_reason", ""),
                        "high_risk_override": decision.get("high_risk_override", False),
                        "repeat_trials_enabled": bool(state.get("allow_repeat_trials", False)),
                        "is_recovery_trial": bool(decision.get("is_recovery", False)),
                        "recovery_target_key": decision.get("recovery_target_key"),
                        "controller_phase": controller_phase(
                            decision_history, target_qoe=float(state["target_qoe"]),
                            target_tolerance=float(state.get("target_tolerance", 0.005)),
                            stable_epochs_required=int(state.get("stable_epochs_required", 2))),
                        "predicted_qoe": prediction.get("qoe"),
                        "predicted_qoe_lower": prediction.get("lower"),
                        "predicted_qoe_upper": prediction.get("upper"),
                        "predicted_target_fraction": prediction.get("target_fraction"),
                        "predicted_outage_fraction": prediction.get("outage_fraction"),
                        "prediction_method": prediction.get("method"),
                        "prediction_domain": prediction.get("domain"),
                        "prediction_run_count": prediction.get("run_count", 0),
                        "prediction_confidence": prediction.get("confidence"),
                        "post_epoch_exploration": bool(decision.get("post_epoch_exploration", False)),
                        "continuation_reason": decision.get("continuation_reason", ""),
                    }
                    state.update({"phase": "DECIDING_NEXT_EPOCH", "latest_measured_qoe_proxy": event["measured_qoe_proxy"],
                                  "controller_phase": current_selection["controller_phase"],
                                  "latest_action": event["controller_action"], "latest_rationale": event["action_rationale"],
                                  "next_prediction": prediction,
                                  "next_configuration_rationale": decision.get("rationale", ""),
                                  "current_config": current, "next_config": current,
                                  "post_epoch_exploration_count": completed_extra_epochs,
                                  "heartbeat_utc": utc_now()})
                    _write_json(folder / "session.json", state)
                    continue
                final_status, final_text = {
                    "NO_ACTION_TARGET_MET": ("TARGET_REACHED", "Target reached using measured simulator output."),
                    "STOP_EPOCH_LIMIT": ("EPOCH_LIMIT_REACHED", "Epoch limit reached while target stability criterion remained unmet."),
                    "STOP_TARGET_STABLE": ("TARGET_REACHED", "Target stability criterion met using consecutive validated measurements."),
                    "STOP_NO_IMPROVEMENT_AFTER_AVAILABLE_TRIALS": (
                        "NO_IMPROVEMENT_AFTER_AVAILABLE_TRIALS", "No improvement after all available candidate trials; target remained unmet."
                    ),
                    "STOP_NO_VALIDATED_CANDIDATES": (
                        "NO_VALIDATED_CANDIDATES",
                        "No eligible untried candidate remains in the validated and currently permitted exploratory pools; target remained unmet."
                    ),
                    "STOP_NO_SAFE_CANDIDATES": (
                        "STOP_NO_SAFE_CANDIDATES",
                        "No candidate remains eligible under the conservative risk policy; high-risk settings were not run."
                    ),
                    "STOP_NO_MEANINGFUL_IMPROVEMENT": (
                        "NO_MEANINGFUL_IMPROVEMENT", "All eligible supported candidates were tried without meaningful QoE improvement; target remained unmet."
                    ),
                    "STOP_POST_EPOCH_LIMIT": (
                        "EXPLORATORY_EPOCH_LIMIT_REACHED",
                        f"Separate limit of {extra_epoch_limit} additional experimental epochs reached; target remained unmet."
                    ),
                    "STOP_NO_EXPERIMENTAL_CANDIDATES": (
                        "NO_EXPERIMENTAL_CANDIDATES",
                        "No eligible untried experimental configurations remain; target remained unmet."
                    ),
                    "STOP_NO_FEASIBLE_CANDIDATES": ("NO_FEASIBLE_CANDIDATES", "No feasible untried candidate remains; target remained unmet."),
                }.get(decision["action"], ("COMPLETED", decision["rationale"]))
                state.update({"status": final_status, "phase": "COMPLETED", "controller_phase": "STOPPED",
                              "last_controller_phase": event.get("controller_phase", "TARGET_SEEKING"),
                              "next_prediction": None,
                              "epoch_state": "COMPLETED",
                              "ended_utc": utc_now(), "final_outcome": final_text,
                              "latest_measured_qoe_proxy": event["measured_qoe_proxy"],
                              "post_epoch_exploration_count": completed_extra_epochs,
                              "latest_action": event["controller_action"], "latest_rationale": event["action_rationale"],
                              "heartbeat_utc": utc_now()})
                _write_json(folder / "session.json", state)
                _log(folder, f"Session ended: {final_status}; {final_text}")
                return
            final_status = "EXPLORATORY_EPOCH_LIMIT_REACHED" if run_remaining_experimental else "EPOCH_LIMIT_REACHED"
            final_text = (f"Separate limit of {extra_epoch_limit} additional experimental epochs reached."
                          if run_remaining_experimental else "Normal epoch limit reached.")
            state.update({"status": final_status, "phase": "COMPLETED", "controller_phase": "STOPPED",
                          "last_controller_phase": state.get("controller_phase", "TARGET_SEEKING"),
                          "ended_utc": utc_now(),
                          "final_outcome": final_text})
            _write_json(folder / "session.json", state)
        finally:
            fcntl.flock(execution_lock.fileno(), fcntl.LOCK_UN)


def worker_main() -> None:
    parser = argparse.ArgumentParser(description="Execute one persisted live ns-3 controller session")
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", default=RUN_TIMEOUT_SECONDS, type=float)
    args = parser.parse_args()
    try:
        run_live_session(args.session_dir, args.timeout_seconds)
    except BaseException as exc:
        _mark_worker_failure(args.session_dir.resolve(), exc)
        raise
