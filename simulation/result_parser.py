"""Parser and metric calculator for ns-3 5G simulation outputs.

Preserves exact physical definitions and distinctions:
- Outage / zero delivery (rx_packets == 0) vs delivered traffic.
- Accurate packet loss calculation with zero-division protection.
- Exact QoE formula from scratch/qoe-dataset.cc.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SIMULATION_CSV_COLUMNS = [
    "scenario_id", "ue_count", "lambda_be", "bandwidth_mhz", "total_tx_power_dbm",
    "scenario_size_m", "flow_id", "tx_packets", "rx_packets", "lost_packets",
    "throughput_mbps", "delay_ms", "jitter_ms", "packet_loss", "qoe_proxy",
]
UE_MEASUREMENT_COLUMNS = [
    "ue_id", "traffic_class", "application_profile", "flow_id", "tx_packets", "rx_packets",
    "throughput_mbps", "delay_ms", "jitter_ms", "packet_loss_fraction", "qoe_proxy",
    "serving_cell_id", "qoe_formula_version",
]
LEGACY_UE_MEASUREMENT_COLUMNS = UE_MEASUREMENT_COLUMNS[:-1]
PREVIOUS_UE_MEASUREMENT_COLUMNS = [c for c in UE_MEASUREMENT_COLUMNS if c not in {"serving_cell_id", "qoe_formula_version"}]
LEGACY_VERSIONED_UE_COLUMNS = [c for c in UE_MEASUREMENT_COLUMNS if c != "qoe_formula_version"]

# Project service profiles. Delay/error targets are analogous to the 5QI 3
# (interactive low-latency) and 5QI 1 (conversational voice) examples in 3GPP
# TS 23.501; the simulator configures the corresponding NR QoS-flow enums for
# gaming/video and does not claim that every external 5QI behavior is modeled.
# Jitter limits and voice/gaming floors are project policy. The low-latency
# throughput follows its default offered load (100 B × 2,500 pps); the video
# throughput reference follows the NGMN generator's documented 64 kbps default.
QOE_FORMULA_VERSION = "service_v2"
QOE_SERVICE_PROFILES = {
    "low_latency": {"throughput_mbps": 2.0, "delay_ms": 50.0, "jitter_ms": 5.0, "loss_fraction": 0.001},
    "voice_qos": {"throughput_mbps": 0.1, "delay_ms": 100.0, "jitter_ms": 30.0, "loss_fraction": 0.01},
    # The gaming reference is a project floor for the NGMN quasi-periodic
    # packet model. Video's 64 kbps reference follows the model's documented
    # default source rate; it is not a player-buffer or resolution QoE model.
    "gaming": {"throughput_mbps": 0.05, "delay_ms": 50.0, "jitter_ms": 10.0, "loss_fraction": 0.001},
    "video_streaming": {"throughput_mbps": 0.064, "delay_ms": 300.0, "jitter_ms": 50.0, "loss_fraction": 0.000001},
}


def calculate_packet_loss(tx_packets: int, rx_packets: int) -> float:
    """Calculate packet loss ratio safely with zero-division protection."""
    if tx_packets <= 0:
        return 0.0
    loss = 1.0 - (float(rx_packets) / float(tx_packets))
    return max(0.0, min(1.0, float(loss)))


def calculate_qoe_components(
    throughput_mbps: float, delay_ms: float, jitter_ms: float, packet_loss: float,
    traffic_class: str, rx_packets: int = 1,
) -> dict[str, float]:
    """Return service_v2 satisfaction components before their QoE weights."""
    if traffic_class not in QOE_SERVICE_PROFILES:
        raise ValueError(f"Unknown traffic class for {QOE_FORMULA_VERSION}: {traffic_class!r}")
    if rx_packets <= 0:
        return {"throughput": 0.0, "delay": 0.0, "jitter": 0.0, "packet_loss": 0.0}
    profile = QOE_SERVICE_PROFILES[traffic_class]
    loss = max(0.0, min(1.0, float(packet_loss)))
    return {
        "throughput": min(max(0.0, float(throughput_mbps)) / profile["throughput_mbps"], 1.0),
        "delay": max(0.0, 1.0 - max(0.0, float(delay_ms)) / profile["delay_ms"]),
        "jitter": max(0.0, 1.0 - max(0.0, float(jitter_ms)) / profile["jitter_ms"]),
        "packet_loss": max(0.0, 1.0 - loss / profile["loss_fraction"]),
    }


def calculate_qoe_proxy(
    throughput_mbps: float,
    delay_ms: float,
    jitter_ms: float,
    packet_loss: float,
    rx_packets: int = 1,
    traffic_class: str | None = None,
    formula_version: str = "legacy_v1",
) -> float:
    """Calculate the explicitly versioned QoE proxy.
    
    Formula from ns-3.48/scratch/qoe-dataset.cc:
        qoeProxy = 0.40 * min(throughputMbps / 50.0, 1.0)
                 + 0.25 * exp(-delayMs / 50.0)
                 + 0.20 * exp(-jitterMs / 20.0)
                 + 0.15 * (1.0 - packetLoss);
    If rxPackets == 0, qoeProxy = 0.0.
    """
    if rx_packets <= 0:
        return 0.0
    clamped_loss = max(0.0, min(1.0, float(packet_loss)))
    if formula_version == QOE_FORMULA_VERSION:
        components = calculate_qoe_components(throughput_mbps, delay_ms, jitter_ms,
                                              clamped_loss, traffic_class or "", rx_packets)
        return (0.40 * components["throughput"] + 0.25 * components["delay"]
                + 0.20 * components["jitter"] + 0.15 * components["packet_loss"])
    if formula_version != "legacy_v1":
        raise ValueError(f"Unsupported QoE formula version: {formula_version}")
    qoe = (
        0.40 * min(max(0.0, float(throughput_mbps)) / 50.0, 1.0)
        + 0.25 * math.exp(-max(0.0, float(delay_ms)) / 50.0)
        + 0.20 * math.exp(-max(0.0, float(jitter_ms)) / 20.0)
        + 0.15 * (1.0 - clamped_loss)
    )
    return max(0.0, min(1.0, float(qoe)))


def parse_ue_measurements(
    csv_path: str | Path, *, expected_ue_count: int | None = None,
    expected_gnb_count: int | None = None,
) -> list[dict[str, Any]]:
    """Read the optional, destination-address-mapped per-UE sidecar artifact.

    This artifact is additive to ``qoe_dataset.csv``. It is deliberately not
    inferred from FlowMonitor ``flow_id`` ordering. A missing file is handled by
    the caller as unavailable for compatibility with older simulator binaries.
    """
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames not in (UE_MEASUREMENT_COLUMNS, LEGACY_VERSIONED_UE_COLUMNS,
                                     LEGACY_UE_MEASUREMENT_COLUMNS, PREVIOUS_UE_MEASUREMENT_COLUMNS):
            raise ValueError(f"Per-UE CSV schema mismatch: {reader.fieldnames!r}")
        has_serving_cell = "serving_cell_id" in reader.fieldnames
        rows = list(reader)
    if expected_ue_count is not None and len(rows) != int(expected_ue_count):
        raise ValueError(f"Expected {expected_ue_count} per-UE rows, found {len(rows)}")
    seen_ues: set[int] = set()
    seen_flows: set[int] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        try:
            ue_id = int(row["ue_id"])
            flow_id = int(row["flow_id"])
            tx = int(row["tx_packets"])
            rx = int(row["rx_packets"])
            throughput = float(row["throughput_mbps"])
            delay = float(row["delay_ms"])
            jitter = float(row["jitter_ms"])
            packet_loss = float(row["packet_loss_fraction"])
            recorded_qoe = float(row["qoe_proxy"])
            traffic_class = str(row["traffic_class"])
            application = str(row["application_profile"])
            serving_cell_id = int(row["serving_cell_id"]) if has_serving_cell else None
            formula_version = str(row.get("qoe_formula_version") or "legacy_v1")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed per-UE measurement row: {exc}") from exc
        if not row["application_profile"] or traffic_class not in QOE_SERVICE_PROFILES:
            raise ValueError(f"Unknown per-UE traffic profile for UE {ue_id}")
        if ue_id < 1 or flow_id < 1 or ue_id in seen_ues or flow_id in seen_flows:
            raise ValueError(f"Duplicate or invalid UE/flow identity: UE {ue_id}, flow {flow_id}")
        if serving_cell_id is not None and serving_cell_id < 1:
            raise ValueError(f"Invalid serving cell identity for UE {ue_id}")
        if expected_gnb_count is not None and serving_cell_id is not None and serving_cell_id > int(expected_gnb_count):
            raise ValueError(f"Serving cell {serving_cell_id} exceeds configured gNB count {expected_gnb_count}")
        seen_ues.add(ue_id)
        seen_flows.add(flow_id)
        numeric = (throughput, delay, jitter, packet_loss, recorded_qoe)
        if tx < 0 or rx < 0 or rx > tx or not all(math.isfinite(v) for v in numeric):
            raise ValueError(f"Invalid counts or non-finite measurements for UE {ue_id}")
        if min(throughput, delay, jitter) < 0 or not 0 <= packet_loss <= 1 or not 0 <= recorded_qoe <= 1:
            raise ValueError(f"Out-of-range per-UE measurement for UE {ue_id}")
        expected_loss = calculate_packet_loss(tx, rx)
        expected_qoe = calculate_qoe_proxy(throughput, delay, jitter, expected_loss, rx_packets=rx,
                                           traffic_class=traffic_class, formula_version=formula_version)
        if not math.isclose(packet_loss, expected_loss, abs_tol=2e-5):
            raise ValueError(f"Packet-loss recomputation mismatch for UE {ue_id}")
        if not math.isclose(recorded_qoe, expected_qoe, abs_tol=2e-5):
            raise ValueError(f"QoE recomputation mismatch for UE {ue_id}")
        if rx == 0 and any(value != 0 for value in (throughput, delay, jitter, recorded_qoe)):
            raise ValueError(f"Zero-reception metrics must be zero for UE {ue_id}")
        result.append({
            "ue_id": ue_id, "traffic_class": traffic_class, "application_profile": application,
            "flow_id": flow_id, "tx_packets": tx, "rx_packets": rx,
            "throughput_mbps": throughput, "delay_ms": delay, "jitter_ms": jitter,
            "packet_loss_fraction": packet_loss, "qoe_proxy": recorded_qoe,
            "qoe_formula_version": formula_version,
            "serving_cell_id": serving_cell_id,
            "is_outage": tx > 0 and rx == 0,
        })
    if expected_ue_count is not None and len(seen_ues) != int(expected_ue_count):
        raise ValueError("Per-UE measurements do not uniquely cover the configured UE set")
    return sorted(result, key=lambda item: item["ue_id"])


def validate_simulation_csv(
    csv_path: str | Path,
    expected_config: dict[str, Any] | None = None,
    expected_flow_count: int | None = None,
) -> tuple[bool, str | None]:
    """Validate the exact ns-3 CSV contract before treating parsed values as measurements.

    This is additive to ``parse_simulation_csv``: it checks schema, row identity,
    finite/physical values, optional requested settings, and independently
    recomputes loss and QoE using the same public metric functions above.
    """
    path = Path(csv_path)
    if not path.is_file() or path.stat().st_size == 0:
        return False, f"Simulation CSV is absent or empty: {path}"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != SIMULATION_CSV_COLUMNS:
                return False, f"CSV schema mismatch: {reader.fieldnames!r}"
            rows = list(reader)
        if not rows:
            return False, "Simulation CSV contains no flow rows"
        if expected_flow_count is not None and len(rows) != int(expected_flow_count):
            return False, f"Expected {expected_flow_count} flow rows, found {len(rows)}"
        seen: set[tuple[int, int]] = set()
        numeric = SIMULATION_CSV_COLUMNS
        expected_columns = {
            "scenario_id": "scenario_id",
            "ue_count": "ue_count",
            "bandwidth_mhz": "bandwidth_mhz",
            "total_tx_power_dbm": "total_tx_power_dbm",
            "scenario_size_m": "scenario_size_m",
            "lambda_be": "lambda_be",
        }
        for row in rows:
            if any(row.get(column, "") in (None, "") for column in numeric):
                return False, "Simulation CSV contains missing values"
            value = {column: float(row[column]) for column in numeric}
            if not all(math.isfinite(number) for number in value.values()):
                return False, "Simulation CSV contains non-finite values"
            flow_key = (int(value["scenario_id"]), int(value["flow_id"]))
            if flow_key in seen:
                return False, f"Duplicate scenario/flow key: {flow_key}"
            seen.add(flow_key)
            for column in ("tx_packets", "rx_packets", "lost_packets", "flow_id", "ue_count", "scenario_id"):
                if value[column] < 0 or not float(value[column]).is_integer():
                    return False, f"Invalid integer/count column {column}={value[column]}"
            if value["throughput_mbps"] < 0 or value["delay_ms"] < 0 or value["jitter_ms"] < 0:
                return False, "Negative throughput, delay or jitter"
            if not 0.0 <= value["packet_loss"] <= 1.0:
                return False, "Packet loss is outside fraction range [0,1]"
            if not 0.0 <= value["qoe_proxy"] <= 1.0:
                return False, "QoE proxy is outside range [0,1]"
            expected_loss = calculate_packet_loss(int(value["tx_packets"]), int(value["rx_packets"]))
            if abs(value["packet_loss"] - expected_loss) > 2e-5:
                return False, f"Packet-loss recomputation mismatch for flow {flow_key}"
            expected_qoe = calculate_qoe_proxy(
                value["throughput_mbps"], value["delay_ms"], value["jitter_ms"],
                expected_loss, rx_packets=int(value["rx_packets"]),
            )
            if abs(value["qoe_proxy"] - expected_qoe) > 2e-5:
                return False, f"QoE recomputation mismatch for flow {flow_key}"
            if int(value["rx_packets"]) == 0 and any(
                value[name] != 0.0 for name in ("throughput_mbps", "delay_ms", "jitter_ms", "qoe_proxy")
            ):
                return False, f"Invalid zero-reception metrics for flow {flow_key}"
            for output_column, config_key in expected_columns.items():
                if expected_config is not None and config_key in expected_config:
                    if not math.isclose(value[output_column], float(expected_config[config_key]), abs_tol=1e-8):
                        return False, f"Applied setting mismatch for {output_column}"
    except (OSError, csv.Error, KeyError, TypeError, ValueError, OverflowError) as exc:
        return False, f"Cannot validate simulation CSV: {type(exc).__name__}: {exc}"
    return True, None


@dataclass
class FlowRecord:
    scenario_id: int
    ue_count: int
    lambda_be: float
    bandwidth_mhz: float
    total_tx_power_dbm: float
    scenario_size_m: float
    flow_id: int
    tx_packets: int
    rx_packets: int
    lost_packets: int
    throughput_mbps: float
    delay_ms: float
    jitter_ms: float
    packet_loss: float
    qoe_proxy: float
    flow_type: str


@dataclass
class SimulationMetrics:
    success: bool
    is_outage: bool
    num_flows: int
    target_flow_type: str
    target_flow_count: int
    mean_throughput_mbps: float
    mean_delay_ms: float
    mean_jitter_ms: float
    mean_packet_loss: float
    mean_qoe_proxy: float
    total_tx_packets: int
    total_rx_packets: int
    flow_records: list[FlowRecord] = field(default_factory=list)
    raw_csv_path: str = ""
    error_message: str | None = None


def parse_simulation_csv(
    csv_path: str | Path,
    target_flow_type: str = "voice_be",
) -> SimulationMetrics:
    """Parse the CSV generated by ns-3 (qoe_dataset.csv) in the run directory.
    
    Target flow type:
    - 'voice_be': Flows with flow_id > ue_count // 2 (UDP 1252B @ lambda_be)
    - 'low_latency': Flows with flow_id <= ue_count // 2 (UDP 100B @ lambda_ull)
    - 'all': All flows combined
    """
    path = Path(csv_path)
    if not path.exists():
        return SimulationMetrics(
            success=False,
            is_outage=False,
            num_flows=0,
            target_flow_type=target_flow_type,
            target_flow_count=0,
            mean_throughput_mbps=0.0,
            mean_delay_ms=0.0,
            mean_jitter_ms=0.0,
            mean_packet_loss=0.0,
            mean_qoe_proxy=0.0,
            total_tx_packets=0,
            total_rx_packets=0,
            error_message=f"Simulation output CSV not found at {path}",
        )

    flow_records: list[FlowRecord] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ue_count = int(row["ue_count"])
                flow_id = int(row["flow_id"])
                is_low_lat = flow_id <= (ue_count // 2)
                flow_type = "low_latency" if is_low_lat else "voice_be"

                tx = int(row["tx_packets"])
                rx = int(row["rx_packets"])
                lost = int(row["lost_packets"])
                thru = float(row["throughput_mbps"])
                delay = float(row["delay_ms"])
                jitter = float(row["jitter_ms"])
                recalc_loss = calculate_packet_loss(tx, rx)
                calc_qoe = calculate_qoe_proxy(thru, delay, jitter, recalc_loss, rx_packets=rx)

                rec = FlowRecord(
                    scenario_id=int(row["scenario_id"]),
                    ue_count=ue_count,
                    lambda_be=float(row["lambda_be"]),
                    bandwidth_mhz=float(row["bandwidth_mhz"]),
                    total_tx_power_dbm=float(row["total_tx_power_dbm"]),
                    scenario_size_m=float(row["scenario_size_m"]),
                    flow_id=flow_id,
                    tx_packets=tx,
                    rx_packets=rx,
                    lost_packets=lost,
                    throughput_mbps=thru,
                    delay_ms=delay,
                    jitter_ms=jitter,
                    packet_loss=recalc_loss,
                    qoe_proxy=calc_qoe,
                    flow_type=flow_type,
                )
                flow_records.append(rec)
            except (KeyError, ValueError) as err:
                return SimulationMetrics(
                    success=False,
                    is_outage=False,
                    num_flows=len(flow_records),
                    target_flow_type=target_flow_type,
                    target_flow_count=0,
                    mean_throughput_mbps=0.0,
                    mean_delay_ms=0.0,
                    mean_jitter_ms=0.0,
                    mean_packet_loss=0.0,
                    mean_qoe_proxy=0.0,
                    total_tx_packets=0,
                    total_rx_packets=0,
                    error_message=f"Malformed row in CSV {path}: {err}",
                )

    if not flow_records:
        return SimulationMetrics(
            success=False,
            is_outage=False,
            num_flows=0,
            target_flow_type=target_flow_type,
            target_flow_count=0,
            mean_throughput_mbps=0.0,
            mean_delay_ms=0.0,
            mean_jitter_ms=0.0,
            mean_packet_loss=0.0,
            mean_qoe_proxy=0.0,
            total_tx_packets=0,
            total_rx_packets=0,
            error_message="Simulation CSV is empty",
        )

    # Filter by target flow type
    norm_target = target_flow_type.lower()
    if norm_target in ["voice_be", "voice", "be"]:
        target_records = [r for r in flow_records if r.flow_type == "voice_be"]
    elif norm_target in ["low_latency", "lowlat", "ull"]:
        target_records = [r for r in flow_records if r.flow_type == "low_latency"]
    else:
        target_records = flow_records

    # Fall back to all flows if target subset is empty
    if not target_records:
        target_records = flow_records

    total_tx = sum(r.tx_packets for r in target_records)
    total_rx = sum(r.rx_packets for r in target_records)
    is_outage = (total_rx == 0 and total_tx > 0)

    n = len(target_records)
    mean_thru = sum(r.throughput_mbps for r in target_records) / n
    mean_delay = sum(r.delay_ms for r in target_records) / n
    mean_jitter = sum(r.jitter_ms for r in target_records) / n
    mean_loss = sum(r.packet_loss for r in target_records) / n
    mean_qoe = sum(r.qoe_proxy for r in target_records) / n

    return SimulationMetrics(
        success=True,
        is_outage=is_outage,
        num_flows=len(flow_records),
        target_flow_type=target_flow_type,
        target_flow_count=n,
        mean_throughput_mbps=float(mean_thru),
        mean_delay_ms=float(mean_delay),
        mean_jitter_ms=float(mean_jitter),
        mean_packet_loss=float(mean_loss),
        mean_qoe_proxy=float(mean_qoe),
        total_tx_packets=total_tx,
        total_rx_packets=total_rx,
        flow_records=flow_records,
        raw_csv_path=str(path),
    )

