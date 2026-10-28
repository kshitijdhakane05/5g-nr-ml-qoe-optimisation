"""Experimental-only pre-run feature derivation for the ns-3 pilot.

Offered load is computed from configured packet rate and packet size. It never
uses FlowMonitor tx_packets, which is a post-run counter.
"""

from __future__ import annotations

import math
from typing import Mapping


def derive_flow_inputs(
    *,
    ue_count: int,
    flow_id: int,
    lambda_be_pps: float,
    lambda_ull_pps: float = 10_000.0,
    packet_size_ull_bytes: int = 100,
    packet_size_be_bytes: int = 1252,
) -> dict[str, float | int | str]:
    """Return the configured flow class, rate, packet size and nominal Mbps.

    The scenario source builds low-latency and voice containers separately;
    FlowMonitor IDs for these are exported as low-latency IDs first, followed
    by best-effort IDs. The pilot uses even UE counts, so both classes have
    ``ue_count // 2`` flows.

    Offered Mbps = configured packets/second * payload bytes * 8 / 1e6.
    This is nominal application payload load, excluding protocol headers and
    any scheduling loss. Applications run from 0.4 s to 1.0 s, but that active
    duration does not change the configured steady-state bit rate.
    """
    if ue_count < 2 or ue_count % 2:
        raise ValueError("This experimental feature mapping requires an even UE count >= 2")
    if not 1 <= flow_id <= ue_count:
        raise ValueError(f"flow_id={flow_id} is outside 1..{ue_count}")
    low_latency_count = ue_count // 2
    if flow_id <= low_latency_count:
        flow_type = "low_latency"
        rate = float(lambda_ull_pps)
        packet_size = int(packet_size_ull_bytes)
    else:
        flow_type = "voice_be"
        rate = float(lambda_be_pps)
        packet_size = int(packet_size_be_bytes)
    if rate <= 0 or packet_size <= 0:
        raise ValueError("Configured packet rate and payload size must be positive")
    return {
        "flow_type": flow_type,
        "flow_lambda_pps": rate,
        "packet_size_bytes": packet_size,
        "offered_throughput_mbps_configured": rate * packet_size * 8.0 / 1_000_000.0,
    }


def derive_effective_band_powers_dbm(
    total_power_dbm: float, bandwidth_band1_mhz: float, bandwidth_band2_mhz: float
) -> tuple[float, float]:
    """Mirror qoe-dataset.cc's proportional total-power split across two BWPs."""
    if total_power_dbm <= 0 or bandwidth_band1_mhz <= 0 or bandwidth_band2_mhz <= 0:
        raise ValueError("Power and both bandwidths must be positive")
    total_bandwidth = bandwidth_band1_mhz + bandwidth_band2_mhz
    return (
        total_power_dbm + 10.0 * math.log10(bandwidth_band1_mhz / total_bandwidth),
        total_power_dbm + 10.0 * math.log10(bandwidth_band2_mhz / total_bandwidth),
    )


def recompute_qoe_proxy(row: Mapping[str, str | float | int]) -> float:
    """Independently reproduce the exported proxy, including zero reception."""
    rx = int(float(row["rx_packets"]))
    if rx == 0:
        return 0.0
    throughput = float(row["throughput_mbps"])
    delay = float(row["delay_ms"])
    jitter = float(row["jitter_ms"])
    loss = float(row["packet_loss"])
    return (
        0.40 * min(throughput / 50.0, 1.0)
        + 0.25 * math.exp(-delay / 50.0)
        + 0.20 * math.exp(-jitter / 20.0)
        + 0.15 * (1.0 - loss)
    )
