"""Search space definition and candidate configuration generator for 5G network optimisation.

Strict rules applied:
1. Uses only values and ranges physically supported by the verified Phase 1 dataset.
2. Does not extrapolate beyond the observed training support.
3. Explicitly distinguishes between controllable network decision variables and environmental context.
4. Documents that total_tx_power_dbm has a single value (35 dBm) in the current dataset.
"""

from __future__ import annotations

from typing import Any

# Verified unique values and ranges supported by the Phase 1 dataset
SUPPORTED_SEARCH_SPACE = {
    # Environmental / Context parameters (demands and cell topology)
    "ue_count": [2, 4, 6, 8, 10],
    "scenario_size_m": [3.0, 30.0, 100.0],
    "flow_type": ["low_latency", "voice_be"],
    "flow_lambda": [5000.0, 10000.0, 20000.0, 40000.0],  # for voice_be; low_lat is fixed at 10000.0
    # Controllable Network Decision Variables
    "bandwidth_mhz": [20.0, 50.0],
    # Fixed in current dataset (single-value support; power optimisation requires future ns-3 runs)
    "total_tx_power_dbm": [35.0],
    # Fixed in current dataset (numerology is fixed: Band1=4, Band2=2; not varied)
    "numerology_bwp1": [4],
    "numerology_bwp2": [2],
}


def validate_network_context(
    ue_count: int,
    scenario_size_m: float,
    flow_type: str,
    lambda_val: float,
) -> tuple[bool, str]:
    """Validate that the given network context lies within the verified training data distribution."""
    if ue_count not in SUPPORTED_SEARCH_SPACE["ue_count"]:
        return (
            False,
            f"ue_count={ue_count} is outside supported values {SUPPORTED_SEARCH_SPACE['ue_count']}",
        )
    if float(scenario_size_m) not in SUPPORTED_SEARCH_SPACE["scenario_size_m"]:
        return (
            False,
            f"scenario_size_m={scenario_size_m} is outside supported values {SUPPORTED_SEARCH_SPACE['scenario_size_m']}",
        )
    norm_flow_type = flow_type.lower()
    if norm_flow_type not in SUPPORTED_SEARCH_SPACE["flow_type"]:
        return (
            False,
            f"flow_type='{flow_type}' is outside supported values {SUPPORTED_SEARCH_SPACE['flow_type']}",
        )
    if norm_flow_type == "voice_be" and float(lambda_val) not in SUPPORTED_SEARCH_SPACE["flow_lambda"]:
        return (
            False,
            f"lambda_val={lambda_val} for voice_be is outside supported values {SUPPORTED_SEARCH_SPACE['flow_lambda']}",
        )
    return True, "Valid context"


def generate_candidate_configurations(
    ue_count: int,
    scenario_size_m: float = 30.0,
    flow_type: str = "voice_be",
    lambda_val: float = 10000.0,
    candidate_bandwidths: list[float] | None = None,
    candidate_powers: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Generate candidate network configurations within the verified dataset support.
    
    Parameters:
        ue_count: Active UE count in cell.
        scenario_size_m: Placement area dimension in meters.
        flow_type: 'voice_be' or 'low_latency'.
        lambda_val: Offered packet rate (pkts/s).
        candidate_bandwidths: Optional subset of supported bandwidths. Defaults to all supported.
        candidate_powers: Optional subset of supported powers. Defaults to all supported.
        
    Returns:
        List of candidate configuration dictionaries.
    """
    valid, msg = validate_network_context(ue_count, scenario_size_m, flow_type, lambda_val)
    if not valid:
        raise ValueError(f"Cannot generate candidates: {msg}")

    # Use specified subset or full supported range
    bw_pool = (
        candidate_bandwidths
        if candidate_bandwidths is not None
        else SUPPORTED_SEARCH_SPACE["bandwidth_mhz"]
    )
    pwr_pool = (
        candidate_powers
        if candidate_powers is not None
        else SUPPORTED_SEARCH_SPACE["total_tx_power_dbm"]
    )

    # Restrict strictly to supported domain (no extrapolation)
    for bw in bw_pool:
        if bw not in SUPPORTED_SEARCH_SPACE["bandwidth_mhz"]:
            raise ValueError(
                f"Requested candidate bandwidth {bw} MHz is outside supported values {SUPPORTED_SEARCH_SPACE['bandwidth_mhz']}"
            )
    for pwr in pwr_pool:
        if pwr not in SUPPORTED_SEARCH_SPACE["total_tx_power_dbm"]:
            raise ValueError(
                f"Requested candidate power {pwr} dBm is outside supported values {SUPPORTED_SEARCH_SPACE['total_tx_power_dbm']}"
            )

    candidates = []
    config_id = 1
    for bw in sorted(bw_pool):
        for pwr in sorted(pwr_pool):
            candidates.append({
                "config_id": config_id,
                "ue_count": ue_count,
                "scenario_size_m": float(scenario_size_m),
                "flow_type": flow_type.lower(),
                "lambda_val": float(lambda_val),
                "bandwidth_mhz": float(bw),
                "total_tx_power_dbm": float(pwr),
            })
            config_id += 1

    return candidates

