"""Human-readable descriptions for the simulator's supported configuration fields."""
from __future__ import annotations

from typing import Any

import pandas as pd


PARAMETERS: tuple[dict[str, Any], ...] = (
    {
        "keys": ("bandwidth_band1_mhz", "bandwidth_mhz"), "name": "Band 1 bandwidth",
        "unit": "MHz", "category": "Radio resources", "valid": "20, 50, or 100 MHz in live sessions",
        "controls": "The width of the first NR operating band (not the total bandwidth).",
        "effect": "More spectrum may provide more capacity, depending on channel and workload.",
        "tradeoff": "Band 2 shares requested transmit power proportionally; wider bandwidth does not guarantee higher QoE.",
    },
    {
        "keys": ("bandwidth_band2_mhz",), "name": "Band 2 bandwidth",
        "unit": "MHz", "category": "Radio resources", "valid": "20, 50, or 100 MHz in live sessions",
        "controls": "The width of the second NR operating band, serving the voice QoS flow in this scenario.",
        "effect": "Changes spectrum available to the second configured bearer class.",
        "tradeoff": "Both bands share requested transmit power proportionally; changing it also changes power allocation.",
    },
    {
        "keys": ("total_tx_power_dbm",), "name": "Requested total transmit power",
        "unit": "dBm", "category": "Radio resources", "valid": "30, 35, or 40 dBm in live sessions",
        "controls": "The requested gNB transmit power before the simulator allocates it across bands.",
        "effect": "More power can improve received signal strength in some conditions.",
        "tradeoff": "The simulator splits power by band bandwidth; propagation and interference also affect results.",
    },
    {
        "keys": ("ue_count",), "name": "Number of user devices",
        "unit": "UEs", "category": "Scenario load", "valid": "2, 6, or 10 in live sessions",
        "controls": "The fixed number of simulated user terminals generating traffic across the configured cells.",
        "effect": "Changes the number of concurrent flows and offered network load.",
        "tradeoff": "UE count is fixed during adaptive exploration and changes scenario composition.",
    },
    {
        "keys": ("gnb_count",), "name": "Number of gNB cells",
        "unit": "gNBs", "category": "Topology", "valid": "1–3 in the bounded live scenario",
        "controls": "The number of NR base stations created by the scenario for each fresh ns-3 process.",
        "effect": "Creates radio cells. Existing UEs can use closest-cell attachment or an explicit round-robin attachment; serving cell IDs are read from UE RRC state.",
        "tradeoff": "The current campaign measured one gNB only. Multi-cell candidates are unvalidated and retain the shared EPC/IP setup; this is not handover or subnet isolation.",
    },
    {
        "keys": ("ue_association_policy",), "name": "UE-to-cell assignment",
        "unit": "policy", "category": "Topology", "valid": "closest or round_robin (round_robin requires at least two gNBs)",
        "controls": "Selects how the scenario attaches the existing UE devices to configured gNBs at simulation setup.",
        "effect": "closest uses ns-3 closest-cell association; round_robin explicitly distributes devices by UE ID across cells.",
        "tradeoff": "This is per-run initial attachment, not mobility or handover. It can place a UE on a farther cell; serving cell IDs must be checked in output.",
    },
    {
        "keys": ("lambda_ull_pps",), "name": "Low-latency traffic rate",
        "unit": "packets/s", "category": "Traffic workload", "valid": "1,000–10,000 packets/s",
        "controls": "Configured UDP packet generation rate for the low-latency QoS class.",
        "effect": "Changes offered packet load for that class.",
        "tradeoff": "Higher load can increase contention; it is not a radio scheduler control.",
    },
    {
        "keys": ("lambda_be_pps", "lambda_be"), "name": "Voice-class traffic rate",
        "unit": "packets/s", "category": "Traffic workload", "valid": "1,000–20,000 packets/s",
        "controls": "Configured UDP packet generation rate for the voice QoS class.",
        "effect": "Changes offered packet load for that class.",
        "tradeoff": "Higher load can increase contention; the class label describes a QoS workload, not a full voice application.",
    },
    {
        "keys": ("lambda_val",), "name": "Selected flow traffic rate",
        "unit": "packets/s", "category": "Traffic workload", "valid": "5,000, 10,000, 20,000, or 40,000 in the legacy search space",
        "controls": "Configured packet rate for the flow class selected in the offline optimizer.",
        "effect": "Changes offered traffic for the selected flow class.",
        "tradeoff": "Higher packet rates increase offered load; this optimizer result is a model estimate, not a simulation measurement.",
    },
    {
        "keys": ("flow_type",), "name": "Traffic class",
        "unit": "class", "category": "Traffic workload", "valid": "voice_be or low_latency in the offline predictor",
        "controls": "Selects the QoS workload represented by the offline optimizer input.",
        "effect": "Changes the model's class-specific input and search comparison.",
        "tradeoff": "This is a modeled flow class; it is not a full application implementation.",
    },
    {
        "keys": ("packet_size_ull_bytes",), "name": "Low-latency UDP payload",
        "unit": "bytes", "category": "Traffic workload", "valid": "64–1,472 bytes",
        "controls": "Payload size of generated low-latency UDP packets.",
        "effect": "Changes the amount of offered data per packet.",
        "tradeoff": "Payload and rate jointly determine offered load; packet headers are additional.",
    },
    {
        "keys": ("packet_size_be_bytes",), "name": "Voice-class UDP payload",
        "unit": "bytes", "category": "Traffic workload", "valid": "64–1,472 bytes",
        "controls": "Payload size of generated voice-class UDP packets.",
        "effect": "Changes the amount of offered data per packet.",
        "tradeoff": "Payload and rate jointly determine offered load; packet headers are additional.",
    },
    {
        "keys": ("scenario_size_m",), "name": "UE placement area",
        "unit": "m", "category": "Scenario", "valid": "30–200 m in live sessions",
        "controls": "The square area used to place user terminals in the grid scenario.",
        "effect": "Changes UE locations and therefore simulated radio conditions.",
        "tradeoff": "Different placements make run-to-run comparisons less directly comparable.",
    },
    {
        "keys": ("sim_time", "sim_time_seconds"), "name": "Simulation duration",
        "unit": "s", "category": "Scenario", "valid": "1–3 s in live sessions",
        "controls": "How long the ns-3 scenario runs and accumulates measurements.",
        "effect": "Longer runs collect traffic over a longer interval.",
        "tradeoff": "Longer runs consume more time; the current outputs remain per-flow run aggregates.",
    },
)


def _value(config: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


def _format_value(value: Any, unit: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        rendered = f"{value:g}"
    else:
        rendered = str(value)
    return f"{rendered} {unit}"


def configuration_change_rows(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    *,
    reason: str | None,
    status: str,
    expected_effect: str | None = None,
    evaluation_result: str | None = None,
    changed_only: bool = True,
) -> list[dict[str, str]]:
    """Return schema-backed comparison rows; effects are explicitly not forecasts."""
    before, after = before or {}, after or {}
    rows: list[dict[str, str]] = []
    why = reason.strip() if isinstance(reason, str) and reason.strip() else "Reason unavailable in the recorded optimizer/controller output."
    for parameter in PARAMETERS:
        old = _value(before, parameter["keys"])
        new = _value(after, parameter["keys"])
        if old is None and new is None:
            continue
        changed = old != new
        if changed_only and not changed:
            continue
        rows.append({
            "Category": parameter["category"],
            "Setting": parameter["name"],
            "Technical field": parameter["keys"][0],
            "Before": _format_value(old, parameter["unit"]),
            "After": _format_value(new, parameter["unit"]),
            "Change": "Changed" if changed else "Unchanged",
            "What it controls": parameter["controls"],
            "Why it changed": why if changed else "No change recorded.",
            "Expected effect": (expected_effect if expected_effect and changed else
                                f"Engineering consideration: {parameter['effect']} Outcome requires simulation."),
            "Evaluation result": evaluation_result or "Not recorded.",
            "Trade-off": parameter["tradeoff"],
            "Status": status,
            "Supported values": parameter["valid"],
        })
    return rows


def render_configuration_change(
    st: Any,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    *,
    reason: str | None,
    status: str,
    expected_effect: str | None = None,
    evaluation_result: str | None = None,
    key: str = "configuration_change",
    changed_only: bool = True,
) -> None:
    rows = configuration_change_rows(
        before, after, reason=reason, status=status, expected_effect=expected_effect,
        evaluation_result=evaluation_result, changed_only=changed_only,
    )
    if not rows:
        st.caption("No supported configuration parameters changed.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", key=key)
    st.caption("Engineering notes describe possible effects, not guaranteed outcomes. Predictions and measured results are reported separately.")
