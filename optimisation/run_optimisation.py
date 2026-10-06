"""Command-line entry point for offline ML-based 5G network optimisation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from optimisation.optimizer import NetworkOptimiser


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline ML-based network configuration optimiser using QoE surrogate"
    )
    # Context / Environment parameters
    parser.add_argument("--ue-count", type=int, default=4, help="Number of active UEs (2, 4, 6, 8, 10)")
    parser.add_argument("--scenario-size", type=float, default=30.0, help="Cell area size in meters (3, 30, 100)")
    parser.add_argument(
        "--flow-type",
        choices=["voice_be", "low_latency"],
        default="voice_be",
        help="Traffic flow class",
    )
    parser.add_argument(
        "--lambda-val",
        type=float,
        default=10000.0,
        help="Offered packet arrival rate in pkts/s (5000, 10000, 20000, 40000)",
    )

    # Baseline / Requested configuration
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=20.0,
        dest="baseline_bandwidth",
        help="Current / baseline bandwidth in MHz (default: 20.0)",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=35.0,
        dest="baseline_power",
        help="Current / baseline total Tx power in dBm (default: 35.0)",
    )

    # Optional constraints
    parser.add_argument(
        "--max-power",
        type=float,
        default=None,
        help="Optional constraint: maximum transmit power in dBm",
    )
    parser.add_argument(
        "--min-qoe",
        type=float,
        default=None,
        help="Optional constraint: minimum acceptable predicted QoE",
    )

    # Paths
    parser.add_argument(
        "--model-path",
        default="models/qoe_model.joblib",
        help="Path to Phase 1 trained model artifact",
    )
    parser.add_argument(
        "--output-csv",
        default="results/optimisation/optimisation_results.csv",
        help="Path to save optimisation results CSV",
    )
    args = parser.parse_args()

    # Build constraint dictionary (only non-None values, never invent defaults)
    constraints = {}
    if args.max_power is not None:
        constraints["max_power_dbm"] = args.max_power
    if args.min_qoe is not None:
        constraints["min_predicted_qoe"] = args.min_qoe

    optimiser = NetworkOptimiser(model_path=args.model_path)
    res = optimiser.optimise(
        ue_count=args.ue_count,
        scenario_size_m=args.scenario_size,
        flow_type=args.flow_type,
        lambda_val=args.lambda_val,
        baseline_bandwidth_mhz=args.baseline_bandwidth,
        baseline_power_dbm=args.baseline_power,
        constraints=constraints if constraints else None,
    )

    base = res["baseline"]
    best = res["best_configuration"]
    context = res["context"]
    improvement = res["predicted_qoe_improvement"]

    print("\n" + "=" * 65)
    print("5G NETWORK CONFIGURATION OPTIMISER (PHASE 2)")
    print("Mode: Offline ML-based Network Configuration Optimisation")
    print("=" * 65)

    print("\n[1] NETWORK CONTEXT & REQUIREMENTS:")
    print(f"  UE Count:            {context['ue_count']}")
    print(f"  Scenario Area:       {context['scenario_size_m']} m")
    print(f"  Traffic Class:       {context['flow_type']}")
    print(f"  Packet Arrival Rate: {context['lambda_val']} pkts/s")
    if constraints:
        print(f"  Active Constraints:  {constraints}")
    else:
        print("  Active Constraints:  None specified")

    print("\n[2] CURRENT / REQUESTED BASELINE CONFIGURATION:")
    print(f"  Bandwidth:           {base['bandwidth_mhz']} MHz")
    print(f"  Transmit Power:      {base['total_tx_power_dbm']} dBm")
    print(f"  Predicted QoE:       {base['predicted_qoe']:.4f}")
    print(f"  Constraints Met:     {base['feasible']}")
    if base["violations"]:
        print(f"  Violations:          {base['violations']}")

    print(f"\n[3] SEARCH EVALUATION ({res['num_candidates_evaluated']} candidates evaluated):")
    print("-" * 65)
    print(f"{'Rank':<5} | {'Bandwidth':<10} | {'Tx Power':<9} | {'Pred QoE':<9} | {'Feasible':<9} | Violations")
    print("-" * 65)
    for c in res["all_candidates"][:5]:
        v_str = ", ".join(c["violations"]) if c["violations"] else "None"
        print(
            f"{c['rank']:<5} | {c['bandwidth_mhz']:<7} MHz | {c['total_tx_power_dbm']:<5} dBm | "
            f"{c['predicted_qoe']:<9.4f} | {str(c['feasible']):<9} | {v_str}"
        )
    print("-" * 65)

    print("\n[4] SELECTED OPTIMAL CONFIGURATION:")
    print(f"  Status:              {res['selection_status']}")
    print(f"  Optimised Bandwidth: {best['bandwidth_mhz']} MHz")
    print(f"  Optimised Tx Power:  {best['total_tx_power_dbm']} dBm")
    print(f"  Optimised Pred QoE:  {best['predicted_qoe']:.4f}")
    print(f"  Constraints Met:     {best['feasible']}")

    print("\n[5] BASELINE VS OPTIMISED COMPARISON:")
    print("-" * 65)
    print(f"  Baseline Predicted QoE:   {base['predicted_qoe']:.4f}")
    print(f"  Optimised Predicted QoE:  {best['predicted_qoe']:.4f}")
    print(f"  Predicted QoE Change:     {improvement:+.4f}")
    print(f"  Label:                    {res['predicted_improvement_label']}")
    print("=" * 65 + "\n")

    # Save results to CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    rows_to_save = []
    for c in res["all_candidates"]:
        is_selected = (c["config_id"] == best["config_id"])
        rows_to_save.append({
            "timestamp": timestamp,
            "context_ue_count": context["ue_count"],
            "context_scenario_size_m": context["scenario_size_m"],
            "context_flow_type": context["flow_type"],
            "context_lambda_val": context["lambda_val"],
            "baseline_bandwidth_mhz": base["bandwidth_mhz"],
            "baseline_power_dbm": base["total_tx_power_dbm"],
            "baseline_predicted_qoe": base["predicted_qoe"],
            "candidate_bandwidth_mhz": c["bandwidth_mhz"],
            "candidate_power_dbm": c["total_tx_power_dbm"],
            "predicted_qoe": c["predicted_qoe"],
            "rank": c["rank"],
            "is_feasible": c["feasible"],
            "violations": ";".join(c["violations"]),
            "is_selected": is_selected,
            "predicted_qoe_improvement": improvement if is_selected else None,
            "status": res["selection_status"],
        })

    df_out = pd.DataFrame(rows_to_save)
    if out_path.exists():
        df_out.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df_out.to_csv(out_path, index=False)
    print(f"Optimisation results saved to {out_path}")


if __name__ == "__main__":
    main()

