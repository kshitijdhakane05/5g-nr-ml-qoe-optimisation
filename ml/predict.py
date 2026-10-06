"""Inference module for 5G QoE prediction using saved model artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def load_model(model_path: str | Path = "models/qoe_model.joblib") -> dict:
    """Load trained model bundle."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Run training first.")
    bundle = joblib.load(path)
    return bundle


def format_input_features(
    ue_count: int,
    bandwidth_mhz: float,
    total_tx_power_dbm: float = 35.0,
    scenario_size_m: float = 30.0,
    flow_type: str = "voice_be",
    lambda_val: float = 10000.0,
) -> pd.DataFrame:
    """Construct model feature vector from network parameters.
    
    flow_type: 'low_latency' (id=0, pkt=100B, lambda=10000) or 'voice_be' (id=1, pkt=1252B, lambda=lambda_val).
    """
    is_low_lat = (flow_type.lower() in ["low_latency", "lowlat", "ull", "0"])

    flow_type_id = 0 if is_low_lat else 1
    packet_size = 100 if is_low_lat else 1252
    flow_lambda = 10000.0 if is_low_lat else float(lambda_val)

    # Active transmission window is 0.6 seconds in the ns-3 simulation
    tx_packets = flow_lambda * 0.6
    offered_throughput_mbps = (tx_packets * packet_size * 8.0) / (0.6 * 1_000_000.0)

    data = {
        "ue_count": [ue_count],
        "bandwidth_mhz": [bandwidth_mhz],
        "total_tx_power_dbm": [total_tx_power_dbm],
        "scenario_size_m": [scenario_size_m],
        "flow_type_id": [flow_type_id],
        "flow_lambda": [flow_lambda],
        "packet_size_bytes": [packet_size],
        "offered_throughput_mbps": [offered_throughput_mbps],
    }
    return pd.DataFrame(data)


def predict_qoe(model_bundle: dict, features_df: pd.DataFrame) -> np.ndarray:
    """Generate QoE prediction using model bundle."""
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_columns"]
    X = features_df[feature_cols]
    preds = model.predict(X)
    # Clip predictions to valid QoE proxy range [0.0, 1.0]
    preds = np.clip(preds, 0.0, 1.0)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict 5G QoE for a given network configuration")
    parser.add_argument("--model-path", default="models/qoe_model.joblib", help="Path to model artifact")
    parser.add_argument("--ue-count", type=int, default=4, help="Number of UEs in scenario")
    parser.add_argument("--bandwidth", type=float, default=50.0, help="Channel bandwidth in MHz")
    parser.add_argument("--power", type=float, default=35.0, help="Total gNB transmit power in dBm")
    parser.add_argument("--size", type=float, default=30.0, help="Scenario area size in meters")
    parser.add_argument("--flow-type", choices=["low_latency", "voice_be"], default="voice_be", help="Traffic flow class")
    parser.add_argument("--lambda-val", type=float, default=10000.0, help="Packet arrival rate (pkts/s)")
    args = parser.parse_args()

    bundle = load_model(args.model_path)
    print(f"Loaded model '{bundle['model_name']}' from {args.model_path}")

    features_df = format_input_features(
        ue_count=args.ue_count,
        bandwidth_mhz=args.bandwidth,
        total_tx_power_dbm=args.power,
        scenario_size_m=args.size,
        flow_type=args.flow_type,
        lambda_val=args.lambda_val,
    )

    pred = predict_qoe(bundle, features_df)[0]

    print("\n" + "=" * 50)
    print("5G NETWORK CONFIGURATION & PREDICTION")
    print("=" * 50)
    print(f"  UE Count:           {args.ue_count}")
    print(f"  Bandwidth:          {args.bandwidth} MHz")
    print(f"  gNB Tx Power:       {args.power} dBm")
    print(f"  Scenario Size:      {args.size} m")
    print(f"  Flow Type:          {args.flow_type}")
    print(f"  Packet Rate:        {args.lambda_val} pkts/s")
    print(f"  Offered Load:       {features_df['offered_throughput_mbps'].iloc[0]:.2f} Mbps")
    print("-" * 50)
    print(f"  Predicted QoE:      {pred:.4f} / 1.0000")
    print("=" * 50)


if __name__ == "__main__":
    main()

