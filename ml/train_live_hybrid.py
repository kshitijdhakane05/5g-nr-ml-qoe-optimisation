"""Train a separate, grouped-held-out KPI model for compatible live ns-3 advice."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from simulation.hybrid_advisor import ARTIFACT_DIR, CONTEXT, FEATURES, TARGETS, file_sha256
from simulation.result_parser import calculate_qoe_proxy

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/qoe_dataset.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data_path = args.dataset.resolve()
    frame = pd.read_csv(data_path)
    required = set(FEATURES + TARGETS + ["scenario_id", "rx_packets", "flow_type"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Dataset contains missing features, targets, or grouping labels")
    duplicate_keys = FEATURES + TARGETS + ["scenario_id"]
    if "flow_id" in frame:
        duplicate_keys.append("flow_id")  # identity is audit-only, never a feature
    if frame.duplicated(subset=duplicate_keys).any():
        raise ValueError("Duplicate scenario/measurement rows detected")
    groups = frame["scenario_id"].astype(str)
    if groups.nunique() < 10:
        raise ValueError("At least 10 independent scenario groups are required")
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.20,
                                                 random_state=args.seed).split(frame, groups=groups))
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    if set(train.scenario_id.astype(str)) & set(test.scenario_id.astype(str)):
        raise AssertionError("Scenario leakage across train/test split")
    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train[TARGETS], test[TARGETS]

    # A simple baseline is retained alongside the more flexible forest.
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
        max_features=1.0, random_state=args.seed, n_jobs=1).fit(X_train, y_train)
    receive_model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
        max_features=1.0, class_weight="balanced", random_state=args.seed, n_jobs=1)
    receive_model.fit(X_train, (train["rx_packets"] > 0).astype(int))
    baseline_pred, pred = baseline.predict(X_test), model.predict(X_test)
    metrics: dict[str, dict[str, float]] = {}
    for i, target in enumerate(TARGETS):
        metrics[target] = {
            "baseline_mae": float(mean_absolute_error(y_test.iloc[:, i], baseline_pred[:, i])),
            "baseline_rmse": float(np.sqrt(mean_squared_error(y_test.iloc[:, i], baseline_pred[:, i]))),
            "model_mae": float(mean_absolute_error(y_test.iloc[:, i], pred[:, i])),
            "model_rmse": float(np.sqrt(mean_squared_error(y_test.iloc[:, i], pred[:, i]))),
            "model_r2": float(r2_score(y_test.iloc[:, i], pred[:, i])),
        }
    receive_probability = receive_model.predict_proba(X_test)[:, list(receive_model.classes_).index(1)]
    receive_actual = (test["rx_packets"].to_numpy() > 0).astype(int)
    receive_metrics = {"accuracy": float(accuracy_score(receive_actual, receive_probability >= .5)),
                       "brier_score": float(brier_score_loss(receive_actual, receive_probability)),
                       "positive_rate": float(receive_actual.mean())}

    qoe_metrics = {}
    test_rows = test.reset_index(drop=True)
    for label, traffic_class in (("low_latency", "low_latency"), ("voice_qos", "voice_qos")):
        mask = test_rows["flow_type"].eq("low_latency" if label == "low_latency" else "voice_be").to_numpy()
        if not mask.any():
            qoe_metrics[label] = {"rows": 0, "mae": None, "baseline_mae": None}
            continue
        y_subset = y_test.to_numpy()[mask]
        predicted_subset = pred[mask]
        baseline_subset = baseline_pred[mask]
        # Compute scores with each test row's observed reception status.
        valid_rows = test_rows.loc[mask].reset_index(drop=True)
        observed_receive = (valid_rows["rx_packets"].to_numpy() > 0).astype(float)
        def qoe(rows: np.ndarray, received_probability: np.ndarray) -> np.ndarray:
            scores = []
            for i, row in enumerate(rows):
                scores.append(calculate_qoe_proxy(
                    max(0.0, float(row[0])), max(0.0, float(row[1])), max(0.0, float(row[2])),
                    min(1.0, max(0.0, float(row[3]))), rx_packets=1,
                    traffic_class=traffic_class, formula_version="service_v2") * received_probability[i])
            return np.asarray(scores)
        mask_positions = np.flatnonzero(mask)
        qoe_metrics[label] = {"rows": int(mask.sum()),
            "mae": float(mean_absolute_error(qoe(y_subset, observed_receive),
                qoe(predicted_subset, receive_probability[mask_positions]))),
            "baseline_mae": float(mean_absolute_error(qoe(y_subset, observed_receive),
                qoe(baseline_subset, receive_probability[mask_positions])))}

    # Refit on all grouped ns-3 data for the separately versioned advisory artifact.
    final_model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
        max_features=1.0, random_state=args.seed, n_jobs=1).fit(frame[FEATURES], frame[TARGETS])
    final_receive_model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
        max_features=1.0, class_weight="balanced", random_state=args.seed, n_jobs=1)
    final_receive_model.fit(frame[FEATURES], (frame["rx_packets"] > 0).astype(int))
    contexts = frame[CONTEXT].drop_duplicates().astype(float).values.tolist()
    bundle = {"model": final_model, "receive_model": final_receive_model,
              "feature_columns": FEATURES, "target_columns": TARGETS,
              "supported_contexts": contexts}
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "kpi_model.joblib"
    metadata_path = args.out / "metadata.json"
    joblib.dump(bundle, model_path)
    metadata = {
        "artifact_version": "live-ns3-kpi-v1", "created_by": "ml/train_live_hybrid.py",
        "dataset_path": str(data_path.relative_to(ROOT) if data_path.is_relative_to(ROOT) else data_path),
        "dataset_sha256": file_sha256(data_path), "rows": int(len(frame)),
        "raw_source_path": "data/raw/qoe_dataset.csv",
        "raw_source_sha256": file_sha256(ROOT / "data/raw/qoe_dataset.csv")
            if (ROOT / "data/raw/qoe_dataset.csv").is_file() else None,
        "dataset_generation": "Project ns-3 scratch/qoe-dataset.cc flow-level measurements; processed features from ml/preprocessing.py. Not the external 5G campus dataset.",
        "independent_scenario_groups": int(groups.nunique()),
        "training_rows": int(len(train)), "training_groups": int(train.scenario_id.nunique()),
        "holdout_rows": int(len(test)), "holdout_groups": int(test.scenario_id.nunique()),
        "split": "GroupShuffleSplit by scenario_id; test_size=0.20; no scenario overlap",
        "random_state": int(args.seed), "features": FEATURES, "targets": TARGETS,
        "target_units": {"throughput_mbps": "Mbps", "delay_ms": "ms", "jitter_ms": "ms",
                         "packet_loss": "fraction [0,1]"},
        "qoe_evaluation": "QoE v2 recomputed from held-out predicted KPI values and predicted receive probability; actual held-out QoE uses observed reception status. QoE is not a trained label.",
        "classification_target": "received (rx_packets > 0); predicts reception probability",
        "qoe_metrics": qoe_metrics, "kpi_metrics": metrics, "reception_classifier_metrics": receive_metrics,
        "supported_context_count": len(contexts),
        "safety_scope": {"traffic_mix": "legacy only", "gNB count": 1,
            "total_tx_power_dbm": 35.0, "context": "exact observed UE/radio/workload tuples only",
            "unsupported": "NGMN gaming/video, multi-cell, other power and unobserved workload contexts",
            "not_modeled": "per-UE location, serving cell, or radio-channel realization; same class/config UEs share a prediction"},
        "software": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "model_sha256": file_sha256(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(model_path), "metadata": str(metadata_path),
                      "rows": len(frame), "groups": groups.nunique(), "metrics": metrics,
                      "qoe_metrics": qoe_metrics}, indent=2))


if __name__ == "__main__":
    main()
