"""Model training and grouped cross-validation pipeline for 5G QoE prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.evaluate import compute_metrics, save_evaluation_results, save_predictions, summarize_cv_scores
from ml.preprocessing import (
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    TARGET_COLUMN,
    get_features_and_target,
    load_raw_dataset,
    preprocess_dataset,
)


def build_models(random_state: int = 42) -> dict[str, object]:
    """Define candidate baseline regressors."""
    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=100,
            random_state=random_state,
        ),
    }


def run_grouped_cv(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_splits: int = 5,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Run GroupKFold cross-validation ensuring scenarios are never split across folds."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train), start=1):
        X_f_trn, y_f_trn = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        X_f_val, y_f_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        model.fit(X_f_trn, y_f_trn)
        y_val_pred = model.predict(X_f_val)
        metrics = compute_metrics(y_f_val, y_val_pred)
        metrics["fold"] = fold
        fold_results.append(metrics)

    summary = summarize_cv_scores(fold_results)
    return summary, fold_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline ML models for 5G QoE prediction")
    parser.add_argument(
        "--dataset",
        default="data/processed/qoe_dataset.csv",
        help="Path to processed dataset CSV",
    )
    parser.add_argument(
        "--model-out",
        default="models/qoe_model.joblib",
        help="Path to save best trained model",
    )
    parser.add_argument(
        "--results-out",
        default="results/ml/model_results.csv",
        help="Path to save evaluation results CSV",
    )
    parser.add_argument(
        "--preds-out",
        default="results/ml/predictions.csv",
        help="Path to save predictions CSV",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits for cross-validation",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of scenarios to reserve for holdout testing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        # Fall back to raw if processed has not been generated yet
        raw_fallback = Path("data/raw/qoe_dataset.csv")
        if raw_fallback.exists():
            print(f"Processed dataset not found. Loading and preprocessing from {raw_fallback}...")
            df_raw = load_raw_dataset(raw_fallback)
            df = preprocess_dataset(df_raw)
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dataset_path, index=False)
        else:
            raise FileNotFoundError(f"Neither {dataset_path} nor {raw_fallback} found.")
    else:
        df = pd.read_csv(dataset_path)

    print(f"Loaded dataset: {len(df)} rows, {df['scenario_id'].nunique()} unique scenarios.")

    # Extract features, target, and grouping column
    X, y, groups = get_features_and_target(df)

    # GroupShuffleSplit to split whole scenarios into train (80%) and holdout test (20%)
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train, groups_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
    X_test, y_test, groups_test = X.iloc[test_idx], y.iloc[test_idx], groups.iloc[test_idx]

    train_scenarios = set(groups_train.unique())
    test_scenarios = set(groups_test.unique())
    scenario_overlap = train_scenarios.intersection(test_scenarios)

    print("\n" + "=" * 60)
    print("SCENARIO SEPARATION AUDIT")
    print("=" * 60)
    print(f"Total scenarios: {len(groups.unique())}")
    print(f"Training scenarios: {len(train_scenarios)} ({len(X_train)} flow rows)")
    print(f"Holdout test scenarios: {len(test_scenarios)} ({len(X_test)} flow rows)")
    print(f"Scenario overlap between train and test: {len(scenario_overlap)} (must be 0)")
    assert len(scenario_overlap) == 0, "Data leakage detected: scenarios overlap between train and test!"

    models = build_models(random_state=args.random_state)
    results_records = []
    trained_models = {}
    test_predictions = {}

    print("\n" + "=" * 60)
    print(f"CROSS-VALIDATION & HOLDOUT EVALUATION ({args.n_splits}-Fold GroupKFold)")
    print("=" * 60)

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # 1. Grouped Cross-Validation on training scenarios
        cv_summary, _ = run_grouped_cv(
            model=model,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            n_splits=args.n_splits,
        )

        # 2. Retrain model on full training set and evaluate on unseen holdout scenarios
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        test_metrics = compute_metrics(y_test, y_test_pred)

        record = {
            "model_name": name,
            "cv_mae_mean": cv_summary["cv_mae_mean"],
            "cv_mae_std": cv_summary["cv_mae_std"],
            "cv_rmse_mean": cv_summary["cv_rmse_mean"],
            "cv_rmse_std": cv_summary["cv_rmse_std"],
            "cv_r2_mean": cv_summary["cv_r2_mean"],
            "cv_r2_std": cv_summary["cv_r2_std"],
            "test_mae": test_metrics["MAE"],
            "test_rmse": test_metrics["RMSE"],
            "test_r2": test_metrics["R2"],
        }
        results_records.append(record)
        trained_models[name] = model
        test_predictions[name] = y_test_pred

        print(f"  CV   -> MAE: {record['cv_mae_mean']:.4f} (+/- {record['cv_mae_std']:.4f}), "
              f"RMSE: {record['cv_rmse_mean']:.4f} (+/- {record['cv_rmse_std']:.4f}), "
              f"R²: {record['cv_r2_mean']:.4f} (+/- {record['cv_r2_std']:.4f})")
        print(f"  Test -> MAE: {record['test_mae']:.4f}, RMSE: {record['test_rmse']:.4f}, R²: {record['test_r2']:.4f}")

    results_df = pd.DataFrame(results_records)

    # Select best model based on holdout test RMSE
    best_record = results_df.sort_values(by="test_rmse").iloc[0]
    best_model_name = best_record["model_name"]
    best_model = trained_models[best_model_name]

    print("\n" + "=" * 60)
    print(f"BEST MODEL SELECTION: {best_model_name}")
    print(f"Holdout Test RMSE: {best_record['test_rmse']:.4f}, Test R²: {best_record['test_r2']:.4f}")
    print("=" * 60)

    # Save model artifact
    model_out_path = Path(args.model_out)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "test_metrics": {
            "MAE": best_record["test_mae"],
            "RMSE": best_record["test_rmse"],
            "R2": best_record["test_r2"],
        },
        "cv_metrics": {
            "cv_mae_mean": best_record["cv_mae_mean"],
            "cv_rmse_mean": best_record["cv_rmse_mean"],
            "cv_r2_mean": best_record["cv_r2_mean"],
        },
    }
    joblib.dump(bundle, model_out_path)
    print(f"Saved best model bundle to {model_out_path}")

    # Save results table
    save_evaluation_results(results_df, args.results_out)

    # Save test predictions for analysis
    preds_df = df.iloc[test_idx][
        [
            "scenario_id",
            "flow_id",
            "ue_count",
            "bandwidth_mhz",
            "total_tx_power_dbm",
            "scenario_size_m",
            "lambda_be",
            "throughput_mbps",
            "delay_ms",
            "jitter_ms",
            "packet_loss",
            "qoe_proxy",
        ]
    ].copy()
    preds_df = preds_df.rename(columns={"qoe_proxy": "actual_qoe_proxy"})

    for name in models:
        preds_df[f"pred_qoe_{name}"] = test_predictions[name]

    preds_df["residual_best_model"] = preds_df["actual_qoe_proxy"] - preds_df[f"pred_qoe_{best_model_name}"]
    save_predictions(preds_df, args.preds_out)


if __name__ == "__main__":
    main()

