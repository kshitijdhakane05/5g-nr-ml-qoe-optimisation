"""Evaluation metrics and reporting utilities for 5G ML models."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """Compute regression metrics: MAE, RMSE, R²."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def compute_outage_breakdown_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    outage_threshold: float = 0.0,
) -> dict[str, dict[str, float | int]]:
    """Compute regression metrics separately for outage (y_true <= threshold) and non-outage records."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    outage_mask = y_true_arr <= outage_threshold
    non_outage_mask = ~outage_mask

    breakdown = {}
    for subset_name, mask in [("all", np.ones_like(outage_mask, dtype=bool)), ("outage", outage_mask), ("non_outage", non_outage_mask)]:
        count = int(np.sum(mask))
        if count == 0:
            breakdown[subset_name] = {"count": 0, "MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
        else:
            sub_true = y_true_arr[mask]
            sub_pred = y_pred_arr[mask]
            sub_metrics = compute_metrics(sub_true, sub_pred)
            sub_metrics["count"] = count
            breakdown[subset_name] = sub_metrics

    return breakdown


def summarize_cv_scores(cv_results: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-fold cross-validation results into mean and std."""
    df_cv = pd.DataFrame(cv_results)
    summary = {}
    for col in ["MAE", "RMSE", "R2"]:
        if col in df_cv.columns:
            summary[f"cv_{col.lower()}_mean"] = float(df_cv[col].mean())
            summary[f"cv_{col.lower()}_std"] = float(df_cv[col].std())
    return summary


def save_evaluation_results(results_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save model evaluation metrics to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)
    print(f"Evaluation metrics saved to {path}")


def save_predictions(predictions_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save model predictions on holdout test set to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(path, index=False)
    print(f"Predictions saved to {path}")

