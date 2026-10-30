"""Reproducible audit and grouped evaluation for the external campus 5G dataset.

This model describes the campus testbeds only. Its output must not be used as an
ns-3 prediction because there is no validated mapping between their settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/raw/5g_campus_network_qos_zenodo_13754300"
FEATURES = ["site", "gnb", "bw", "slots", "ratio", "mbpsoffered_uplink", "mbpsoffered_downlink"]
TARGETS = [
    "mbpsactual_uplink", "mbpsactual_downlink",
    "meanjitterms_uplink", "meanjitterms_downlink",
    "meanloss_uplink", "meanloss_downlink",
]
NUMERIC = ["bw", "slots", "ratio", "mbpsoffered_uplink", "mbpsoffered_downlink"]
CATEGORICAL = ["site", "gnb"]
SOURCE_FILES = {
    "ntnu": "ntnu_tput_all_Throughput.csv",
    "wue": "wue_tput_all_Throughput.csv",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for site, name in SOURCE_FILES.items():
        path = data_dir / name
        frame = pd.read_csv(path)
        if "Unnamed: 0" in frame:
            frame = frame.rename(columns={"Unnamed: 0": "source_row_id"})
        elif "" in frame:
            frame = frame.rename(columns={"": "source_row_id"})
        required = set(FEATURES[1:] + TARGETS + ["rep", "scenario"])
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing required columns {sorted(missing)}")
        frame["site"] = site
        frame["source_file"] = name
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result[FEATURES + TARGETS].isna().any().any():
        raise ValueError("Missing values in model features or measured targets")
    for col in NUMERIC + TARGETS:
        result[col] = pd.to_numeric(result[col], errors="raise")
    if result.duplicated(subset=FEATURES + TARGETS + ["rep", "scenario"]).any():
        raise ValueError("Duplicate measurement rows detected")
    # Group all repetitions of each exact configuration across the two sites.
    # Run labels, repetition number and CSV row number are metadata, never features.
    result["configuration_group"] = result[FEATURES[1:] + ["mbpsoffered_uplink", "mbpsoffered_downlink"]].astype(str).agg("|".join, axis=1)
    return result


def make_model(kind: str) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])
    regressors = {
        "mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=100, l2_regularization=1.0, random_state=42),
    }
    return Pipeline([("preprocess", prep), ("regressor", regressors[kind])])


def predict_target(bundle: dict, inputs: pd.DataFrame, target: str) -> dict:
    """Predict within the separate campus domain and report simple support status."""
    if target not in bundle["models"]:
        raise ValueError(f"Unsupported target: {target}")
    if set(FEATURES) - set(inputs.columns):
        raise ValueError(f"Missing features: {sorted(set(FEATURES) - set(inputs.columns))}")
    model = bundle["models"][target]
    prediction = model.predict(inputs[FEATURES])
    categories = bundle["support"]["categories"]
    ranges = bundle["support"]["ranges"]
    warnings = []
    for column, observed in categories.items():
        if any(value not in observed for value in inputs[column]):
            warnings.append(f"{column} includes an unseen category")
    for column, (low, high) in ranges.items():
        if ((inputs[column] < low) | (inputs[column] > high)).any():
            warnings.append(f"{column} is outside the training range [{low}, {high}]")
    return {"target": target, "prediction": prediction, "domain": bundle["domain"],
            "support_status": "outside_training_range" if warnings else "within_marginal_ranges",
            "warnings": warnings,
            "limitation": "Marginal range checks do not prove support for a joint configuration; no ns-3 mapping exists."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/campus_5g_zenodo_13754300")
    args = parser.parse_args()
    df = load_dataset(args.data_dir)
    groups = df["configuration_group"]
    if groups.nunique() < 5:
        raise ValueError("At least five distinct configuration groups are required")
    splitter = GroupKFold(n_splits=min(5, groups.nunique()))
    metrics = []
    bundles = {}
    selected_models = {}
    preds = df[["site", "scenario", "rep", "configuration_group"]].copy()
    for target in TARGETS:
        y = df[target]
        target_results = []
        for kind in ("mean", "ridge", "random_forest", "hist_gradient_boosting"):
            model = make_model(kind)
            prediction = cross_val_predict(model, df[FEATURES], y, groups=groups, cv=splitter, n_jobs=1)
            result = {
                "target": target, "reported_unit": "as provided by source; see provenance limitations",
                "model": kind, "rows": len(df), "configuration_groups": groups.nunique(),
                "folds": min(5, groups.nunique()), "MAE": mean_absolute_error(y, prediction),
                "RMSE": mean_squared_error(y, prediction) ** 0.5,
                "R2": r2_score(y, prediction),
            }
            metrics.append(result)
            target_results.append(result)
            if kind == "random_forest":
                preds[f"{target}_actual"] = y
                preds[f"{target}_predicted_grouped_cv"] = prediction
        selected = min(target_results, key=lambda result: result["RMSE"])["model"]
        selected_models[target] = selected
        final = make_model(selected).fit(df[FEATURES], y)
        bundles[target] = final

    args.output_dir.mkdir(parents=True, exist_ok=True)
    support = {
        "categories": {col: sorted(df[col].astype(str).unique().tolist()) for col in CATEGORICAL},
        "ranges": {col: [float(df[col].min()), float(df[col].max())] for col in NUMERIC},
    }
    joblib.dump({"models": bundles, "selected_models": selected_models,
                 "features": FEATURES, "targets": TARGETS,
                 "support": support,
                 "domain": "Zenodo 13754300 5G campus testbeds; not ns-3"}, args.output_dir / "model.joblib")
    pd.DataFrame(metrics).to_csv(args.output_dir / "evaluation.csv", index=False)
    preds.to_csv(args.output_dir / "grouped_cv_predictions.csv", index=False)
    audit = {
        "rows": len(df), "features": FEATURES, "targets": TARGETS,
        "target_units": {"throughput": "source metadata says MBps while field names say mbps; unresolved",
                         "jitter": "ms", "loss": "source reported mean loss; scale not specified"},
        "missing_values": int(df[FEATURES + TARGETS].isna().sum().sum()),
        "duplicate_rows": int(df.duplicated(subset=FEATURES + TARGETS + ["rep", "scenario"]).sum()),
        "configuration_groups": int(groups.nunique()),
        "folds": min(5, groups.nunique()),
        "group_definition": "site-independent combination of gNB, bandwidth, slots, DL:UL ratio, offered uplink/downlink rates",
        "observed_categories": {col: sorted(df[col].astype(str).unique().tolist()) for col in CATEGORICAL},
        "numeric_ranges": {col: [float(df[col].min()), float(df[col].max())] for col in NUMERIC},
        "target_summaries": {
            target: {"min": float(df[target].min()), "median": float(df[target].median()), "max": float(df[target].max())}
            for target in TARGETS
        },
        "source_checksums_sha256": {name: sha256(args.data_dir / name) for name in SOURCE_FILES.values()},
        "limitations": ["No transmit power, UE count, scenario size, traffic class or project-specific traffic rate.",
                        "No validated equivalence between campus bandwidth/slot/gNB settings and this project's ns-3 knobs.",
                        "Throughput unit and packet-loss scale are not unambiguous in the dataset metadata.",
                        "The fitted models are external-domain research artifacts and must not score ns-3 candidates."],
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    training_metadata = {
        "artifact_version": "campus-zenodo-13754300-v1-20261029",
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "random_state": 42, "cross_validation": "5-fold GroupKFold by configuration_group",
        "selection_rule": "lowest grouped-CV RMSE per target; refit selected estimator on all 302 rows",
        "selected_models": selected_models,
        "feature_columns": FEATURES, "target_columns": TARGETS,
        "source": "https://doi.org/10.5281/zenodo.13754300",
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(training_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"Saved external-domain artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
