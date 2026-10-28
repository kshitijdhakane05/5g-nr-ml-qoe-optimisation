"""Validate and evaluate the completed bounded expansion without touching production artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import expansion
from .preprocessing import derive_effective_band_powers_dbm, derive_flow_inputs, recompute_qoe_proxy


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results/experiments/ns3_qoe_expansion_20261026_r1_expansion"
FEATURES = [
    "ue_count", "bandwidth_mhz", "total_tx_power_dbm", "scenario_size_m",
    "flow_type_id", "flow_lambda", "packet_size_bytes", "offered_throughput_mbps",
]
TARGET = "qoe_proxy"
TARGET_UNITS = {
    "qoe_proxy": "unitless [0,1] (synthetic network KPI proxy)",
    "throughput_mbps": "Mbps per flow",
    "delay_ms": "ms per flow",
    "jitter_ms": "ms per flow",
    "packet_loss": "fraction [0,1] per flow (multiply by 100 for percent)",
}
MODEL_SEED = 20261026


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _fit_models(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[DummyRegressor, RandomForestRegressor]:
    mean_model = DummyRegressor(strategy="mean")
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=12, random_state=MODEL_SEED,
        n_jobs=1, min_samples_leaf=1,
    )
    mean_model.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)
    return mean_model, rf_model


def _manual_svg(path: Path, title: str, labels: list[str], series: dict[str, list[float]],
                y_label: str) -> None:
    """Create a small dependency-free SVG grouped bar chart."""
    width, height = 1040, 480
    left, right, top, bottom = 70, 25, 65, 105
    chart_w, chart_h = width - left - right, height - top - bottom
    finite_vals = [v for values in series.values() for v in values if math.isfinite(v)]
    max_value = max(finite_vals, default=1.0) or 1.0
    max_value *= 1.12
    names = list(series)
    group_w = chart_w / max(1, len(labels))
    bar_w = min(28.0, group_w / (len(names) + 1.5))
    colors = ["#64748b", "#2563eb", "#f59e0b"]
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="32" font-family="sans-serif" font-size="19" font-weight="bold">{title}</text>',
        f'<text transform="translate(20 {top + chart_h / 2}) rotate(-90)" font-family="sans-serif" font-size="13">{y_label}</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width-right}" y2="{top + chart_h}" stroke="#475569"/>',
    ]
    for tick in range(5):
        value = max_value * tick / 4
        y = top + chart_h - chart_h * tick / 4
        chunks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        chunks.append(f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.3g}</text>')
    for i, label in enumerate(labels):
        center = left + (i + 0.5) * group_w
        chunks.append(f'<text x="{center:.1f}" y="{top+chart_h+22}" text-anchor="end" transform="rotate(-35 {center:.1f} {top+chart_h+22})" font-family="sans-serif" font-size="10">{label}</text>')
        total_bar_w = len(names) * bar_w + (len(names)-1) * 3
        start = center - total_bar_w / 2
        for j, name in enumerate(names):
            val = series[name][i]
            if not math.isfinite(val):
                continue
            h = chart_h * val / max_value
            x = start + j * (bar_w + 3)
            y = top + chart_h - h
            chunks.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[j % len(colors)]}"/>')
    legend_x = left
    for j, name in enumerate(names):
        x = legend_x + j * 175
        chunks.append(f'<rect x="{x}" y="{height-26}" width="12" height="12" fill="{colors[j % len(colors)]}"/>')
        chunks.append(f'<text x="{x+18}" y="{height-16}" font-family="sans-serif" font-size="12">{name}</text>')
    chunks.append('</svg>')
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def _validate_expanded_table(df: pd.DataFrame, results_dir: Path) -> dict:
    required = {
        "config_id", "repetition_id", "rng_seed", "rng_run", "run_id", "scenario_id",
        "ue_count", "lambda_be", "bandwidth_mhz", "total_tx_power_dbm", "flow_id",
        "tx_packets", "rx_packets", "throughput_mbps", "delay_ms", "jitter_ms",
        "packet_loss", "qoe_proxy", "flow_type", "flow_type_id", "flow_lambda",
        "packet_size_bytes", "offered_throughput_mbps", "scenario_size_m",
    }
    absent = sorted(required - set(df.columns))
    if absent:
        raise ValueError(f"Expanded flow table is missing columns: {absent}")
    if len(df) != 124 or df["config_id"].nunique() != 11 or df["run_id"].nunique() != 22:
        raise ValueError("Expanded table does not have the expected 124 rows / 11 configs / 22 runs")
    if df.isna().any().any() or not np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Expanded table contains missing or non-finite values")
    if df.duplicated().any():
        raise ValueError("Expanded table contains exact duplicate rows")
    key = ["config_id", "repetition_id", "flow_id"]
    if df.duplicated(key).any():
        raise ValueError("Expanded table contains duplicate config/repetition/flow keys")
    pair_counts = df.groupby("config_id")["rng_run"].agg(lambda s: sorted(set(map(int, s))))
    if len(pair_counts) != 11 or not all(v == [1, 2] for v in pair_counts):
        raise ValueError("Each configuration must contain exactly RNG runs 1 and 2")

    mismatch_loss = 0.0
    mismatch_qoe = 0.0
    offered_mismatch = 0
    for row in df.to_dict(orient="records"):
        tx, rx = int(row["tx_packets"]), int(row["rx_packets"])
        expected_loss = max(0.0, min(1.0, 1.0 - rx / tx)) if tx > 0 else 0.0
        mismatch_loss = max(mismatch_loss, abs(float(row["packet_loss"]) - expected_loss))
        mismatch_qoe = max(mismatch_qoe, abs(float(row["qoe_proxy"]) - recompute_qoe_proxy(row)))
        flow = derive_flow_inputs(
            ue_count=int(row["ue_count"]), flow_id=int(row["flow_id"]),
            lambda_be_pps=int(row["lambda_be"]), lambda_ull_pps=10_000,
            packet_size_ull_bytes=100, packet_size_be_bytes=1252,
        )
        if not math.isclose(float(row["offered_throughput_mbps"]),
                            float(flow["offered_throughput_mbps_configured"]), abs_tol=1e-9):
            offered_mismatch += 1
        if rx == 0 and not (float(row["qoe_proxy"]) == 0.0 and float(row["packet_loss"]) == 1.0
                            and float(row["throughput_mbps"]) == 0.0
                            and float(row["delay_ms"]) == 0.0 and float(row["jitter_ms"]) == 0.0):
            raise ValueError("Outage rows do not follow the verified zero-reception output convention")
    if mismatch_loss > 1e-5 or mismatch_qoe > 2e-5 or offered_mismatch:
        raise ValueError("Independent loss/QoE/configured-offered-load validation failed")
    manifest = list(csv.DictReader((results_dir / "run_manifest.csv").open(encoding="utf-8")))
    if len(manifest) != 22 or any(r["status"] != "PASS" or int(r["process_returncode"]) != 0 for r in manifest):
        raise ValueError("Manifest indicates incomplete or unsuccessful designated runs")
    return {
        "rows": int(len(df)), "configuration_groups": int(df.config_id.nunique()),
        "configuration_repetition_pairs": int(df[["config_id", "repetition_id"]].drop_duplicates().shape[0]),
        "repetitions_per_configuration": 2,
        "runs": int(df.run_id.nunique()), "missing_values": int(df.isna().sum().sum()),
        "non_finite_numeric_values": 0, "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_config_repetition_flow_keys": 0,
        "outage_rows": int((df.rx_packets == 0).sum()),
        "zero_reception_rows_validated": int((df.rx_packets == 0).sum()),
        "max_packet_loss_recompute_abs_error": mismatch_loss,
        "max_qoe_recompute_abs_error": mismatch_qoe,
        "offered_load_mismatches": offered_mismatch,
        "schema_and_data_quality_pass": True,
    }


def _load_or_rebuild_feature_table(results_dir: Path) -> tuple[pd.DataFrame, str]:
    source = results_dir / "expanded_flow_dataset.csv"
    df = pd.read_csv(source)
    if "bandwidth_band2_mhz" in df.columns:
        return df, source.name
    # The first campaign export omitted some explicit run metadata columns.
    # Preserve it and write the reconciled table under a distinct filename.
    output = results_dir / "expanded_flow_dataset_reconciled.csv"
    if output.exists():
        existing = pd.read_csv(output)
        if "bandwidth_band2_mhz" in existing.columns:
            return existing, output.name
        raise FileExistsError(f"Existing reconciled dataset is incomplete; refusing to overwrite: {output}")
    cfg_doc = json.loads((results_dir / "campaign_config.json").read_text(encoding="utf-8"))
    configs = {c["run_id"]: c for c in cfg_doc["configurations"]}
    manifest = list(csv.DictReader((results_dir / "run_manifest.csv").open(encoding="utf-8")))
    rows = []
    for record in manifest:
        cfg = configs[record["run_id"]]
        rows.extend(expansion._record_feature_rows(cfg, Path(record["csv_path"])))
    if not rows:
        raise ValueError("Cannot reconstruct configured feature table from run artifacts")
    pd.DataFrame(rows).to_csv(output, index=False)
    return pd.read_csv(output), output.name


def evaluate(results_dir: Path) -> None:
    results_dir = results_dir.resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(results_dir)
    df, dataset_file = _load_or_rebuild_feature_table(results_dir)
    validation = _validate_expanded_table(df, results_dir)
    campaign_validation = json.loads((results_dir / "campaign_validation.json").read_text(encoding="utf-8"))
    for key in ("success_count", "failure_count", "retry_count", "attempted_run_count",
                "total_runtime_seconds", "status"):
        validation[key] = campaign_validation[key]
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)
    groups = df["config_id"].astype(str)
    unique_groups = sorted(groups.unique())
    oof_mean = np.full(len(df), np.nan)
    oof_rf = np.full(len(df), np.nan)
    fold_for_row = np.full(len(df), -1, dtype=int)
    fold_records = []

    # Leave one whole configuration out: neither repetition nor any flow from
    # the held condition can appear in the corresponding training fold.
    for fold, group in enumerate(unique_groups, start=1):
        test_mask = groups == group
        train_mask = ~test_mask
        train_groups, test_groups = set(groups[train_mask]), set(groups[test_mask])
        if train_groups & test_groups:
            raise AssertionError("Configuration leakage across LOCO split")
        mean_model, rf_model = _fit_models(X.loc[train_mask], y.loc[train_mask])
        oof_mean[test_mask.to_numpy()] = mean_model.predict(X.loc[test_mask])
        oof_rf[test_mask.to_numpy()] = rf_model.predict(X.loc[test_mask])
        fold_for_row[test_mask.to_numpy()] = fold
        fold_records.append({
            "fold": fold, "held_out_config_id": group,
            "train_config_groups": len(train_groups), "test_config_groups": len(test_groups),
            "train_rows": int(train_mask.sum()), "test_rows": int(test_mask.sum()),
            "train_repetitions_in_test_config": 0,
        })

    if not np.isfinite(oof_mean).all() or not np.isfinite(oof_rf).all():
        raise ValueError("Incomplete grouped OOF predictions")
    df_pred = df.copy()
    df_pred["loco_fold"] = fold_for_row
    df_pred["actual_qoe_proxy"] = y
    df_pred["pred_mean_baseline"] = oof_mean
    df_pred["pred_experimental_random_forest"] = oof_rf
    df_pred.to_csv(results_dir / "loco_oof_predictions.csv", index=False)

    pred_rows = []
    for model_name, preds in (("training_fold_mean", oof_mean), ("experimental_random_forest", oof_rf)):
        m = _metrics(y.to_numpy(), preds)
        pred_rows.append({"evaluation": "leave_one_configuration_out", "model": model_name,
                          "config_id": "ALL", **m})
        for group in unique_groups:
            mask = groups.to_numpy() == group
            pred_rows.append({"evaluation": "leave_one_configuration_out", "model": model_name,
                              "config_id": group, **_metrics(y.to_numpy()[mask], preds[mask])})
        outage_mask = (df["rx_packets"].to_numpy() == 0)
        for subset, mask in (("outage", outage_mask), ("non_outage", ~outage_mask)):
            pred_rows.append({"evaluation": f"LOCO_{subset}", "model": model_name,
                              "config_id": "ALL", **_metrics(y.to_numpy()[mask], preds[mask])})
    pd.DataFrame(pred_rows).to_csv(results_dir / "loco_metrics.csv", index=False)
    pd.DataFrame(fold_records).to_csv(results_dir / "loco_folds.csv", index=False)

    factor_predictions = []
    factor_metric_rows = []
    factor_specs = (
        ("Band1_100MHz", df.bandwidth_mhz == 100),
        ("Power_40dBm", df.total_tx_power_dbm == 40),
        ("UE_10", df.ue_count == 10),
    )
    factor_folds = []
    for name, held_rows in factor_specs:
        held_groups = set(df.loc[held_rows, "config_id"].astype(str))
        test_mask = df.config_id.astype(str).isin(held_groups)
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            raise ValueError(f"Cannot evaluate factor holdout {name}")
        if set(groups[train_mask]) & set(groups[test_mask]):
            raise AssertionError(f"Factor holdout {name} leaks a configuration")
        mean_model, rf_model = _fit_models(X.loc[train_mask], y.loc[train_mask])
        fold_predictions = {
            "training_fold_mean": mean_model.predict(X.loc[test_mask]),
            "experimental_random_forest": rf_model.predict(X.loc[test_mask]),
        }
        factor_folds.append({
            "holdout": name, "held_out_config_ids": ";".join(sorted(held_groups)),
            "train_config_groups": int(df.loc[train_mask, "config_id"].nunique()),
            "test_config_groups": len(held_groups), "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "training_and_test_config_overlap": 0,
        })
        for method, preds in fold_predictions.items():
            y_test = y.loc[test_mask].to_numpy()
            for ix, (pred_idx, pred_val) in enumerate(zip(df.loc[test_mask].index, preds)):
                row = df.loc[pred_idx].to_dict()
                row.update({"holdout": name, "model": method,
                            "actual_qoe_proxy": float(y.loc[pred_idx]), "prediction": float(pred_val)})
                factor_predictions.append(row)
            factor_metric_rows.append({"holdout": name, "model": method, "config_id": "ALL",
                                       **_metrics(y_test, preds)})
            for config_id in sorted(held_groups):
                m = df.loc[test_mask, "config_id"].astype(str).to_numpy() == config_id
                factor_metric_rows.append({"holdout": name, "model": method, "config_id": config_id,
                                           **_metrics(y_test[m], preds[m])})
    pd.DataFrame(factor_predictions).to_csv(results_dir / "factor_holdout_predictions.csv", index=False)
    pd.DataFrame(factor_metric_rows).to_csv(results_dir / "factor_holdout_metrics.csv", index=False)
    pd.DataFrame(factor_folds).to_csv(results_dir / "factor_holdout_folds.csv", index=False)

    # The production bundle has matching feature names, but its historical
    # training value for offered_throughput_mbps was computed from tx_packets
    # in ml/preprocessing.py; this experiment uses configured rate × payload.
    # That target-correlated definition mismatch makes a scored comparison invalid.
    production_comparison = {
        "scored": False,
        "reason": "Not scored: ml/preprocessing.py generated training offered_throughput_mbps from post-run tx_packets over 0.6 s, while this experiment and ml/predict.py use configured rate × payload. Matching column names do not establish matching feature semantics; using the frozen production bundle would conflate this leakage/definition shift with model quality.",
        "production_model_artifact_modified": False,
        "compatible_feature_names": FEATURES,
    }

    # No model was selected from held-out scores. This artifact is a separate
    # experimental fit on all 11 validated conditions for future inspection.
    final_model = RandomForestRegressor(n_estimators=100, max_depth=12,
                                        random_state=MODEL_SEED, n_jobs=1, min_samples_leaf=1)
    final_model.fit(X, y)
    model_path = results_dir / "experimental_qoe_random_forest_all_conditions.joblib"
    joblib.dump({
        "model_name": "ExperimentalRandomForestRegressor",
        "model": final_model,
        "feature_columns": FEATURES,
        "target_column": TARGET,
        "target_definition": "synthetic ns-3 qoe_proxy, not subjective user QoE",
        "fit_data": "all 124 flow rows from 11 validated configurations × 2 RNG repetitions",
        "not_for_deployment": True,
        "evaluation_method": "leave-one-configuration-out predictions; model class predeclared",
        "random_seed": MODEL_SEED,
    }, model_path)

    # Per-configuration observed target summaries for the final report.
    summary = df.groupby("config_id", sort=True).agg(
        repetitions=("repetition_id", "nunique"), flow_rows=("flow_id", "size"),
        ue_count=("ue_count", "first"), bandwidth_band1_mhz=("bandwidth_mhz", "first"),
        bandwidth_band2_mhz=("bandwidth_band2_mhz", "first"),
        total_tx_power_dbm=("total_tx_power_dbm", "first"),
        offered_load_mean_mbps=("offered_throughput_mbps", "mean"),
        outage_rows=("rx_packets", lambda s: int((s == 0).sum())),
        qoe_proxy_mean=("qoe_proxy", "mean"), throughput_mean_mbps=("throughput_mbps", "mean"),
        delay_mean_ms=("delay_ms", "mean"), jitter_mean_ms=("jitter_ms", "mean"),
        packet_loss_mean_fraction=("packet_loss", "mean"),
    ).reset_index()
    rng_qoe = df.pivot_table(index="config_id", columns="rng_run", values="qoe_proxy", aggfunc="mean")
    rng_qoe = rng_qoe.rename(columns={1: "qoe_proxy_rng_run_1_mean", 2: "qoe_proxy_rng_run_2_mean"}).reset_index()
    summary = summary.merge(rng_qoe, on="config_id", validate="one_to_one")
    summary["qoe_proxy_rng_pair_abs_difference"] = (
        summary["qoe_proxy_rng_run_1_mean"] - summary["qoe_proxy_rng_run_2_mean"]
    ).abs()
    summary.to_csv(results_dir / "configuration_kpi_summary.csv", index=False)

    # Dependency-free plots because matplotlib is not installed in the project venv.
    loco_frame = pd.DataFrame(pred_rows)
    qoe_mae = loco_frame[(loco_frame.evaluation == "leave_one_configuration_out") & (loco_frame.config_id != "ALL")]
    label_order = sorted(qoe_mae.config_id.unique())
    mean_series = [float(qoe_mae[(qoe_mae.model == "training_fold_mean") & (qoe_mae.config_id == c)].MAE.iloc[0]) for c in label_order]
    rf_series = [float(qoe_mae[(qoe_mae.model == "experimental_random_forest") & (qoe_mae.config_id == c)].MAE.iloc[0]) for c in label_order]
    _manual_svg(results_dir / "loco_mae_by_configuration.svg", "Grouped OOF QoE prediction error by held-out configuration",
                label_order, {"training-fold mean": mean_series, "experimental RF": rf_series}, "MAE (QoE proxy units)")
    _manual_svg(results_dir / "observed_qoe_by_configuration.svg", "Observed mean synthetic QoE proxy by configuration",
                summary.config_id.tolist(), {"mean qoe_proxy": summary.qoe_proxy_mean.astype(float).tolist()}, "mean QoE proxy [0,1]")

    validation["model_evaluation"] = {
        "target": TARGET, "target_units": TARGET_UNITS[TARGET], "features": FEATURES,
        "model": "RandomForestRegressor(n_estimators=100,max_depth=12,random_state=20261026)",
        "baseline": "DummyRegressor(strategy='mean') fitted separately inside each training fold",
        "groups": 11, "loco_folds": len(fold_records), "fold_group_overlap": 0,
        "factor_holdouts": factor_folds,
        "production_model_comparison": production_comparison,
        "experimental_model_artifact": model_path.name,
        "evaluated_dataset": dataset_file,
        "full_data_export_is_not_an_evaluation_artifact": True,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "joblib": joblib.__version__},
    }
    if dataset_file == "expanded_flow_dataset_reconciled.csv":
        validation["export_note"] = (
            "The initial expanded_flow_dataset.csv combined table omitted explicit Band 2/effective-power metadata and is preserved. "
            "expanded_flow_dataset_reconciled.csv was rebuilt from per-run CSVs and run metadata without changing raw measurement values."
        )
    else:
        validation["export_note"] = (
            "The expanded flow table was generated with per-run Band 2/effective-power metadata and configured offered load."
        )
    (results_dir / "model_validity.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(results_dir, validation, summary, pd.DataFrame(pred_rows), pd.DataFrame(factor_metric_rows), production_comparison)


def _write_report(results_dir: Path, validation: dict, summary: pd.DataFrame,
                  loco: pd.DataFrame, factor_metrics: pd.DataFrame,
                  production_comparison: dict) -> None:
    overall = loco[(loco.evaluation == "leave_one_configuration_out") & (loco.config_id == "ALL")]
    outages = int(validation["outage_rows"])
    total = int(validation["rows"])
    durations = [float(r["duration_seconds"]) for r in csv.DictReader((results_dir / "run_manifest.csv").open(encoding="utf-8"))]
    table = [
        "# Bounded ns-3 QoE expansion and evaluation",
        "",
        f"Completed (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Objective and design",
        "",
        "Evaluate QoE-proxy response across the predeclared L9 fractional array plus two anchors, with Band 2 fixed at 50 MHz. The design has 11 conditions and two RNG repetitions each (22 designated runs). It is one ns-3 scenario family, not broad 5G deployment evidence.",
        "",
        "The exact design and source rationale are recorded in `experiments/ns3_qoe_expansion_20261026_r1/EXPANSION_DESIGN.md`; the commands and per-run metadata are in `run_manifest.csv` and each run's `run_config.json`.",
        "",
        "| Config | Band1 MHz | Band2 MHz | Total power dBm | Effective B1 dBm | Effective B2 dBm | UEs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg, b1, pwr, ues in expansion.EXPANSION_TABLE:
        p1, p2 = derive_effective_band_powers_dbm(pwr, b1, 50)
        table.append(f"| {cfg} | {b1} | 50 | {pwr} | {p1:.3f} | {p2:.3f} | {ues} |")
    table.extend([
        "",
        "RNG seed was 20261026 for all runs; each condition used `RngRun=1,2`. They are paired by run number for bookkeeping, but changing conditions can alter event sequences, so equal RNG numbers do not guarantee identical realized random events.",
        "",
        "## Environment and execution",
        "",
        f"- Designated runs: **{validation['success_count']} successful, {validation['failure_count']} failed, {validation['retry_count']} retried**.",
        f"- Total simulator time: {sum(durations):.2f} s; per-run min/median/max: {min(durations):.2f}/{float(np.median(durations)):.2f}/{max(durations):.2f} s.",
        f"- Flow rows: {total} across 11 configurations × two repetitions; timeout was 120 s per run.",
        "- Simulator: prebuilt ns-3.48 scratch executable; no dependencies installed.",
        "- Band 2 explicitly 50 MHz; one gNB; 30 m area; 1 s simulation; app starts 0.4 s; BWP numerologies 4 and 2.",
        "- Total requested power is split by bandwidth ratio in the simulator. Effective per-band power is recorded in `run_manifest.csv`.",
        "",
        "The best-effort rate was explicitly raised from the pilot's 5,000 to 20,000 packets/s, following the pilot report's proposal to expose bandwidth/power sensitivity under higher load. The low-load pilot is preserved as a separate reference. Configured payload offered load is 20,000 × 1,252 × 8 / 1e6 = **200.32 Mbps per best-effort flow**; low-latency traffic remains 10,000 × 100 × 8 / 1e6 = **8 Mbps per low-latency flow**. This is application payload rate, excluding headers, calculated from configured rate and packet size, not TX packet counters.",
        "",
        "## Data validation",
        "",
        f"- Schema: 15 simulator fields plus experiment metadata/derived configured features in `{validation['model_evaluation']['evaluated_dataset']}`; {validation['missing_values']} missing, {validation['non_finite_numeric_values']} non-finite, {validation['exact_duplicate_rows']} exact duplicate rows.",
        f"- Unique condition/repetition/flow keys: passed. Manifest: all {validation['runs']} run IDs and scenario IDs unique; each condition has RNG runs 1 and 2.",
        f"- Independent packet-loss recomputation max absolute error: {validation['max_packet_loss_recompute_abs_error']:.3g}; independent QoE recomputation max absolute error: {validation['max_qoe_recompute_abs_error']:.3g}.",
        f"- Configured offered-load mismatch count: {validation['offered_load_mismatches']}.",
        f"- Export note: {validation['export_note']}",
        f"- Outages: **{outages}/{total} flow rows** ({100*outages/total:.1f}%). Outage rows have zero received packets, throughput/delay/jitter/QoE proxy = 0, and packet-loss fraction = 1.",
        "- All 36 outage rows occurred in the three Band 1=20 MHz conditions. No outage rows were removed or imputed.",
        "",
        "## Measured KPI summary",
        "",
        "Values below are means over flow rows and both repetitions within each configuration. Packet loss is a fraction; multiply by 100 for percent.",
        "",
        "| Config | UEs | B1 MHz | power dBm | QoE mean | QoE RngRun1 / RngRun2 | Throughput Mbps/flow | Delay ms | Jitter ms | Loss fraction | Outage rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in summary.itertuples(index=False):
        table.append(f"| {r.config_id} | {r.ue_count} | {r.bandwidth_band1_mhz} | {r.total_tx_power_dbm} | {r.qoe_proxy_mean:.4f} | {r.qoe_proxy_rng_run_1_mean:.4f} / {r.qoe_proxy_rng_run_2_mean:.4f} | {r.throughput_mean_mbps:.3f} | {r.delay_mean_ms:.3f} | {r.jitter_mean_ms:.4f} | {r.packet_loss_mean_fraction:.4f} | {r.outage_rows} |")
    table.extend([
        "",
        "The RNG1/RNG2 QoE values are per-run means; their differences are descriptive only (two repetitions per condition are not enough for a stable uncertainty estimate, and shared RNG indices do not guarantee common random-number coupling).",
        "",
        "The 20 MHz conditions yielded complete outage at all three tested requested powers/UE-count combinations. Among non-outage settings, throughput, latency, loss and QoE varied across power, bandwidth and load. These effects are descriptive: the fractional design confounds factors within some rows, two repetitions give little precision, and no monotonic response is expected or claimed.",
        "",
        "## Experimental QoE model evaluation",
        "",
        "Target: synthetic `qoe_proxy` in unitless [0,1], computed from measured throughput, delay, jitter and packet loss by the simulator's fixed formula; it is not subjective user QoE. Predictors are pre-run: UE count, Band 1 MHz, requested total power dBm, scenario area m, flow type ID, configured packet rate (packets/s), configured payload bytes, and `offered_throughput_mbps` from configured rate × payload. No measured KPI, TX/RX counter, scenario/run ID, RNG seed or repetition ID is a predictor.",
        "",
        "The experimental Random Forest was predeclared (100 trees, max depth 12, seed 20261026). The mean baseline is refit on each training fold. Evaluation is leave-one-configuration-out (LOCO): all flow rows and both repetitions from one condition are held out together; there are 11 folds and no configuration overlap. Features are numeric and complete, so no imputation, scaling or data-dependent feature transforms were fit outside the fold.",
        "",
        "| Model | LOCO rows | MAE | RMSE |",
        "|---|---:|---:|---:|",
    ])
    for r in overall.itertuples(index=False):
        table.append(f"| {r.model} | {r.n} | {r.MAE:.5f} | {r.RMSE:.5f} |")
    table.extend([
        "",
        "LOCO performance by held-out configuration:",
        "",
        "| Config | Rows | Mean MAE | Mean RMSE | RF MAE | RF RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for config_id in sorted(loco.loc[(loco.evaluation == "leave_one_configuration_out") & (loco.config_id != "ALL"), "config_id"].unique()):
        base = loco[(loco.evaluation == "leave_one_configuration_out") &
                    (loco.config_id == config_id) & (loco.model == "training_fold_mean")].iloc[0]
        model = loco[(loco.evaluation == "leave_one_configuration_out") &
                     (loco.config_id == config_id) & (loco.model == "experimental_random_forest")].iloc[0]
        table.append(f"| {config_id} | {int(base.n)} | {base.MAE:.5f} | {base.RMSE:.5f} | {model.MAE:.5f} | {model.RMSE:.5f} |")
    table.extend([
        "",
        "LOCO errors by observed outage status:",
        "",
        "| Subset | Model | Rows | MAE | RMSE |",
        "|---|---|---:|---:|---:|",
    ])
    outage_metrics = loco[loco.evaluation.isin(["LOCO_outage", "LOCO_non_outage"])]
    for r in outage_metrics.itertuples(index=False):
        subset = "Outage" if r.evaluation == "LOCO_outage" else "Non-outage"
        table.append(f"| {subset} | {r.model} | {r.n} | {r.MAE:.5f} | {r.RMSE:.5f} |")
    table.extend([
        "",
        "Per-configuration LOCO metrics are in `loco_metrics.csv`; row-level out-of-fold predictions are in `loco_oof_predictions.csv`. Outage and non-outage errors are reported separately in the same metrics file.",
        "",
        "### Factor-level held-out checks",
        "",
        "Separate evaluations exclude every condition containing the factor level, with all flows/repetitions for those conditions held out. These tests are overlapping, descriptive subsets, not independent untouched final tests.",
        "",
        "| Held-out level | Model | Held-out configurations | Rows | MAE | RMSE |",
        "|---|---|---|---:|---:|---:|",
    ])
    folds = pd.read_csv(results_dir / "factor_holdout_folds.csv")
    for f in folds.itertuples(index=False):
        rows = factor_metrics[(factor_metrics.holdout == f.holdout) & (factor_metrics.config_id == "ALL")]
        for r in rows.itertuples(index=False):
            table.append(f"| {f.holdout} | {r.model} | {f.held_out_config_ids} | {r.n} | {r.MAE:.5f} | {r.RMSE:.5f} |")
    table.extend([
        "",
        "### Production-model comparison",
        "",
        production_comparison["reason"],
        "The production artifact `models/qoe_model.joblib` was read only and not modified. Its feature names match, but the training feature semantics do not; no production-vs-experimental accuracy claim is reported. The experimental artifact is saved separately as `experimental_qoe_random_forest_all_conditions.joblib`; it is a final fit on all 124 rows, **not** the evaluation model and not for deployment.",
        "",
        "## Report-ready plots",
        "",
        "- `loco_mae_by_configuration.svg`: grouped out-of-fold MAE by condition for the training-fold mean and experimental Random Forest.",
        "- `observed_qoe_by_configuration.svg`: measured mean synthetic QoE proxy by condition.",
        "",
        "## Limitations and conclusion",
        "",
        f"Within this dataset only, the predeclared RF achieved grouped LOCO MAE {float(overall[overall.model == 'experimental_random_forest'].MAE.iloc[0]):.5f} versus {float(overall[overall.model == 'training_fold_mean'].MAE.iloc[0]):.5f} for the fold-mean baseline (RMSE {float(overall[overall.model == 'experimental_random_forest'].RMSE.iloc[0]):.5f} versus {float(overall[overall.model == 'training_fold_mean'].RMSE.iloc[0]):.5f}). This supports a limited within-study predictive advantage over that simple baseline on this simulated proxy. It does not establish improvement over the production model or an optimizer.",
        "This is one simulated scenario family and 11 design conditions, with two RNG runs per condition. The repetitions are limited, flow rows within a condition are correlated, and shared RNG indices do not guarantee paired random events. Only one best-effort load level was used in the expansion. The DOE is fractional, so main effects/interactions are not fully identifiable. QoE is a deterministic proxy over simulated KPIs. The existing production feature-generation mismatch prevents a fair frozen-model score. These results do not establish robust model generalization, an optimization benefit, live-network control, or real-world 5G performance.",
        "",
        "## Reproduction commands",
        "",
        "Run the campaign only with a fresh output directory:",
        "",
        "```bash",
        ".venv/bin/python -m experiments.ns3_qoe_expansion_20261026_r1.expansion --output-root results/experiments/ns3_qoe_expansion_20261026_r1_expansion_reproduction",
        "```",
        "",
        "After all 22 runs pass validation, reproduce the model/evaluation artifacts:",
        "",
        "```bash",
        ".venv/bin/python -m experiments.ns3_qoe_expansion_20261026_r1.evaluate_expansion --results-dir results/experiments/ns3_qoe_expansion_20261026_r1_expansion",
        "```",
        "",
        "Campaign raw per-run CSV/log/summary artifacts, the reconciled validated flow table, manifests, predictions, metrics, plots and experimental model all reside under `results/experiments/ns3_qoe_expansion_20261026_r1_expansion/`. No production model, optimizer, controller or dashboard path was changed.",
    ])
    (results_dir / "expansion_report.md").write_text("\n".join(table) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    evaluate(args.results_dir)


if __name__ == "__main__":
    main()
