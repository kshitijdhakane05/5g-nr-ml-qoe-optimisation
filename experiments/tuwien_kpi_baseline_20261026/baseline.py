"""Reproducible, isolated baselines for the processed TU Wien UDP trials.

This module intentionally uses only the Python standard library so it does not
import the production ML stack or require package installation. It reads the
already-audited CSV sample, verifies joins, then runs six-fold condition-grouped
out-of-fold predictions for four separate KPI targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


EXPERIMENT_NAME = "tuwien_kpi_baseline_20261026"
RANDOM_SEED = 20261026
N_SPLITS = 6
EXPECTED_TRIALS = 210
EXPECTED_GROUPS = 42
EXPECTED_REPETITIONS = 5
RIDGE_ALPHA = 1.0
FEATURE_COLUMNS = ("target_Mbps", "qIdx")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "results"
    / "dataset_audit"
    / "20261026_ranalyzer_tuwien"
    / "tuwien_data"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "experiments" / EXPERIMENT_NAME
)

TARGETS = {
    "throughput": {
        "column": "achieved_Mbps_server",
        "unit": "Mbps",
        "definition": "UDP achieved throughput reported by the iPerf server",
        "source": "achieved_bandwidth_210.csv",
    },
    "latency": {
        "column": "latency_avg_ms",
        "unit": "ms",
        "definition": "raw mean one-way latency; only valid_owd=1 labels",
        "source": "latency_trials_raw_210.csv",
    },
    "jitter": {
        "column": "jitter_ms_raw",
        "unit": "ms",
        "definition": "iPerf Server Report jitter",
        "source": "latency_with_achieved_bw_210.csv, cross-checked to raw CSV",
    },
    "packet_loss": {
        "column": "loss_pct_raw",
        "unit": "percentage points",
        "definition": "iPerf Server Report packet-loss percentage, not a fraction",
        "source": "latency_with_achieved_bw_210.csv, cross-checked to raw CSV",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _key(row: dict[str, str], rate_column: str) -> tuple[str, int, int]:
    try:
        location = str(int(row["qIdx"]))
        rate_value = float(row[rate_column])
        if not rate_value.is_integer():
            raise ValueError(f"Non-integral configured rate: {rate_value}")
        rate = int(rate_value)
        trial = int(row["trial"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid trial key in row: {row}") from exc
    return location, rate, trial


def _unique_index(
    rows: Sequence[dict[str, str]], rate_column: str, label: str
) -> dict[tuple[str, int, int], dict[str, str]]:
    indexed: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        key = _key(row, rate_column)
        if key in indexed:
            raise ValueError(f"Duplicate key in {label}: {key}")
        indexed[key] = row
    return indexed


def _same_float(left: str, right: str, *, label: str, key: object) -> None:
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric {label} for {key}: {left!r}, {right!r}") from exc
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(
        a, b, rel_tol=1e-10, abs_tol=1e-10
    ):
        raise ValueError(f"Mismatched {label} for {key}: {left!r} != {right!r}")


def merge_trial_tables(
    raw_rows: Sequence[dict[str, str]],
    joined_rows: Sequence[dict[str, str]],
    achieved_rows: Sequence[dict[str, str]],
    *,
    expected_rows: int = EXPECTED_TRIALS,
) -> list[dict[str, object]]:
    """Match the 3 source tables by (qIdx, configured rate, trial).

    Duplicate keys, missing/extra keys, count mismatches, inconsistent target
    copies, or mismatched latency-validity flags fail closed.
    """
    for label, rows in (
        ("raw", raw_rows), ("joined", joined_rows), ("achieved", achieved_rows)
    ):
        if len(rows) != expected_rows:
            raise ValueError(f"{label} row count {len(rows)} != {expected_rows}")

    raw = _unique_index(raw_rows, "target_Mbps", "raw")
    joined = _unique_index(joined_rows, "B_Mbps", "joined")
    achieved = _unique_index(achieved_rows, "B_target_Mbps", "achieved")
    if set(raw) != set(joined):
        raise ValueError("Raw and joined trial key sets differ")
    if set(raw) != set(achieved):
        raise ValueError("Raw and achieved-throughput trial key sets differ")

    merged: list[dict[str, object]] = []
    for key in sorted(raw, key=lambda item: (int(item[0]), item[1], item[2])):
        source = raw[key]
        joined_row = joined[key]
        achieved_row = achieved[key]
        valid_owd = int(source["valid_owd"])
        corrupt_flag = int(joined_row["latency_corrupted_flag"])
        if valid_owd not in (0, 1) or corrupt_flag != 1 - valid_owd:
            raise ValueError(f"Latency validity flags disagree for {key}")

        _same_float(
            source["achieved_Mbps"], achieved_row["achieved_Mbps_server"],
            label="raw/server achieved throughput", key=key,
        )
        _same_float(
            source["achieved_Mbps"], joined_row["achieved_Mbps_server"],
            label="raw/joined achieved throughput", key=key,
        )
        _same_float(
            source["jitter_ms"], joined_row["jitter_ms_raw"],
            label="jitter", key=key,
        )
        _same_float(
            source["loss_pct"], joined_row["loss_pct_raw"],
            label="packet-loss percentage", key=key,
        )

        item: dict[str, object] = {
            "qIdx": key[0],
            "target_Mbps": key[1],
            "trial": key[2],
            "group_id": f"q{key[0]}_rate{key[1]}",
            "valid_owd": valid_owd,
            "throughput": float(achieved_row["achieved_Mbps_server"]),
            "latency": float(source["latency_avg_ms"]),
            "jitter": float(joined_row["jitter_ms_raw"]),
            "packet_loss": float(joined_row["loss_pct_raw"]),
        }
        for target in TARGETS:
            value = float(item[target])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {target} target for {key}")
        merged.append(item)

    groups: dict[str, int] = defaultdict(int)
    for row in merged:
        groups[str(row["group_id"])] += 1
    if len(groups) != EXPECTED_GROUPS:
        raise ValueError(f"Condition group count {len(groups)} != {EXPECTED_GROUPS}")
    bad_sizes = {group: n for group, n in groups.items() if n != EXPECTED_REPETITIONS}
    if bad_sizes:
        raise ValueError(f"Expected five repetitions in each condition: {bad_sizes}")
    locations = {str(row["qIdx"]) for row in merged}
    rates = {int(row["target_Mbps"]) for row in merged}
    expected_trials = set(range(1, EXPECTED_REPETITIONS + 1))
    if locations != {str(location) for location in range(1, 8)}:
        raise ValueError(f"Unexpected location set: {sorted(locations, key=int)}")
    if rates != {1, 10, 50, 100, 200, 500}:
        raise ValueError(f"Unexpected configured-rate set: {sorted(rates)}")
    for group, group_rows in _group_rows(merged).items():
        trials = {int(row["trial"]) for row in group_rows}
        if trials != expected_trials:
            raise ValueError(f"Unexpected repetition IDs for {group}: {sorted(trials)}")
    return merged


def _group_rows(
    rows: Sequence[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(dict(row))
    return grouped


def load_trials(data_dir: Path = DEFAULT_DATA_DIR) -> list[dict[str, object]]:
    raw = read_csv(data_dir / "latency_trials_raw_210.csv")
    joined = read_csv(data_dir / "latency_with_achieved_bw_210.csv")
    achieved = read_csv(data_dir / "achieved_bandwidth_210.csv")
    return merge_trial_tables(raw, joined, achieved)


def make_condition_folds(
    rows: Sequence[dict[str, object]], *, n_splits: int = N_SPLITS,
    seed: int = RANDOM_SEED,
) -> dict[str, int]:
    """Assign complete location/rate conditions to balanced, seeded folds.

    The dataset has 7 locations at each of 6 rates. A seeded shuffle within
    each rate, followed by a rotating round-robin assignment, yields 7 groups
    per fold. No trial from one condition can cross a fold boundary.
    """
    conditions: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        condition = (str(row["qIdx"]), int(row["target_Mbps"]))
        conditions[condition].append(row)

    rates = sorted({rate for _, rate in conditions})
    if len(conditions) != EXPECTED_GROUPS or len(rates) != 6:
        raise ValueError("Expected 42 condition groups across six configured rates")
    rng = random.Random(seed)
    assignment: dict[str, int] = {}
    for rate_index, rate in enumerate(rates):
        rate_groups = [condition for condition in conditions if condition[1] == rate]
        if len(rate_groups) != 7:
            raise ValueError(f"Expected 7 location groups at {rate} Mbps")
        rng.shuffle(rate_groups)
        for offset, (location, configured_rate) in enumerate(rate_groups):
            fold = (offset + rate_index) % n_splits
            assignment[f"q{location}_rate{configured_rate}"] = fold

    if len(set(assignment.values())) != n_splits:
        raise ValueError("Fold assignment failed to populate every fold")
    fold_sizes = [sum(value == fold for value in assignment.values()) for fold in range(n_splits)]
    if fold_sizes != [7] * n_splits:
        raise ValueError(f"Condition folds are not balanced: {fold_sizes}")
    return assignment


def target_rows(
    rows: Sequence[dict[str, object]], target_name: str
) -> list[dict[str, object]]:
    if target_name not in TARGETS:
        raise KeyError(target_name)
    result = [row for row in rows if target_name != "latency" or row["valid_owd"] == 1]
    return result


def _design_matrix(
    training: Sequence[dict[str, object]],
    predicting: Sequence[dict[str, object]],
) -> tuple[list[list[float]], list[list[float]]]:
    """Fit numeric scaling and categorical levels on training rows only."""
    rates = [float(row["target_Mbps"]) for row in training]
    rate_mean = statistics.fmean(rates)
    rate_std = math.sqrt(statistics.fmean((value - rate_mean) ** 2 for value in rates))
    if not rate_std:
        rate_std = 1.0
    locations = sorted({str(row["qIdx"]) for row in training}, key=int)
    if not locations:
        raise ValueError("No training location categories")

    def encode(row: dict[str, object]) -> list[float]:
        z_rate = (float(row["target_Mbps"]) - rate_mean) / rate_std
        location = str(row["qIdx"])
        # Unknown categories map to the reference level; this dataset's folds
        # retain every location in training, so this is defensive only.
        loc_values = [1.0 if location == level else 0.0 for level in locations[1:]]
        return [1.0, z_rate, z_rate * z_rate, *loc_values]

    return [encode(row) for row in training], [encode(row) for row in predicting]


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for a small ridge system."""
    n = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ArithmeticError("Singular ridge normal-equation matrix")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor:
                for j in range(col, n + 1):
                    augmented[row][j] -= factor * augmented[col][j]
    return [augmented[i][n] for i in range(n)]


def _ridge_fit_predict(
    x_train: Sequence[Sequence[float]], y_train: Sequence[float],
    x_predict: Sequence[Sequence[float]], alpha: float = RIDGE_ALPHA,
) -> list[float]:
    width = len(x_train[0])
    gram = [[0.0] * width for _ in range(width)]
    cross = [0.0] * width
    for x, y in zip(x_train, y_train):
        for i in range(width):
            cross[i] += x[i] * y
            for j in range(width):
                gram[i][j] += x[i] * x[j]
    # Leave intercept unpenalized; penalize standardized rate terms and site dummies.
    for i in range(1, width):
        gram[i][i] += alpha
    coefficients = _solve_linear_system(gram, cross)
    return [sum(a * b for a, b in zip(x, coefficients)) for x in x_predict]


def _mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return statistics.fmean(abs(a - p) for a, p in zip(actual, predicted))


def _rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return math.sqrt(statistics.fmean((a - p) ** 2 for a, p in zip(actual, predicted)))


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate percentile of empty sample")
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _group_bootstrap(
    predictions: Sequence[dict[str, object]], *, seed: int, replicates: int = 1000
) -> dict[str, tuple[float, float]]:
    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        by_group[str(row["group_id"])].append(row)
    groups = sorted(by_group)
    rng = random.Random(seed)
    maes: list[float] = []
    rmses: list[float] = []
    deltas: list[float] = []
    for _ in range(replicates):
        sampled = [rng.choice(groups) for _ in groups]
        draws = [row for group in sampled for row in by_group[group]]
        actual = [float(row["y_true"]) for row in draws]
        mean_pred = [float(row["mean_prediction"]) for row in draws]
        ridge_pred = [float(row["ridge_prediction"]) for row in draws]
        maes.append(_mae(actual, ridge_pred))
        rmses.append(_rmse(actual, ridge_pred))
        deltas.append(_mae(actual, ridge_pred) - _mae(actual, mean_pred))
    return {
        "ridge_mae_ci95": (_percentile(maes, 0.025), _percentile(maes, 0.975)),
        "ridge_rmse_ci95": (_percentile(rmses, 0.025), _percentile(rmses, 0.975)),
        "ridge_minus_mean_mae_ci95": (
            _percentile(deltas, 0.025), _percentile(deltas, 0.975)
        ),
    }


def run_cross_validation(
    rows: Sequence[dict[str, object]], *, seed: int = RANDOM_SEED,
    n_splits: int = N_SPLITS,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Run four separate target CV experiments, returning predictions/metrics."""
    all_predictions: list[dict[str, object]] = []
    fold_metrics: list[dict[str, object]] = []
    assignment = make_condition_folds(rows, n_splits=n_splits, seed=seed)

    for target_index, (target_name, spec) in enumerate(TARGETS.items()):
        eligible = target_rows(rows, target_name)
        predictions: list[dict[str, object]] = []
        for fold in range(n_splits):
            test = [
                row for row in eligible
                if assignment[str(row["group_id"])] == fold
            ]
            train = [
                row for row in eligible
                if assignment[str(row["group_id"])] != fold
            ]
            if not train or not test:
                raise ValueError(f"Empty {target_name} train/test data in fold {fold}")
            if {str(row["qIdx"]) for row in train} != {
                str(row["qIdx"]) for row in eligible
            }:
                raise ValueError(
                    f"Fold {fold + 1} removes a complete location from training; "
                    "this experiment is defined for known-location combinations"
                )
            # Explicit schema-level guard: exactly the two documented pre-run inputs.
            x_train, x_test = _design_matrix(train, test)
            y_train = [float(row[target_name]) for row in train]
            y_test = [float(row[target_name]) for row in test]
            mean_value = statistics.fmean(y_train)
            mean_preds = [mean_value] * len(test)
            ridge_preds = _ridge_fit_predict(x_train, y_train, x_test)
            for row, actual, mean_pred, ridge_pred in zip(
                test, y_test, mean_preds, ridge_preds
            ):
                predictions.append({
                    "target": target_name,
                    "unit": spec["unit"],
                    "qIdx": row["qIdx"],
                    "target_Mbps": row["target_Mbps"],
                    "trial": row["trial"],
                    "group_id": row["group_id"],
                    "fold": fold + 1,
                    "y_true": actual,
                    "mean_prediction": mean_pred,
                    "ridge_prediction": ridge_pred,
                })
            for model_name, preds in (("mean", mean_preds), ("ridge", ridge_preds)):
                fold_metrics.append({
                    "target": target_name,
                    "unit": spec["unit"],
                    "model": model_name,
                    "fold": fold + 1,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_groups": len({str(row["group_id"]) for row in train}),
                    "test_groups": len({str(row["group_id"]) for row in test}),
                    "mae": _mae(y_test, preds),
                    "rmse": _rmse(y_test, preds),
                })
        all_predictions.extend(predictions)

    summaries: dict[str, object] = {}
    for target_index, target_name in enumerate(TARGETS):
        target_predictions = [p for p in all_predictions if p["target"] == target_name]
        actual = [float(p["y_true"]) for p in target_predictions]
        mean_preds = [float(p["mean_prediction"]) for p in target_predictions]
        ridge_preds = [float(p["ridge_prediction"]) for p in target_predictions]
        ci = _group_bootstrap(
            target_predictions, seed=seed + target_index, replicates=1000
        )
        valid_groups = {str(p["group_id"]) for p in target_predictions}
        summaries[target_name] = {
            "unit": TARGETS[target_name]["unit"],
            "n_rows": len(target_predictions),
            "n_condition_groups": len(valid_groups),
            "n_invalid_latency_rows_excluded": (
                EXPECTED_TRIALS - len(target_predictions) if target_name == "latency" else 0
            ),
            "mean_baseline_mae": _mae(actual, mean_preds),
            "mean_baseline_rmse": _rmse(actual, mean_preds),
            "ridge_mae": _mae(actual, ridge_preds),
            "ridge_rmse": _rmse(actual, ridge_preds),
            "ridge_mae_ci95_group_bootstrap": list(ci["ridge_mae_ci95"]),
            "ridge_rmse_ci95_group_bootstrap": list(ci["ridge_rmse_ci95"]),
            "ridge_minus_mean_mae_ci95_group_bootstrap": list(
                ci["ridge_minus_mean_mae_ci95"]
            ),
            "zero_target_rows": (
                sum(float(row["packet_loss"]) == 0.0 for row in rows)
                if target_name == "packet_loss" else None
            ),
        }
    return all_predictions, fold_metrics, summaries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    rows = []
    for target, values in summary.items():
        row = {"target": target}
        row.update(values)  # type: ignore[arg-type]
        for key, value in list(row.items()):
            if isinstance(value, (list, dict)):
                row[key] = json.dumps(value, sort_keys=True)
        rows.append(row)
    _write_csv(path, rows)


def write_results_report(path: Path, summary: dict[str, object]) -> None:
    """Write a concise, human-readable report without overwriting an artifact."""
    lines = [
        "# TU Wien KPI baseline results",
        "",
        "This report summarizes deterministic six-fold out-of-fold predictions from the isolated baseline experiment. Metrics are trial-level and use the target units shown below.",
        "",
        "| Target | n | Groups | Unit | Mean baseline MAE / RMSE | Ridge MAE (95% group-bootstrap CI) | Ridge RMSE (95% group-bootstrap CI) | Ridge minus mean MAE (95% paired group-bootstrap CI) |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for target, values in summary.items():
        mae_ci = values["ridge_mae_ci95_group_bootstrap"]
        rmse_ci = values["ridge_rmse_ci95_group_bootstrap"]
        delta_ci = values["ridge_minus_mean_mae_ci95_group_bootstrap"]
        lines.append(
            f"| {target} | {values['n_rows']} | {values['n_condition_groups']} | {values['unit']} | "
            f"{values['mean_baseline_mae']:.6g} / {values['mean_baseline_rmse']:.6g} | "
            f"{values['ridge_mae']:.6g} ({mae_ci[0]:.6g}, {mae_ci[1]:.6g}) | "
            f"{values['ridge_rmse']:.6g} ({rmse_ci[0]:.6g}, {rmse_ci[1]:.6g}) | "
            f"{delta_ci[0]:.6g}, {delta_ci[1]:.6g} |"
        )
    loss = summary["packet_loss"]
    lines.extend([
        "",
        f"The packet-loss set contains {loss['zero_target_rows']} zero-loss rows and "
        f"{loss['n_rows'] - loss['zero_target_rows']} nonzero-loss rows. Loss is scored in percentage points.",
        "",
        "## Interpretation limits",
        "",
        "The bootstrap resamples the 42 condition groups, not individual trials, and gives uncertainty conditional on this one deployment and its sampled locations/rates. The paired MAE interval describes ridge-minus-mean error in this experiment; it is not evidence of transfer to another network. The packet-loss paired interval includes zero, so the observed MAE difference is unresolved at this resampling level. Other intervals should also be read as descriptive given only 42 condition groups.",
        "",
        "Latency uses 207 raw valid labels; the three flagged Q1/50 Mbps trials are excluded. Throughput, jitter, and loss use all 210 matched UDP trials. Predictors are configured rate and categorical location; no measured KPI is used as input. This is not a subjective QoE evaluation and does not represent all 5G deployments.",
        "",
        "See `run_metadata.json`, `fold_metrics.csv`, and `fold_predictions.csv` for the split assignment, software/runtime details, input checksums, fold-level scores, and out-of-fold records.",
        "",
    ])
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_experiment(
    *, data_dir: Path = DEFAULT_DATA_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = RANDOM_SEED,
) -> dict[str, object]:
    """Run and save the experiment; refuses to overwrite an output directory."""
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; refusing overwrite: {output_dir}")
    rows = load_trials(data_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions, fold_metrics, summary = run_cross_validation(rows, seed=seed)

    _write_csv(output_dir / "fold_predictions.csv", predictions)
    _write_csv(output_dir / "fold_metrics.csv", fold_metrics)
    _write_summary(output_dir / "metrics_summary.csv", summary)
    write_results_report(output_dir / "README.md", summary)

    assignment = make_condition_folds(rows, seed=seed)
    input_files = [
        data_dir / "latency_trials_raw_210.csv",
        data_dir / "latency_with_achieved_bw_210.csv",
        data_dir / "achieved_bandwidth_210.csv",
    ]
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "software": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependencies": "Python standard library only; no package installation",
        },
        "dataset": {
            "title": "Measurement dataset and analysis code for Measurement-Driven Modeling of End-to-End Latency in an Indoor Private Standalone 5G Network",
            "doi": "10.48436/nw34t-zt678",
            "authors": ["Osman Bodur", "Sami Çağlayan", "Neslihan Demir", "Günther Poszvek", "Friedrich Bleicher"],
            "associated_article_doi": "10.3390/network6030078",
            "license": "CC BY 4.0 for data; cite TU Wien record and associated article",
            "data_dir": str(data_dir),
            "input_files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in input_files
            ],
        },
        "preprocessing": {
            "key": ["qIdx", "configured rate in Mbps", "trial"],
            "input_row_counts": {"raw": EXPECTED_TRIALS, "joined": EXPECTED_TRIALS, "achieved": EXPECTED_TRIALS},
            "verified_condition_groups": EXPECTED_GROUPS,
            "repetitions_per_condition": EXPECTED_REPETITIONS,
            "predictors": ["configured rate in Mbps", "qIdx (categorical, known-location estimand)"],
            "excluded_predictor_fields": [
                "trial", "group_id", "date", "time", "source_file", "achieved throughput",
                "latency", "jitter", "packet loss", "packet counters", "validity flags",
                "derived summaries and cleaned/imputed latency",
            ],
            "latency_exclusion": "valid_owd=0: qIdx=1, 50 Mbps, trials 3, 4, 5; 207 observed labels retained; never use latency_210.csv imputed labels",
            "packet_loss": "iPerf Server Report percentage; evaluated in percentage points, not fraction",
        },
        "validation": {
            "method": "6-fold out-of-fold validation grouped by qIdx x configured rate condition; each fold holds out seven complete condition groups",
            "fold_group_counts": [
                sum(fold == assigned for assigned in assignment.values())
                for fold in range(N_SPLITS)
            ],
            "group_assignment": assignment,
            "mean_baseline": "training-fold arithmetic mean",
            "regressor": "ridge regression (alpha=1), intercept unpenalized; standardized numeric rate, its square, and categorical qIdx one-hot features",
            "fold_local_preprocessing": "rate mean/std and location categories fit from each training fold only",
        "uncertainty": "1000-replicate percentile bootstrap resampling whole condition groups; paired MAE-difference interval; descriptive because OOF models have overlapping training folds",
        },
        "targets": TARGETS,
        "limitations": [
            "One testbed deployment, seven fixed locations, and six configured rates; not representative of all 5G deployments.",
            "Condition-grouped folds test held-out combinations of known locations and rates, not unseen deployments or independent network conditions.",
            "Group-bootstrap intervals summarize these OOF predictions and do not fully capture training-set variability from overlapping folds.",
            "No subjective QoE labels; these are four separate network KPI regressions.",
            "iPerf aggregate KPI definitions and testbed path differ from ns-3 per-flow outcomes.",
        ],
        "metrics": summary,
    }
    with (output_dir / "run_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    manifest = run_experiment(data_dir=args.data_dir, output_dir=args.output_dir, seed=args.seed)
    print(f"Saved experiment results to {args.output_dir}")
    for target, metrics in manifest["metrics"].items():
        print(
            f"{target} ({metrics['unit']}): mean MAE={metrics['mean_baseline_mae']:.6g}; "
            f"ridge MAE={metrics['ridge_mae']:.6g}; ridge RMSE={metrics['ridge_rmse']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
