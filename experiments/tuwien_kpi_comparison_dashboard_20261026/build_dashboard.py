"""Build a self-contained offline comparison dashboard from saved OOF results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from experiments.tuwien_kpi_baseline_20261026.baseline import (
    DEFAULT_DATA_DIR,
    EXPECTED_GROUPS,
    EXPECTED_TRIALS,
    REPO_ROOT,
    TARGETS,
    load_trials,
)


EXPERIMENT_DIR = REPO_ROOT / "results" / "experiments" / "tuwien_kpi_baseline_20261026"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "experiments" / "tuwien_kpi_comparison_dashboard_20261026"
)
TARGET_VALUE_FIELDS = {
    "throughput": "throughput",
    "latency": "latency",
    "jitter": "jitter",
    "packet_loss": "packet_loss",
}
EXPECTED_NEGATIVE_RIDGE = {"jitter": 35, "packet_loss": 10}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Metric inputs must have equal nonzero lengths")
    errors = [prediction - observed for observed, prediction in zip(actual, predicted)]
    return {
        "mae": statistics.fmean(abs(error) for error in errors),
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
    }


def rate_only_prediction(
    rows: Sequence[dict[str, object]],
    group_assignment: dict[str, int],
    *,
    test_fold: int,
    target: str,
    configured_rate: int,
) -> float:
    """Mean target at one configured rate from training groups only."""
    field = TARGET_VALUE_FIELDS[target]
    eligible = [
        row for row in rows
        if int(row["target_Mbps"]) == configured_rate
        and group_assignment[str(row["group_id"])] != test_fold
        and (target != "latency" or row["valid_owd"] == 1)
    ]
    if not eligible:
        raise ValueError(f"No training observations for {target} at {configured_rate} Mbps")
    if any(group_assignment[str(row["group_id"])] == test_fold for row in eligible):
        raise AssertionError("Held-out condition entered rate-only training baseline")
    return statistics.fmean(float(row[field]) for row in eligible)


def build_comparison(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    experiment_dir: Path = EXPERIMENT_DIR,
) -> dict[str, object]:
    """Join saved predictions to source targets and add leak-safe rate means."""
    metadata = json.loads((experiment_dir / "run_metadata.json").read_text(encoding="utf-8"))
    saved_metrics_rows = read_csv(experiment_dir / "metrics_summary.csv")
    saved_metrics = {row["target"]: row for row in saved_metrics_rows}
    saved_predictions = read_csv(experiment_dir / "fold_predictions.csv")
    source_rows = load_trials(data_dir)
    group_assignment = {
        str(group): int(fold) for group, fold in metadata["validation"]["group_assignment"].items()
    }

    source = {
        (str(row["qIdx"]), str(row["target_Mbps"]), str(row["trial"])): row
        for row in source_rows
    }
    if len(source) != EXPECTED_TRIALS or len(group_assignment) != EXPECTED_GROUPS:
        raise ValueError("Source trial/group counts do not match the audited dataset")
    if len(saved_predictions) != 837:
        raise ValueError(f"Expected 837 saved OOF rows, got {len(saved_predictions)}")

    predictions: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for saved in saved_predictions:
        target = saved["target"]
        if target not in TARGETS:
            raise ValueError(f"Unexpected saved target: {target}")
        key = (saved["qIdx"], str(int(float(saved["target_Mbps"]))), saved["trial"])
        source_row = source.get(key)
        if source_row is None:
            raise ValueError(f"Saved prediction has no source key: {key}")
        composite = (target, *key)
        if composite in seen:
            raise ValueError(f"Duplicate OOF prediction key: {composite}")
        seen.add(composite)
        if target == "latency" and source_row["valid_owd"] != 1:
            raise ValueError(f"Invalid latency label present in OOF predictions: {key}")
        fold = int(saved["fold"]) - 1
        group_id = str(saved["group_id"])
        if group_assignment.get(group_id) != fold:
            raise ValueError(f"Prediction fold does not match saved group assignment: {group_id}")

        source_target = float(source_row[TARGET_VALUE_FIELDS[target]])
        observed = float(saved["y_true"])
        if not math.isclose(source_target, observed, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"Saved target differs from source for {target} {key}")

        configured_rate = int(key[1])
        rate_prediction = rate_only_prediction(
            source_rows, group_assignment, test_fold=fold, target=target,
            configured_rate=configured_rate,
        )
        ridge_prediction = float(saved["ridge_prediction"])
        physically_invalid = ridge_prediction < 0.0
        predictions.append({
            "target": target,
            "unit": TARGETS[target]["unit"],
            "qIdx": key[0],
            "configured_rate_Mbps": configured_rate,
            "trial": int(key[2]),
            "group_id": group_id,
            "fold": fold + 1,
            "observed": observed,
            "rate_only_prediction": rate_prediction,
            "ridge_prediction_raw": ridge_prediction,
            "ridge_physical_status": (
                "INVALID_NEGATIVE" if physically_invalid else "NONNEGATIVE_NOT_PHYSICALLY_CERTIFIED"
            ),
            "global_mean_prediction": float(saved["mean_prediction"]),
        })

    by_target: dict[str, dict[str, object]] = {}
    for target in TARGETS:
        target_rows = [row for row in predictions if row["target"] == target]
        actual = [float(row["observed"]) for row in target_rows]
        rate_pred = [float(row["rate_only_prediction"]) for row in target_rows]
        ridge_pred = [float(row["ridge_prediction_raw"]) for row in target_rows]
        mean_pred = [float(row["global_mean_prediction"]) for row in target_rows]
        rate_metric = _metric(actual, rate_pred)
        ridge_metric = _metric(actual, ridge_pred)
        mean_metric = _metric(actual, mean_pred)
        saved = saved_metrics[target]
        for metric_name, calculated in (
            ("mean_baseline_mae", mean_metric["mae"]),
            ("mean_baseline_rmse", mean_metric["rmse"]),
            ("ridge_mae", ridge_metric["mae"]),
            ("ridge_rmse", ridge_metric["rmse"]),
        ):
            if not math.isclose(calculated, float(saved[metric_name]), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Recomputed {metric_name} disagrees for {target}")
        negatives = [value for value in ridge_pred if value < 0.0]
        by_target[target] = {
            "unit": TARGETS[target]["unit"],
            "n_rows": len(target_rows),
            "n_condition_groups": len({str(row["group_id"]) for row in target_rows}),
            "primary_benchmark": "rate_only",
            "rate_only": rate_metric,
            "ridge": ridge_metric,
            "global_mean_context": mean_metric,
            "rate_only_beats_ridge_mae": rate_metric["mae"] < ridge_metric["mae"],
            "ridge_negative_prediction_count": len(negatives),
            "ridge_negative_min": min(negatives) if negatives else None,
            "ridge_negative_values_retained": True,
        }

    return {
        "validation": {
            "name": "Saved six-fold grouped out-of-fold validation",
            "seed": metadata["random_seed"],
            "grouping": "qIdx × configured rate; all repetitions of each condition stay together",
            "condition_groups": EXPECTED_GROUPS,
            "train_groups_per_fold": 35,
            "test_groups_per_fold": 7,
            "no_condition_overlap": True,
            "features": ["configured rate (Mbps)", "qIdx categorical known-site label"],
        },
        "warning": (
            "Offline experimental study from one TU Wien testbed. No arbitrary-input prediction is exposed. "
            "Negative ridge jitter/loss outputs are retained and marked invalid, never clipped."
        ),
        "targets": by_target,
        "predictions": predictions,
    }


def _html_document(comparison: dict[str, object]) -> str:
    payload = json.dumps(comparison, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TU Wien 5G KPI Offline Comparison</title>
<style>
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#172033;background:#f3f6fb;--blue:#1769aa;--orange:#c05a17;--red:#a52626;--muted:#526174}body{margin:0}main{max-width:1280px;margin:auto;padding:28px}h1{margin:0 0 6px;font-size:1.8rem}.sub,.muted{color:var(--muted)}.banner{margin:18px 0;padding:14px 18px;border-left:5px solid var(--red);background:#fff0ee;border-radius:5px}.method{background:#eaf3fb;padding:14px 18px;border-radius:6px;margin-bottom:20px}.controls{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}.controls label{font-weight:650}select{display:block;margin-top:5px;padding:8px 10px;border:1px solid #a9b6c5;border-radius:5px;background:white;min-width:190px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}.card,.panel{background:white;border:1px solid #d8e0ea;border-radius:8px;padding:16px}.card h3{margin:0 0 9px;font-size:.92rem;color:var(--muted)}.score{font-size:1.35rem;font-weight:700}.primary{border-top:4px solid var(--blue)}.secondary{border-top:4px solid var(--orange)}.context{border-top:4px solid #9aa6b2}.panel{margin-top:16px;overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #e0e6ed;padding:8px 9px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#f5f7fa;position:sticky;top:0}.negative{color:var(--red);font-weight:700;background:#fff0ee}.badge{font-size:.82rem;color:var(--blue)}.small{font-size:.9rem}.none{padding:14px;color:var(--muted)}footer{padding:20px 0;color:var(--muted);font-size:.86rem}
</style></head><body><main>
<h1>TU Wien 5G KPI comparison</h1><div class="sub">Offline experimental results · one private 5G testbed · 42 condition groups</div>
<div class="banner"><strong>Experimental only; not deployment-ready.</strong> No arbitrary-input prediction is shown: the available predictors are configured rate and seven fixed testbed location labels, and the rate-only benchmark beats ridge in this grouped check. Ridge produced 35 negative jitter predictions and 10 negative packet-loss predictions. Those raw values remain visible and are marked invalid; none are clipped or presented as physically valid.</div>
<div class="method"><strong>Validation:</strong> saved six-fold out-of-fold predictions, grouped by qIdx × configured rate. Every repetition in a condition stays in one fold; each fold uses 35 training and 7 held-out condition groups. Rate-only predictions use only training-fold observations at the same configured rate. Rate is in Mbps; throughput MAE/RMSE are Mbps, latency and jitter are ms, and loss is percentage points. The data describe one deployment and do not establish generalization to other 5G networks.</div>
<section class="controls"><label>Target<select id="target"></select></label><label>Breakdown<select id="breakdown"><option value="rate">Configured rate</option><option value="location">Location</option></select></label><label>Condition rows<select id="condition"></select></label></section>
<section id="cards" class="cards"></section><section class="panel"><h2>MAE by condition</h2><div id="breakdownTable"></div></section><section class="panel"><h2>Saved out-of-fold predictions</h2><div id="predTable"></div></section>
<footer>Rate-only is the primary benchmark. The global-mean baseline is shown only as context. Ridge is a comparison model, not the preferred predictor. Nonnegative outputs are not thereby certified physically valid. No subjective QoE labels are included.</footer>
<script id="payload" type="application/json">__PAYLOAD__</script><script>
const D=JSON.parse(document.getElementById('payload').textContent), targets=D.targets, rows=D.predictions;
const names={throughput:'Throughput',latency:'One-way latency',jitter:'Jitter',packet_loss:'Packet loss'};
const $=id=>document.getElementById(id), fmt=(x,n=4)=>Number(x).toFixed(n), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const t of Object.keys(targets))$('target').add(new Option(names[t],t));
function targetRows(){return rows.filter(r=>r.target===$('target').value)}
function renderCards(){const t=$('target').value,m=targets[t],u=m.unit; $('cards').innerHTML=`<article class="card primary"><h3>Primary · rate-only</h3><div class="score">MAE ${fmt(m.rate_only.mae)} ${esc(u)}</div><div class="small">RMSE ${fmt(m.rate_only.rmse)} ${esc(u)}</div><div class="badge">${m.rate_only_beats_ridge_mae?'Lower MAE than ridge':'Higher MAE than ridge'} in this split</div></article><article class="card secondary"><h3>Ridge · comparison only</h3><div class="score">MAE ${fmt(m.ridge.mae)} ${esc(u)}</div><div class="small">RMSE ${fmt(m.ridge.rmse)} ${esc(u)}</div><div class="small">Negative outputs: ${m.ridge_negative_prediction_count}; raw values retained</div></article><article class="card context"><h3>Global mean · context</h3><div class="score">MAE ${fmt(m.global_mean_context.mae)} ${esc(u)}</div><div class="small">RMSE ${fmt(m.global_mean_context.rmse)} ${esc(u)}</div><div class="small">Training-fold target mean</div></article><article class="card"><h3>Evaluation rows</h3><div class="score">${m.n_rows}</div><div class="small">${m.n_condition_groups} conditions · six-fold OOF</div></article>`}
function renderBreakdown(){const t=$('target').value,by=$('breakdown').value,items=targetRows(), groups=[...new Set(items.map(x=>by==='rate'?String(x.configured_rate_Mbps):x.qIdx))].sort((a,b)=>Number(a)-Number(b));let out='<table><thead><tr><th>'+ (by==='rate'?'Rate (Mbps)':'Location')+'</th><th>Rows</th><th>Rate-only MAE ('+esc(targets[t].unit)+')</th><th>Ridge MAE ('+esc(targets[t].unit)+')</th><th>Global mean MAE ('+esc(targets[t].unit)+')</th></tr></thead><tbody>';for(const g of groups){const rs=items.filter(x=>(by==='rate'?String(x.configured_rate_Mbps):x.qIdx)===g);const mae=k=>rs.reduce((s,r)=>s+Math.abs(Number(r.observed)-Number(r[k])),0)/rs.length;out+=`<tr><td>${by==='rate'?esc(g):'Q'+esc(g)}</td><td>${rs.length}</td><td>${fmt(mae('rate_only_prediction'))}</td><td>${fmt(mae('ridge_prediction_raw'))}</td><td>${fmt(mae('global_mean_prediction'))}</td></tr>`} $('breakdownTable').innerHTML=out+'</tbody></table>'}
function renderConditionOptions(){const items=targetRows(),select=$('condition'),old=select.value;const groups=[...new Set(items.map(x=>x.group_id))].sort((a,b)=>{let A=a.match(/q(\d+)_rate(\d+)/),B=b.match(/q(\d+)_rate(\d+)/);return +A[2]-+B[2]||+A[1]-+B[1]});select.innerHTML='<option value="all">All conditions</option>';for(const g of groups){const r=items.find(x=>x.group_id===g);select.add(new Option(`Q${r.qIdx} · ${r.configured_rate_Mbps} Mbps`,g))}if([...select.options].some(o=>o.value===old))select.value=old}
function renderPredictions(){let items=targetRows();if($('condition').value!=='all')items=items.filter(r=>r.group_id===$('condition').value);let out='<table><thead><tr><th>Condition</th><th>Trial</th><th>Fold</th><th>Observed ('+esc(targets[$('target').value].unit)+')</th><th>Rate-only</th><th>Ridge raw</th><th>Ridge status</th><th>Global mean</th></tr></thead><tbody>';for(const r of items){const neg=r.ridge_physical_status==='INVALID_NEGATIVE';out+=`<tr><td>Q${esc(r.qIdx)} · ${r.configured_rate_Mbps} Mbps</td><td>${r.trial}</td><td>${r.fold}</td><td>${fmt(r.observed,6)}</td><td>${fmt(r.rate_only_prediction,6)}</td><td class="${neg?'negative':''}">${fmt(r.ridge_prediction_raw,9)}</td><td class="${neg?'negative':''}">${esc(r.ridge_physical_status)}</td><td>${fmt(r.global_mean_prediction,6)}</td></tr>`} $('predTable').innerHTML=out+'</tbody></table>'}
function render(){renderCards();renderBreakdown();renderConditionOptions();renderPredictions()}
$('target').addEventListener('change',render);$('breakdown').addEventListener('change',render);$('condition').addEventListener('change',renderPredictions);render();
</script></main></body></html>'''
    return html.replace("__PAYLOAD__", payload)


def _write_exclusive(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)


def build_dashboard(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    experiment_dir: Path = EXPERIMENT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    comparison = build_comparison(data_dir=data_dir, experiment_dir=experiment_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_rows = comparison["predictions"]
    if not prediction_rows:
        raise ValueError("No comparison predictions to write")
    with (output_dir / "comparison_predictions.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    metrics = {key: value for key, value in comparison.items() if key != "predictions"}
    _write_exclusive(
        output_dir / "comparison_metrics.json",
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
    )
    _write_exclusive(output_dir / "offline_comparison_dashboard.html", _html_document(comparison))
    return comparison


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    comparison = build_dashboard(
        data_dir=args.data_dir,
        experiment_dir=args.experiment_dir,
        output_dir=args.output_dir,
    )
    print(f"Offline dashboard saved to {args.output_dir / 'offline_comparison_dashboard.html'}")
    for target, values in comparison["targets"].items():
        print(
            f"{target} ({values['unit']}): rate-only MAE={values['rate_only']['mae']:.6g}; "
            f"ridge MAE={values['ridge']['mae']:.6g}; negative ridge rows="
            f"{values['ridge_negative_prediction_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
