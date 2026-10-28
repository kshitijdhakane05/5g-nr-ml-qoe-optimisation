# TU Wien offline KPI comparison dashboard

This standalone, dependency-free dashboard compares the saved six-fold out-of-fold ridge predictions with a rate-only benchmark computed strictly from each fold’s training condition groups. It also displays the saved global-mean baseline for context. It does not import or alter the production dashboard, QoE model, optimizer, controller, or simulation code.

## Build

From the repository root:

```bash
python3 -m experiments.tuwien_kpi_comparison_dashboard_20261026.build_dashboard
```

The command reads the existing experiment outputs under `results/experiments/tuwien_kpi_baseline_20261026/` and the audited TU Wien CSV files. It writes a new `results/experiments/tuwien_kpi_comparison_dashboard_20261026/` directory and refuses to overwrite it if it already exists. The HTML is self-contained and works offline; no Streamlit, Plotly, pandas, network access, or new model fit is required.

The rate-only predictor for each held-out trial is the mean target at that configured rate among the **other training folds’ condition groups**. No held-out target or same-condition repetition enters that mean. Saved ridge and global-mean predictions are reused as-is. Raw negative ridge outputs are retained and marked invalid; they are not clipped.

## Interpretation

Rate-only MAE is the primary benchmark. The dashboard is an experimental, single-testbed study with six configured rates, seven fixed sites, one deployment, and repeated trials. It has filters for target and saved condition rows, not arbitrary-input prediction controls. The available features do not support defensible deployment predictions: rate-only scoring is tied to these six rates, `qIdx` is a site label for this testbed, and ridge performed worse than rate-only in the audited grouped split. The results are not deployment-ready and do not represent all 5G deployments.

Run focused tests with:

```bash
python3 -m unittest discover -s tests -p 'test_tuwien_kpi_comparison_dashboard_20261026.py' -v
```
