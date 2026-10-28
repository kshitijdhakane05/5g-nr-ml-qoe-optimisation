# TU Wien KPI baseline experiment

This isolated experiment evaluates four separate trial-level regression targets from the already-audited processed TU Wien 5G data. It does not import the project’s production ML, optimizer, dashboard, or simulation modules and does not load or write any production model.

## Data and provenance

Input CSVs are read from `results/dataset_audit/20261026_ranalyzer_tuwien/tuwien_data/`:

- `latency_trials_raw_210.csv` — raw UDP trial records and `valid_owd`.
- `latency_with_achieved_bw_210.csv` — joined jitter and loss records, including the corruption flag.
- `achieved_bandwidth_210.csv` — server-reported achieved throughput.

The loader requires exactly 210 rows and unique, identical `(qIdx, rate, trial)` keys in all three tables. It checks that throughput, jitter, loss, and latency-validity copies agree. This yields 42 condition groups (`qIdx × configured rate`) with five trials each. The packet-loss target is the official iPerf Server Report percentage and is scored in percentage points.

The three `valid_owd=0` trials (Q1, 50 Mbps, trials 3–5) are excluded from the latency target. The code never reads `latency_210.csv` as a target source, so the author-imputed 14.2395 ms values cannot enter latency fitting or evaluation. Those runs remain eligible for throughput, jitter, and loss as documented by the dataset authors.

## Prediction contract

Only two documented pre-run features are used: configured offered rate in Mbps and `qIdx` as a categorical location label for a known-location estimand. Trial ID, condition key, timestamps, filenames, measured KPI fields, packet counters, validity flags, and derived summaries are not predictors.

Each target is evaluated independently:

| Target | Source definition | Unit |
|---|---|---|
| Throughput | UDP achieved throughput from iPerf server | Mbps |
| Latency | Raw valid mean one-way delay | ms |
| Jitter | iPerf Server Report jitter | ms |
| Packet loss | iPerf Server Report percentage | percentage points |

There are two estimators: a training-fold mean predictor and a low-complexity ridge regression using standardized configured rate, its square, and one-hot categorical location. Feature scaling/category discovery occur within each training fold. No target scaling, feature selection, clipping, or imputation is performed.

## Validation and outputs

The experiment uses deterministic six-fold out-of-fold validation. The 42 complete location/rate conditions are assigned as seven groups to each fold; all repetitions of one condition remain together. This evaluates held-out combinations of known locations and rates. It does not evaluate a new deployment, new rates, unseen site characteristics, or ns-3 transfer.

Dataset citation: Bodur, Osman; Çağlayan, Sami; Demir, Neslihan; Poszvek, Günther; Bleicher, Friedrich. *Measurement dataset and analysis code for “Measurement-Driven Modeling of End-to-End Latency in an Indoor Private Standalone 5G Network”*. TU Wien Research Data, DOI [10.48436/nw34t-zt678](https://doi.org/10.48436/nw34t-zt678). Associated article DOI [10.3390/network6030078](https://doi.org/10.3390/network6030078). Data license: CC BY 4.0.

For each target, results include MAE and RMSE, fold-level metrics, trial-level out-of-fold predictions, and 95% percentile intervals from a 1000-replicate condition-group bootstrap. The report also includes a paired bootstrap interval for ridge-minus-mean MAE. These intervals summarize OOF condition clusters; overlapping training folds mean they do not fully capture training-set variability. Treat them as descriptive, and do not interpret small score differences as meaningful without stronger independent groups.

Run from the repository root:

```bash
python3 -m experiments.tuwien_kpi_baseline_20261026.baseline
```

The default output directory is `results/experiments/tuwien_kpi_baseline_20261026/`. Existing output directories are rejected to avoid overwriting. Outputs are `fold_predictions.csv`, `fold_metrics.csv`, `metrics_summary.csv`, and `run_metadata.json`; metadata records input checksums, seed, Python/platform versions, target definitions, exclusions, features, split assignment, model settings, metrics, and limitations.

Run focused tests with:

```bash
python3 -m unittest discover -s tests -p 'test_tuwien_kpi_baseline_20261026.py' -v
```

## Scope and limitations

This is a small, single-deployment indoor private standalone 5G testbed with one user, six configured UDP rates, seven fixed sites, and repeated trials. Condition grouping prevents the five repeats of a condition from leaking across folds, but does not make the 42 design cells independent networks. No subjective QoE label is available. The models are not integrated into the project or claimed to generalize to all 5G deployments.
