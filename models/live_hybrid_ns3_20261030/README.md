# Live hybrid ns-3 KPI model v1

This artifact is a **guarded advisor** for the persistent live controller. It
predicts measured throughput, delay, jitter, packet-loss fraction, and reception
probability. The application recomputes service QoE v2 from those estimated
KPIs; QoE itself is not a training label. Live telemetry and the short-horizon
trend forecast remain separate inputs, and post-action ns-3 measurements are
the only verification of an action's effect.

## Source and training

- Processed source: `data/processed/qoe_dataset.csv` (720 flow rows).
- Raw source: `data/raw/qoe_dataset.csv`; the processed features were produced
  by `ml/preprocessing.py` from the project's ns-3 `scratch/qoe-dataset.cc`
  scenario. These are generated simulator runs, not external campus data.
- Group split: `GroupShuffleSplit` by `scenario_id`, seed 42; 96 groups / 590
  rows for fit and 24 groups / 130 rows for holdout evaluation. No scenario
  group overlaps between the two sets.
- Reproduce with:
  `PYTHONPATH=.venv/lib/python3.12/site-packages python3 -m ml.train_live_hybrid`
- Full hashes, feature/target schema, metrics, package versions, and permitted
  contexts are in `metadata.json`.

## Validation snapshot

On the held-out scenarios, QoE v2 recomputed using estimated KPIs had MAE
0.0039 for low-latency flows and 0.0144 for voice-QoS flows. The corresponding
mean-baseline MAEs were 0.1380 and 0.0722. These are one grouped holdout from
the generated scenario grid; they are not independent evidence of performance
on another simulator version, topology, or live NGMN traffic.

The runtime permits advice only for exact represented UE/radio/workload
contexts, one gNB, 35 dBm, and the existing low-latency + voice mix. NGMN
gaming/video, multi-cell, unsupported powers, or unseen contexts are explicitly
out of scope and return to the existing telemetry rule path. The controller
compares a rate proposal against no change, and requires predicted mean gain to
exceed half the paired-tree QoE spread while no represented service profile
loses more than 0.01. This is advisory evidence, not a guarantee.

The production artifact `models/qoe_model.joblib` and the separate campus
benchmark artifact are not replaced or used by this live advisor.
