# QoE-Aware 5G Network Control Simulator

A research prototype for observing 5G-LENA/ns-3 traffic, estimating per-user Quality of Experience (QoE), and evaluating bounded network-control actions. The main workflow is a supervised live ns-3 session in a Streamlit dashboard. It is a simulation study, not a real-network controller.

## What the live system does

- Collects measured per-UE and network telemetry from ns-3, including throughput, delay, jitter, packet loss, traffic class, serving cell, and QoE.
- Displays a five-second, linear-trend QoE forecast after enough telemetry is available. This is a low-confidence baseline, not a validated prediction guarantee.
- Supports **Monitor**, **Recommend**, and **Automatic** controller modes. Recommend is the default.
- Uses adjustable per-service safety floors to identify UEs that are near or forecast to cross a floor.
- Applies verified live offered-load pacing for the legacy low-latency + voice traffic mix. Structural changes such as bandwidth, topology, association policy, and transmit power require an ns-3 restart epoch. NGMN gaming/video traffic does not have a verified live rate-control hook.
- Records session metadata, telemetry, controller decisions, simulator acknowledgements, and post-action verification under `results/live_sessions/`.

The optional one-shot load disturbance is available only for the legacy traffic mix. It raises offered voice workload; it does not model fading or a radio outage. It requires Automatic mode and three consecutive fresh windows with every UE at or above target. Its purpose is to demonstrate measured detection and controller response to a controlled workload shock, not guaranteed advance prediction.

## QoE definition and scope

New live sessions calculate a service-specific score for each UE:

```text
QoE = 0.40 × throughput satisfaction
    + 0.25 × delay satisfaction
    + 0.20 × jitter satisfaction
    + 0.15 × packet-loss satisfaction
```

Throughput satisfaction is `min(measured throughput / service reference, 1)`. Delay, jitter, and packet-loss satisfaction are each `max(0, 1 − measured value / service limit)`. A UE with no received packets scores zero. The overall user QoE is the equal-weight mean of configured UEs; per-service means and a legacy pooled-flow diagnostic are shown separately.

This score is a network-metric proxy, not a subjective user survey or MOS measurement. In particular, the legacy voice workload is a high-rate UDP stress generator rather than a realistic codec model, and the video generator does not model playback buffering, startup delay, or adaptive resolution. The exact profile references and units are documented in the dashboard and [`docs/live-simulation.md`](docs/live-simulation.md).

## Run the dashboard

From the project root in WSL/Linux, with the Python dependencies installed:

```bash
python3 -m streamlit run app.py
```

If using the copied environment whose package directory is populated but whose Python launcher is unavailable, use:

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages python3 -m streamlit run app.py
```

Open the local URL printed by Streamlit. Select **Run simulation**, choose the UE mix and controller mode, set the target and service floors, then start a session. Select **Automatic** only when you want eligible actions to be applied without approval. The dashboard distinguishes measured telemetry, forecast values, model estimates, and verified simulator outcomes.

The dashboard requires the live ns-3 executable at `ns-3.48/build-live-release/scratch/ns3.48-qoe-live-control`. If it is missing, rebuild it using the commands in [`docs/live-simulation.md`](docs/live-simulation.md). The simulator uses ns-3's real-time scheduler on a best-effort basis; a heavy scenario or a slow host can make telemetry stale. A structural restart creates a new process segment and resets that segment's simulation clock while retaining the session history.

## Models and data

- The **live trend forecast** is calculated from recent measurements; it is not a trained ML model.
- A separate guarded ML advisor compares selected offered-rate candidates for the supported legacy traffic contexts. It was trained on generated ns-3 KPI data and is not used as a general time-series forecaster. Unsupported contexts, including NGMN gaming/video, multi-cell scenarios, and unrepresented configurations, are rejected. See [`models/live_hybrid_ns3_20261030/README.md`](models/live_hybrid_ns3_20261030/README.md).
- The external 5G campus dataset is retained as a separate benchmark. It is not interchangeable with this ns-3 scenario and has not replaced the active production model. See [`docs/external-campus-dataset-migration.md`](docs/external-campus-dataset-migration.md).
- Run `python3 scripts/evaluate_live_forecast.py` to produce a session-grouped forecast evaluation in `results/live_forecast_calibration/`. The report requires more independent Monitor sessions before it can support a reliability claim.

## Dashboard sections

- **Run simulation:** live session control, per-UE topology and metrics, forecast, safety-floor assessment, action feed, history, and diagnostics.
- **Overview, Experiments, Analytics:** saved simulation summaries, comparisons, and trends.
- **Tools:** offline dataset, model, and benchmark utilities. These are separate from live ns-3 measurements.

## Reproducibility and records

Session data are retained at `results/live_sessions/<session-id>/`. Each session includes metadata and events; each ns-3 process segment stores its telemetry, simulator output, run configuration, and control acknowledgements. Do not interpret a proposed or model-estimated gain as an observed improvement until later ns-3 measurements verify it.

Useful references:

- [`docs/live-simulation.md`](docs/live-simulation.md) — build, launch, QoE formula, controller actions, and live-session behavior.
- [`docs/predictive-live-control.md`](docs/predictive-live-control.md) — separate scheduled-demand experiment and its limits.
- [`docs/data-and-measurement-audit.md`](docs/data-and-measurement-audit.md) — audit of the original generated dataset and measurement semantics.
- [`docs/external-campus-dataset-migration.md`](docs/external-campus-dataset-migration.md) — external dataset scope, provenance, licensing, and compatibility decision.

## Git safety

This project may contain local simulation records, datasets, and model artifacts. Review `git status` before staging; stage only the files intended for a commit. Do not commit raw data, backups, or generated session outputs unless they are deliberately part of the research deliverable.
