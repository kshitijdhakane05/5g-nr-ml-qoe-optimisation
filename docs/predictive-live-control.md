# Predictive live QoE control

## Opt-in scheduled-demand protection trial

The dashboard provides a separate **Predictive demand protection · matched
ns-3 trial** workflow. It does not change ordinary session startup or control.
The trial runs a Monitor reference and an Automatic session sequentially with
the same legacy 4-UE, 1-gNB, 50-MHz profile and ns-3 seed. Both begin at
7,500 best-effort packets/s/UE and receive a scheduled request for 20,000
packets/s/UE after warmup. If the first run measures any UE below the trial
QoE target, the Automatic controller can apply a 5,000 packets/s/UE admission
cap before the scheduled demand onset.

This control path separates offered demand from admitted source pacing using
the live scenario's command files. Telemetry reports requested, admitted, and
unadmitted rates. QoE is scored from the traffic that the simulator actually
transmits; the interface explicitly reports rejected demand, so it does not
claim the blocked portion received service. The no-control value is a measured
result from the matched reference run and is labeled as the estimate of what
the controlled run would have experienced without intervention. It is not a
simultaneous or directly observed counterfactual.

The experiment tests protection against **known scheduled upcoming demand**.
It does not predict an unknown arrival, and sequential ns-3 runs can still
differ despite matched seeds. A successful result requires the simulator to
acknowledge the cap before the demand onset and every UE's subsequent measured
QoE to remain at or above target. A failed or non-degrading reference is shown
as such rather than presented as successful prevention. Durable study output
is written under `results/live_control/predictive_demand_studies/`; raw
simulator sessions remain under `results/live_sessions/`.

### Direct simulator-path smoke test

The rebuilt `scratch_qoe-live-control` binary was exercised with real paired
ns-3 runs using the same 4-UE, 1-gNB, 50-MHz profile and seed. At simulation
time 9.4 s, both runs received the 20,000 packets/s/UE request. In the
no-control run, the complete 9.4–10.4 s window measured mean per-UE QoE
0.798938; voice UEs 2 and 4 measured 0.598666 and 0.602289. In the capped
run, ns-3 acknowledged a 5,000 packets/s/UE admission cap at 5.4 s, before the
demand onset; the complete post-onset window measured mean per-UE QoE 0.997305
and all four UEs were above the 0.85 trial target. The simulator reported
15,000 requested packets/s/UE as unadmitted. Full JSONL and summaries are in
`results/live_control/predictive_demand_direct_smoke/`.

This smoke test directly exercised the simulator's demand and cap command
files; the cap was issued by the test harness, not by the Python controller.
The dashboard's paired study runner exercises the controller path, but it was
not run in the current shell because system Python lacks `joblib`, NumPy,
pandas, and Streamlit, while the checked-in `.venv/bin/python` launchers are
empty files. Install the listed `requirements.txt` in a working environment
before presenting the dashboard-run as verified end to end.

## Runtime contract

The controller is supervised and epoch based. Each measurement epoch starts a
new ns-3 process; the simulator does not support hot-changing the verified
radio parameters in this scenario. The application labels a selected bandwidth
change `SCHEDULED_RESTART`, starts the next epoch with the changed CLI
parameter, validates the generated `run_config.json` and QoS CSV, then writes a
`VERIFIED` or `APPLICATION_FAILED` intervention record. A process failure never
creates a measurement. Recommend mode is the dashboard default and waits for
approval of a specific configuration fingerprint; Monitor repeats the selected
configuration; Automatic applies eligible actions without waiting for approval.
Stop remains available while a recommendation is pending.

## Forecast and QoE definition

`simulation.qoe_forecast.forecast_qoe_degradation` forecasts one epoch ahead.
It requires at least three completed successful epochs with the same
configuration, and uses at most the latest five. For each UE it fits a simple
linear trend to throughput, delay, jitter and packet-loss fraction, projects
those metrics one epoch, and recalculates QoE with the existing formula. The
aggregate is the mean across forecast UEs. Risk means the projected UE QoE is
below the configured target; the reported cause is the largest adverse
contribution among the projected metric changes. A descriptive residual band
is shown when available. It is not a calibrated confidence interval or a
validated probabilistic guarantee. Configuration changes reset forecast
history, and a prediction is scored only against a subsequent measurement of
the same configuration.

The existing project QoE proxy is unitless in `[0,1]`:

```text
0.40 × min(throughput_mbps / 50, 1)
+ 0.25 × exp(-delay_ms / 50)
+ 0.20 × exp(-jitter_ms / 20)
+ 0.15 × (1 - packet_loss_fraction)
```

It is zero when no packets are received. This is an ns-3-derived proxy, not a
subjective user QoE measurement. Throughput is Mbps, delay and jitter are ms,
and packet loss is a fraction in `[0,1]`.

The controller still reacts to an observed target miss when no forecast can be
made, but the UI records `INSUFFICIENT_HISTORY` instead of presenting that
reaction as predictive. It degrades to monitoring when data are stale,
invalid, or too sparse.

## Supported action

The initial action registry contains only `bandwidth_band1_mhz`, with validated
values 20, 50 and 100 MHz. The parameter maps to the scenario CLI argument
`--bandwidthBand1` and takes effect in the next epoch process. The candidate
must be present in the validated ns-3 experiment registry and preserve the
session's UE count, transmit power, Band 2 bandwidth, traffic, packet sizes,
topology and scenario settings. Minimum expected QoE benefit, action budget,
candidate history, and the no-action/controller stop decisions bound changes.
The prior setting is retained in the session history as recovery context; a
restart is the recovery mechanism. No action is described as live hot-swap.

Candidate estimates come from exact-condition ns-3 run summaries, not the
external campus dataset or the production ML model. The production model is
disabled for this control path because its training feature contract does not
match pre-run inference. The exact-condition estimates have limited run
support and descriptive, uncalibrated variation. They rank candidates; they do
not establish generalization to unmeasured configurations.

## Data and audit records

Each session directory under `results/live_control/` retains `session.json`,
`events.jsonl`, `epoch_history.csv`, logs, per-epoch `run_config.json` and
validated output CSVs. `interventions.jsonl` records proposal, approval or
restart scheduling, application verification, predicted impact and measured
impact. Forecast outcomes and simulator measurements are separate fields.
These records remain after the session stops.

The external campus dataset and model migration are maintained as a separate
benchmark/training workflow; they are not treated as ns-3 measurements and are
not used to claim a control action's effect.

## Verified real execution

On 2026-10-29, Automatic session `20261029T191424_9eb12ebe` used scenario L9-02 with 6 UEs, 20 MHz Band
1, 50 MHz Band 2, 35 dBm requested power, validated traffic defaults, one gNB,
and a target QoE proxy of 0.80. The actual first epoch completed at QoE 0.0000
(outage). The controller selected the validated 50 MHz Band 1 candidate using
exact-condition empirical campaign data. It launched a second ns-3 process
with a distinct configuration fingerprint. The second run's validated CSV
reported 27.08565 Mbps mean throughput, 111.6920 ms mean delay, 0.136524 ms
mean jitter, 0.392931 mean packet-loss fraction, and QoE proxy 0.630932. The
candidate estimate was 0.652332 (absolute error 0.021401). The ledger marked
the restart verified from the run fingerprint and validated output file. The
target remained unmet at the configured two-epoch limit. The originating epoch
record was updated to `APPLIED_AND_VERIFIED` in both `events.jsonl` and
`epoch_history.csv`; the ledger contains `PROPOSED`, `SCHEDULED_RESTART` and
`VERIFIED` records. Run evidence is retained at
`results/live_control/predictive_e2e_verified/20261029T191424_9eb12ebe/`.

This run verifies the restart-based parameter path and records a measured
outcome. It does not demonstrate forecast-led early warning: the first epoch
had insufficient history, so the action was a response to measured outage.
That live action session is `results/live_control/predictive_e2e_verified/20261029T191424_9eb12ebe/`.

A separate Monitor-only live session,
`results/live_control/early_warning_trace/20261029T191729_c44a120d/`, verified
the forecast loop at target 0.658. After epoch 3, the predictor forecast UEs 4
and 6 below target (0.65134 and 0.65714); their current values were 0.658624
and 0.658689, so both were still above target at prediction time. In epoch 4,
UE 4 measured 0.52658 (risk realized) and UE 6 measured 0.65888 (risk not
realized). The aggregate QoE forecast was 0.66527 versus measured 0.64743,
absolute error 0.01784. This is a small trace, not calibration proof: it shows
one true positive and one false positive and does not establish general
predictive reliability.

## Rollback

The previous production model, original dataset backup, and pre-existing
session records are retained. To restore the prior dashboard/control workflow,
revert only the task-specific source changes using version control after
reviewing the current worktree; do not run `git reset`, `git clean`, or restore
over unrelated local changes. An individual live session can be stopped from
the dashboard. A new bounded recovery session can be started from its recorded
previous configuration; old session evidence remains immutable.
