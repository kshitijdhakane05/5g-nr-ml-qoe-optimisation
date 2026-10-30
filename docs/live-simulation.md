# Live ns-3 QoE control

The primary **Run simulation** dashboard route starts `qoe-live-control`, a separate executable copied from the existing `qoe-dataset.cc` scenario. The live binary uses `RealtimeSimulatorImpl` and emits one-second simulation-window JSONL records. Monitoring and source-pacing control use one persistent ns-3 process. Startup-only radio/topology actions use a **restart epoch**: the current process stops, a new configuration starts in a separate segment directory, and the logical dashboard session continues with history retained. Each segment has its own simulation clock, telemetry, logs and final outputs.

## Measurements and QoE

Each telemetry row is measured from FlowMonitor counter deltas for the current simulation-time window. Throughput is received bytes × 8 divided by the one-second simulation window, in Mbps. Delay and jitter are delta averages in milliseconds. Packet loss is computed from cumulative received/transmitted packets to avoid negative loss from packets crossing a window boundary.

New live sessions use **service-profile QoE v2**. For each UE, calculate four satisfaction components and weight them:

```text
QoE = 0.40 × throughput satisfaction
    + 0.25 × delay satisfaction
    + 0.20 × jitter satisfaction
    + 0.15 × packet-loss satisfaction
```

Throughput satisfaction is `min(measured Mbps / profile reference Mbps, 1)`. Delay, jitter, and loss satisfaction are each `max(0, 1 − measured / profile limit)`. A UE with no received packets scores zero. The profile references are:

| Traffic class | Throughput reference | Delay limit | Jitter limit | Packet-loss limit |
| --- | ---: | ---: | ---: | ---: |
| low_latency | 2 Mbps | 50 ms | 5 ms | 0.1% |
| voice_qos | 0.1 Mbps | 100 ms | 30 ms | 1% |
| gaming | 0.05 Mbps | 50 ms | 10 ms | 0.1% |
| video_streaming | 0.064 Mbps | 300 ms | 50 ms | 0.0001% |

The low-latency throughput reference follows its default offered workload (100-byte packets × 2,500 packets/s). The voice throughput reference is a 0.1 Mbps project service floor, not a standardized bitrate. The current `voice_qos` generator is a high-rate 1,252-byte UDP stress workload, not a realistic codec-rate voice source; therefore its QoE score is not a validated voice-quality estimate. Gaming uses the bundled NGMN downlink generator with a 0.05 Mbps project throughput floor and the simulator's `GBR_GAMING` profile (50 ms packet-delay budget, 0.1% packet-error target). Video uses the bundled NGMN frame/slice generator, whose documented default source rate is 64 kbps; the 300 ms/0.0001% delay/error limits follow `NGBR_VIDEO_TCP_DEFAULT`. Voice's 100 ms/1% limits follow `GBR_CONV_VOICE` ([ETSI TS 23.501 Release 18](https://www.etsi.org/deliver/etsi_ts/123500_123599/123501/18.11.00_60/ts_123501v181100p.pdf)). Jitter limits, throughput floors, and score weights are project policy, not standards-defined QoE. The score remains a network-derived proxy, not a user survey or MOS.

The displayed **overall measured QoE** is the equal-weight arithmetic mean of per-UE service-profile scores, including zero for configured UEs with no received packets. Per-class means are shown as diagnostics. Equal weighting prevents high-volume UEs from dominating the user aggregate. The old pooled-flow formula remains only as `network_qoe` for legacy diagnostics and is not directly comparable to v2. Telemetry records `qoe_formula_version`; historical sessions retain their original formula and are labeled accordingly. The controller uses the current session’s per-UE v2 values and continues to track low-scoring users individually.

## Prediction and actions

The forecast is a five-second linear trend over the latest five one-second simulation windows. It requires four observations, reports low confidence, and has not been calibrated as a reliable predictor. A projected QoE decrease of at least 0.04 or a forecast below the configured QoE target is treated as a risk signal. Recommend mode requires explicit dashboard approval.

The live action changes the `Interval` attribute on installed best-effort `UdpClient` applications, bounded to 2,500–20,000 packets/s per UE with a 10-second cooldown. The C++ scenario polls an atomic command file and records the applied value and simulation time only after `SetAttribute` executes. The controller allows one live pacing adjustment per segment, then waits for ns-3 acknowledgement and two post-action measurement windows. If individual users remain below target after that verification, it evaluates bounded structural candidates; in Automatic mode it may launch a restart epoch, while Recommend mode requires approval. A materially worse result triggers one bounded restoration attempt; measured deltas remain observational and are not causal proof.

Controller records now identify the decision basis as `predictive_preemptive` when a forecast identifies target risk while measured QoE is still above target, or `reactive_target_recovery` after a measured breach. The dashboard labels these separately and includes the measured QoE, target, forecast value and horizon where available. This makes a forecast-triggered action auditable; it does not improve the trend model's accuracy or imply that a prediction is correct. A sudden disturbance without a measured precursor remains detectable only after its first affected telemetry window.

### Per-service QoE safety floors

New sessions can set a lower safety floor for each traffic class: low-latency, voice QoS, gaming, and video streaming. Defaults are 0.08–0.12 QoE below the selected target; these are adjustable project policy values, not standardized service guarantees. The controller uses the existing five-second linear trend to warn when a complete per-UE history projects below its class floor. It can also warn when a UE is within 0.05 QoE of its floor and still declining. A floor warning for one UE can bypass the normal two-user target-crossing gate. Actions remain limited to existing verified controls: live offered-load pacing for the legacy mix, or a bounded simulator restart epoch for supported changes in the gaming/video mix. A sudden unannounced shock can cross the floor within a single telemetry window; that case can only be recovered from after measurement. Floor values are preserved in session metadata and shown with the selected UE's QoE history.

The start form's optional **NGMN gaming + video mix** assigns alternating UEs to the bundled NGMN Gaming and NGMN Video Streaming traffic generators, with separate 5G-LENA QoS flows (`GBR_GAMING` and `NGBR_VIDEO_TCP_DEFAULT`). The original low-latency/voice constant-rate mix remains the default. The NGMN video model sends frame/slice traffic but does not model a player buffer, startup delay, adaptive resolution, or rebuffer events, so its displayed QoE is only a network-metric proxy. NGMN generators do not have a verified runtime rate-control hook in this integration; for this mix the controller considers verified network restart actions and does not offer or claim live packet pacing.

gNB count/topology, UE placement/association, channel bandwidth, numerology and transmit power are set during scenario construction and require restart epochs. In Recommend mode a structural proposal requires approval. In Automatic mode the controller can start a bounded restart epoch when QoE remains below target or a forecast threatens an attained target after live pacing is insufficient. Supported restart fields are gNB count (1–4), association policy (`closest`, `round_robin`, or valid two-plus-cell `by_traffic_class`), Band 1 bandwidth (20/50/100 MHz), and total power (10–50 dBm). The controller prioritizes adding a gNB under congestion and widening bandwidth when throughput is the apparent limiter. Candidates without matching evidence are explicitly unvalidated trials, not predicted gains. Startup values are checked against the new segment's telemetry; the first three post-change windows are compared with the prior segment. If QoE does not improve toward target, Recommend mode surfaces rollback and Automatic mode attempts one rollback. The restart causes an observable telemetry gap and resets simulation time for that segment.

The controller objective is to reach the configured target when QoE is below it and to hold the measured-safe configuration once the target is attained. It does not explore for a higher score while the target is stable; it responds to a forecast target threat using the measured QoE components. The trend model remains low-confidence and uncalibrated.

## Controlled load-shock demonstration

The **Controlled test disturbance** expander on the live page provides an opt-in, one-shot offered-load shock for the legacy constant-rate profile. It is enabled only in Automatic mode when the latest three fresh telemetry windows each contain every configured UE, every UE is at or above the configured QoE target, the offered rate is stable and below its 20,000 packets/s/UE bound, and no controller action is pending or awaiting verification. The request is persisted and is reported as applied only after ns-3 records the requested rate in `controls.jsonl`.

This raises offered voice workload to 20,000 packets/s/UE. It is a controlled workload disturbance, not radio fading, a channel outage, or an external network fault. The test shock is separate from the controller action ledger and does not consume the controller's one live-rate intervention. Automatic mode then observes actual measurements and may reduce the load using the existing supported live pacing action. If the controller does not act or cannot restore the target, the load remains at the test setting until the operator stops or otherwise changes the session. The feature can be triggered only once per session.

Rate-change requests snapshot the simulator control-log position before writing the command. Each request clears prior verification state and receives its own action ID; only a subsequent matching ns-3 acknowledgement can mark it applied, and only subsequent telemetry windows can verify its measured outcome. `target_reached` in the post-action record means every configured UE had complete verification data and met target; `network_mean_target_reached` separately reports whether the arithmetic mean met target.

For a repeatable demonstration, start with 2 UEs, 1 gNB, `closest` association, the legacy profile, 50 MHz, 35 dBm, and 2,500 packets/s for both rates; use Automatic mode and a 0.70 target. Wait for the shock control to enable, trigger it once, then follow the per-UE QoE, action acknowledgement, and verified post-action measurement in the feed. This is a demonstration workload, not a realistic voice source or a calibrated user-perception study. Abrupt disturbances are detected from subsequent measurements; the existing trend model does not guarantee warning before a sudden event.

The implementation was exercised through the real background worker and ns-3 binary in session `live_20261030_190815_b0e8cf` (2 UEs, 1 gNB, 50 MHz, 35 dBm, target 0.70). After all UEs stayed above target for three windows, ns-3 acknowledged the shock at 20,000 packets/s/UE. The voice UE then measured QoE 0.694, 62.1 ms delay and 24.3% loss; the controller's 10,000 packets/s/UE reduction was acknowledged and verified. Across the two verification windows, mean UE QoE measured 0.8442 before and 0.9834 after; both UEs met the target afterward. This is one controlled simulation, not a repeated study or causal guarantee. The trend warning followed the first degraded window, so the test demonstrates detection and recovery, not prediction before an abrupt disturbance.

To collect repeatable paper/demo evidence, run `PYTHONPATH=.venv/lib/python3.12/site-packages python3 scripts/run_live_load_shock_study.py --runs 3`. The script starts a fresh real ns-3 session for each replicate, triggers only through the same three-window eligibility gate, and writes per-replicate results plus measured telemetry to a new timestamped directory under `results/live_control/`. Each session is stopped after verified action outcome or timeout; failures are retained in the study record rather than counted as success. Use `--runs`, `--target`, and `--timeout` to set the replicate count, per-UE target, and per-session time limit.

## Build and launch

From the project root:

```bash
cmake -S ns-3.48 -B ns-3.48/cmake-cache-live-release -G Ninja \
  -DCMAKE_BUILD_TYPE=release \
  -DNS3_NATIVE_OPTIMIZATIONS=OFF \
  -DNS3_OUTPUT_DIRECTORY="$PWD/ns-3.48/build-live-release" \
  -DNS3_EXAMPLES=OFF -DNS3_TESTS=OFF
cmake --build ns-3.48/cmake-cache-live-release --target scratch_qoe-live-control -j8
PYTHONPATH=.venv/lib/python3.12/site-packages python3 -m streamlit run app.py
```

The copied `.venv` in this workspace currently contains zero-byte, non-executable Python launchers. Python packages are present under `.venv/lib/python3.12/site-packages`, so the command above uses the system Python with that package directory. Recreate the virtual environment from `requirements.txt` with a working Python installation for normal development.

Sessions and logs are kept under `results/live_sessions/<session-id>/`. Session-level events and metadata are retained with per-process outputs under `segments/segment_NNN/` (`telemetry.jsonl`, `controls.jsonl`, simulator log, and standard final ns-3 CSV outputs). Historical segments are not removed. Older sessions without segment metadata continue to support live packet pacing but must be restarted as a new session to use structural restart epochs.

## Runtime limits

The live scenario uses a separate optimized (`release`, `-O3`, assertions/logging disabled) binary. The original `build-copyfix` executable is preserved for rollback and other workflows. ns-3's real-time scheduler remains best effort: if a configuration generates events faster than this host can process them, simulated time still falls behind. The page marks measurements stale after 15 seconds without an update and blocks actions unless status is Running. Graph traces are separated by segment so reset simulation clocks are not drawn as continuous time. The topology view is built from live per-UE FlowMonitor windows, scenario positions and serving-cell IDs. During a restart, it retains historical telemetry but does not present the previous segment's topology as current; it updates after the first new-segment measurement. The trend predictor is a baseline, not a validated anticipation guarantee.

## Runtime check in this project copy

Session `live_20261029_201334_b8eb4e` used 2 UEs, 1 gNB, 50 MHz, 35 dBm total requested power, 10,000 best-effort packets/s per UE at startup, and 2,500 low-latency packets/s per UE. Its first complete one-second sample ended at simulation time 1.4 s (the workload starts at 0.4 s). The live adapter changed best-effort pacing from 10,000 to 5,000 packets/s per UE at simulation time 3.4 s. The same ns-3 process acknowledged the command; the first full post-change window reported 53.83 Mbps / QoE 0.99604, and later windows settled at 53.76 Mbps / QoE 0.99605. The recorded pre-action mean (three windows, including the acknowledgement-boundary window) was 0.99496; the four post-action windows averaged 0.99604. This short, uncontrolled comparison is observational, not causal proof or evidence of a reliable QoE improvement; delivered throughput also decreased from about 104.96 to 53.76 Mbps. The one-second telemetry tick took several seconds of wall time on this host.

A corrected release-build benchmark used the higher-load 6 UE / 3 gNB / 100 MHz / 15,000 best-effort / 2,000 low-latency configuration. It emitted five distinct one-second simulation windows through simulation time 5.4 s in 59.5 wall-clock seconds: approximately 12.0 wall seconds per simulation second, compared with the earlier default-profile run that produced a window after roughly 30 seconds in that configuration. The measured QoE and other metrics matched the previous run for the same timestamps. This brings the tested profile under the current 15-second stale threshold, with limited headroom; heavier settings may still go stale. Benchmark records are in `results/live_performance/release_rate_20261029_204106/`.
