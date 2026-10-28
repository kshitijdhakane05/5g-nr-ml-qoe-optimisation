# Bounded expansion design (pre-execution)

## Scope

Run exactly 11 configurations twice (22 designated invocations), with Band 2 fixed at 50 MHz. This follows the approved table in `results/experiments/ns3_qoe_expansion_20261026_r1_run2/pilot_report.md`. It is a fractional design plus two anchors, not a full factorial.

| Config ID | Band 1 MHz | Requested total power dBm | UEs |
|---|---:|---:|---:|
| L9-01 | 20 | 30 | 2 |
| L9-02 | 20 | 35 | 6 |
| L9-03 | 20 | 40 | 10 |
| L9-04 | 50 | 30 | 6 |
| L9-05 | 50 | 35 | 10 |
| L9-06 | 50 | 40 | 2 |
| L9-07 | 100 | 30 | 10 |
| L9-08 | 100 | 35 | 2 |
| L9-09 | 100 | 40 | 6 |
| ANCHOR-UE2 | 50 | 35 | 2 |
| ANCHOR-UE6 | 50 | 35 | 6 |

Each configuration uses `RngSeed=20261026` and `RngRun=1,2`. Scenario IDs are unique per invocation; `config_id` identifies the condition and `repetition_id` identifies its paired RNG run. Equal seed/run values support paired bookkeeping but do not guarantee identical realized random events when configuration changes alter event sequences.

## Traffic decision

The pilot report explicitly proposed `lambdaBe=20000` packets/s because its 5,000 packet/s best-effort condition delivered almost all traffic and did not distinguish the tested 50/100 MHz or 30/35/40 dBm settings. This campaign uses that documented rate; the pilot remains the 5,000 packet/s low-load reference. Source inspection confirms `lambdaBe` is passed to the UDP client's `Interval=1/lambdaBe`, with a configured 1,252-byte best-effort payload. Nominal offered payload load is therefore 200.32 Mbps per best-effort flow. The low-latency class remains 10,000 packets/s × 100 bytes = 8 Mbps per flow. These are configured application payload rates, not post-run TX-counter estimates, and exclude protocol overhead.

Other fixed settings: one gNB; Band 2=50 MHz; 28.0/28.2 GHz center frequencies; BWP numerologies 4/2; 30 m scenario area; 1 s simulation; application start 0.4 s; double operational band enabled. The source divides total requested power proportionally by bandwidth: `P_i[dBm] = P_total[dBm] + 10*log10(BW_i/(BW1+BW2))`.

## Execution and validation policy

Use the existing compiled ns-3 executable, explicit CLI settings, new output directory per invocation, and 120-second per-run timeout. Preserve stdout/stderr and a manifest row for every designated run. No automatic retry: any failure remains recorded and must be diagnosed before a separately identified retry. Continue the design unless a failure indicates a shared configuration/build defect; do not train unless all designated output data pass schema, identity, numeric, outage, packet-loss, QoE, and configured-offered-load checks.

Group model evaluation by the 11 `config_id` values; all flow rows and both repetitions for a condition stay together. This is only 11 distinct conditions and two stochastic repetitions per condition. It cannot establish broad generalization or real-network behavior. The two RNG repetitions are not guaranteed paired in realized event sequence.

## Read-only source checks

- The compiled scenario exposes `--ueNumPergNb`, `--bandwidthBand1`, `--bandwidthBand2`, `--totalTxPower`, `--lambdaBe`, `--lambdaUll`, `--packetSizeBe`, `--packetSizeUll`, `--scenarioId`, `--outputDir`, `--RngSeed`, and `--RngRun`.
- `ueNumPergNb*gNbNum` sets the total UE count.
- The two bandwidths and total power are applied to the actual BWP PHYs using the proportional split above.
- The proposed conditions vary each factor and hold Band 2 explicitly fixed.
- The report's traffic increase is an explicit campaign setting, not a silent mutation of pilot conditions.
