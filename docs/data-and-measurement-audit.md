# Data and measurement audit

Audit performed 2026-10-25 against the checked-in raw and processed CSVs and the current source.

## Baseline

| Dataset | Rows | Columns | Scenarios | Rows per scenario |
|---|---:|---:|---:|---|
| `data/raw/qoe_dataset.csv` | 720 | 15 | 120 | 2, 4, 6, 8, or 10 |
| `data/processed/qoe_dataset.csv` | 720 | 22 | 120 | 2, 4, 6, 8, or 10 |

Neither CSV has missing cells or duplicate complete rows. The processed CSV retains the raw 15 columns and adds seven engineered fields; it does not represent additional simulation observations. Each scenario has UE count 2/4/6/8/10, background rate 5000/10000/20000/40000 packets/s, bandwidth 20/50 MHz, and scenario size 3/30/100 m. Transmit power is 35 dBm for every row. These match the 5 x 4 x 2 x 3 generator grid.

Observed raw ranges: packet loss [0.000333333, 1], throughput [0, 189.696] Mbps, delay [0, 298.496] ms, jitter [0, 2.05813] ms, and QoE proxy [0, 0.994839]. There are 360 rows with RX=0. TX is never below RX, `lost_packets` is zero for all rows, including all 360 zero-RX rows. The source writes FlowMonitor's `lostPackets` into that field, and independently derives `packet_loss` as clamped `(txPackets-rxPackets)/txPackets` (zero when TX=0). The two values are different counters and must not be conflated.

Rows with no received packets currently export zero throughput, delay, and jitter. Their QoE proxy is explicitly zero. Zero delay/jitter for these rows are unavailable measurements, not evidence of good service. The existing CSV schema has no missing-value marker for these metrics; downstream outage classification must use RX packet count / `is_outage`.

The C++ QoE proxy is `0.40*min(throughputMbps/50, 1) + 0.25*exp(-delayMs/50) + 0.20*exp(-jitterMs/20) + 0.15*(1-packetLoss)` for flows with RX packets, else zero. Inputs are nonnegative measurements and packet loss is bounded to [0,1], making the intended output [0,1]. It is a hand-designed simulation proxy, not a validated subjective QoE score or a standards-defined metric.

In the output schema, `lost_packets` is FlowMonitor's `FlowStats::lostPackets` value after `CheckForLostPackets()`. `packet_loss` is a separate quantity calculated as `(tx_packets-rx_packets)/tx_packets`, clamped to [0,1], or zero for zero TX. The generated one-scenario check had zero FlowMonitor lost counts despite zero RX, while the derived packet loss was 1.0; these values are intentionally not interchangeable.

## Modeling and generalization limits

The model's explicit features are configuration/context fields and derived traffic-profile fields. `scenario_id` and `flow_id` are excluded. Training uses `GroupShuffleSplit` and `GroupKFold` by scenario ID, so rows from one scenario are kept together. There are only 120 configurations and a single fixed power value; results cannot establish generalization to other transmit powers, seeds, radio environments, or real networks. `offered_throughput_mbps` is computed from observed `tx_packets` and packet size even though the comment describes it as pre-simulation; because offered load is derivable from configured flow rate and packet size, the feature should be based on those inputs in future data preparation rather than relying on a traffic counter.

## Execution baseline

The source target was built with `./ns3 build qoe-dataset`. A single run (scenario ID 987654; 2 UEs, 5000 packets/s, 20 MHz, 35 dBm, 30 m, 1 s) wrote two 15-column flow rows to a new directory under `/tmp`. Both flows had TX>0, RX=0, FlowMonitor `lost_packets=0`, derived packet loss 1.0, and QoE proxy 0.0. CSV row widths, counter bounds, QoE bounds, and output-directory placement were checked. No full campaign was run. Pytest availability and the full unittest result are reported in the verification turn's final report.
