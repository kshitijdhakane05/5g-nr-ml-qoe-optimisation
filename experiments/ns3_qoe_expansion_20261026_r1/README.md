# Isolated ns-3 QoE pilot and bounded expansion

This module contains the original seven-run pilot and the approved bounded
11-configuration × 2-RNG expansion. Both bypass the 120-run generator and write
each simulation to its own directory. They do not write to production datasets,
the production model, optimizer outputs, or phase-3 histories.

## Reproduce

The seven-run result set is in
`results/experiments/ns3_qoe_expansion_20261026_r1_run2/`. The first runner
attempt completed one baseline simulation and then hit a manifest-serialization
bug; its extra duplicate baseline output is preserved under the unsuffixed
`_r1` path. The runner refuses to reuse that path. The seven designated runs
were then completed in `_run2`; eight simulator processes ran in total because
of the duplicate baseline. Reproduce to a fresh output path:

```bash
cd ~/5g-project
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/ns3_qoe_expansion_20261026_r1/pilot.py \
  --output-root results/experiments/ns3_qoe_expansion_20261026_r1_reproduction
```

The built binary's `--PrintHelp` and `--PrintGlobals` were checked first.
Accepted names include `--bandwidthBand1`, `--bandwidthBand2`, `--totalTxPower`,
`--ueNumPergNb`, `--RngSeed` and `--RngRun`. The runner also checks for the
corresponding application-source UE and power-allocation expressions before
running. Each invocation has a 120-second timeout and must exit zero and emit
both `qoe_dataset.csv` and `sim_summary.txt`.

## Fixed traffic and radio settings

- One gNB, 30 m placement square; 1 second simulation, applications active
  from 0.4 to 1.0 seconds.
- Band 2 is explicitly 50 MHz; Band 1 is varied per pilot configuration.
- `lambdaBe=5000` packets/s, `lambdaUll=10000` packets/s, payloads 1252/100 B.
- Numerologies 4/2 at 28.0/28.2 GHz; two operational bands enabled.
- `RngSeed=20261026` on every run. `RngRun=1` for each primary condition;
  the repeated baseline uses `RngRun=2`.
- Scenario IDs 900001–900007; each run has an independent output directory.

The source divides requested total power in linear units by BWP bandwidth:
`P_i(dBm)=P_total+10*log10(BW_i/(BW1+BW2))`. Both effective powers are saved
in the manifest and per-run config. The source CSV logs Band 1 and requested
total power but does not include Band 2 or the effective split.

## Experimental preprocessing

`preprocessing.py` derives nominal offered throughput as
`configured packets/s * payload bytes * 8 / 1e6`. It does not use the
post-run `tx_packets` counter. The result is nominal UDP payload Mbps and
excludes protocol headers. The low-latency class uses 10,000 packets/s × 100 B;
the best-effort class uses configured `lambdaBe` × 1252 B.

After a fully passing pilot, export a separate analysis table:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m \
  experiments.ns3_qoe_expansion_20261026_r1.postprocess \
  --pilot-root results/experiments/ns3_qoe_expansion_20261026_r1_run2
```

The exporter refuses incomplete pilots and refuses to overwrite an existing
`pilot_preprocessed_configured_load.csv`.

## Bounded 22-run expansion

The approved configuration table, source checks, traffic rationale, grouping
and RNG caveat are recorded in `EXPANSION_DESIGN.md`. The proposed increase to
`lambdaBe=20000` packets/s is explicit: the pilot's 5,000 packets/s load was
insensitive to several bandwidth/power settings, and the scenario accepts this
rate through the actual UDP interval configuration. The low-load pilot is
preserved as the reference.

Run only with an unused output root:

```bash
cd ~/5g-project
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m \
  experiments.ns3_qoe_expansion_20261026_r1.expansion \
  --output-root results/experiments/ns3_qoe_expansion_20261026_r1_expansion_reproduction
```

The runner verifies command-line support before execution, records exact
commands and per-run metadata, uses 120-second timeouts, and does not retry
failures automatically. After all 22 runs pass:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m \
  experiments.ns3_qoe_expansion_20261026_r1.evaluate_expansion \
  --results-dir results/experiments/ns3_qoe_expansion_20261026_r1_expansion_reproduction
```

Evaluation validates each raw run and the combined configured-load features,
then performs leave-one-configuration-out and factor-level grouped checks. It
saves an experimental-only model below the results directory. It does not
load or replace that model in the production optimizer/controller. The
production model is not scored because its training-time offered-load feature
was derived from post-run TX counters while this experiment uses configured
rate × payload.

## Validation and scope

The runner checks process exit status, required outputs, exact 15-column CSV
schema, expected one-flow-per-UE count and IDs, configured row values, missing
and non-finite data, duplicate rows/keys, physical value ranges, packet loss
from TX/RX, outage behavior, and an independent QoE recomputation. Pilot
failures stop the remaining pilot; expansion failures are preserved without
automatic retries. Both studies are limited ns-3 simulation evidence, not
real-world 5G validation. No production model, optimizer, controller, or
dashboard behavior is changed.
