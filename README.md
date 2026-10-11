# 5G Network Simulation & ML Optimisation Project

An end-to-end machine learning-based 5G network prediction and optimisation framework built on top of **ns-3 5G NR (3GPP TR 38.901 mmWave / FR2)** simulations running in WSL Ubuntu 24.04.

---

## Phase 1: Machine Learning Prediction Pipeline

Phase 1 establishes a reliable, non-leaking ML surrogate model to predict end-to-end Quality of Experience (`qoe_proxy`) directly from 5G network configurations and environmental parameters.

### 1. What the Model Predicts
The primary prediction target is **`qoe_proxy`** $\in [0.0, 1.0]$, representing user satisfaction calculated across four key quality-of-service dimensions:
- **Throughput utility**: normalized against a 50 Mbps reference limit.
- **Latency penalty**: exponential decay with a 50 ms characteristic scaling factor.
- **Jitter penalty**: exponential decay with a 20 ms characteristic scaling factor.
- **Reliability (packet delivery)**: scaled delivery success rate $(1 - \text{packet\_loss})$.

When packets fail to be delivered ($\text{rx\_packets} = 0$), $\text{qoe\_proxy} = 0.0$ indicates total outage.

---

### 2. Feature Representation
Features represent **pre-transmission network configurations and environment context**, strictly preventing data leakage from post-simulation measurements:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `ue_count` | Integer | Total number of active user equipments (UEs) |
| `bandwidth_mhz` | Float | Configured 5G NR carrier bandwidth (MHz) |
| `total_tx_power_dbm` | Float | Base station (gNB) total transmit power (dBm) |
| `scenario_size_m` | Float | Physical placement grid dimension ($L \times L$ meters) |
| `flow_type_id` | Binary | Flow traffic class: `0` = Low-Latency, `1` = Voice / Best-Effort |
| `flow_lambda` | Float | Offered packet arrival rate for this flow (pkts/s) |
| `packet_size_bytes` | Integer | Packet payload size (`100` B for Low-Lat, `1252` B for Voice) |
| `offered_throughput_mbps` | Float | Nominal offered traffic load (Mbps) |

> **Note on Transmit Power**: In the initial 120-scenario dataset, `total_tx_power_dbm` is constant at 35 dBm. It is preserved in the feature schema to ensure seamless compatibility when power-varying datasets are generated in subsequent phases.

---

### 3. Data Splitting & Why Scenario-Level Grouping is Mandatory

A 5G simulation scenario models an entire multi-UE radio cell where all flows concurrently share base station scheduling, resource blocks, beamforming vectors, and channel fading.

- **The Leakage Risk**: If individual flow rows are partitioned randomly into train and test sets (`train_test_split`), flows belonging to the same scenario appear in both sets. The model would memorize scenario-specific scheduling states and channel realizations, producing artificially inflated test scores that fail on new network deployments.
- **The Grouped Solution**: The dataset is strictly partitioned by **`scenario_id`** using `GroupShuffleSplit` (80% train scenarios / 20% holdout test scenarios) and evaluated via 5-Fold `GroupKFold` cross-validation:
  - **Train Scenarios**: 96 scenarios (590 flow rows)
  - **Holdout Test Scenarios**: 24 scenarios (130 flow rows)
  - **Scenario Overlap**: **0 scenarios** (100% clean isolation)

---

### 4. Reproducing the Pipeline

#### Setup Environment
```bash
# From project root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Run Full Automated Pipeline
```bash
./scripts/run_ml_pipeline.sh
```

#### Run Individual Pipeline Steps
1. **Preprocess and validate dataset**:
   ```bash
   python -m ml.preprocessing --raw-path data/raw/qoe_dataset.csv --output-path data/processed/qoe_dataset.csv
   ```
2. **Train and cross-validate baseline regressors**:
   ```bash
   python -m ml.train --dataset data/processed/qoe_dataset.csv
   ```
3. **Run inference on a new network configuration**:
   ```bash
   python -m ml.predict \
     --model-path models/qoe_model.joblib \
     --ue-count 6 \
     --bandwidth 50 \
     --power 35 \
     --size 30 \
     --flow-type voice_be \
     --lambda-val 10000
   ```

---

### 5. Benchmark Performance

Evaluated on completely unseen test scenarios:

| Model | CV MAE | CV RMSE | CV $R^2$ | Test MAE | Test RMSE | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DummyRegressor (Mean)** | 0.3482 | 0.3617 | -0.5768 | 0.3144 | 0.3265 | -0.0097 |
| **LinearRegression** | 0.0582 | 0.0893 | 0.9021 | 0.0552 | 0.0883 | 0.9261 |
| **HistGradientBoostingRegressor** | 0.0234 | 0.0523 | 0.9607 | 0.0184 | 0.0462 | 0.9798 |
| **RandomForestRegressor** *(Best)* | **0.0189** | **0.0525** | **0.9618** | **0.0162** | **0.0445** | **0.9812** |

The trained model artifact is saved at `models/qoe_model.joblib`.
Evaluation logs are stored at `results/ml/model_results.csv` and detailed predictions at `results/ml/predictions.csv`.

---

## Phase 2: Offline ML-based Network Configuration Optimisation

Phase 2 builds an **offline ML-based network configuration optimiser** using the verified Phase 1 surrogate model.

### 1. What the Optimiser Does
The optimiser solves the constrained network decision problem:
$$\max_{\mathbf{c} \in \mathcal{C}} \widehat{\text{QoE}}(\mathbf{c}, \mathbf{x}) \quad \text{subject to } g_i(\mathbf{c}, \widehat{\text{QoE}}) \le 0$$
where:
- $\mathbf{x} = (\text{ue\_count}, \text{scenario\_size\_m}, \text{flow\_type}, \lambda)$ represents the operational network context.
- $\mathbf{c} = (\text{bandwidth\_mhz}, \text{total\_tx\_power\_dbm})$ represents the candidate network configuration.
- $\widehat{\text{QoE}}$ is evaluated rapidly via `models/qoe_model.joblib` (`qoe_proxy` / predicted QoE is a dimensionless score in the project's QoE scale; it is not measured in milliseconds).
- $g_i$ represents configurable operational constraints (e.g., maximum transmit power, minimum acceptable QoE).

The system executes an exhaustive grid search over all supported candidates, filters by feasibility, ranks candidates by predicted QoE, and compares the top selection against a requested baseline configuration.

---

### 2. Supported Optimisation Variables vs Dataset Limitations

| Parameter | Role | Supported Range / Values | Status in Current Optimiser |
| :--- | :--- | :--- | :--- |
| `bandwidth_mhz` | Decision Variable | $\{20.0, 50.0\}$ MHz | **Active**: Evaluates bandwidth allocation |
| `total_tx_power_dbm` | Decision Variable | $\{35.0\}$ dBm | **Fixed**: Single-value support; true power optimisation requires additional ns-3 runs |
| `ue_count` | Environment Context | $\{2, 4, 6, 8, 10\}$ | **Active**: Context parameter (no extrapolation) |
| `scenario_size_m` | Environment Context | $\{3.0, 30.0, 100.0\}$ m | **Active**: Context parameter (no extrapolation) |
| `flow_type` | Traffic Demand | `low_latency`, `voice_be` | **Active**: Context parameter |
| `flow_lambda` | Traffic Demand | $\{5000, 10000, 20000, 40000\}$ pkts/s | **Active**: Context parameter |
| `numerology` | PHY Numerology | $\mu=4$ (Band 1), $\mu=2$ (Band 2) | **Fixed**: Fixed by simulation architecture; not varied |

> **Scientific Transparency Notice**: Because `total_tx_power_dbm` was constant at 35 dBm in the original dataset, the optimiser does not pretend to optimize power consumption. Power remains in the configuration schema for forward compatibility with future ns-3 datasets.

---

### 3. Candidate Generation & Search Space Rules
- Defined in `optimisation/search_space.py`.
- Enforces strict support boundaries: candidates are never extrapolated outside observed data.
- Input context is validated before generation. Any unobserved parameter (e.g., 50 UEs, 500m cell size) raises an explicit `ValueError`.

---

### 4. Constraints Mechanism
Constraints are modular and configurable:
- `max_power_dbm`: Rejects candidates whose transmit power exceeds the threshold.
- `min_predicted_qoe`: Rejects candidates failing to satisfy SLA quality targets.
- `allowed_bandwidths`: Restricts search to a specific subset of bandwidths.

If no constraints are passed, none are silently assumed. If no candidate satisfies the constraint, the status `NO_FEASIBLE_CANDIDATE_HIGHEST_QOE_RETURNED` is flagged.

---

### 5. Running the Optimiser

#### Basic Execution
```bash
python -m optimisation.run_optimisation \
  --ue-count 4 \
  --flow-type voice_be \
  --lambda-val 10000 \
  --bandwidth 20 \
  --power 35
```

#### Execution with Constraints
```bash
python -m optimisation.run_optimisation \
  --ue-count 10 \
  --flow-type voice_be \
  --lambda-val 40000 \
  --bandwidth 20 \
  --min-qoe 0.70
```

#### Running Automated Unit Tests
```bash
python -m unittest tests/test_optimisation.py -v
```

All optimization decisions, candidates, and constraint statuses are automatically logged to `results/optimisation/optimisation_results.csv`.

---

### 6. ns-3 Verification Status
All reported improvements are strictly labeled:
> **"Predicted improvement — pending ns-3 verification"**

The offline optimiser uses the ML surrogate to select promising configurations. **Phase 3** introduces the closed-loop ns-3 runner to simulate selected configurations in the 3GPP NR environment and verify actual network KPIs against ML predictions.

---

## Phase 3: Closed-Loop ns-3 Simulation Verification

Phase 3 implements an **offline, simulation-based closed-loop verification workflow** (NOT live network control). It automates the transition from ML recommendation to physical 5G simulation, actual metric parsing, and error quantification.

### 1. Workflow Architecture
```
Network Context (UEs, Area, Traffic Demand)
           ↓
Phase 2 Optimiser (selects candidate & predicts QoE)
           ↓
ns-3 Runner (launches isolated 5G simulation via subprocess)
           ↓
Result Parser (extracts throughput, delay, jitter, packet loss)
           ↓
QoE Calculation (calculates measured QoE using exact formula)
           ↓
Verification & Error Analysis (|QoE_pred - QoE_meas|, gain vs baseline)
           ↓
Persistent Logging (results/phase3/verification_results.csv)
```

### 2. Execution Isolation & Artifact Preservation
- Each simulation run executes with `cwd = results/phase3/runs/<run_id>/` and an absolute `--outputDir`.
- Generates per-run simulation logs (`stdout.log`, `stderr.log`, `run_config.json`, `sim_summary.txt`, and per-run `qoe_dataset.csv`).
- The original dataset (`data/raw/qoe_dataset.csv`), the `ns-3.48/` repository, and Phase 1/2 outputs are **never modified or overwritten**.

### 3. Running Closed-Loop Experiments

#### Run Closed-Loop Verification
```bash
python -m simulation.closed_loop \
  --ue-count 2 \
  --flow-type voice_be \
  --lambda-val 5000 \
  --baseline-bandwidth 20
```

#### Run All Unit Tests Across Phases
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 4. Interpreting Prediction Error
- **Absolute Error**: $|\widehat{\text{QoE}} - \text{QoE}_{\text{meas}}|$, measuring deviation on the dimensionless $[0, 1]$ QoE scale.
- **Relative Error**: $\frac{|\widehat{\text{QoE}} - \text{QoE}_{\text{meas}}|}{\text{QoE}_{\text{meas}}}$, evaluated only when $\text{QoE}_{\text{meas}} > 0$.
- **Measured Improvement**: $\text{QoE}_{\text{meas, opt}} - \text{QoE}_{\text{meas, base}}$, reporting verified gain against the physical baseline run.

