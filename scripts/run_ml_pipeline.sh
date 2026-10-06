#!/usr/bin/env bash
set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Determine python executable: prefer .venv if present
if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "=========================================================="
echo "5G Network ML Pipeline (Phase 1)"
echo "Using Python: $PYTHON_BIN"
echo "=========================================================="

echo ""
echo "[Step 1/3] Running Data Preprocessing & Validation..."
"$PYTHON_BIN" -m ml.preprocessing \
    --raw-path data/raw/qoe_dataset.csv \
    --output-path data/processed/qoe_dataset.csv

echo ""
echo "[Step 2/3] Training Baseline ML Models (Grouped Scenario Evaluation)..."
"$PYTHON_BIN" -m ml.train \
    --dataset data/processed/qoe_dataset.csv \
    --model-out models/qoe_model.joblib \
    --results-out results/ml/model_results.csv \
    --preds-out results/ml/predictions.csv \
    --n-splits 5 \
    --test-size 0.20

echo ""
echo "[Step 3/3] Testing Inference with Sample Network Configuration..."
"$PYTHON_BIN" -m ml.predict \
    --model-path models/qoe_model.joblib \
    --ue-count 4 \
    --bandwidth 50 \
    --power 35 \
    --size 30 \
    --flow-type voice_be \
    --lambda-val 10000

echo ""
echo "=========================================================="
echo "Phase 1 ML Pipeline completed successfully!"
echo "Model artifact:    models/qoe_model.joblib"
echo "Evaluation stats:  results/ml/model_results.csv"
echo "Predictions:       results/ml/predictions.csv"
echo "=========================================================="

