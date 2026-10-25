#!/bin/bash
# P2: re-implemented baselines (XGBoost+SHAP, NNN, CausalMMM) on seeds 0-4. Resumable.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
for s in 0 1 2 3 4; do
  python3 -u run_experiments.py --scenarios S1 S2 S3 --seeds $s --methods XGB-SHAP NNN NNN-AR CausalMMM --out results/raw_reimpl.jsonl
done
echo CHAIN_REIMPL_DONE
