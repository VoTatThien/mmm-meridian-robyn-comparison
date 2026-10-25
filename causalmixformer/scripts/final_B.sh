#!/bin/bash
# FINAL EVALUATION, chain B: re-implemented baselines on S1-S3 seeds 100-104, robustness scenarios seeds 100-102.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
for s in 100 101 102 103 104; do
  python3 -u run_experiments.py --scenarios S1 S2 S3 --seeds $s --methods XGB-SHAP NNN NNN-AR CausalMMM --out results_final/main.jsonl
done
for s in 100 101 102; do
  python3 -u run_experiments.py --scenarios S6 S3u S3l --seeds $s --methods CMF CMF-noCT1 Ridge Ridge+M NNN NNN-AR --out results_final/robust.jsonl
  python3 -u run_experiments.py --scenarios S3c S5 --seeds $s --methods CMF Ridge Ridge+M NNN NNN-AR --out results_final/robust.jsonl
  python3 -u run_experiments.py --scenarios S4 --seeds $s --methods CMF CMF-linbase Ridge NNN XGB-SHAP --out results_final/robust.jsonl
done
echo FINAL_B_DONE
