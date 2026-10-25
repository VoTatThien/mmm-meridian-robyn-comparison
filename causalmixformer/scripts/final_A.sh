#!/bin/bash
# FINAL EVALUATION, chain A: fresh seeds never used during development.
# S1-S3 seeds 100-104: Ridge, Ridge+M, CausalMixFormer; ablations on seeds 100-102; benchmark rerun.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
for s in 100 101 102 103 104; do
  python3 -u run_experiments.py --scenarios S1 S2 S3 --seeds $s --methods Ridge Ridge+M CMF --out results_final/main.jsonl
done
for s in 100 101 102; do
  python3 -u run_experiments.py --scenarios S1 S2 S3 --seeds $s --methods CMF-noCT1 CMF-noCT2 CMF-noCT3 CMF-linbase --out results_final/ablation.jsonl
done
python3 -u run_benchmark.py
echo FINAL_A_DONE
