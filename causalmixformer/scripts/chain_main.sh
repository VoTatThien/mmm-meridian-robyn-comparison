#!/bin/bash
# P1: CausalMixFormer, ablations and ridge baselines for seeds 3-4 (seeds 0-2 done earlier),
# (real-data runs: scripts/chain_realdata.sh). Resumable: finished (scenario, seed, method) are skipped.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
for s in 3 4; do
  python3 -u run_experiments.py --scenarios S1 S2 S3 --seeds $s \
    --methods Ridge Ridge+M CMF CMF-noCT1 CMF-noCT2 CMF-noCT3 CMF-linbase --out results/raw_seeds3to9.jsonl
done
echo CHAIN_MAIN_DONE
