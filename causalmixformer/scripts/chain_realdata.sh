#!/bin/bash
# Real-data runs at three rolling origins (after the spend-scale floor fix). Resumable.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
for c in 101 81 61; do
  python3 -u run_realdata.py --cutoff $c --seeds 0 1 2
done
echo CHAIN_REALDATA_DONE
