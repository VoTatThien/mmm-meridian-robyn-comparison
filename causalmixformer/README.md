# CausalMixFormer

Code, data and results for the paper

> **CausalMixFormer: A Mediation-Aware Causal Temporal Transformer for Nonlinear Marketing Mix Modeling**
> Vo T. Thien, Phan T. B. Tran, Bui T. Hung
> Data Science Laboratory, Industrial University of Ho Chi Minh City (manuscript in preparation)

CausalMixFormer is a marketing mix model (MMM) that encodes the causal graph media → mediator
(e.g. branded search) → sales as two jointly trained structural equations, so that channel ROI is a
counterfactual *total* effect that includes the indirect path. It combines channel-specific lag
attention for carryover, Hill saturation, a Transformer encoder restricted to the media-free baseline,
non-negativity constraints with weak ROI priors, a bootstrap-over-regions ensemble for uncertainty, and
budget reallocation on the learned response curves.

## Layout

```
cmf/
  simulate.py          geo-level simulator with ground-truth ROI; scenarios S1-S3 (main),
                       S3c, S3l, S3u, S4, S5, S6 (robustness)
  model.py             CausalMixFormer (mediation, constraints + priors, lag attention,
                       bootstrap ensemble) with ablation switches
  baselines.py         Robyn-style ridge MMM (with / without the mediator), no-media forecaster
  baselines_reimpl.py  re-implementations of methods without public code: XGBoost + SHAP, NNN, CausalMMM
  realdata.py          loader for the public 190-DMA panel (data/dcm_190dma.csv)
  budget.py            budget reallocation and uplift-vs-oracle scoring
  benchmark_adapter.py runs CausalMixFormer on the national benchmark datasets with their ROI definition
run_experiments.py     simulation runner (resumable)
run_realdata.py        real-data runner, rolling origins via --cutoff
run_benchmark.py       CausalMixFormer and two ablations on the benchmark datasets in bench/
make_tables.py         results -> LaTeX tables and figures in paper/ (created on first run)
plot_kernels.py        learned vs. true carryover kernels (scenario S2)
scripts/               the exact sequence of runs used for the paper
results_final/         final evaluation results (seeds 100-104, never used during development)
results/               development runs (seeds 0-4) and the real-data results
bench/                 benchmark of seven open-source MMM configurations (Robyn, Meridian,
                       PyMC-Marketing, LightweightMMM, Orbit KTR, DeepCausalMMM x2) and its datasets
data/                  public 190-DMA panel (MIT licence, see data/README.md)
```

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.13, PyTorch 2.14 (CPU), NumPy 2.5, SciPy 1.18, scikit-learn 1.9, pandas 3.0,
matplotlib 3.11 and XGBoost 3.4. The benchmark scripts in `bench/src/` additionally need the
respective frameworks (R + Robyn, google-meridian, pymc-marketing, lightweight_mmm, orbit-ml,
deepcausalmmm); their results are already included in `bench/results/`.

## Reproducing the paper

All numbers in the paper come from `results_final/`, `results/realdata.jsonl` and `bench/results/`.
Development runs (seeds 0-4) are kept in `results/raw_*.jsonl` for transparency but are not used in
the paper.

```bash
export OMP_NUM_THREADS=1
bash scripts/final_A.sh          # S1-S3, seeds 100-104: Ridge, Ridge+M, CausalMixFormer; ablations; benchmark
bash scripts/final_B.sh          # S1-S3, seeds 100-104: XGBoost+SHAP, NNN, CausalMMM; robustness scenarios
bash scripts/chain_realdata.sh   # 190-DMA panel, three rolling origins
python plot_kernels.py           # carryover kernels (evaluation seed 100)
python make_tables.py            # all tables and figures -> paper/tables, paper/figures
```

The runners are resumable: finished (scenario, seed, method) combinations are skipped. On a 2-core
CPU one CausalMixFormer fit with 5 ensemble members and 1000 steps takes about 7 minutes; run
scenarios in parallel processes with `OMP_NUM_THREADS=1`.

Final CausalMixFormer configuration: 5 ensemble members, bootstrap over regions, randomised initial
ROI (sd 0.5 on the log scale) for members 2-5, LogNormal(0.2, 0.9) ROI prior, media share in
[0.05, 0.60], paid search excluded as a parent of the mediator.

## Quick start

```bash
python run_experiments.py --scenarios S3 --seeds 0 --methods Ridge Ridge+M CMF --members 3
python run_realdata.py --seeds 0                 # holdout = last 8 weeks
python run_realdata.py --cutoff 61 --seeds 0     # rolling origin
python make_tables.py
```

## Model in one paragraph

Spend → channel-specific lag attention (free-form carryover, recency-biased initialisation) → Hill
saturation → two structural equations: mediator `M = m(u) + Σ α_c h_c` and sales
`y = b(u) + Σ β_c h_c + δ M`, where `b(u)` is a Transformer over controls and calendar features only
(no media). `α, β, δ ≥ 0`, the media share is kept in a prior range and a weak LogNormal ROI prior is
used. The total ROI of channel c is computed by switching the channel off and propagating the change
through the mediator; an ensemble over bootstrapped regions gives intervals, and the response curves
drive the budget optimisation.

## Data and licences

`data/dcm_190dma.csv` is the public example panel of DeepCausalMMM
(https://github.com/adityapt/deepcausalmmm), redistributed under its MIT licence
(`data/LICENSE-deepcausalmmm`). The simulated panels are generated on the fly by `cmf/simulate.py`.
The code in this folder is released under the MIT licence (`LICENSE`).
