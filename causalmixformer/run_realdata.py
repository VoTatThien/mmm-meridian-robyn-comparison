"""Experiment on the public 190-DMA dataset (DeepCausalMMM example data).

Holdout = last 8 weeks, as in the DeepCausalMMM paper. No ground-truth ROI exists, so we
report predictive accuracy (pooled R^2, within-geo R^2, WAPE), the gap to forecasters that
use no media, the media share, and the stability of channel contribution shares across seeds.

    python run_realdata.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

torch.set_num_threads(1)

from cmf.baselines import NoMediaForecaster, RidgeMMM
from cmf.model import EnsembleCMF, ModelConfig
from cmf.realdata import load_dcm190

HOLD = 8


def _truncate(d, T):
    """Keep weeks 0..T-1 so that the holdout is the 8 weeks before T (rolling origin)."""
    from dataclasses import replace
    return replace(d, spend=d.spend[:, :T], controls=d.controls[:, :T], mediator=d.mediator[:, :T],
                   sales=d.sales[:, :T])


def metrics(pred, y, tT):
    out = {}
    for name, sl in [("train", slice(0, tT)), ("hold", slice(tT, None))]:
        p, a = pred[:, sl], y[:, sl]
        out[f"r2_pooled_{name}"] = float(1 - ((p - a) ** 2).sum() / ((a - a.mean()) ** 2).sum())
        gm = a.mean(1, keepdims=True)
        out[f"r2_within_{name}"] = float(1 - ((p - a) ** 2).sum() / ((a - gm) ** 2).sum())
        out[f"wape_{name}"] = float(np.abs(p - a).sum() / np.abs(a).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=["Naive", "NoMedia", "Ridge", "CMF-linbase", "CMF"])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--out", default="results/realdata.jsonl")
    ap.add_argument("--cutoff", type=int, default=None,
                    help="first holdout week (rolling origin); default = last 8 weeks")
    a = ap.parse_args()
    d = load_dcm190()
    tT = d.T - HOLD if a.cutoff is None else a.cutoff
    d = _truncate(d, tT + HOLD)
    print(f"panel: {d.G} DMAs x {d.T} weeks, {d.C} channels; holdout {HOLD} weeks", flush=True)
    done = set()
    if os.path.exists(a.out):
        done = {(r["method"], r["seed"]) for r in map(json.loads, open(a.out)) if r.get("cutoff", 101) == tT}
    sum_y = d.sales[:, :tT].sum()
    sum_x = d.spend[:, :tT].sum((0, 1))
    for meth in a.methods:
        seeds = [0] if meth in ("Naive", "NoMedia") else a.seeds
        for seed in seeds:
            if (meth, seed) in done:
                continue
            t0 = time.time()
            share = None
            if meth == "Naive":  # each DMA's mean of the last 8 training weeks, flat
                lvl = d.sales[:, tT - HOLD:tT].mean(1, keepdims=True)
                pred = np.concatenate([d.sales[:, :tT], np.repeat(lvl, d.T - tT, 1)], 1)
            elif meth == "NoMedia":
                pred = NoMediaForecaster().fit(d, tT).predict_holdout()
            elif meth == "Ridge":
                m = RidgeMMM(mediator_as_control=False, seed=seed, n_trials=60).fit(d, tT)
                pred = m.predict_holdout()
                share = m.roi()[0] * d.spend.sum((0, 1)) / d.sales.sum()
            else:
                cfg = ModelConfig(mediation=False, init="share", lambda_roi=0.0, epochs=a.epochs,
                                  baseline="linear" if meth == "CMF-linbase" else "transformer")
                e = EnsembleCMF(cfg, n_members=1).fit(d, tT, seed=seed)
                pred = e.predict_holdout()
                share = e.roi()[0] * d.spend.sum((0, 1)) / d.sales.sum()
            res = dict(method=meth, seed=seed, cutoff=tT, seconds=time.time() - t0, **metrics(pred, d.sales, tT))
            if share is not None:
                res.update(share=np.asarray(share).tolist(), media_share=float(np.sum(share)))
            with open(a.out, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"{meth:12s} seed={seed} R2pool={res['r2_pooled_hold']:.3f} R2within={res['r2_within_hold']:.3f} "
                  f"WAPE={res['wape_hold']:.3f} media={res.get('media_share', float('nan')):.2f} "
                  f"({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
