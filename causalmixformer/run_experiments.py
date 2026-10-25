"""Pilot experiments for CausalMixFormer on scenarios S1-S3.

Usage:
    python run_experiments.py --seeds 0 1 2 --members 5 --epochs 1200
Results are appended to results/raw.jsonl (one line per scenario/seed/method),
already-finished combinations are skipped, so the script can be resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

torch.set_num_threads(1)
from scipy.stats import spearmanr

from cmf.baselines import NoMediaForecaster, RidgeMMM
from cmf.budget import evaluate_allocation
from cmf.simulate import CHANNELS, SimConfig, simulate

HOLDOUT = 13


def make_method(name, members, epochs):
    from cmf.model import EnsembleCMF, ModelConfig

    base = dict(epochs=epochs)
    table = {
        "CMF": ModelConfig(**base),
        "CMF-noCT1": ModelConfig(mediation=False, **base),
        "CMF-noCT2": ModelConfig(constrained=False, **base),
        "CMF-noCT3": ModelConfig(carryover="geometric", **base),
        "CMF-linbase": ModelConfig(baseline="linear", **base),
    }
    return EnsembleCMF(table[name], n_members=members)


def r2(pred, y):
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


_NNN_CACHE = {}


def run_one(method, d, members, epochs, seed):
    tT = d.T - HOLDOUT
    t0 = time.time()
    if method == "XGB-SHAP":
        from cmf.baselines_reimpl import XGBShapMMM
        m = XGBShapMMM(seed=seed).fit(d, tT)
    elif method in ("NNN", "NNN-AR"):
        from cmf.baselines_reimpl import NNNLite
        key = (d.scenario, int(d.meta["seed"]), seed)
        if key not in _NNN_CACHE:  # one fit serves both attribution modes
            _NNN_CACHE.clear()
            _NNN_CACHE[key] = NNNLite(seed=seed).fit(d, tT)
        m = _NNN_CACHE[key].set_mode(method == "NNN-AR")
    elif method == "CausalMMM":
        from cmf.baselines_reimpl import CausalMMMLite
        m = CausalMMMLite(seed=seed).fit(d, tT)
    elif method == "Ridge":
        m = RidgeMMM(mediator_as_control=False, seed=seed).fit(d, tT)
    elif method == "Ridge+M":
        m = RidgeMMM(mediator_as_control=True, seed=seed).fit(d, tT)
    else:
        m = make_method(method, members, epochs).fit(d, tT, seed=seed)
    roi, lo, hi = m.roi()
    pred = m.predict_holdout()
    true = d.true_roi
    ape = np.abs(roi - true) / true
    out = dict(
        roi=roi.tolist(), roi_lo=np.asarray(lo).tolist(), roi_hi=np.asarray(hi).tolist(),
        true_roi=true.tolist(),
        roi_mape=float(ape.mean()),
        spearman=float(spearmanr(roi, true).statistic),
        tv_bias_pct=float(100 * (roi[0] - true[0]) / true[0]),
        r2_holdout=r2(pred[:, tT:], d.sales[:, tT:]),
        coverage=float(np.mean((true >= lo) & (true <= hi))) if method.startswith("CMF") else None,
        interval_width=float(np.mean((np.asarray(hi) - np.asarray(lo)) / true)) if method.startswith("CMF") else None,
    )
    out["true_direct_roi"] = np.asarray(d.true_direct_roi).tolist()
    if hasattr(m, "rois_direct"):  # CausalMixFormer: direct part of the ROI (beta path only)
        out["roi_direct"] = m.rois_direct.mean(0).tolist()
        out["roi_members"] = np.asarray(m.rois).tolist()
    bud = evaluate_allocation(d, m.media_value)
    out.update(uplift_captured=float(bud["uplift_captured"]), true_gain_pct=float(bud["true_gain_pct"]),
               oracle_gain_pct=float(bud["oracle_gain_pct"]), seconds=time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+",
                    default=["Ridge", "Ridge+M", "CMF", "CMF-noCT1", "CMF-noCT2", "CMF-noCT3", "CMF-linbase"])
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--out", default="results/raw.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["scenario"], r["seed"], r["method"]))
    for s in a.scenarios:
        for seed in a.seeds:
            d = simulate(SimConfig(scenario=s, seed=seed))
            tT = d.T - HOLDOUT
            if (s, seed, "NoMedia") not in done:
                p = NoMediaForecaster().fit(d, tT).predict_holdout()
                with open(a.out, "a") as f:
                    f.write(json.dumps(dict(scenario=s, seed=seed, method="NoMedia",
                                            r2_holdout=r2(p[:, tT:], d.sales[:, tT:]))) + "\n")
            for meth in a.methods:
                if (s, seed, meth) in done:
                    continue
                res = run_one(meth, d, a.members, a.epochs, seed)
                res.update(scenario=s, seed=seed, method=meth, channels=CHANNELS)
                with open(a.out, "a") as f:
                    f.write(json.dumps(res) + "\n")
                print(f"{s} seed={seed} {meth:12s} MAPE={res['roi_mape']:.3f} rho={res['spearman']:.2f} "
                      f"TV={res['tv_bias_pct']:+.0f}% R2={res['r2_holdout']:.3f} "
                      f"uplift={res['uplift_captured']:.2f} ({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
