"""Run CausalMixFormer on the national-level datasets of the preliminary benchmark.

The benchmark (bench/ folder, author's code) contains 9 simulated national series
(S1-S3 x seeds 1-3, 156 weeks, 16 test weeks, channels tv/video/social/display/search) and
results of Robyn, Meridian, PyMC-Marketing, LightweightMMM, Orbit KTR and DeepCausalMMM (x2).
To compare on identical terms, this adapter
  * reads the benchmark's CSV/JSON files unchanged,
  * computes ROI exactly as the benchmark's DGP defines it: channel switched off during the
    training weeks only, effect summed over the training weeks + 12 carry-over weeks,
    divided by the training-period spend,
  * scores train/test fit with the benchmark's own formulas (MAPE in %, R^2),
  * writes one JSON per run in the benchmark's result format (results/<model>/<dataset>.json).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .model import EnsembleCMF, ModelConfig
from .simulate import MMMData

CARRY_L = 12  # same L as the benchmark DGP


def load_bench(csv_path, meta_path):
    df = pd.read_csv(csv_path)
    meta = json.load(open(meta_path))
    ch = meta["channels"]
    T = len(df)
    spend = df[[f"{c}_spend" for c in ch]].to_numpy(float)[None]  # (1,T,C)
    controls = df[["promo"]].to_numpy(float)[None]  # calendar/trend features are built by the model
    has_med = "brand_search" in df.columns
    mediator = (df["brand_search"].to_numpy(float) if has_med else np.ones(T))[None]
    d = MMMData(spend=spend, controls=controls, mediator=mediator, sales=df["revenue"].to_numpy(float)[None],
                true_roi=np.array([meta["true_roi"][c] for c in ch]), true_direct_roi=np.full(len(ch), np.nan),
                true_contrib_share=float("nan"), channels=list(ch), scenario=meta["scenario"],
                meta=dict(seed=meta["seed"], has_mediator=has_med))
    return d, meta


def _metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return dict(mape=float(np.mean(np.abs(y - yhat) / np.abs(y)) * 100),
                r2=float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)),
                nrmse=float(np.sqrt(np.mean((y - yhat) ** 2)) / (y.max() - y.min())))


@torch.no_grad()
def bench_roi(ens: EnsembleCMF, train_T: int):
    """ROI with the benchmark's definition, averaged over ensemble members."""
    prep = ens.full
    T = prep.T
    win = slice(0, min(T, train_T + CARRY_L))
    spend_tr = prep.raw_spend[:, :train_T].sum((0, 1))
    rois = []
    for m, _ in ens.members:
        beta, alpha, delta = m.coefs()
        w = beta + (delta * alpha if m.cfg.mediation else 0.0)  # total-effect weight per channel
        h = m.media_response(prep.x)
        r = np.zeros(prep.C)
        for c in range(prep.C):
            x0 = prep.x.clone()
            x0[:, :train_T, c] = 0
            dh = (h - m.media_response(x0))[..., c] * w[c]  # (G,T)
            r[c] = float((dh[:, win].numpy() * prep.y_scale).sum() / spend_tr[c])
        rois.append(r)
    rois = np.array(rois)
    bench_roi.last_members = rois  # kept for the post-hoc first-member analysis
    return rois.mean(0), rois.std(0, ddof=1) if len(rois) > 1 else np.zeros(prep.C)


def run(csv_path, meta_path, out_path, model_name="CausalMixFormer", members=5, seed=0, **cfg_kw):
    torch.set_num_threads(1)
    d, meta = load_bench(csv_path, meta_path)
    T = d.T
    train_T = T - meta["test_weeks"]
    cfg_kw.setdefault("mediation", bool(d.meta["has_mediator"]))
    cfg = ModelConfig(**cfg_kw)
    t0 = time.time()
    ens = EnsembleCMF(cfg, n_members=members).fit(d, train_T, seed=seed)
    roi, roi_sd = bench_roi(ens, train_T)
    yhat = ens.predict_holdout()[0]
    y = d.sales[0]
    tr = np.arange(T) < train_T
    ch = meta["channels"]
    res = dict(model=model_name, dataset=os.path.basename(csv_path).replace(".csv", ""), scenario=meta["scenario"],
               seed=meta["seed"], runtime_s=round(time.time() - t0, 1),
               train=_metrics(y[tr], yhat[tr]), test=_metrics(y[~tr], yhat[~tr]),
               roi={c: float(v) for c, v in zip(ch, roi)}, roi_sd={c: float(v) for c, v in zip(ch, roi_sd)},
               roi_members=[{c: float(v) for c, v in zip(ch, m)} for m in bench_roi.last_members],
               true_roi=meta["true_roi"], notes=json.dumps({k: v for k, v in cfg_kw.items()}))
    tru = meta["true_roi"]
    res["roi_mape"] = float(np.mean([abs(res["roi"][c] - tru[c]) / abs(tru[c]) for c in tru]) * 100)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)
    return res
