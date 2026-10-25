"""Simple reference baselines run inside this repository.

RidgeMMM  : Robyn-style pipeline - geometric adstock + Hill transform with
            hyper-parameters chosen by random search on a time-based validation
            split, non-negative ridge regression on the transformed media.
            `mediator_as_control=True` reproduces the common practice of adding
            brand-search volume as a control variable.
NoMediaForecaster: same baseline features (+ mediator) without any media input;
            used to check whether holdout R^2 actually reflects media modelling.

The full benchmark (Meridian, Robyn, PyMC-Marketing, LightweightMMM, Orbit KTR,
DeepCausalMMM) lives in the author's benchmark repository and is plugged in
through `cmf.benchmark_adapter`.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .simulate import MMMData, apply_carryover, calendar_features, geometric_kernel, safe_scale


def _hill(a, half, slope=1.0):
    a = np.maximum(a, 0)
    return a**slope / (a**slope + half**slope)


def _design(d: MMMData, decays, halves, L, mediator_as_control, include_media=True, scale_T=None):
    G, T, C = d.spend.shape
    sT = slice(0, scale_T)
    y_scale = d.sales[:, sT].mean(1, keepdims=True)
    x_scale = safe_scale(d.spend, scale_T)
    xn = d.spend / x_scale
    cols = []
    if include_media:
        for c in range(C):
            a = apply_carryover(xn[:, :, c], geometric_kernel(decays[c], L))
            cols.append(_hill(a, halves[c]))
    cal = np.broadcast_to(calendar_features(T)[None], (G, T, calendar_features(T).shape[-1]))
    z = (d.controls - d.controls[:, sT].mean((0, 1))) / (d.controls[:, sT].std((0, 1)) + 1e-8)
    extra = [z, cal]
    if mediator_as_control:
        extra.append((d.mediator / d.mediator[:, sT].mean(1, keepdims=True))[..., None])
    geo_dummies = np.broadcast_to(np.eye(G)[:, None, :], (G, T, G))
    Xm = np.stack(cols, -1) if cols else np.zeros((G, T, 0))
    Xo = np.concatenate(extra + [geo_dummies], -1)
    return Xm, Xo, d.sales / y_scale, y_scale


class RidgeMMM:
    def __init__(self, mediator_as_control=False, n_trials=150, L=13, alpha=1e-2, seed=0):
        self.mac, self.n_trials, self.L, self.alpha, self.seed = mediator_as_control, n_trials, L, alpha, seed

    def _fit_once(self, d, decays, halves, train_T, val_T=None):
        Xm, Xo, yn, ys = _design(d, decays, halves, self.L, self.mac, scale_T=train_T)
        C = Xm.shape[-1]
        X = np.concatenate([Xm, Xo], -1)
        end = val_T if val_T is not None else train_T
        Xtr, ytr = X[:, :end].reshape(-1, X.shape[-1]), yn[:, :end].reshape(-1)
        # non-negative media coefficients: sklearn Ridge(positive=True) applies to all,
        # so flip sign of non-media columns by duplicating them (+/-).
        Xtr2 = np.concatenate([Xtr[:, :C], Xtr[:, C:], -Xtr[:, C:]], 1)
        r = Ridge(alpha=self.alpha, positive=True, fit_intercept=True).fit(Xtr2, ytr)
        w = r.coef_
        coef = np.concatenate([w[:C], w[C:C + Xo.shape[-1]] - w[C + Xo.shape[-1]:]])
        pred = (X @ coef + r.intercept_) * ys
        return coef, r.intercept_, pred, Xm, ys

    def fit(self, d: MMMData, train_T: int):
        rng = np.random.default_rng(self.seed)
        C = d.C
        val_T = train_T - 13
        best = None
        for _ in range(self.n_trials):
            decays = rng.uniform(0.0, 0.85, C)
            halves = rng.uniform(0.3, 2.0, C)
            _, _, pred, _, _ = self._fit_once(d, decays, halves, train_T=val_T, val_T=val_T)
            err = np.mean((pred[:, val_T:train_T] - d.sales[:, val_T:train_T]) ** 2)
            if best is None or err < best[0]:
                best = (err, decays, halves)
        _, self.decays, self.halves = best
        self.d, self.train_T = d, train_T
        self.coef, self.intercept, self.pred, Xm, ys = self._fit_once(d, self.decays, self.halves, train_T)
        # ROI: media enters linearly after transform; zeroing channel c removes Xm[...,c]*beta_c
        # (Hill(0)=0). The mediator, if used as control, is held fixed.
        eff = Xm * self.coef[None, None, :C] * ys[:, :, None]
        self.roi_ = eff.sum((0, 1)) / d.spend.sum((0, 1))
        return self

    def roi(self):
        return self.roi_, self.roi_, self.roi_

    def media_value(self, mult):
        d = self.d
        sT = slice(0, self.train_T)
        ys = d.sales[:, sT].mean(1, keepdims=True)
        xn = d.spend * np.asarray(mult)[None, None, :] / safe_scale(d.spend, self.train_T)
        tot = 0.0
        for c in range(d.C):
            a = apply_carryover(xn[:, :, c], geometric_kernel(self.decays[c], self.L))
            tot += (self.coef[c] * _hill(a, self.halves[c]) * ys).sum()
        return float(tot)

    def predict_holdout(self):
        return self.pred


class NoMediaForecaster:
    """Ridge on calendar, controls, geo and mediator - no media at all."""

    def fit(self, d: MMMData, train_T: int):
        _, Xo, yn, ys = _design(d, None, None, 13, mediator_as_control=True, include_media=False, scale_T=train_T)
        Xtr, ytr = Xo[:, :train_T].reshape(-1, Xo.shape[-1]), yn[:, :train_T].reshape(-1)
        r = Ridge(alpha=1e-2).fit(Xtr, ytr)
        self.pred = (Xo @ r.coef_ + r.intercept_) * ys
        return self

    def predict_holdout(self):
        return self.pred
