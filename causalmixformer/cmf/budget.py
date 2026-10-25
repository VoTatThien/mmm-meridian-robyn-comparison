"""Budget re-allocation driven by each model's estimated response curves.

Every channel's whole spend path is multiplied by m_c in [lo, hi] subject to an
unchanged total budget. Each model chooses m to maximise its *own* predicted
media-driven sales; the choice is then scored with the simulator's true outcome.

  uplift_captured = (true(m_model) - true(1)) / (true(m_oracle) - true(1))

1.0 means the model found the oracle allocation, 0 means no gain over the
current plan, negative means the model's advice destroys value.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .simulate import MMMData


def _optimise(value_fn, spend_tot, lo=0.5, hi=1.5, n_starts=4, seed=0):
    C = len(spend_tot)
    budget = spend_tot.sum()
    cons = [{"type": "eq", "fun": lambda m: (m * spend_tot).sum() / budget - 1.0}]
    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_starts):
        m0 = np.ones(C) if k == 0 else rng.uniform(lo, hi, C)
        m0 = m0 / ((m0 * spend_tot).sum() / budget)
        m0 = np.clip(m0, lo, hi)
        res = minimize(lambda m: -value_fn(m) / 1e3, m0, method="SLSQP", bounds=[(lo, hi)] * C,
                       constraints=cons, options=dict(maxiter=200, ftol=1e-9, eps=1e-3))
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def true_value(d: MMMData, m):
    sp = d.spend * np.asarray(m)[None, None, :]
    return (d.meta["outcome"](sp) - d.meta["y_nomedia"]).sum()


def evaluate_allocation(d: MMMData, model_value_fn, lo=0.5, hi=1.5):
    spend_tot = d.spend.sum((0, 1))
    m_model = _optimise(model_value_fn, spend_tot, lo, hi)
    m_oracle = _optimise(lambda m: true_value(d, m), spend_tot, lo, hi)
    v1, vm, vo = true_value(d, np.ones(d.C)), true_value(d, m_model), true_value(d, m_oracle)
    captured = (vm - v1) / (vo - v1) if vo - v1 > 1e-9 else np.nan
    return dict(m_model=m_model, m_oracle=m_oracle, uplift_captured=captured,
                true_gain_pct=100 * (vm - v1) / v1, oracle_gain_pct=100 * (vo - v1) / v1)
