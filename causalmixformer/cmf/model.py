"""CausalMixFormer: a mediation-aware causal temporal Transformer for MMM.

Structure (per geo g, week t, channel c):

  a_{g,t,c} = sum_l w_{c,l} x_{g,t-l,c}                       (CT3 lag-attention carryover)
  h_{g,t,c} = Hill(a_{g,t,c}; K_c, n_c)                       (saturation)
  M_{g,t}   = m_{g,t} + sum_c alpha_c h_{g,t,c} + e^M          (CT1 mediator equation)
  y_{g,t}   = b_{g,t} + sum_c beta_c h_{g,t,c} + delta M_{g,t} + e^y
  b_{g,t}   = Transformer(controls, calendar, geo)            (non-linear baseline, media-free)

beta, alpha, delta >= 0 and the media contribution share is softly kept inside a
prior range (CT2). Uncertainty comes from a bootstrap-over-geos deep ensemble (CT4).
All variables are normalised by their per-geo means, so effects are converted back
to currency with the geo mean of sales before computing ROI.

Ablation switches (ModelConfig):
  mediation=False      -> mediator used as an ordinary control (common practice)
  carryover='geometric'-> classic geometric adstock instead of lag attention
  constrained=False    -> unconstrained (signed) coefficients, no share prior
  baseline='linear'    -> linear calendar/controls baseline instead of Transformer
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .simulate import MMMData, calendar_features, safe_scale


@dataclass
class ModelConfig:
    lag_window: int = 13
    d_model: int = 16
    n_heads: int = 2
    n_layers: int = 1
    dropout: float = 0.1
    mediation: bool = True
    carryover: str = "attention"  # attention | geometric
    constrained: bool = True
    baseline: str = "transformer"  # transformer | linear
    share_prior: tuple = (0.05, 0.60)
    lambda_med: float = 1.0
    lambda_share: float = 10.0
    lambda_alpha_l1: float = 1e-3
    lambda_tf: float = 1e-2
    roi_prior: tuple = (0.2, 0.9)  # LogNormal(mu, sd) on channel ROI, Meridian's default
    lambda_roi: float = 1e-5
    init: str = "roi"  # roi | share  (how beta is initialised, see train_member)
    init_jitter: float = 0.5  # sd of the log-normal jitter of each channel's initial ROI (members k>=1)
    # channels allowed to affect the mediator; "no_paid_search" encodes that paid-search ads are
    # served in response to queries, so paid search cannot cause organic query volume
    mediator_parents: str = "no_paid_search"  # all | no_paid_search
    lr: float = 1e-2
    epochs: int = 1000
    weight_decay: float = 1e-4


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------
class Prepared:
    """Normalised tensors for one dataset."""

    def __init__(self, d: MMMData, geo_idx=None, scale_T=None):
        geo_idx = np.arange(d.G) if geo_idx is None else np.asarray(geo_idx)
        self.geo_idx = geo_idx
        sp, y, m, z = d.spend[geo_idx], d.sales[geo_idx], d.mediator[geo_idx], d.controls[geo_idx]
        sT = slice(0, scale_T)  # normalise with training-period means only (no holdout leakage)
        self.y_scale = y[:, sT].mean(1, keepdims=True)  # (G,1)
        self.x_scale = safe_scale(sp, scale_T)  # (G,1,C)
        self.m_scale = m[:, sT].mean(1, keepdims=True)
        z_mu, z_sd = z[:, sT].mean((0, 1), keepdims=True), z[:, sT].std((0, 1), keepdims=True) + 1e-8
        cal = np.broadcast_to(calendar_features(d.T)[None], (len(geo_idx), d.T, calendar_features(d.T).shape[-1]))
        f = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
        self.x = f(sp / self.x_scale)
        self.y = f(y / self.y_scale)
        self.m = f(m / self.m_scale)
        self.z = f(np.concatenate([(z - z_mu) / z_sd, cal], -1))
        self.raw_spend = sp
        self.channels = list(d.channels)
        self.G, self.T, self.C = sp.shape
        self.geo_id = torch.arange(len(geo_idx))


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
class LagAttentionCarryover(nn.Module):
    """Channel-specific attention over a window of past lags (CT3).

    w_{c,l} = softmax_l( q_c . k_l / sqrt(d) ), keys k_l are learned lag embeddings
    (initialised with a sinusoidal position code). The weights form a free-form,
    non-negative lag distribution that can peak at any lag, unlike geometric decay.
    """

    def __init__(self, C, L, d=16):
        super().__init__()
        self.L, self.d = L, d
        pos = torch.arange(L).float()[:, None]
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(100.0) / d))
        pe = torch.zeros(L, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.keys = nn.Parameter(pe.clone())
        self.query = nn.Parameter(0.1 * torch.randn(C, d))
        # recency prior: logits start at -0.7*l (equivalent to geometric decay ~0.5),
        # the query-key term then reshapes the distribution freely (delayed peaks etc.)
        self.recency = nn.Parameter(torch.full((C, 1), 0.7))

    def weights(self):
        lags = torch.arange(self.L, device=self.keys.device).float()[None]
        logits = self.query @ self.keys.T / math.sqrt(self.d) - F.softplus(self.recency) * lags
        return torch.softmax(logits, -1)  # (C, L)

    def forward(self, x):  # x (G,T,C)
        return causal_conv(x, self.weights())


class GeometricCarryover(nn.Module):
    def __init__(self, C, L):
        super().__init__()
        self.L = L
        self.logit = nn.Parameter(torch.zeros(C))

    def weights(self):
        decay = torch.sigmoid(self.logit)[:, None] * 0.95
        w = decay ** torch.arange(self.L).float()[None]
        return w / w.sum(-1, keepdim=True)

    def forward(self, x):
        return causal_conv(x, self.weights())


def causal_conv(x, w):
    """x (G,T,C), w (C,L) -> (G,T,C), a_t = sum_l w_l x_{t-l}."""
    G, T, C = x.shape
    L = w.shape[1]
    xp = F.pad(x.permute(0, 2, 1), (L - 1, 0))  # (G,C,T+L-1)
    kernel = w.flip(-1)[:, None, :]  # (C,1,L)
    return F.conv1d(xp, kernel, groups=C).permute(0, 2, 1)


class Hill(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.log_half = nn.Parameter(torch.zeros(C))
        self.slope_raw = nn.Parameter(torch.zeros(C))

    def forward(self, a):
        half = torch.exp(self.log_half)
        n = 0.5 + 2.5 * torch.sigmoid(self.slope_raw)
        a = a.clamp_min(1e-6)
        return a**n / (a**n + half**n)


class TransformerBaseline(nn.Module):
    def __init__(self, n_feat, n_geo, cfg: ModelConfig):
        super().__init__()
        self.inp = nn.Linear(n_feat, cfg.d_model)
        self.geo = nn.Embedding(n_geo, cfg.d_model)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, 2 * cfg.d_model, cfg.dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(cfg.d_model, 1)
        self.lin = nn.Linear(n_feat, 1)  # interpretable linear part
        nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)

    def forward(self, z, geo_id):
        h = self.inp(z) + self.geo(geo_id)[:, None, :]
        resid = self.out(self.enc(h)).squeeze(-1)
        return self.lin(z).squeeze(-1) + resid, resid


class LinearBaseline(nn.Module):
    def __init__(self, n_feat, n_geo, cfg):
        super().__init__()
        self.lin = nn.Linear(n_feat, 1)
        self.geo = nn.Embedding(n_geo, 1)
        nn.init.zeros_(self.geo.weight)

    def forward(self, z, geo_id):
        out = self.lin(z).squeeze(-1) + self.geo(geo_id)
        return out, torch.zeros_like(out)


# ---------------------------------------------------------------------------
# full model
# ---------------------------------------------------------------------------
class CausalMixFormer(nn.Module):
    def __init__(self, C, n_feat, n_geo, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.carry = (LagAttentionCarryover(C, cfg.lag_window) if cfg.carryover == "attention"
                      else GeometricCarryover(C, cfg.lag_window))
        self.hill = Hill(C)
        Base = TransformerBaseline if cfg.baseline == "transformer" else LinearBaseline
        self.base_y = Base(n_feat, n_geo, cfg)
        self.base_m = LinearBaseline(n_feat, n_geo, cfg)
        self.beta_raw = nn.Parameter(torch.full((C,), -1.0))
        self.alpha_raw = nn.Parameter(torch.full((C,), -3.0))
        self.register_buffer("alpha_mask", torch.ones(C))
        self.delta_raw = nn.Parameter(torch.tensor(-2.0))
        with torch.no_grad():
            self.base_y.lin.bias.fill_(0.7)
            self.base_m.lin.bias.fill_(1.0)

    def pos(self, v):
        return F.softplus(v) if self.cfg.constrained else v

    def coefs(self):
        return self.pos(self.beta_raw), self.pos(self.alpha_raw) * self.alpha_mask, self.pos(self.delta_raw)

    def media_response(self, x):
        return self.hill(self.carry(x))  # (G,T,C)

    def forward(self, x, z, geo_id, m_obs):
        h = self.media_response(x)
        beta, alpha, delta = self.coefs()
        b, resid = self.base_y(z, geo_id)
        direct = h * beta  # (G,T,C)
        if self.cfg.mediation:
            m_base, _ = self.base_m(z, geo_id)
            m_hat = m_base + (h * alpha).sum(-1)
        else:
            m_hat = None
        y_hat = b + direct.sum(-1) + delta * m_obs
        return dict(y=y_hat, m=m_hat, direct=direct, h=h, base=b, resid=resid)

    # ---- counterfactual total effect of each channel, in normalised units
    @torch.no_grad()
    def channel_effects(self, x, z, geo_id, m_obs, direct_only: bool = False):
        """Return (G,T,C) total effect: y(x) - y(x with channel c off).

        With mediation=True the mediator is updated through its structural equation,
        so the effect contains the indirect path. With mediation=False the mediator
        is held at its observed value (what a control-variable MMM implicitly does).
        """
        beta, alpha, delta = self.coefs()
        h = self.media_response(x)
        out = torch.zeros_like(h)
        for c in range(h.shape[-1]):
            x0 = x.clone()
            x0[..., c] = 0
            h0 = self.media_response(x0)
            dh = h - h0
            eff = (dh * beta).sum(-1)
            if self.cfg.mediation and not direct_only:
                eff = eff + delta * (dh * alpha).sum(-1)
            out[..., c] = eff
        return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_member(prep: Prepared, cfg: ModelConfig, train_T: int, seed: int, verbose=False, jitter: float = 0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CausalMixFormer(prep.C, prep.z.shape[-1], prep.G, cfg)
    if cfg.mediator_parents == "no_paid_search" and getattr(prep, "channels", None):
        mask = torch.tensor([0.0 if "search" in c.lower() else 1.0 for c in prep.channels])
        model.alpha_mask.copy_(mask)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    sl = slice(0, train_T)
    lo, hi = cfg.share_prior
    ys = torch.tensor(prep.y_scale, dtype=torch.float32)  # (G,1)
    spend_tot = torch.tensor(prep.raw_spend[:, :train_T].sum((0, 1)), dtype=torch.float32)  # (C,)
    mu, sd = cfg.roi_prior

    def roi_of(h, beta, alpha, delta):
        w = beta + (delta * alpha if cfg.mediation else 0.0)
        return w * (h[:, sl] * ys[:, :, None]).sum((0, 1)) / spend_tot

    # initialise beta so that every channel starts at the prior-median ROI instead of an
    # equal normalised coefficient (which would imply ROI inversely proportional to spend)
    with torch.no_grad():
        h0 = model.media_response(prep.x)
        if cfg.init == "roi":
            target = math.exp(mu) * spend_tot / (h0[:, sl] * ys[:, :, None]).sum((0, 1))
        else:  # "share": one common coefficient so that media explain the mid-point share
            # (used when inputs are impressions rather than spend, so an ROI prior has no scale).
            # A per-channel equal share would give a channel that is active in only a few regions
            # a huge coefficient, which explodes once the channel is switched on elsewhere.
            s_mid = 0.5 * (lo + hi)
            common = s_mid * prep.y[:, sl].sum() / h0[:, sl].sum().clamp_min(1e-6)
            target = torch.full((prep.C,), float(common))
        if jitter > 0:  # randomised initialisation: each channel's starting ROI drawn around the prior median
            g = torch.Generator().manual_seed(seed + 7919)
            target = target * torch.exp(jitter * torch.randn(prep.C, generator=g))
        target = target.clamp_min(1e-4)
        model.beta_raw.copy_(torch.log(torch.expm1(target)) if cfg.constrained else target)
    for ep in range(cfg.epochs):
        model.train()
        out = model(prep.x, prep.z, prep.geo_id, prep.m)
        loss = F.mse_loss(out["y"][:, sl], prep.y[:, sl])
        if cfg.mediation:
            loss = loss + cfg.lambda_med * F.mse_loss(out["m"][:, sl], prep.m[:, sl])
            loss = loss + cfg.lambda_alpha_l1 * model.coefs()[1].abs().sum()
        if cfg.constrained:
            # media share of predicted sales (direct + mediated part of delta*M)
            beta, alpha, delta = model.coefs()
            media = out["direct"].sum(-1)
            if cfg.mediation:
                media = media + delta * (out["h"] * alpha).sum(-1)
            share = media[:, sl].sum() / out["y"][:, sl].sum().clamp_min(1e-6)
            loss = loss + cfg.lambda_share * (F.relu(lo - share) ** 2 + F.relu(share - hi) ** 2)
            roi = roi_of(out["h"], beta, alpha, delta)
            loss = loss + cfg.lambda_roi * (((torch.log(roi.clamp_min(1e-4)) - mu) / sd) ** 2).mean()
        loss = loss + cfg.lambda_tf * out["resid"][:, sl].pow(2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if verbose and ep % 200 == 0:
            print(ep, float(loss))
    model.eval()
    return model


def roi_from_model(model, prep: Prepared, direct_only: bool = False):
    eff_n = model.channel_effects(prep.x, prep.z, prep.geo_id, prep.m, direct_only=direct_only).numpy()  # (G,T,C)
    eff = eff_n * prep.y_scale[:, :, None]
    return eff.sum((0, 1)) / prep.raw_spend.sum((0, 1))


def predict(model, prep: Prepared):
    with torch.no_grad():
        out = model(prep.x, prep.z, prep.geo_id, prep.m)
    return out["y"].numpy() * prep.y_scale


class EnsembleCMF:
    """Bootstrap-over-geos deep ensemble (CT4)."""

    def __init__(self, cfg: ModelConfig, n_members: int = 5, bootstrap: bool = True):
        self.cfg, self.n, self.bootstrap = cfg, n_members, bootstrap

    def fit(self, d: MMMData, train_T: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.d, self.train_T = d, train_T
        self.full = Prepared(d, scale_T=train_T)
        self.members, self.rois, self.rois_direct = [], [], []
        for k in range(self.n):
            idx = rng.choice(d.G, d.G, replace=True) if (self.bootstrap and k > 0) else np.arange(d.G)
            prep = Prepared(d, idx, scale_T=train_T)
            m = train_member(prep, self.cfg, train_T, seed=seed * 100 + k,
                             jitter=self.cfg.init_jitter if k > 0 else 0.0)
            self.members.append((m, idx))
            # ROI is evaluated on the full panel (geo embeddings are only used by the baseline,
            # which does not enter the counterfactual difference)
            self.rois.append(roi_from_model(m, self.full_for(idx)))
            self.rois_direct.append(roi_from_model(m, self.full_for(idx), direct_only=True))
        self.rois = np.array(self.rois)
        self.rois_direct = np.array(self.rois_direct)
        return self

    def full_for(self, idx):
        # members trained on a bootstrap sample have geo embeddings indexed by position in idx;
        # for counterfactual ROI only media parameters matter, so reuse the full panel.
        return self.full

    def roi(self):
        """Ensemble mean and a 90% interval mean +/- t_{0.95, E-1} * sd over members."""
        from scipy.stats import t as _t
        mu = self.rois.mean(0)
        if self.n < 2:
            return mu, mu, mu
        sd = self.rois.std(0, ddof=1)
        q = _t.ppf(0.95, self.n - 1)
        return mu, mu - q * sd, mu + q * sd

    def media_value(self, mult):
        """Ensemble-mean predicted media-driven sales (currency) if each channel's
        spend path is scaled by mult[c]. Includes the mediated path when modelled."""
        x = self.full.x * torch.tensor(np.asarray(mult), dtype=torch.float32)[None, None, :]
        vals = []
        with torch.no_grad():
            for m, _ in self.members:
                beta, alpha, delta = m.coefs()
                w = beta + (delta * alpha if m.cfg.mediation else 0.0)
                h = m.media_response(x)
                vals.append(float(((h * w).sum(-1).numpy() * self.full.y_scale).sum()))
        return float(np.mean(vals))

    def predict_holdout(self):
        """Holdout prediction from the member trained on all geos (member 0)."""
        m, _ = self.members[0]
        return predict(m, self.full)
