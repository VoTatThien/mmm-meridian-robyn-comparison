"""Re-implementations of three published MMM methods that release no code.

All three are simplified versions written from the papers' descriptions and adapted to scalar
weekly spend panels, so that they run on exactly the same data and metrics as the other methods.
Deviations from the original papers are listed in each class docstring and must be reported.

  XGBShapMMM   gradient-boosted trees + Shapley attribution (Chen & Guestrin 2016; Lundberg & Lee 2017)
  NNNLite      NNN, attention-based neural MMM (Mulc et al. 2025, arXiv:2504.06212)
  CausalMMMLite CausalMMM, Granger structure learning + response decoder (Gong et al., WSDM 2024)

Common interface (as RidgeMMM / EnsembleCMF):
  fit(d, train_T) ; roi() -> (roi, lo, hi) ; predict_holdout() -> (G,T) ; media_value(mult) -> float
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .simulate import MMMData, calendar_features, safe_scale


# ---------------------------------------------------------------------------
# shared normalisation (training-period statistics only)
# ---------------------------------------------------------------------------
class _Norm:
    def __init__(self, d: MMMData, train_T: int):
        sT = slice(0, train_T)
        self.ys = d.sales[:, sT].mean(1, keepdims=True)  # (G,1)
        self.xs = safe_scale(d.spend, train_T)  # (G,1,C)
        self.ms = d.mediator[:, sT].mean(1, keepdims=True)
        z = d.controls
        self.z = (z - z[:, sT].mean((0, 1))) / (z[:, sT].std((0, 1)) + 1e-8)
        self.cal = np.broadcast_to(calendar_features(d.T)[None], (d.G, d.T, calendar_features(d.T).shape[-1]))
        self.x = d.spend / self.xs
        self.y = d.sales / self.ys
        self.m = d.mediator / self.ms
        self.u = np.concatenate([self.z, self.cal], -1)  # (G,T,K) controls + calendar


def _t(a):
    return torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)


# ===========================================================================
# 1. XGBoost + SHAP
# ===========================================================================
class XGBShapMMM:
    """Gradient-boosted trees on lagged spend, controls, calendar and the mediator, with
    channel attribution by exact group Shapley values against a zero-media baseline.

    Deviations / choices: the original XGBoost+SHAP(+LLM) MMM studies use SHAP for explanation;
    for ROI we use *baseline* Shapley values with all media switched off as the reference
    (rather than the data mean), so that the contributions of the channels sum to the total
    media effect f(x) - f(x | media = 0). Lags 0..7 of each channel are given as features.
    Branded search enters as an ordinary feature, as in common practice. The LLM component
    (natural-language reporting) is irrelevant for attribution and omitted.
    """

    def __init__(self, n_lags=8, seed=0):
        self.n_lags, self.seed = n_lags, seed

    def _features(self, x):  # x (G,T,C) normalised spend
        G, T, C = x.shape
        lags = []
        for l in range(self.n_lags):
            xl = np.zeros_like(x)
            xl[:, l:] = x[:, : T - l]
            lags.append(xl)
        X = np.concatenate(lags + [self.n.u, self.n.m[..., None]], -1)
        return X.reshape(G * T, -1)

    def fit(self, d: MMMData, train_T: int):
        from xgboost import XGBRegressor

        self.d, self.tT, self.n = d, train_T, _Norm(d, train_T)
        X = self._features(self.n.x).reshape(d.G, d.T, -1)
        self.model = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=5, random_state=self.seed,
                                  n_jobs=1, verbosity=0)
        self.model.fit(X[:, :train_T].reshape(-1, X.shape[-1]), self.n.y[:, :train_T].reshape(-1))
        self.pred = self._predict(self.n.x) * self.n.ys
        self.roi_ = self._shapley_roi()
        return self

    def _predict(self, x):
        return self.model.predict(self._features(x)).reshape(self.d.G, self.d.T)

    def _shapley_roi(self):
        C = self.d.C
        cache = {}
        for S in itertools.product([0, 1], repeat=C):  # 2^C coalitions, channels on/off
            mask = np.array(S, dtype=float)
            cache[S] = (self._predict(self.n.x * mask[None, None, :]) * self.n.ys).sum()
        phi = np.zeros(C)
        for c in range(C):
            for S in cache:
                if S[c] == 1:
                    continue
                k = sum(S)
                w = math.factorial(k) * math.factorial(C - k - 1) / math.factorial(C)
                S1 = tuple(1 if i == c else s for i, s in enumerate(S))
                phi[c] += w * (cache[S1] - cache[S])
        return phi / self.d.spend.sum((0, 1))

    def roi(self):
        return self.roi_, self.roi_, self.roi_

    def predict_holdout(self):
        return self.pred

    def media_value(self, mult):
        x = self.n.x * np.asarray(mult)[None, None, :]
        return float(((self._predict(x) - self._predict(np.zeros_like(x))) * self.n.ys).sum())


# ===========================================================================
# 2. NNN (simplified, scalar inputs)
# ===========================================================================
class _FactoredAttention(nn.Module):
    """Temporal attention a_time(lag/omega, onehot(c)) x channel attention softmax(psi)."""

    def __init__(self, Cp, omega, tau=1.0):
        super().__init__()
        self.Cp, self.omega, self.tau = Cp, omega, tau
        self.time_mlp = nn.Sequential(nn.Linear(1 + Cp, 128), nn.ReLU(), nn.Linear(128, 1))
        self.psi = nn.Parameter(torch.eye(Cp) * 3.0)  # identity init (sharpened)

    def kernels(self):
        lag = torch.arange(self.omega).float() / self.omega
        inp = torch.cat([lag[None, :, None].expand(self.Cp, -1, 1),
                         torch.eye(self.Cp)[:, None, :].expand(-1, self.omega, -1)], -1)
        return torch.softmax(self.time_mlp(inp).squeeze(-1) / self.tau, -1)  # (Cp, omega)

    def forward(self, X):  # X (G,T,Cp,D)
        G, T, Cp, D = X.shape
        Wc = torch.softmax(self.psi / self.tau, -1)  # (Cp target, Cp source)
        mixed = torch.einsum("cs,gtsd->gtcd", Wc, X)
        k = self.kernels()  # (Cp, omega), lag 0 first
        xp = F.pad(mixed.permute(0, 2, 3, 1).reshape(G, Cp * D, T), (self.omega - 1, 0))
        ker = k.flip(-1).repeat_interleave(D, 0)[:, None, :]  # (Cp*D,1,omega)
        out = F.conv1d(xp, ker, groups=Cp * D).reshape(G, Cp, D, T).permute(0, 3, 1, 2)
        return out


class _NNNNet(nn.Module):
    def __init__(self, C, n_feat, n_geo, D=16, layers=2, omega=13):
        super().__init__()
        self.C, self.Cp, self.D = C, C + 1, D  # media + organic search
        self.lift = nn.Parameter(torch.zeros(self.Cp, D))  # scalar -> first slot, learned extras
        self.attn = nn.ModuleList([_FactoredAttention(self.Cp, omega) for _ in range(layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(D, 4 * D), nn.ReLU(), nn.Linear(4 * D, D))
                                  for _ in range(layers)])
        self.P = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, self.Cp))
        self.scale = nn.Parameter(torch.full((self.Cp,), -1.0))
        self.search_head = nn.Sequential(nn.Linear(self.Cp * D, 64), nn.ReLU(), nn.Linear(64, 1))
        self.base = nn.Linear(n_feat, 1)
        self.geo = nn.Embedding(n_geo, 1)
        nn.init.zeros_(self.geo.weight)
        with torch.no_grad():
            self.base.bias.fill_(0.6)

    def forward(self, x, s, u, geo_id):
        """x (G,T,C) media, s (G,T) organic search, u (G,T,K) controls+calendar."""
        v = torch.cat([x, s[..., None]], -1)  # (G,T,Cp)
        X = torch.zeros(*v.shape, self.D)
        X[..., 0] = v
        X = X + v[..., None] * self.lift  # scalar placed in slot 0 (+ learned extra slots)
        vol = None
        for k, (att, mlp) in enumerate(zip(self.attn, self.mlp)):
            A = att(X)
            if k == 0:
                vol = A.norm(dim=-1)  # carryover-smoothed volume (see class docstring)
            X = X + A
            X = X + mlp(X)
        dirn = X / (X.norm(dim=-1, keepdim=True) + 1e-6)
        return X, vol, dirn

    def sales(self, x, s, u, geo_id):
        X, vol, dirn = self.forward(x, s, u, geo_id)
        G, T, Cp, D = X.shape
        rate = torch.sigmoid(self.P(dirn))  # (G,T,Cp,Cp): head output per channel slot
        rate = torch.diagonal(rate, dim1=-2, dim2=-1)  # P(.; theta^c) for channel c
        contrib = vol * rate * F.softplus(self.scale)  # (G,T,Cp)
        base = self.base(u).squeeze(-1) + self.geo(geo_id)
        return base + contrib.sum(-1), contrib, X

    def search_next(self, X):
        G, T, Cp, D = X.shape
        return self.search_head(X.reshape(G, T, Cp * D)).squeeze(-1)  # predicts s_{t+1}


class NNNLite:
    """Simplified NNN for scalar spend data.

    Kept from the paper: scalar channels lifted into D-dim vectors without normalisation
    layers; factored attention (temporal attention scored by an MLP on the normalised lag and
    the target channel's one-hot code, causal window omega; channel attention through a
    learned matrix initialised at the identity); residual channel-wise MLPs; sales head
    sum_c V_c * P(direction of channel representation) with a sigmoid output; a search head
    that predicts next-week organic search from the media and search representations;
    loss alpha*L_sales + (1-alpha)*L_search + lambda*L1(all parameters); counterfactual
    attribution by zeroing a channel.

    Deviations: no content embeddings (scalar volumes only, D=16 instead of hundreds);
    omega=13 weeks instead of 52 to match the other models; the volume V is the norm of the
    first attention output (carryover-smoothed volume) so that carried-over effects can
    contribute in weeks without spend; a linear baseline on controls, calendar and geo is
    added because NNN has no explicit baseline. `propagate_search=True` attributes the change
    of predicted search to media (one-step version of the paper's autoregressive unrolling);
    with False, observed search is held fixed (the paper's standard inference).
    """

    def __init__(self, propagate_search=False, steps=1500, lr=3e-3, l1=1e-6, alpha=0.5, seed=0):
        self.prop, self.steps, self.lr, self.l1, self.alpha, self.seed = propagate_search, steps, lr, l1, alpha, seed

    def fit(self, d: MMMData, train_T: int):
        torch.manual_seed(self.seed)
        self.d, self.tT, self.n = d, train_T, _Norm(d, train_T)
        self.x, self.s, self.u, self.y = _t(self.n.x), _t(self.n.m), _t(self.n.u), _t(self.n.y)
        self.gid = torch.arange(d.G)
        self.net = _NNNNet(d.C, self.u.shape[-1], d.G)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sl = slice(0, train_T)
        for _ in range(self.steps):
            yhat, _, X = self.net.sales(self.x, self.s, self.u, self.gid)
            s_next = self.net.search_next(X)
            l_sales = F.mse_loss(yhat[:, sl], self.y[:, sl])
            l_search = F.mse_loss(s_next[:, : train_T - 1], self.s[:, 1:train_T])
            l1 = sum(p.abs().sum() for p in self.net.parameters())
            loss = self.alpha * l_sales + (1 - self.alpha) * l_search + self.l1 * l1
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
        self.net.eval()
        with torch.no_grad():
            self.pred = self.net.sales(self.x, self.s, self.u, self.gid)[0].numpy() * self.n.ys
            mode = self.prop
            self.prop = False
            self.roi_std = self._roi()
            self.prop = True
            self.roi_prop = self._roi()
            self.prop = mode
        return self

    def set_mode(self, propagate_search: bool):
        """Switch between standard inference and search propagation without refitting."""
        self.prop = propagate_search
        return self

    @property
    def roi_(self):
        return self.roi_prop if self.prop else self.roi_std

    @torch.no_grad()
    def _sales_cf(self, x):
        """Predicted sales for media x; search held fixed or propagated through the search head."""
        if not self.prop:
            return self.net.sales(x, self.s, self.u, self.gid)[0]
        _, _, X_f = self.net.sales(self.x, self.s, self.u, self.gid)
        _, _, X_c = self.net.sales(x, self.s, self.u, self.gid)
        ds = self.net.search_next(X_c) - self.net.search_next(X_f)  # change of predicted s_{t+1}
        s_cf = self.s.clone()
        s_cf[:, 1:] = s_cf[:, 1:] + ds[:, :-1]
        return self.net.sales(x, s_cf, self.u, self.gid)[0]

    @torch.no_grad()
    def _roi(self):
        y_f = self._sales_cf(self.x)
        out = np.zeros(self.d.C)
        for c in range(self.d.C):
            x0 = self.x.clone()
            x0[..., c] = 0
            out[c] = ((y_f - self._sales_cf(x0)).numpy() * self.n.ys).sum()
        return out / self.d.spend.sum((0, 1))

    def roi(self):
        return self.roi_, self.roi_, self.roi_

    def predict_holdout(self):
        return self.pred

    def media_value(self, mult):
        x = self.x * torch.tensor(np.asarray(mult), dtype=torch.float32)[None, None, :]
        return float(((self._sales_cf(x) - self._sales_cf(torch.zeros_like(x))).numpy() * self.n.ys).sum())


# ===========================================================================
# 3. CausalMMM (simplified)
# ===========================================================================
class _CausalMMMNet(nn.Module):
    """Nodes: C media (exogenous), organic search, sales. Edges i->j into the two endogenous
    nodes are sampled from an amortised encoder (per geo) with Gumbel-softmax."""

    def __init__(self, C, T_enc, n_feat, n_geo, H=32, tau=0.5):
        super().__init__()
        self.C, self.N, self.tau = C, C + 2, tau
        N = self.N
        self.f_emb = nn.Sequential(nn.Linear(T_enc, H), nn.ReLU(), nn.Linear(H, H))
        self.f_edge1 = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Linear(H, H))
        self.f_vertex = nn.Sequential(nn.Linear(H, H), nn.ReLU())
        self.f_edge2 = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Linear(H, 2))
        # allowed edges: any node -> search (except search), any node -> sales (except sales)
        mask = torch.zeros(N, N)
        mask[:, C] = 1
        mask[:, C + 1] = 1
        mask[C, C] = 0
        mask[C + 1, C + 1] = 0
        self.register_buffer("mask", mask)
        self.msg = nn.Linear(1, 8)  # f~_edge on the source value
        self.node_emb = nn.Parameter(0.1 * torch.randn(N, 8))
        self.gru = nn.GRU(8, 16, batch_first=True)
        self.f_pre = nn.ModuleList([nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1)) for _ in range(2)])
        self.f_alpha = nn.Linear(n_feat, 1)
        self.f_gamma = nn.Linear(n_feat, 1)
        self.amp = nn.Parameter(torch.tensor(0.0))
        self.base_y = nn.Linear(n_feat, 1)
        self.base_s = nn.Linear(n_feat, 1)
        self.geo_y = nn.Embedding(n_geo, 1)
        self.geo_s = nn.Embedding(n_geo, 1)
        for e in (self.geo_y, self.geo_s):
            nn.init.zeros_(e.weight)
        with torch.no_grad():
            self.base_y.bias.fill_(0.6)
            self.base_s.bias.fill_(1.0)

    def edge_logits(self, series):  # series (G,N,T_enc)
        h1 = self.f_emb(series)
        G, N, H = h1.shape
        pair = torch.cat([h1[:, :, None, :].expand(G, N, N, H), h1[:, None, :, :].expand(G, N, N, H)], -1)
        e1 = self.f_edge1(pair)  # (G,N,N,H)  i->j
        h2 = self.f_vertex((e1 * self.mask[None, :, :, None]).sum(1))  # aggregate incoming
        pair2 = torch.cat([h2[:, :, None, :].expand(G, N, N, H), h2[:, None, :, :].expand(G, N, N, H)], -1)
        return self.f_edge2(pair2)  # (G,N,N,2) logits for (off, on)

    def sample_z(self, logits, hard=False, train=True):
        if train:
            z = F.gumbel_softmax(logits, tau=self.tau, hard=hard)[..., 1]
        else:
            z = torch.softmax(logits, -1)[..., 1]
        return z * self.mask

    def decode(self, x, s, u, gid, z):
        """x (G,T,C), s (G,T) search input, returns (sales_hat, search_hat)."""
        G, T, C = x.shape
        vals = torch.cat([x, s[..., None], torch.zeros(G, T, 1)], -1)  # sales node not used as a source value
        m = torch.tanh(self.msg(vals[..., None])) + self.node_emb  # (G,T,N,8)
        outs = []
        for k, j in enumerate([C, C + 1]):
            msg = (z[:, :, j][:, None, :, None] * m).sum(2)  # (G,T,8) gated by edges into j
            h, _ = self.gru(msg)  # temporal (carryover) module
            outs.append(self.f_pre[k](h).squeeze(-1))
        p_s, p_y = outs
        search_hat = self.base_s(u).squeeze(-1) + self.geo_s(gid) + p_s
        p = F.softplus(p_y) + 1e-4
        a = 0.5 + 2.5 * torch.sigmoid(self.f_alpha(u).squeeze(-1))
        g = F.softplus(self.f_gamma(u).squeeze(-1)) + 1e-3
        resp = p**a / (p**a + g**a)  # context-conditioned saturation of the target
        sales_hat = self.base_y(u).squeeze(-1) + self.geo_y(gid) + F.softplus(self.amp) * resp
        return sales_hat, search_hat


class CausalMMMLite:
    """Simplified CausalMMM.

    Kept from the paper: an amortised structure encoder (node embeddings of each geo's
    series, two rounds of edge/vertex MLP message passing, Gumbel-softmax edge sampling with
    a KL penalty to a Bernoulli prior); a decoder whose node messages are gated by the sampled
    edges and aggregated by a recurrent network (carryover); a saturation module on the target
    whose shape and inflection depend on the context variables; Gaussian (MSE) likelihood.

    Deviations: only the two endogenous nodes (organic search and sales) are predicted, and
    media are treated as exogenous; the target node does not receive its own lag, so that
    effects must flow from media; the target is predicted contemporaneously; a linear
    baseline on controls/calendar/geo is added. The paper defines no ROI; we use a
    counterfactual that zeroes a channel and propagates the change of predicted search to
    sales, so that edges media -> search -> sales carry indirect effects.
    """

    def __init__(self, steps=800, lr=3e-3, kl=1.0, prior=0.2, seed=0):
        self.steps, self.lr, self.kl, self.prior, self.seed = steps, lr, kl, prior, seed

    def fit(self, d: MMMData, train_T: int):
        torch.manual_seed(self.seed)
        self.d, self.tT, self.n = d, train_T, _Norm(d, train_T)
        self.x, self.s, self.u, self.y = _t(self.n.x), _t(self.n.m), _t(self.n.u), _t(self.n.y)
        self.gid = torch.arange(d.G)
        series = torch.cat([self.x, self.s[..., None], self.y[..., None]], -1)[:, :train_T].permute(0, 2, 1)
        self.series = (series - series.mean(-1, keepdim=True)) / (series.std(-1, keepdim=True) + 1e-6)
        self.net = _CausalMMMNet(d.C, train_T, self.u.shape[-1], d.G)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sl = slice(0, train_T)
        p0 = torch.tensor([1 - self.prior, self.prior])
        for _ in range(self.steps):
            logits = self.net.edge_logits(self.series)
            z = self.net.sample_z(logits)
            yh, sh = self.net.decode(self.x, self.s, self.u, self.gid, z)
            q = torch.softmax(logits, -1)
            kl = ((q * (torch.log(q + 1e-8) - torch.log(p0))).sum(-1) * self.net.mask).sum() / self.net.mask.sum() / self.d.G
            loss = F.mse_loss(yh[:, sl], self.y[:, sl]) + F.mse_loss(sh[:, sl], self.s[:, sl]) + self.kl * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
        self.net.eval()
        with torch.no_grad():
            self.z = self.net.sample_z(self.net.edge_logits(self.series), train=False)
            self.pred = self.net.decode(self.x, self.s, self.u, self.gid, self.z)[0].numpy() * self.n.ys
            self.roi_ = self._roi()
        return self

    @torch.no_grad()
    def _sales(self, x):
        _, sh_f = self.net.decode(self.x, self.s, self.u, self.gid, self.z)
        _, sh_c = self.net.decode(x, self.s, self.u, self.gid, self.z)
        s_cf = self.s + (sh_c - sh_f)  # mediator responds to the intervention
        return self.net.decode(x, s_cf, self.u, self.gid, self.z)[0]

    @torch.no_grad()
    def _roi(self):
        y_f = self._sales(self.x)
        out = np.zeros(self.d.C)
        for c in range(self.d.C):
            x0 = self.x.clone()
            x0[..., c] = 0
            out[c] = ((y_f - self._sales(x0)).numpy() * self.n.ys).sum()
        return out / self.d.spend.sum((0, 1))

    def roi(self):
        return self.roi_, self.roi_, self.roi_

    def predict_holdout(self):
        return self.pred

    def media_value(self, mult):
        x = self.x * torch.tensor(np.asarray(mult), dtype=torch.float32)[None, None, :]
        return float(((self._sales(x) - self._sales(torch.zeros_like(x))).numpy() * self.n.ys).sum())

    def edges(self):
        """Mean edge probability into search and sales over geos: (N,) each."""
        z = self.z.mean(0).numpy()
        return z[:, self.d.C], z[:, self.d.C + 1]
