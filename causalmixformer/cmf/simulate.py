"""Geo-level weekly MMM simulator with ground-truth channel ROI.

Three scenarios (matching the S1-S3 design of the benchmark):
  S1  well-specified: geometric adstock + Hill saturation, no mediation.
  S2  carryover misspecification: delayed (Weibull-shaped) carryover whose
      effect peaks 2-4 weeks after spend, which geometric adstock cannot fit.
  S3  mediation: TV (and to a lesser degree Social) raises organic brand-search
      volume, which in turn drives sales. Part of TV's effect is indirect.

Robustness scenarios (added for the revision; they use a separate random stream, so S1-S3
are unchanged):
  S3c confounded search, no mediation: an unobserved demand shock drives both branded search
      and sales, media do not affect search. A mediation model must not credit TV with it.
  S3l lagged mediator: as S3, but search responds to TV/Social with a delayed carryover and a
      different saturation curve than sales (violates the shared-response assumption).
  S3u unmeasured mediator-outcome confounding: as S3 plus a demand shock that drives both
      search and sales (strong violation of assumption A2).
  S4  nonlinear baseline: region-specific seasonal amplitude, holiday spikes of region-specific
      size, a structural break in half of the regions and interactions of promotion and price
      with the season (tests whether the Transformer baseline helps).
  S5  endogenous spend: TV and paid-search budgets follow an unobserved demand shock that also
      drives sales (violates assumption A1; no method in the study corrects for it).
  S6  search as mediator and confounder (the setting of Chen et al., 2018): branded search is
      raised by TV and by an unobserved demand shock; paid-search spend follows the demand shock
      (ads are served when queries occur); demand affects sales only through search. Including
      search as a control biases TV, omitting it biases paid search.

Ground-truth ROI of channel c is the total causal effect
    ROI_c = sum_{g,t} [ y(x) - y(x with channel c set to 0) ] / sum_{g,t} x_c
where the mediator is recomputed under the counterfactual, so ROI_c includes
the indirect path. Noise is additive and cancels in the difference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CHANNELS = ["TV", "Search", "Social", "Display", "Radio"]


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
def geometric_kernel(decay: float, L: int) -> np.ndarray:
    w = decay ** np.arange(L)
    return w / w.sum()


def weibull_kernel(shape: float, scale: float, L: int) -> np.ndarray:
    """Discretised Weibull pdf over lags 0..L-1 (delayed peak when shape>1)."""
    lag = np.arange(L) + 0.5
    w = (shape / scale) * (lag / scale) ** (shape - 1) * np.exp(-((lag / scale) ** shape))
    return w / w.sum()


def apply_carryover(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """x: (G, T). Causal convolution with kernel over lags."""
    G, T = x.shape
    out = np.zeros_like(x)
    for l, w in enumerate(kernel):
        out[:, l:] += w * x[:, : T - l]
    return out


def hill(a: np.ndarray, half_sat: float, slope: float) -> np.ndarray:
    a = np.maximum(a, 0.0)
    return a**slope / (a**slope + half_sat**slope)


def calendar_features(T: int, n_fourier: int = 3) -> np.ndarray:
    t = np.arange(T)
    feats = [t / T]
    for k in range(1, n_fourier + 1):
        feats += [np.sin(2 * np.pi * k * t / 52), np.cos(2 * np.pi * k * t / 52)]
    return np.stack(feats, -1)  # (T, F)



def safe_scale(sp, train_T=None):
    """Per-geo training mean of each channel, floored at 1% of the channel's overall mean.

    Without the floor, a geo that never used a channel during training gets a scale of ~0 and
    any later spend explodes after normalisation (happens on real panels with sparse channels).
    """
    sT = slice(0, train_T)
    m = sp[:, sT].mean(1, keepdims=True)
    glob = sp[:, sT].mean((0, 1), keepdims=True)
    return np.maximum(m, 0.01 * glob) + 1e-8


# --------------------------------------------------------------------------
# config / container
# --------------------------------------------------------------------------
@dataclass
class SimConfig:
    scenario: str = "S1"  # S1 | S2 | S3
    n_geos: int = 30
    n_weeks: int = 156
    lag_window: int = 13
    noise_sd: float = 0.02  # relative to mean baseline
    spend_noise: float = 0.4  # geo-week idiosyncratic spend variation (identifying variation)
    seed: int = 0


@dataclass
class MMMData:
    spend: np.ndarray  # (G, T, C)
    controls: np.ndarray  # (G, T, K)
    mediator: np.ndarray  # (G, T)   observed brand-search index
    sales: np.ndarray  # (G, T)
    true_roi: np.ndarray  # (C,)  total effect ROI
    true_direct_roi: np.ndarray  # (C,)
    true_contrib_share: float
    channels: list = field(default_factory=lambda: list(CHANNELS))
    scenario: str = "S1"
    meta: dict = field(default_factory=dict)

    @property
    def G(self):
        return self.spend.shape[0]

    @property
    def T(self):
        return self.spend.shape[1]

    @property
    def C(self):
        return self.spend.shape[2]


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------
def _spend_paths(rng, G, T, spend_noise=0.25):
    """Correlated, flighted spend in thousands of currency units."""
    C = len(CHANNELS)
    geo_size = rng.lognormal(0.0, 0.35, size=G)  # market size
    base_level = np.array([60.0, 25.0, 20.0, 18.0, 15.0])  # avg weekly spend/channel
    t = np.arange(T)
    season = 1 + 0.25 * np.sin(2 * np.pi * t / 52 - 0.8)  # common budget seasonality
    common = np.exp(np.cumsum(rng.normal(0, 0.05, T)))  # shared budget shocks -> collinearity
    common /= common.mean()
    spend = np.zeros((G, T, C))
    for c in range(C):
        own = np.exp(np.cumsum(rng.normal(0, 0.08, T)))
        own /= own.mean()
        lvl = base_level[c] * season * (0.6 * common + 0.4 * own)
        if CHANNELS[c] in ("TV", "Radio"):  # flighting: on/off bursts
            on = (rng.random(T) < 0.55).astype(float)
            on = np.convolve(on, np.ones(3) / 3, mode="same") > 0.5
            lvl = lvl * on * 1.8
        noise = rng.lognormal(0, spend_noise, size=(G, T))
        spend[:, :, c] = geo_size[:, None] * lvl[None, :] * noise
    return spend, geo_size


def simulate(cfg: SimConfig) -> MMMData:
    rng = np.random.default_rng(cfg.seed)
    G, T, L = cfg.n_geos, cfg.n_weeks, cfg.lag_window
    C = len(CHANNELS)
    spend, geo_size = _spend_paths(rng, G, T, cfg.spend_noise)
    rng2 = np.random.default_rng(cfg.seed + 100_000)  # separate stream for robustness scenarios
    demand = np.zeros((G, T))
    if cfg.scenario in ("S3c", "S3u", "S5", "S6"):
        eps = rng2.normal(0, 1, (G, T))
        for i in range(T):  # AR(1) demand shock with unit stationary variance
            demand[:, i] = (0.7 * demand[:, i - 1] if i else 0) + np.sqrt(1 - 0.7 ** 2) * eps[:, i]
    if cfg.scenario == "S5":  # budgets follow expected demand (TV and paid search most strongly)
        phi = np.array([0.25, 0.5, 0.1, 0.1, 0.1])
        spend = spend * np.exp(phi[None, None, :] * demand[..., None])
    if cfg.scenario == "S6":  # paid search is served in response to queries, which follow demand
        spend[:, :, 1] = spend[:, :, 1] * np.exp(0.5 * demand)

    # ---- controls: price index, promotion, competitor activity
    t = np.arange(T)
    price = 1 + 0.05 * np.sin(2 * np.pi * t / 26)[None] + rng.normal(0, 0.02, (G, T))
    promo = (rng.random((G, T)) < 0.12).astype(float)
    comp = np.clip(rng.normal(0, 1, (G, T)).cumsum(1) * 0.05, -1, 1)
    controls = np.stack([price, promo, comp], -1)

    # ---- baseline (nonlinear: trend x seasonality interaction + control effects)
    base_g = 400 * geo_size
    trend = 1 + 0.15 * t / T + 0.05 * np.sin(2 * np.pi * t / T)
    seas = 1 + 0.12 * np.sin(2 * np.pi * t / 52) + 0.05 * np.cos(4 * np.pi * t / 52)
    baseline = base_g[:, None] * trend[None] * seas[None]
    baseline = baseline * (1 - 0.8 * (price - 1)) * (1 + 0.10 * promo) * (1 - 0.05 * comp)
    if cfg.scenario == "S4":
        woy = t % 52
        amp = rng2.uniform(0.0, 0.35, G)  # region-specific seasonal amplitude
        seas_g = 1 + amp[:, None] * np.sin(2 * np.pi * t / 52 + rng2.uniform(-0.5, 0.5, G)[:, None])
        holiday = np.isin(woy, [46, 47, 50, 51]).astype(float)
        spike = rng2.uniform(0.1, 0.6, G)  # region-specific holiday uplift
        brk = (rng2.random(G) < 0.5)[:, None] * (t[None] >= 100) * 0.15  # structural break
        season_c = np.sin(2 * np.pi * t / 52)[None]
        elastic = 0.8 + 0.6 * season_c  # price elasticity varies with the season
        baseline = (base_g[:, None] * trend[None] * seas_g * (1 + spike[:, None] * holiday[None]) * (1 + brk)
                    * (1 - elastic * (price - 1)) * (1 + 0.10 * promo * (1 + season_c)) * (1 - 0.05 * comp))
        controls = np.concatenate([controls, np.broadcast_to(holiday[None, :, None], (G, T, 1))], -1)

    # ---- media response parameters (per channel)
    #               TV    Search Social Display Radio
    decay = np.array([0.6, 0.15, 0.35, 0.25, 0.45])
    half = np.array([0.9, 0.7, 0.8, 1.0, 1.0])  # half-saturation relative to mean adstock
    slope = np.array([1.6, 1.2, 1.4, 1.1, 1.8])
    # target direct ROI (approx., before saturation curvature) -> scale beta
    roi_target = np.array([1.6, 2.6, 1.9, 0.8, 1.1]) * rng.lognormal(0, 0.15, C)

    kernels = []
    for c in range(C):
        if cfg.scenario == "S2" and CHANNELS[c] in ("TV", "Social", "Radio"):
            kernels.append(weibull_kernel(shape=2.5, scale=[4.0, 3.0, 3.5][["TV", "Social", "Radio"].index(CHANNELS[c])], L=L))
        else:
            kernels.append(geometric_kernel(decay[c], L))

    def media_effects(sp):
        """Return (G,T,C) direct effects and (G,T) adstocked-TV/Social response for mediator."""
        eff = np.zeros_like(sp)
        sat = np.zeros_like(sp)
        for c in range(C):
            a = apply_carryover(sp[:, :, c], kernels[c])
            ref = ref_level[c]
            s = hill(a / ref, half[c], slope[c])
            sat[:, :, c] = s
            eff[:, :, c] = beta[c] * geo_size[:, None] * s
        return eff, sat

    # reference adstock level (from factual spend) so Hill is unit-free
    ref_level = np.array([apply_carryover(spend[:, :, c], kernels[c]).mean() for c in range(C)])
    # calibrate beta so that direct ROI ~ roi_target
    beta = np.ones(C)
    _, sat0 = media_effects(spend)
    for c in range(C):
        beta[c] = roi_target[c] * spend[:, :, c].sum() / (geo_size[:, None] * sat0[:, :, c]).sum()

    # ---- mediator (brand search index): driven by TV (and Social) in S3
    med_base = 50 * geo_size[:, None] * seas[None] * (1 + 0.1 * t / T)[None]
    if cfg.scenario in ("S3", "S3l", "S3u", "S6"):
        med_coef = np.array([60.0, 0.0, 8.0, 0.0, 0.0])  # TV strongly, Social mildly
        med_to_sales = 4.0  # sales per unit of brand-search index
        direct_scale = np.array([0.45, 1.0, 0.7, 1.0, 1.0])  # TV keeps 45% direct
    else:
        med_coef = np.zeros(C)
        med_to_sales = 0.0
        direct_scale = np.ones(C)
    med_noise = rng.normal(0, 0.05, (G, T)) * med_base

    # S3l: search responds with its own (delayed) carryover and a steeper saturation curve
    med_kern = [weibull_kernel(2.5, 4.0, L) if CHANNELS[c] == "TV" else
                (weibull_kernel(2.0, 3.0, L) if CHANNELS[c] == "Social" else kernels[c]) for c in range(C)]
    med_ref = np.array([apply_carryover(spend[:, :, c], med_kern[c]).mean() for c in range(C)])

    def mediator_response(sp, sat):
        if cfg.scenario != "S3l":
            return sat
        out = np.zeros_like(sp)
        for c in range(C):
            if med_coef[c] > 0:
                out[:, :, c] = hill(apply_carryover(sp[:, :, c], med_kern[c]) / med_ref[c], 0.6, 2.5)
        return out

    def outcome(sp):
        eff, sat = media_effects(sp)
        eff = eff * direct_scale[None, None, :]
        med = med_base + (mediator_response(sp, sat) * med_coef[None, None, :]).sum(-1) * geo_size[:, None]
        y = baseline + eff.sum(-1) + med_to_sales * (med - med_base)
        return y, med, eff

    y_det, med_det, eff = outcome(spend)
    sales_noise = rng.normal(0, cfg.noise_sd, (G, T)) * baseline.mean()
    sales = y_det + sales_noise
    mediator = med_det + med_noise
    if cfg.scenario in ("S3", "S3l"):  # brand search also responds to sales-noise shocks (demand) -> realism
        mediator = mediator + 0.02 * sales_noise
    if cfg.scenario == "S6":  # demand raises queries; it reaches sales only through search
        dm = 0.15 * demand * med_base
        mediator = mediator + dm
        sales = sales + med_to_sales * dm
    if cfg.scenario in ("S3c", "S3u", "S5"):  # unobserved demand shock (does not depend on spend)
        sales = sales + 0.06 * demand * baseline
        if cfg.scenario in ("S3c", "S3u"):
            mediator = mediator + 0.15 * demand * med_base

    # ---- ground-truth ROI by counterfactual zeroing
    true_roi = np.zeros(C)
    true_direct = np.zeros(C)
    for c in range(C):
        sp0 = spend.copy()
        sp0[:, :, c] = 0
        y0, _, eff0 = outcome(sp0)
        true_roi[c] = (y_det - y0).sum() / spend[:, :, c].sum()
        true_direct[c] = (eff[:, :, c] - eff0[:, :, c]).sum() / spend[:, :, c].sum()
    y_nomedia, _, _ = outcome(np.zeros_like(spend))
    share = (y_det - y_nomedia).sum() / y_det.sum()

    return MMMData(
        spend=spend,
        controls=controls,
        mediator=mediator,
        sales=sales,
        true_roi=true_roi,
        true_direct_roi=true_direct,
        true_contrib_share=float(share),
        scenario=cfg.scenario,
        meta=dict(kernels=np.array(kernels), seed=cfg.seed,
                  outcome=lambda sp: outcome(sp)[0], y_nomedia=y_nomedia),
    )


if __name__ == "__main__":
    for s in ["S1", "S2", "S3", "S3c", "S3l", "S3u", "S4", "S5", "S6"]:
        d = simulate(SimConfig(scenario=s, seed=1))
        print(s, "ROI", np.round(d.true_roi, 2), "direct", np.round(d.true_direct_roi, 2),
              "media share %.2f" % d.true_contrib_share)
