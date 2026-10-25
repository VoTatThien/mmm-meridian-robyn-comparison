"""Bộ sinh dữ liệu mô phỏng MMM có ground-truth ROI.

Kịch bản:
  S1 – chỉ định đúng: adstock hình học + Hill tĩnh (đúng giả định của Meridian/PyMC/LightweightMMM).
  S2 – lưu dư có độ trễ: TV và Video có adstock Weibull đạt đỉnh sau 2–4 tuần.
  S3 – tác động gián tiếp: TV, Video -> tìm kiếm thương hiệu (biến trung gian) -> doanh thu.
       Biến trung gian được đưa cho mô hình như một biến kiểm soát (thực hành phổ biến),
       nên các mô hình dễ quy nhầm hiệu ứng gián tiếp của TV/Video cho biến này.

ROI thật = (doanh thu kỳ vọng khi có kênh – doanh thu kỳ vọng khi tắt kênh) / chi tiêu của kênh,
tính trên giai đoạn huấn luyện, có kéo dài thêm L tuần để tính đủ hiệu ứng lưu dư.
"""
import json, sys, os
import numpy as np, pandas as pd

CH = ["tv", "video", "social", "display", "search"]
L = 12  # độ trễ tối đa


def geo_w(lam, L=L):
    w = lam ** np.arange(L)
    return w / w.sum()


def weib_w(k, theta, L=L):
    l = np.arange(L) + 0.5
    w = (k / theta) * (l / theta) ** (k - 1) * np.exp(-(l / theta) ** k)
    return w / w.sum()


def adstock(x, w):
    out = np.zeros_like(x, dtype=float)
    for l, wl in enumerate(w):
        out[l:] += wl * x[: len(x) - l] if l > 0 else wl * x
    return out


def hill(x, K, S):
    return x ** S / (x ** S + K ** S)


def simulate(scenario, seed, T=156, test_weeks=16):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    season = np.sin(2 * np.pi * t / 52) + 0.4 * np.cos(4 * np.pi * t / 52)
    common = 0.5 * season + rng.normal(0, 0.3, T)  # nhân tố chung -> đa cộng tuyến vừa phải
    mean_spend = dict(tv=60000, video=30000, social=20000, display=15000, search=25000)
    phi = dict(tv=0.6, video=0.5, social=0.4, display=0.3, search=0.5)
    spend = {}
    for c in CH:
        z = np.zeros(T)
        for i in range(T):
            z[i] = (phi[c] * z[i - 1] if i else 0) + 0.6 * rng.normal()
        s = mean_spend[c] * np.exp(0.5 * z + 0.4 * common - 0.2)
        if c == "tv":  # TV phát theo đợt: một số tuần không chi
            s = s * (rng.random(T) > 0.25)
        spend[c] = s
    promo = (rng.random(T) < 0.12).astype(float)

    # tham số thật
    lam = dict(tv=0.6, video=0.45, social=0.3, display=0.2, search=0.1)
    W = {c: geo_w(lam[c]) for c in CH}
    if scenario == "S2":
        W["tv"] = weib_w(2.5, 4.0)
        W["video"] = weib_w(2.0, 3.0)
    K = dict(tv=0.9, video=0.8, social=0.7, display=0.6, search=0.8)
    S = dict(tv=1.8, video=1.5, social=1.2, display=1.0, search=1.3)
    # hiệu ứng cực đại (đơn vị doanh thu / tuần) -> ROI thật khoảng 0,5–4
    beta = dict(tv=180000, video=90000, social=45000, display=18000, search=120000)

    base = 1_000_000 + 1500 * t
    seas_y = 120000 * season
    mediator_used = scenario == "S3"
    a_m = dict(tv=60000, video=30000)  # tác động lên lượng tìm kiếm thương hiệu
    beta_m = 2.0                       # doanh thu / 1 lượt tìm kiếm

    def mean_outcome(sp):
        contrib = {}
        m = 50000 + 8000 * season
        for c in CH:
            xa = adstock(sp[c] / mean_spend[c], W[c])
            h = hill(xa, K[c], S[c])
            if mediator_used and c in a_m:
                contrib[c] = 0.5 * beta[c] * h  # nửa hiệu ứng trực tiếp
                m = m + a_m[c] * h               # nửa còn lại đi qua trung gian
            else:
                contrib[c] = beta[c] * h
        y = base + seas_y + 90000 * promo + sum(contrib.values())
        if mediator_used:
            y = y + beta_m * m
        return y, m, contrib

    y_mean, m_mean, contrib = mean_outcome(spend)
    noise_y = rng.normal(0, 0.03 * y_mean.mean(), T)
    noise_m = rng.normal(0, 2000, T)
    y = y_mean + noise_y
    m_obs = m_mean + noise_m

    # ROI thật trên giai đoạn huấn luyện (kéo dài thêm L tuần cho lưu dư)
    T_tr = T - test_weeks
    roi = {}
    for c in CH:
        sp0 = {k: v.copy() for k, v in spend.items()}
        sp0[c][:T_tr] = 0.0           # tắt kênh trong giai đoạn huấn luyện
        y0, _, _ = mean_outcome(sp0)
        win = slice(0, min(T, T_tr + L))
        roi[c] = float((y_mean[win] - y0[win]).sum() / spend[c][:T_tr].sum())

    df = pd.DataFrame({"date": pd.date_range("2022-01-03", periods=T, freq="W-MON").strftime("%Y-%m-%d"), "revenue": y})
    for c in CH:
        df[f"{c}_spend"] = spend[c]
    df["promo"] = promo
    if mediator_used:
        df["brand_search"] = m_obs
    meta = dict(scenario=scenario, seed=seed, T=T, test_weeks=test_weeks, channels=CH,
                controls=["promo"] + (["brand_search"] if mediator_used else []),
                true_roi=roi, adstock_weights={c: W[c].round(4).tolist() for c in CH},
                hill_K=K, hill_S=S, beta=beta)
    return df, meta


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/sim"
    os.makedirs(out, exist_ok=True)
    for sc in ["S1", "S2", "S3"]:
        for seed in [1, 2, 3]:
            df, meta = simulate(sc, seed)
            name = f"{sc}_seed{seed}"
            df.to_csv(f"{out}/{name}.csv", index=False)
            json.dump(meta, open(f"{out}/{name}.json", "w"), indent=1)
            print(name, {k: round(v, 2) for k, v in meta["true_roi"].items()},
                  "y_mean=%.0f" % df.revenue.mean(), "media_share=%.1f%%" % (100 * sum(meta["true_roi"][c] * df[f'{c}_spend'][:-16].sum() for c in meta['channels']) / df.revenue[:-16].sum()))
