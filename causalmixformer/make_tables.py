"""Aggregate experiment results into LaTeX tables and figures for the paper.

    python make_tables.py      # writes paper/tables/*.tex, paper/figures/*.pdf, prints summaries

Inputs:  results_final/*.jsonl (final evaluation on fresh seeds 100-104),
         results/realdata.jsonl (190-DMA data), bench/results/*/ (national benchmark).
Development runs (seeds 0-4) are kept in results/raw_*.jsonl and are NOT used in the paper.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT_T = "paper/tables"
OUT_F = "paper/figures"
SCEN = ["S1", "S2", "S3"]
MAIN = ["Ridge", "Ridge+M", "XGB-SHAP", "NNN", "NNN-AR", "CausalMMM", "CMF"]
ABL = ["CMF", "CMF-noCT1", "CMF-noCT2", "CMF-noCT3", "CMF-linbase"]
LABEL = {
    "Ridge": "Ridge MMM (no mediator)",
    "Ridge+M": "Ridge MMM (mediator as control)",
    "XGB-SHAP": "XGBoost + SHAP$^\\dagger$",
    "NNN": "NNN$^\\dagger$ (standard inference)",
    "NNN-AR": "NNN$^\\dagger$ (search propagated)",
    "CausalMMM": "CausalMMM$^\\dagger$",
    "CMF": "\\textbf{CausalMixFormer}",
    "CMF-noCT1": "\\quad w/o mediation (CT1)",
    "CMF-noCT2": "\\quad w/o constraints, priors (CT2)",
    "CMF-noCT3": "\\quad w/o lag attention (CT3)",
    "CMF-linbase": "\\quad linear baseline",
}
COLS = [("roi_mape", "min", 2), ("spearman", "max", 2), ("tv_bias_pct", "absmin", 0),
        ("r2_holdout", None, 3), ("uplift_captured", "max", 2)]


EVAL_SEEDS = [100, 101, 102, 103, 104]  # never inspected during development


def _read(paths):
    rows = []
    for f in paths:
        if os.path.exists(f):
            rows += [json.loads(line) for line in open(f)]
    return pd.DataFrame(rows)


def load_sim():
    df = _read(["results_final/main.jsonl", "results_final/ablation.jsonl"])
    df = df[df.seed.isin(EVAL_SEEDS)]
    return df.drop_duplicates(subset=["scenario", "seed", "method"], keep="first")


def load_robust():
    df = _read(["results_final/robust.jsonl"])
    if df.empty:
        return df
    df = df[df.method != "NoMedia"]
    return df.drop_duplicates(subset=["scenario", "seed", "method"], keep="first")


BENCH_ORDER = ["Robyn", "Meridian", "PyMC-Marketing", "LightweightMMM", "Orbit KTR", "DeepCausalMMM",
               "DeepCausalMMM (NOTEARS)", "CausalMixFormer", "CausalMixFormer (first member, post hoc)",
               "CausalMixFormer w/o mediation", "CausalMixFormer (linear baseline)"]


def load_bench():
    rows = []
    for f in glob.glob("bench/results/*/S*_seed*.json"):
        r = json.load(open(f))
        tr = r["true_roi"]
        rr = dict(model=r["model"], scenario=r["scenario"], seed=r["seed"], roi_mape=r["roi_mape"],
                  test_r2=r["test"]["r2"], test_mape=r["test"]["mape"],
                  tv_bias=100 * (r["roi"]["tv"] - tr["tv"]) / tr["tv"],
                  video_bias=100 * (r["roi"]["video"] - tr["video"]) / tr["video"])
        from scipy.stats import spearmanr
        cs = list(tr)
        rr["rank"] = float(spearmanr([r["roi"][c] for c in cs], [tr[c] for c in cs])[0])
        rows.append(rr)
        if r["model"] == "CausalMixFormer" and "roi_members" in r:  # post-hoc: first member as point estimate
            m0 = r["roi_members"][0]
            rows.append(dict(rr, model="CausalMixFormer (first member, post hoc)",
                             roi_mape=float(np.mean([abs(m0[c] - tr[c]) / tr[c] for c in cs]) * 100),
                             tv_bias=100 * (m0["tv"] - tr["tv"]) / tr["tv"],
                             video_bias=100 * (m0["video"] - tr["video"]) / tr["video"],
                             rank=float(spearmanr([m0[c] for c in cs], [tr[c] for c in cs])[0])))
    return pd.DataFrame(rows)


def bench_table(b):
    lines = []
    for sc in SCEN:
        sub = b[b.scenario == sc]
        lines.append(f"\\midrule\n\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{Scenario {sc}}} (3 datasets)}}\\\\")
        st = {m: sub[sub.model == m] for m in BENCH_ORDER if (sub.model == m).any()}
        if sc != "S3":
            st.pop("CausalMixFormer w/o mediation", None)
        post = st.get("CausalMixFormer (first member, post hoc)")
        mean = lambda m, c: st[m][c].mean()
        cand = [m for m in st if "post hoc" not in m]  # the post-hoc row is not eligible for bold
        best_mape = min(cand, key=lambda m: mean(m, "roi_mape"))
        best_rank = max(cand, key=lambda m: mean(m, "rank"))
        best_tv = min(cand, key=lambda m: abs(mean(m, "tv_bias")))
        best_r2 = max(cand, key=lambda m: mean(m, "test_r2"))
        for m, r in st.items():
            sd = lambda c: r[c].std(ddof=1)
            name = "\\textbf{CausalMixFormer}" if m == "CausalMixFormer" else (
                "\\quad " + m.replace("CausalMixFormer ", "").strip("()") if m.startswith("CausalMixFormer") else m)
            B = lambda txt, cond: f"\\textbf{{{txt}}}" if cond else txt
            lines.append(" & ".join([
                name,
                B(f"{r.roi_mape.mean():.1f} $\\pm$ {sd('roi_mape'):.1f}", m == best_mape),
                B(f"{r['rank'].mean():.2f}", m == best_rank),
                B(f"{r.tv_bias.mean():+.0f}", m == best_tv),
                B(f"{r.test_r2.mean():.2f}", m == best_r2),
                f"{r.test_mape.mean():.2f}"]) + " \\\\")
    return (r"""\begin{table}[t]
\caption{Comparison with the open-source frameworks of the preliminary benchmark on its nine national-level datasets (three per scenario, 140 training and 16 test weeks). ROI MAPE in \%, $\rho$: Spearman correlation of estimated and true ROI, TV bias: relative error of the TV ROI in \%, test $R^2$ and test MAPE (\%) on the 16 test weeks. All methods are scored with the benchmark's code; the best value per scenario is in bold. The row ``first member (post hoc)'' uses the first ensemble member as point estimate and was added after the evaluation results were seen; it is not eligible for bold.}
\label{tab:bench}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
Method & ROI MAPE (\%) $\downarrow$ & $\rho$ $\uparrow$ & TV bias (\%) & Test $R^2$ $\uparrow$ & Test MAPE (\%) $\downarrow$ \\
""" + "\n".join(lines) + r"""
\botrule
\end{tabular}
\end{table}
""")


def bench_tv_figure(b):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = b[b.scenario == "S3"]
    order = [m for m in BENCH_ORDER if (sub.model == m).any() and m not in ("CausalMixFormer (linear baseline)",
             "CausalMixFormer (first member, post hoc)")]
    order = sorted(order, key=lambda m: sub[sub.model == m].tv_bias.mean())
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(order))
    for k, m in enumerate(order):
        v = sub[sub.model == m].tv_bias
        col = "#2f5d8a" if m == "CausalMixFormer" else ("#7c9fc9" if m.startswith("CausalMixFormer") else "#b9bec5")
        ax.barh(k, v.mean(), color=col, height=0.62)
        ax.plot(v.values, [k] * len(v), "o", color="#222222", ms=2.5)
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_yticks(y, order, fontsize=7.5)
    ax.set_xlabel("Relative error of the TV ROI in scenario S3 (%)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT_F}/bench_tv_s3.pdf")
    fig.savefig(f"{OUT_F}/bench_tv_s3.png", dpi=200)


def cell(m, s, digits, pct=False, bold=False):
    if np.isnan(m):
        return "--"
    if pct:
        txt = f"{(0 if round(m) == 0 else m):+.0f} $\\pm$ {s:.0f}".replace("+0 ", "0 ")
    else:
        txt = f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def table(df, methods, label, caption, seeds=None, nomedia=True):
    d = df[df.method.isin(methods)]
    if seeds is not None:
        d = d[d.seed.isin(seeds)]
    lines = []
    for sc in SCEN:
        sub = d[d.scenario == sc]
        if sub.empty:
            continue
        stats = {}
        for m in methods:
            r = sub[sub.method == m]
            if r.empty:
                continue
            stats[m] = {c: (r[c].astype(float).mean(), r[c].astype(float).std(ddof=1) if len(r) > 1 else 0.0)
                        for c, _, _ in COLS}
            stats[m]["n"] = len(r)
        ns = sorted({v["n"] for v in stats.values()})
        ntxt = f"{ns[0]} seeds" if len(ns) == 1 else f"{ns[0]}--{ns[-1]} seeds"
        lines.append(f"\\midrule\n\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{Scenario {sc}}} ({ntxt})}}\\\\")
        best = {}
        for c, rule, _ in COLS:
            if rule is None:
                continue
            vals = {m: v[c][0] for m, v in stats.items()}
            key = {"min": lambda m: vals[m], "max": lambda m: -vals[m], "absmin": lambda m: abs(vals[m])}[rule]
            best[c] = min(vals, key=key)
        for m, v in stats.items():
            cells = [LABEL[m]] + [cell(*v[c], dg, pct=(c == "tv_bias_pct"), bold=(best.get(c) == m))
                                  for c, _, dg in COLS]
            lines.append(" & ".join(cells) + " \\\\")
        if nomedia:
            nm = df[(df.scenario == sc) & (df.method == "NoMedia")]
            if seeds is not None:
                nm = nm[nm.seed.isin(seeds)]
            if not nm.empty:
                lines.append("No-media forecaster & -- & -- & -- & "
                             f"{cell(nm.r2_holdout.mean(), nm.r2_holdout.std(ddof=1) if len(nm) > 1 else 0, 3)} & -- \\\\")
    tex = (r"""\begin{table}[t]
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\centering
\footnotesize
\setlength{\tabcolsep}{1.5pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
Method & ROI MAPE $\downarrow$ & $\rho$ $\uparrow$ & TV bias (\%) & Holdout $R^2$ & Uplift $\uparrow$ \\
""" + "\n".join(lines) + r"""
\botrule
\end{tabular}
\end{table}
""")
    return tex


def coverage_table(df, seeds):
    d = df[df.method.isin(ABL) & df.seed.isin(seeds)]
    lines = []
    for m in ABL:
        r = d[d.method == m]
        if r.empty:
            continue
        cells = [LABEL[m]]
        for sc in SCEN:
            rr = r[r.scenario == sc]
            cells.append("--" if rr.empty else f"{rr.coverage.mean():.2f} / {rr.interval_width.mean():.2f}")
        lines.append(" & ".join(cells) + " \\\\")
    return (r"""\begin{table}[t]
\caption{Empirical coverage of the nominal 90\% ROI intervals and mean relative interval width (coverage / width), averaged over channels and over the seeds of Table~\ref{tab:ablation}.}
\label{tab:coverage}
\centering
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
Method & S1 & S2 & S3 \\
\midrule
""" + "\n".join(lines) + r"""
\botrule
\end{tabular}
\end{table}
""")


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (dict name -> p)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def wilcoxon_table(df):
    """CausalMixFormer vs each method, paired over all (scenario, seed) runs; Holm-adjusted p-values
    within each metric family."""
    cmf = df[df.method == "CMF"].set_index(["scenario", "seed"])
    lines, out = [], {}
    for m in MAIN[:-1]:
        o = df[df.method == m].set_index(["scenario", "seed"])
        idx = cmf.index.intersection(o.index)
        if len(idx) < 5:
            continue
        a, b = cmf.loc[idx, "roi_mape"].values, o.loc[idx, "roi_mape"].values
        ta, tb = np.abs(cmf.loc[idx, "tv_bias_pct"].values), np.abs(o.loc[idx, "tv_bias_pct"].values)
        ua, ub = cmf.loc[idx, "uplift_captured"].values, o.loc[idx, "uplift_captured"].values
        p1 = wilcoxon(a, b).pvalue
        p2 = wilcoxon(ta, tb).pvalue
        p3 = wilcoxon(ua, ub).pvalue
        out[m] = dict(n=len(idx), win_mape=float(np.mean(a < b)), p_mape=p1, win_tv=float(np.mean(ta < tb)), p_tv=p2,
                      win_up=float(np.mean(ua > ub)), p_up=p3)
    for key in ["p_mape", "p_tv", "p_up"]:
        adj = holm({m: v[key] for m, v in out.items()})
        for m in out:
            out[m][key + "_holm"] = adj[m]
    f = lambda p: "$<$0.001" if p < 1e-3 else f"{p:.3f}"
    for m, v in out.items():
        lines.append(f"{LABEL[m]} & {v['n']} & {100*v['win_mape']:.0f}\\% ({f(v['p_mape_holm'])}) & "
                     f"{100*v['win_tv']:.0f}\\% ({f(v['p_tv_holm'])}) & {100*v['win_up']:.0f}\\% ({f(v['p_up_holm'])}) \\\\")
    tex = (r"""\begin{table}[t]
\caption{Paired comparison of CausalMixFormer with each method over all simulated runs (scenarios $\times$ seeds). Each cell gives the share of runs in which CausalMixFormer is better and, in parentheses, the two-sided $p$-value of the Wilcoxon signed-rank test, adjusted with the Holm procedure over the six comparisons of each column.}
\label{tab:wilcoxon}
\centering
\footnotesize
\begin{tabular}{@{}lcccc@{}}
\toprule
Compared method & Runs & Lower ROI MAPE & Lower $|$TV bias$|$ & Higher uplift \\
\midrule
""" + "\n".join(lines) + r"""
\botrule
\end{tabular}
\end{table}
""")
    return tex, out


def realdata_table():
    path = "results/realdata.jsonl"
    if not os.path.exists(path):
        return None, None
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        r.setdefault("cutoff", 101)
    df = pd.DataFrame(rows)
    cut = sorted(df.cutoff.unique())
    names = {"Naive": "Naive (last 8 weeks, flat)", "NoMedia": "No-media forecaster", "Ridge": "Ridge MMM",
             "CMF-linbase": "CausalMixFormer, linear baseline", "CMF": "\\textbf{CausalMixFormer}"}
    lines, summ = [], {}
    for m in ["Naive", "NoMedia", "Ridge", "CMF-linbase", "CMF"]:
        r = df[df.method == m]
        if r.empty:
            continue
        s = {}
        s["r2w_train"] = r.groupby("cutoff").r2_within_train.mean().mean()
        s["wape"] = {int(c): float(r[r.cutoff == c].wape_hold.mean()) for c in cut}
        s["r2p"] = r.groupby("cutoff").r2_pooled_hold.mean().mean()
        if "share" in r and r.share.notna().any():
            sh = r.dropna(subset=["share"])
            s["media"] = sh.media_share.mean()
            sds = [np.stack(g.share.values).std(0, ddof=1).mean() for _, g in sh.groupby("cutoff") if len(g) > 1]
            s["stab"] = float(np.mean(sds)) if sds else np.nan
        else:
            s["media"] = s["stab"] = np.nan
        summ[m] = s
        wcells = " & ".join(f"{100*s['wape'][int(c)]:.1f}" for c in cut)
        mtxt = "--" if np.isnan(s["media"]) else f"{100*s['media']:.0f}\\%"
        stxt = "--" if np.isnan(s["stab"]) else f"{100*s['stab']:.2f}"
        r2t = "--" if m == "Naive" else f"{s['r2w_train']:.2f}"
        lines.append(f"{names[m]} & {r2t} & {wcells} & {s['r2p']:.3f} & {mtxt} & {stxt} \\\\")
    head_w = " & ".join(f"wk {c}--{c+7}" for c in cut)
    tex = (r"""\begin{table}[t]
\caption{Results on the public 190-region dataset (184 regions with complete series, 109 weeks, 13 channels). Within-geo $R^2$ is computed on the training weeks after removing each region's mean. Holdout WAPE (\%) is reported for three rolling origins of 8 weeks each, pooled $R^2$ is averaged over origins. Media share is the estimated share of visits attributed to media; attribution spread is the standard deviation of channel shares across seeds (percentage points, averaged over channels). Averages over three seeds.}
\label{tab:realdata}
\centering
\footnotesize
\setlength{\tabcolsep}{1.5pt}
\begin{tabular}{@{}l""" + "c" * (len(cut) + 5) + r"""@{}}
\toprule
 & Within-geo & \multicolumn{""" + str(len(cut)) + r"""}{c}{Holdout WAPE (\%) $\downarrow$} & Pooled & Media & Attribution \\
\cmidrule(lr){3-""" + str(2 + len(cut)) + r"""}
Method & $R^2$ (train) & """ + head_w + r""" & $R^2$ & share & spread $\downarrow$ \\
\midrule
""" + "\n".join(lines) + r"""
\botrule
\end{tabular}
\end{table}
""")
    return tex, summ


ROBUST_ORDER = [("S6", "S6: search mediates TV and confounds paid search"),
                ("S3u", "S3u: mediator--sales confounding (A2 violated)"),
                ("S3l", "S3l: mediator with its own carryover and saturation"),
                ("S3c", "S3c: search confounded by demand, no mediation"),
                ("S4", "S4: nonlinear baseline"),
                ("S5", "S5: endogenous spend (A1 violated)")]
ROBUST_METHODS = ["Ridge", "Ridge+M", "XGB-SHAP", "NNN", "NNN-AR", "CMF", "CMF-noCT1", "CMF-linbase"]


def robust_table(r):
    lines = []
    for sc, title in ROBUST_ORDER:
        sub = r[r.scenario == sc]
        if sub.empty:
            continue
        n = sub.seed.nunique()
        lines.append(f"\\midrule\n\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{title}}} ({n} seeds)}}\\\\")
        for m in ROBUST_METHODS:
            x = sub[sub.method == m]
            if x.empty:
                continue
            ps = [100 * (a[1] - t[1]) / t[1] for a, t in zip(x.roi, x.true_roi)]
            lines.append(" & ".join([
                LABEL[m],
                f"{x.roi_mape.mean():.2f}",
                f"{x.tv_bias_pct.mean():+.0f}",
                f"{np.mean(ps):+.0f}",
                f"{x.uplift_captured.mean():.2f}"]) + " \\\\")
    return (r"""\begin{table*}[t]
\caption{Robustness scenarios (seeds 100--102). TV and paid-search bias are relative errors of the corresponding ROI in \%; uplift as in Table~\ref{tab:main}. $^\dagger$Re-implemented.}
\label{tab:robust}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & ROI MAPE $\downarrow$ & TV bias (\%) & Paid-search bias (\%) & Uplift $\uparrow$ \\
""" + "\n".join(lines) + r"""
\botrule
\end{tabular}
\end{table*}
""")


def decomposition_figure(df, r):
    """TV ROI split into direct and indirect parts (S3, S6) and paid-search ROI in S6."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("S3", df, 0, "TV, scenario S3"), ("S6", r, 0, "TV, scenario S6"), ("S6", r, 1, "Paid search, scenario S6")]
    meths = [("Ridge", "Ridge"), ("Ridge+M", "Ridge+M"), ("NNN", "NNN"), ("CMF", "CMF")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=False)
    for ax, (sc, src, ch, title) in zip(axes, panels):
        sub = src[src.scenario == sc] if not src.empty else src
        if sub.empty:
            ax.set_visible(False)
            continue
        cm = sub[sub.method == "CMF"]
        tr_tot = np.mean([t[ch] for t in cm.true_roi])
        tr_dir = np.mean([t[ch] for t in cm.true_direct_roi]) if "true_direct_roi" in cm else tr_tot
        x0 = 0
        ax.bar(x0, tr_dir, 0.6, color="#222222")
        ax.bar(x0, tr_tot - tr_dir, 0.6, bottom=tr_dir, color="#888888")
        labels = ["True"]
        for k, (m, lab) in enumerate(meths, start=1):
            x = sub[sub.method == m]
            if x.empty:
                continue
            tot = np.mean([a[ch] for a in x.roi])
            if m == "CMF" and "roi_direct" in x and x.roi_direct.notna().all():
                dr = np.mean([a[ch] for a in x.roi_direct])
                ax.bar(k, dr, 0.6, color="#2f5d8a")
                ax.bar(k, tot - dr, 0.6, bottom=dr, color="#9cb9d9")
            else:
                ax.bar(k, tot, 0.6, color={"Ridge": "#9aa5b1", "Ridge+M": "#d18f52", "NNN": "#7c9fc9"}[m])
            labels.append(lab)
        ax.set_xticks(range(len(labels)), labels, fontsize=6.5)
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("ROI", fontsize=8)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#222222", label="true direct"), Patch(color="#888888", label="true indirect"),
                        Patch(color="#2f5d8a", label="CausalMixFormer direct"),
                        Patch(color="#9cb9d9", label="CausalMixFormer indirect")],
               loc="lower center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{OUT_F}/decomposition.pdf")
    fig.savefig(f"{OUT_F}/decomposition.png", dpi=200)


def roi_figure(df):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meths = ["Ridge", "Ridge+M", "XGB-SHAP", "NNN", "CMF"]
    d = df[(df.scenario == "S3") & df.method.isin(meths)]
    if d.empty:
        return
    meths = [m for m in meths if m in set(d.method)]
    ch = d.iloc[0]["channels"]
    true = np.mean(np.stack(d[d.method == "CMF"].true_roi.values), 0)
    names = {"Ridge": "Ridge (no mediator)", "Ridge+M": "Ridge (mediator as control)", "XGB-SHAP": "XGBoost+SHAP",
             "NNN": "NNN", "CMF": "CausalMixFormer"}
    colors = {"Ridge": "#9aa5b1", "Ridge+M": "#d18f52", "XGB-SHAP": "#c9b458", "NNN": "#7c9fc9", "CMF": "#2f5d8a"}
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    w = 0.84 / (len(meths) + 1)
    x = np.arange(len(ch))
    ax.bar(x - 0.42 + w / 2, true, w, color="#222222", label="True ROI")
    for k, m in enumerate(meths):
        est = np.stack(d[d.method == m].roi.values)
        ax.bar(x - 0.42 + w * (k + 1.5), est.mean(0), w, yerr=est.std(0, ddof=1) if len(est) > 1 else None,
               color=colors[m], label=names[m], capsize=1.5, error_kw=dict(lw=0.5))
    ax.axhline(0, color="#444444", lw=0.6)
    ax.set_xticks(x, ch)
    ax.set_ylabel("ROI")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.8, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(f"{OUT_F}/roi_s3.pdf")
    fig.savefig(f"{OUT_F}/roi_s3.png", dpi=200)


def fit_width(path):
    """Shrink a generated table to the text width if it is wider (needs \\usepackage{adjustbox})."""
    t = open(path).read()
    if "adjustbox" not in t:
        t = t.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}", 1)
        t = t.replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}", 1)
        open(path, "w").write(t)


if __name__ == "__main__":
    os.makedirs(OUT_T, exist_ok=True)
    os.makedirs(OUT_F, exist_ok=True)
    df = load_sim()
    open(f"{OUT_T}/main_results.tex", "w").write(table(
        df, MAIN, "tab:main",
        "Results on the simulated scenarios (mean $\\pm$ standard deviation over seeds). ROI MAPE: mean absolute "
        "percentage error of channel ROI. $\\rho$: Spearman correlation between estimated and true ROI. TV bias: "
        "relative error of the TV ROI. Holdout $R^2$: last 13 weeks. Uplift: share of the oracle budget-reallocation "
        "gain achieved by the model's allocation (1 is optimal; negative values lose sales). The best value per "
        "scenario is in bold. $^\\dagger$Re-implemented from the paper's description (see text)."))
    abl_seeds = sorted(set.intersection(*[set(df[df.method == m].seed) for m in ABL]))
    open(f"{OUT_T}/ablation.tex", "w").write(table(
        df, ABL, "tab:ablation",
        f"Ablation study on the simulated scenarios ({len(abl_seeds)} seeds per scenario; mean $\\pm$ standard "
        "deviation). Each row removes one component of CausalMixFormer. Metrics as in Table~\\ref{tab:main}.",
        seeds=abl_seeds, nomedia=False))
    open(f"{OUT_T}/coverage.tex", "w").write(coverage_table(df, abl_seeds))
    wtex, wsum = wilcoxon_table(df)
    open(f"{OUT_T}/wilcoxon.tex", "w").write(wtex)
    rtex, rsum = realdata_table()
    if rtex:
        open(f"{OUT_T}/realdata.tex", "w").write(rtex)
    roi_figure(df)
    rob = load_robust()
    if not rob.empty:
        open(f"{OUT_T}/robust.tex", "w").write(robust_table(rob))
        cols_r = ["roi_mape", "tv_bias_pct", "uplift_captured"]
        rr = rob.copy()
        rr["ps_bias"] = [100 * (a[1] - t[1]) / t[1] for a, t in zip(rr.roi, rr.true_roi)]
        print(rr.groupby(["scenario", "method"])[cols_r + ["ps_bias", "r2_holdout"]].mean().round(3).to_string())
    decomposition_figure(df, rob if not rob.empty else pd.DataFrame())
    if "true_gain_pct" in df:
        print("\nabsolute gains (% of media-driven sales): model / oracle")
        print(df.groupby(["scenario", "method"])[["true_gain_pct", "oracle_gain_pct"]].mean().round(2).unstack(0).to_string())
    b = load_bench()
    if not b.empty:
        open(f"{OUT_T}/benchmark.tex", "w").write(bench_table(b))
    # wide tables use the full-width table* environment of sn-jnl (as in the lab's sample paper)
    for name in ["main_results", "ablation", "realdata", "benchmark"]:
        if os.path.exists(f"{OUT_T}/{name}.tex"):
            t = open(f"{OUT_T}/{name}.tex").read()
            t = t.replace("\\begin{table}[t]", "\\begin{table*}[t]", 1).replace("\\end{table}", "\\end{table*}", 1)
            open(f"{OUT_T}/{name}.tex", "w").write(t)
    if not b.empty:
        bench_tv_figure(b)
        print(b.groupby(["scenario", "model"])[["roi_mape", "rank", "tv_bias", "video_bias", "test_r2", "test_mape"]]
              .mean().round(2).to_string())
    # ---- console summaries used when writing the text
    pd.set_option("display.width", 200)
    cols = ["roi_mape", "spearman", "tv_bias_pct", "r2_holdout", "uplift_captured", "coverage"]
    print(df[df.method != "NoMedia"].groupby(["scenario", "method"])[cols].mean().round(3).to_string())
    print("\nseeds per method:", df.groupby("method").seed.nunique().to_dict())
    print("ablation seeds:", abl_seeds)
    print("\nWilcoxon:", json.dumps({k: {a: round(b, 4) for a, b in v.items()} for k, v in wsum.items()}, indent=1))
    if rsum:
        print("\nReal data:", json.dumps({k: {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()} for k, v in rsum.items()}, indent=1, default=float))
