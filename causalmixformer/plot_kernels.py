"""Learned lag-attention weights vs. true carryover kernels in scenario S2 (CT3 evidence).

    python plot_kernels.py   -> paper/figures/kernels_s2.pdf / .png, prints peak lags
"""
import numpy as np
import torch

torch.set_num_threads(1)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmf.model import EnsembleCMF, ModelConfig
from cmf.simulate import CHANNELS, SimConfig, simulate

d = simulate(SimConfig(scenario="S2", seed=100))  # an evaluation seed
tT = d.T - 13
att = EnsembleCMF(ModelConfig(), n_members=1).fit(d, tT, seed=100)
geo = EnsembleCMF(ModelConfig(carryover="geometric"), n_members=1).fit(d, tT, seed=100)
w_att = att.members[0][0].carry.weights().detach().numpy()
w_geo = geo.members[0][0].carry.weights().detach().numpy()
true = d.meta["kernels"]

show = [0, 2, 4]  # TV, Social, Radio have delayed (Weibull) carryover in S2
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
lags = np.arange(true.shape[1])
for ax, c in zip(axes, show):
    ax.plot(lags, true[c], color="#222222", lw=1.8, label="True kernel")
    ax.plot(lags, w_att[c], color="#2f5d8a", lw=1.5, marker="o", ms=3, label="Lag attention (CT3)")
    ax.plot(lags, w_geo[c], color="#d18f52", lw=1.5, ls="--", label="Geometric adstock")
    ax.set_title(CHANNELS[c], fontsize=9)
    ax.set_xlabel("Lag (weeks)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("Weight", fontsize=8)
axes[0].legend(fontsize=6.5, frameon=False)
fig.tight_layout()
import os
os.makedirs("paper/figures", exist_ok=True)
fig.savefig("paper/figures/kernels_s2.pdf")
fig.savefig("paper/figures/kernels_s2.png", dpi=200)
for c in show:
    print(CHANNELS[c], "true peak", true[c].argmax(), "attention peak", w_att[c].argmax(), "geometric peak", w_geo[c].argmax(),
          "L1 dist att %.2f geo %.2f" % (np.abs(w_att[c] - true[c]).sum(), np.abs(w_geo[c] - true[c]).sum()))
