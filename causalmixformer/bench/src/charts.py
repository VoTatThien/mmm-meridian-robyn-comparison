import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#9a9994', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
ORDER = ['Robyn', 'Meridian', 'PyMC-Marketing', 'LightweightMMM', 'Orbit KTR', 'DeepCausalMMM', 'DeepCausalMMM (NOTEARS)']
COL = dict(zip(ORDER, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']))
d = pd.read_csv('../report/all_runs.csv'); sim = d[d.scenario != 'PUBLIC']
models = [m for m in ORDER if m in set(sim.model)]
SC = ['S1', 'S2', 'S3']; SCN = {'S1': 'S1 – chỉ định đúng', 'S2': 'S2 – lưu dư có độ trễ', 'S3': 'S3 – tác động gián tiếp'}

# Hình 1: sai số ROI theo kịch bản (thang log)
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), dpi=200, sharey=True)
for ax, sc in zip(axes, SC):
    g = sim[sim.scenario == sc].groupby('model')['roi_mape'].agg(['mean', 'min', 'max'])
    ms = [m for m in models if m in g.index]
    y = np.arange(len(ms))
    ax.barh(y, g.loc[ms, 'mean'], color=[COL[m] for m in ms], height=0.62, edgecolor='white', linewidth=1)
    ax.hlines(y, g.loc[ms, 'min'], g.loc[ms, 'max'], color='#0b0b0b', lw=1)
    for yi, m in zip(y, ms): ax.text(g.loc[m, 'max'] * 1.12, yi, f"{g.loc[m, 'mean']:.0f}%", va='center', fontsize=8, color='#0b0b0b')
    ax.set_xscale('log'); ax.set_xlim(10, 3000); ax.set_xticks([10, 100, 1000]); ax.set_xticklabels(['10', '100', '1000']); ax.set_title(SCN[sc], fontsize=9.5, color='#0b0b0b')
    ax.set_yticks(y); ax.set_yticklabels(ms); ax.invert_yaxis(); ax.grid(axis='x', color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel('Sai số ROI tương đối (%) – thang log')
plt.tight_layout(); plt.savefig('../report/fig_roi_mape.png'); plt.close()

# Hình 2: S3 – ROI ước lượng của TV và Video so với ROI thật (tổng) và ROI trực tiếp
s3 = sim[sim.scenario == 'S3']; s1 = sim[sim.scenario == 'S1']
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), dpi=200, sharey=True)
for ax, c in zip(axes, ['tv', 'video']):
    g = s3.groupby('model')[f'roi_{c}'].mean(); ms = [m for m in models if m in g.index]
    x = np.arange(len(ms))
    ax.bar(x, g.loc[ms], color=[COL[m] for m in ms], width=0.62, edgecolor='white')
    tot = s3[f'true_{c}'].mean(); direct = s1[f'true_{c}'].mean() / 2
    ax.axhline(tot, color='#0b0b0b', lw=1.5, label='ROI thật (trực tiếp + gián tiếp)')
    ax.axhline(direct, color='#0b0b0b', lw=1.2, ls='--', label='ROI thật – chỉ phần trực tiếp')
    for xi, m in zip(x, ms):
        v = g.loc[m]
        if v > 5.6: ax.text(xi, 5.55, f'{v:.2f}', ha='center', va='top', fontsize=7.5, color='white', fontweight='bold')
        else: ax.text(xi, v + 0.08, f'{v:.2f}', ha='center', fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([m.replace('-Marketing', '-Mkt').replace('LightweightMMM', 'Lightweight') for m in ms], rotation=20, ha='right')
    ax.set_title(f"Kênh {'TV' if c == 'tv' else 'Video'} – kịch bản S3", fontsize=9.5); ax.set_ylim(0, 6)
    ax.grid(axis='y', color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel('ROI ước lượng (trung bình 3 seed)')
axes[0].legend(loc='upper left', fontsize=7.5, frameon=False)
plt.tight_layout(); plt.savefig('../report/fig_s3_indirect.png'); plt.close()

# Hình 3: thời gian chạy trung bình
g = sim.groupby('model')['runtime_s'].mean(); ms = [m for m in models if m in g.index]
fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=200)
y = np.arange(len(ms)); ax.barh(y, g.loc[ms], color=[COL[m] for m in ms], height=0.6, edgecolor='white')
for yi, m in zip(y, ms): ax.text(g.loc[m] + 3, yi, f'{g.loc[m]:.0f} s', va='center', fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(ms); ax.invert_yaxis(); ax.set_xlabel('Thời gian chạy trung bình mỗi bộ dữ liệu (giây, CPU 2 nhân)')
ax.grid(axis='x', color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig('../report/fig_runtime.png'); plt.close()
print('charts ok', models)
