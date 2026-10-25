import sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from common import load, save
from orbit.models import KTR
csv, mj, out = sys.argv[1:4]
df, meta, tr = load(csv, mj); t0 = time.time()
ch = meta['channels']; sp = [f'{c}_spend' for c in ch]
ctl = [c for c in meta['controls'] if c not in ('trend', 'sin1', 'cos1', 'sin2', 'cos2')]  # KTR có level + mùa vụ riêng
# KTR là hồi quy hệ số biến thiên theo thời gian, không có adstock/bão hòa tích hợp:
# dùng adstock hình học cố định (lambda = 0,5; 8 tuần) và chuẩn hóa theo trung bình, như cách dùng KTR cho MMM.
w = 0.5 ** np.arange(8); w /= w.sum()
X = df.copy(); X['date'] = pd.to_datetime(X['date'])
scale = {}
for s in sp:
    a = np.convolve(df[s].values, w)[:len(df)]
    scale[s] = a[tr].mean(); X[s + '_ad'] = a / scale[s]
for c in ctl:
    X[c] = (df[c] - df.loc[tr, c].mean()) / (df.loc[tr, c].std() + 1e-9)
reg = [s + '_ad' for s in sp] + ctl
ysc = df.loc[tr, 'revenue'].mean(); X['revenue'] = df['revenue'] / ysc  # chuẩn hóa thang đo cho các tiên nghiệm mặc định
ktr = KTR(response_col='revenue', date_col='date', regressor_col=reg,
          regressor_sign=['+'] * len(sp) + ['='] * len(ctl),
          seasonality=[52], seasonality_fs_order=[2], level_segments=6, regression_segments=4,
          estimator='pyro-svi', num_steps=1500, num_sample=300, seed=int(meta['seed']))
ktr.fit(X[tr])
pred = ktr.predict(X)
yhat = pred['prediction'].values * ysc
coef = ktr.get_regression_coefs()            # hệ số theo từng tuần huấn luyện
roi = {}
for c, s in zip(ch, sp):
    contrib = (coef[s + '_ad'].values * X.loc[tr, s + '_ad'].values).sum() * ysc
    roi[c] = contrib / df.loc[tr, s].sum()
save(out, 'Orbit KTR', meta, csv, t0, df['revenue'].values, yhat, tr, roi, notes='pyro-svi; adstock hình học cố định λ=0,5')
