import sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import jax.numpy as jnp, numpyro
numpyro.set_host_device_count(2)
from common import load, save
from lightweight_mmm import lightweight_mmm, preprocessing
csv, mj, out = sys.argv[1:4]
df, meta, tr = load(csv, mj); t0 = time.time()
ch = meta['channels']; sp = [f'{c}_spend' for c in ch]
ctl = [c for c in meta['controls'] if c not in ('trend', 'sin1', 'cos1', 'sin2', 'cos2')]  # LW có sẵn trend + mùa vụ
media = jnp.array(df[sp].values, dtype=jnp.float32)
extra = jnp.array(df[ctl].values, dtype=jnp.float32) if ctl else None
target = jnp.array(df['revenue'].values, dtype=jnp.float32)
costs = jnp.array(df.loc[tr, sp].sum().values, dtype=jnp.float32)
ms = preprocessing.CustomScaler(divide_operation=jnp.mean)
ts = preprocessing.CustomScaler(divide_operation=jnp.mean)
cs = preprocessing.CustomScaler(divide_operation=jnp.mean)
pos_ctl = bool(ctl) and bool((df[ctl].values >= 0).all())
es = preprocessing.CustomScaler(divide_operation=jnp.mean) if pos_ctl else None  # biến đã chuẩn hóa (có giá trị âm) thì giữ nguyên
trn = np.where(tr)[0]
m_tr = ms.fit_transform(media[trn]); y_tr = ts.fit_transform(target[trn]); c_s = cs.fit_transform(costs)
kw = {}
if ctl:
    e_all = extra + 1e-6 if pos_ctl else extra
    kw['extra_features'] = es.fit_transform(e_all[trn]) if es else e_all[trn]
mmm = lightweight_mmm.LightweightMMM(model_name='hill_adstock')
mmm.fit(media=m_tr, media_prior=c_s, target=y_tr, number_warmup=400, number_samples=300, number_chains=2,
        weekday_seasonality=False, seasonality_frequency=52, degrees_seasonality=2, seed=int(meta['seed']), **kw)
# dự báo toàn chuỗi (để lưu dư từ giai đoạn huấn luyện được tính vào tuần kiểm tra)
tei = np.where(~tr)[0]
pk = dict(media=ms.transform(media[tei]), target_scaler=ts, seed=1)
if ctl: pk['extra_features'] = es.transform(e_all[tei]) if es else e_all[tei]
pred_te = np.asarray(mmm.predict(**pk)).mean(0)
fit_tr = np.asarray(ts.inverse_transform(mmm.trace['mu'])).mean(0)
yhat = np.concatenate([fit_tr, pred_te])
_, roi_hat = mmm.get_posterior_metrics(target_scaler=ts, cost_scaler=cs)
roi = dict(zip(ch, np.asarray(roi_hat).mean(0)))
save(out, 'LightweightMMM', meta, csv, t0, df['revenue'].values, yhat, tr, roi, notes='hill_adstock')
