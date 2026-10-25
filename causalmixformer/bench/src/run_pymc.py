import sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from common import load, save
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
csv, mj, out = sys.argv[1:4]
df, meta, tr = load(csv, mj); t0 = time.time()
ch = meta['channels']; sp = [f'{c}_spend' for c in ch]
df['date'] = pd.to_datetime(df['date'])
X = df[['date'] + sp + meta['controls']]; y = df['revenue']
mmm = MMM(date_column='date', channel_columns=sp, control_columns=meta['controls'],
          adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation())
mmm.fit(X[tr], y[tr], chains=2, cores=2, draws=300, tune=400, target_accept=0.9,
        random_seed=int(meta['seed']), progressbar=False)
p_tr = mmm.sample_posterior_predictive(X[tr], extend_idata=False, progressbar=False, random_seed=1)
p_te = mmm.sample_posterior_predictive(X[~tr], extend_idata=False, include_last_observations=True, progressbar=False, random_seed=1)
yhat = np.concatenate([np.asarray(p_tr['y'].mean('sample')).ravel(), np.asarray(p_te['y'].mean('sample')).ravel()])
contrib = mmm.compute_channel_contribution_original_scale().mean(('chain', 'draw'))  # date x channel
roi = {c: float(contrib.sel(channel=s).sum()) / df.loc[tr, s].sum() for c, s in zip(ch, sp)}
import arviz as az
rh = float(az.rhat(mmm.idata).to_array().max())
save(out, 'PyMC-Marketing', meta, csv, t0, y.values, yhat, tr, roi, notes=f'max_rhat={rh:.3f}')
