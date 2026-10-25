import sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from common import load, save
from meridian.data import load as mload
from meridian.model import model, spec, prior_distribution
from meridian.analysis import analyzer
csv, mj, out = sys.argv[1:4]
df, meta, tr = load(csv, mj); t0 = time.time()
ch = meta['channels']; sp = [f'{c}_spend' for c in ch]
c2c = mload.CoordToColumns(time='date', kpi='revenue', controls=meta['controls'], media=sp, media_spend=sp)
data = mload.DataFrameDataLoader(df=df, coord_to_columns=c2c, kpi_type='revenue',
        media_to_channel=dict(zip(sp, ch)), media_spend_to_channel=dict(zip(sp, ch))).load()
ms = spec.ModelSpec(max_lag=8, holdout_id=~tr)
mmm = model.Meridian(input_data=data, model_spec=ms)
mmm.sample_posterior(n_chains=2, n_adapt=400, n_burnin=300, n_keep=300, seed=int(meta['seed']) + 1)
an = analyzer.Analyzer(mmm)
eo = np.asarray(an.expected_outcome(aggregate_geos=True, aggregate_times=False))  # chains, draws, times
yhat = eo.reshape(-1, eo.shape[-1]).mean(0)
times = list(df['date'][tr])
roi_s = np.asarray(an.roi(selected_times=times))
roi_m = roi_s.reshape(-1, roi_s.shape[-1]).mean(0)
rhat = an.get_rhat()
mx = max(float(np.nanmax(np.asarray(v))) for v in rhat.values())
save(out, 'Meridian', meta, csv, t0, df['revenue'].values, yhat, tr, dict(zip(ch, roi_m)), notes=f'max_rhat={mx:.3f}')
