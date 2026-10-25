import sys, time, warnings, logging, os, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
import matplotlib; matplotlib.use('Agg')
from common import load, save
import robyn
from robyn.robyn import Robyn
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.enums import DependentVarType, AdstockType, ProphetVariableType, ProphetSigns, PaidMediaSigns, ContextSigns
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
csv, mj, out = sys.argv[1:4]
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
df, meta, tr = load(csv, mj); t0 = time.time()
ch = meta['channels']; sp = [f'{c}_spend' for c in ch]
ctx = [c for c in meta['controls'] if c not in ('trend', 'sin1', 'cos1', 'sin2', 'cos2')]  # Robyn dùng Prophet cho trend + mùa vụ
d = df[['date', 'revenue'] + sp + ctx].copy(); d['date'] = pd.to_datetime(d['date'])
np.random.seed(int(meta['seed']))
spec = MMMData.MMMDataSpec(dep_var='revenue', dep_var_type=DependentVarType.REVENUE, date_var='date',
    window_start=d['date'][0], window_end=d['date'][tr].iloc[-1],
    paid_media_spends=sp, paid_media_vars=sp, paid_media_signs=[PaidMediaSigns.POSITIVE] * len(sp),
    context_vars=ctx, context_signs=[ContextSigns.DEFAULT] * len(ctx), organic_vars=[], organic_signs=[], factor_vars=[])
hol = pd.read_csv(os.path.join(os.path.dirname(robyn.__file__), 'tutorials/resources/dt_prophet_holidays.csv'))
hd = HolidaysData(dt_holidays=hol, prophet_vars=[ProphetVariableType.TREND, ProphetVariableType.SEASON],
                  prophet_signs=[ProphetSigns.DEFAULT] * 2, prophet_country='US')
hp = Hyperparameters(hyperparameters={s: ChannelHyperparameters(thetas=[0, 0.8], alphas=[0.5, 3], gammas=[0.3, 1]) for s in sp},
                     adstock=AdstockType.GEOMETRIC, lambda_=0.0, train_size=[0.5, 0.8])
wd = f"../work/robyn/{os.path.basename(csv)[:-4]}"
r = Robyn(working_dir=wd, console_log_level='ERROR', file_log_level='ERROR')
r.initialize(MMMData(data=d, mmmdata_spec=spec), hd, hp)
r.feature_engineering(display_plots=False)
r.train_models(trials_config=TrialsConfig(trials=3, iterations=iters), cores=2, display_plots=False)
r.evaluate_models(display_plots=False)
pr = r.pareto_result; h = pr.result_hyp_param.copy(); xa = pr.x_decomp_agg
# chọn mô hình trên mặt Pareto có khoảng cách nhỏ nhất tới gốc (NRMSE, DECOMP.RSSD đã chuẩn hóa)
cand = h[h['sol_id'].isin(pr.pareto_solutions)].copy()
nz = lambda s: (s - h[s.name].min()) / (h[s.name].max() - h[s.name].min() + 1e-12)
cand['dist'] = np.sqrt(nz(cand['nrmse']) ** 2 + nz(cand['decomp.rssd']) ** 2)
best = cand.sort_values('dist').iloc[0]; sid = best['sol_id']
xs = xa[xa['sol_id'] == sid].set_index('rn')
# tái dựng dự báo cho toàn chuỗi (huấn luyện + kiểm tra) từ siêu tham số và hệ số của mô hình được chọn
dm = r.featurized_mmm_data.dt_mod.reset_index(drop=True)
win = np.where(tr)[0]
yhat = np.full(len(dm), xs.loc['intercept', 'coef'] if 'intercept' in xs.index else 0.0)
for rn in xs.index:
    if rn == 'intercept': continue
    if rn in sp:
        x = dm[rn].values.astype(float); th = best[f'{rn}_thetas']; al = best[f'{rn}_alphas']; ga = best[f'{rn}_gammas']
        a = np.zeros_like(x); a[0] = x[0]
        for i in range(1, len(x)): a[i] = x[i] + th * a[i - 1]
        infl = a[win].max() * ga
        feat = a ** al / (a ** al + infl ** al)
    else:
        feat = dm[rn].values.astype(float)
    yhat = yhat + xs.loc[rn, 'coef'] * feat
roi = {c: float(xs.loc[s, 'roi_total']) for c, s in zip(ch, sp)}
from common import metrics
chk = metrics(df['revenue'][tr], yhat[tr])['r2']
save(out, 'Robyn', meta, csv, t0, df['revenue'].values, yhat, tr, roi,
     notes=f"sol={sid}; robyn rsq_train={best['rsq_train']:.3f} vs reconstructed {chk:.3f}; nrmse={best['nrmse']:.3f}; decomp.rssd={best['decomp.rssd']:.3f}; iters={iters}x3")
