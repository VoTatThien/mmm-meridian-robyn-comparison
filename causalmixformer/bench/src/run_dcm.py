import sys, time, warnings, logging, numpy as np, pandas as pd, torch
torch.set_num_threads(1)
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
from common import load, save
from deepcausalmmm.core import get_default_config
from deepcausalmmm.core.trainer import ModelTrainer
from deepcausalmmm.core.data import UnifiedDataPipeline
csv, mj, out = sys.argv[1:4]
mode = sys.argv[4] if len(sys.argv) > 4 else 'triangular'
df, meta, tr = load(csv, mj); t0 = time.time()
ch = meta['channels']; sp = [f'{c}_spend' for c in ch]; ctl = meta['controls']
torch.manual_seed(int(meta['seed'])); np.random.seed(int(meta['seed']))
cfg = get_default_config(); cfg['random_seed'] = int(meta['seed']); cfg['dag_mode'] = mode
T = len(df); cfg['holdout_ratio'] = (meta['test_weeks'] + 0.5) / T
Xm = df[sp].values[None].astype(np.float32); Xc = df[ctl].values[None].astype(np.float32); y = df['revenue'].values[None].astype(np.float32)
pipe = UnifiedDataPipeline(cfg)
trd, hod = pipe.temporal_split(Xm, Xc, y)
trt = pipe.fit_and_transform_training(trd); hot = pipe.transform_holdout(hod)
trainer = ModelTrainer(cfg)
model = trainer.create_model(n_media=trt['X_media'].shape[2], n_control=trt['X_control'].shape[2], n_regions=trt['X_media'].shape[0])
trainer.create_optimizer_and_scheduler()
tres = trainer.train(trt['X_media'], trt['X_control'], trt['R'], trt['y'], hot['X_media'], hot['X_control'], hot['R'], hot['y'], pipeline=pipe, verbose=False)
res = pipe.predict_and_postprocess(model, Xm, Xc, ch, ctl, combine_with_holdout=True)
yhat = np.asarray(res['predictions']).ravel()
import torch.nn.functional as F
ps = F.softplus(model.prediction_scale).detach() if hasattr(model, 'prediction_scale') else None
mc = pipe.scaler.inverse_transform_contributions(res['media_contributions'], prediction_scale=ps)['media']
mc = np.asarray(mc.detach() if hasattr(mc, 'detach') else mc)[0]  # weeks x channels

roi = {c: mc[tr, i].sum() / df.loc[tr, s].sum() for i, (c, s) in enumerate(zip(ch, sp))}
name = 'DeepCausalMMM' + ('' if mode == 'triangular' else ' (NOTEARS)')
save(out, name, meta, csv, t0, df['revenue'].values, yhat, tr, roi, notes=f"dag_mode={mode}; package-reported train_r2={tres.get('final_train_r2', float('nan')):.3f}, holdout_r2={tres.get('final_holdout_r2', float('nan')):.3f}")
