"""Tái lập kết quả công bố của DeepCausalMMM trên dữ liệu bảng 190 DMA × 109 tuần (holdout 12%)."""
import sys, time, warnings, logging, json, numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
from deepcausalmmm.core import get_default_config
from deepcausalmmm.core.trainer import ModelTrainer
from deepcausalmmm.core.data import UnifiedDataPipeline
mode = sys.argv[1] if len(sys.argv) > 1 else 'triangular'
t0 = time.time()
d = pd.read_csv('../data/raw/dcm_mmm_data.csv')
idx = pd.MultiIndex.from_product([sorted(d.dmacode.unique()), sorted(d.weekid.unique())], names=['dmacode', 'weekid'])
d = d.set_index(['dmacode', 'weekid']).reindex(idx, fill_value=0).reset_index()  # bảng không cân bằng -> điền 0
imp = [c for c in d.columns if c.startswith('impressions_')]; ctl = [c for c in d.columns if c.startswith('control_')]
G = d.dmacode.nunique(); T = d.weekid.nunique()
Xm = d[imp].values.reshape(G, T, -1).astype(np.float32); Xc = d[ctl].values.reshape(G, T, -1).astype(np.float32)
y = d['target_visits'].values.reshape(G, T).astype(np.float32)
torch.manual_seed(42); np.random.seed(42)
cfg = get_default_config(); cfg['dag_mode'] = mode  # cấu hình mặc định, holdout_ratio = 0.12 như bài báo
pipe = UnifiedDataPipeline(cfg)
trd, hod = pipe.temporal_split(Xm, Xc, y)
trt = pipe.fit_and_transform_training(trd); hot = pipe.transform_holdout(hod)
tr = ModelTrainer(cfg)
model = tr.create_model(n_media=trt['X_media'].shape[2], n_control=trt['X_control'].shape[2], n_regions=G)
tr.create_optimizer_and_scheduler()
res = tr.train(trt['X_media'], trt['X_control'], trt['R'], trt['y'], hot['X_media'], hot['X_control'], hot['R'], hot['y'], pipeline=pipe, verbose=False)
out = dict(mode=mode, runtime_s=round(time.time() - t0, 1), train_r2=float(res['final_train_r2']), holdout_r2=float(res['final_holdout_r2']),
           train_rmse=float(res['final_train_rmse']), holdout_rmse=float(res['final_holdout_rmse']), paper_holdout_r2=0.843, paper_train_r2=0.949)
json.dump(out, open(f'../results/dcm_panel_{mode}.json', 'w'), indent=1); print(out)
