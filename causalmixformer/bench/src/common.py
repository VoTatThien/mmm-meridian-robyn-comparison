import json, sys, time, os
import numpy as np, pandas as pd

def load(csv, meta_json):
    df = pd.read_csv(csv); meta = json.load(open(meta_json))
    n_te = meta['test_weeks']; T = len(df)
    tr = np.arange(T) < T - n_te
    return df, meta, tr

def metrics(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    mape = float(np.mean(np.abs(y - yhat) / np.abs(y)) * 100)
    r2 = float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))
    nrmse = float(np.sqrt(np.mean((y - yhat) ** 2)) / (y.max() - y.min()))
    return dict(mape=mape, r2=r2, nrmse=nrmse)

def save(out, model, meta, csv, t0, y, yhat, tr, roi, notes=''):
    res = dict(model=model, dataset=os.path.basename(csv).replace('.csv', ''), scenario=meta['scenario'],
               seed=meta['seed'], runtime_s=round(time.time() - t0, 1),
               train=metrics(np.asarray(y)[tr], np.asarray(yhat)[tr]),
               test=metrics(np.asarray(y)[~tr], np.asarray(yhat)[~tr]),
               roi={k: float(v) for k, v in roi.items()} if roi else None,
               true_roi=meta.get('true_roi'), notes=notes)
    if roi and meta.get('true_roi'):
        tru = meta['true_roi']
        res['roi_mape'] = float(np.mean([abs(roi[c] - tru[c]) / abs(tru[c]) for c in tru]) * 100)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ['model', 'dataset', 'runtime_s', 'test']}), 'roi_mape=', res.get('roi_mape'))
