import json, glob, os, numpy as np, pandas as pd
rows = []
for f in [f for f in glob.glob('../results/*/*.json') if '_old' not in f]:
    r = json.load(open(f))
    row = dict(model=r['model'], dataset=r['dataset'], scenario=r['scenario'], seed=r['seed'], runtime_s=r['runtime_s'],
               train_r2=r['train']['r2'], train_mape=r['train']['mape'], test_r2=r['test']['r2'], test_mape=r['test']['mape'],
               test_nrmse=r['test']['nrmse'], roi_mape=r.get('roi_mape'), notes=r.get('notes', ''))
    if r.get('roi') and r.get('true_roi'):
        for c, v in r['roi'].items():
            row[f'roi_{c}'] = v; row[f'true_{c}'] = r['true_roi'][c]
        from scipy.stats import spearmanr
        cs = list(r['true_roi']); row['roi_rank_corr'] = float(spearmanr([r['roi'][c] for c in cs], [r['true_roi'][c] for c in cs])[0])
    rows.append(row)
df = pd.DataFrame(rows).sort_values(['scenario', 'model', 'seed'])
os.makedirs('../report', exist_ok=True)
df.to_csv('../report/all_runs.csv', index=False)
sim = df[df.scenario != 'PUBLIC']
agg = sim.groupby(['scenario', 'model']).agg(roi_mape=('roi_mape', 'mean'), roi_mape_sd=('roi_mape', 'std'), rank_corr=('roi_rank_corr', 'mean'),
        test_mape=('test_mape', 'mean'), test_r2=('test_r2', 'mean'), train_r2=('train_r2', 'mean'), runtime_s=('runtime_s', 'mean'), n=('seed', 'count')).round(3)
print(agg.to_string())
pub = df[df.scenario == 'PUBLIC'][['model', 'train_r2', 'train_mape', 'test_r2', 'test_mape', 'runtime_s']].round(3)
print(pub.to_string())
agg.to_csv('../report/summary_sim.csv'); pub.to_csv('../report/summary_public.csv', index=False)
