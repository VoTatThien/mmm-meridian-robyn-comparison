"""Run CausalMixFormer (and ablations) on the 9 national datasets of the preliminary benchmark.

    python run_benchmark.py            # writes bench/results/<model>/<dataset>.json
Then `cd bench/src && python aggregate.py` includes the new rows next to the 7 configurations.
"""
import glob
import json
import os
import sys

from cmf.benchmark_adapter import run

VARIANTS = {
    "CausalMixFormer": dict(),
    "CausalMixFormer w/o mediation": dict(mediation=False),
    "CausalMixFormer (linear baseline)": dict(baseline="linear"),
}
FOLDER = {"CausalMixFormer": "causalmixformer_v2", "CausalMixFormer w/o mediation": "cmf_nomed_v2",
          "CausalMixFormer (linear baseline)": "cmf_linbase_v2"}

if __name__ == "__main__":
    only = sys.argv[1:] or list(VARIANTS)
    for csv in sorted(glob.glob("bench/data/sim/S*_seed*.csv")):
        name = os.path.basename(csv)[:-4]
        meta = csv[:-4] + ".json"
        for v in only:
            if v == "CausalMixFormer w/o mediation" and not name.startswith("S3"):
                continue  # identical to the full model when no mediator exists
            out = f"bench/results/{FOLDER[v]}/{name}.json"
            if os.path.exists(out):
                continue
            r = run(csv, meta, out, model_name=v, **VARIANTS[v])
            tv = 100 * (r["roi"]["tv"] - r["true_roi"]["tv"]) / r["true_roi"]["tv"]
            print(f"{name} {v:34s} ROI MAPE={r['roi_mape']:.1f}% TV={tv:+.0f}% test R2={r['test']['r2']:.3f} "
                  f"test MAPE={r['test']['mape']:.2f}% ({r['runtime_s']:.0f}s)", flush=True)
