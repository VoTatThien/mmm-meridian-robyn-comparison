"""Loader for the public 190-DMA dataset distributed with DeepCausalMMM (MIT licence).

Source: https://github.com/adityapt/deepcausalmmm, file examples/data/MMM Data.csv
190 DMAs x 109 weeks (weekid 0..108), 13 media channels given as impressions,
target = weekly visits, 7 anonymised controls. No mediator is provided.

Only DMAs with all 109 weeks are kept, so every series is a complete panel.
Because media are impressions, "ROI" from the models is an effect per impression;
only relative quantities (contribution shares, stability) are interpreted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate import MMMData


def load_dcm190(path="data/dcm_190dma.csv") -> MMMData:
    df = pd.read_csv(path)
    n_weeks = df.weekid.nunique()
    full = df.groupby("dmacode").weekid.transform("size") == n_weeks
    df = df[full].sort_values(["dmacode", "weekid"])
    media = [c for c in df.columns if c.startswith("impressions_")]
    ctrl = [c for c in df.columns if c.startswith("control_")]
    geos = df.dmacode.unique()
    G, T = len(geos), n_weeks

    def panel(cols):
        a = df[cols].to_numpy(dtype=float)
        return a.reshape(G, T, len(cols))

    spend = panel(media)
    controls = panel(ctrl)
    sales = panel(["target_visits"])[..., 0]
    names = [c.split("_")[1] + "_" + c.split("_")[2] for c in media]  # e.g. Channel_01
    C = len(media)
    return MMMData(
        spend=spend,
        controls=controls,
        mediator=np.ones((G, T)),  # no mediator in this dataset; constant column is inert
        sales=sales,
        true_roi=np.full(C, np.nan),
        true_direct_roi=np.full(C, np.nan),
        true_contrib_share=float("nan"),
        channels=names,
        scenario="DCM190",
        meta=dict(geos=geos, n_dropped=int(df.dmacode.nunique() != G)),
    )
