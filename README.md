# MMM Meridian vs Robyn: A Three-Configuration Comparative Study

[![DOI](https://img.shields.io/badge/Conference-ICATSD%202026-blue)](https://icatsd.iuh.edu.vn/)

## Projects in this repository

| Folder | Paper |
|---|---|
| `data/`, `models/`, `analysis/` (this page) | *Analyzing the Differences Between Google Meridian and Meta Robyn in Marketing Mix Modeling: A Three-Configuration Comparative Study* (ICATSD 2026) |
| [`causalmixformer/`](causalmixformer/) | *CausalMixFormer: A Mediation-Aware Causal Temporal Transformer for Nonlinear Marketing Mix Modeling* (manuscript in preparation) — see [`causalmixformer/README.md`](causalmixformer/README.md) |

## Overview

This repository contains the analysis scripts and experimental configurations for the paper:

> **"Analyzing the Differences Between Google Meridian and Meta Robyn in Marketing Mix Modeling: A Three-Configuration Comparative Study"**
> 
> Thien T. Vo, Tran T. B. Phan, Kien C. Nguyen
> 
> The 3rd International Conference on Advanced Technology and Sustainable Development (ICATSD 2026)

## Abstract

This study conducts a systematic three-configuration comparative analysis of Google Meridian and Meta Robyn using a synthetic FMCG dataset from the Indian market (1,248 observations spanning 156 weeks across 8 geographic regions). The three configurations—Meridian-Geo, Meridian-National, and Robyn-National—disentangle the effects of data structure from modeling methodology.

## Repository Structure

```
├── data/
│   └── preprocess_india.py          # Data preprocessing script
├── models/
│   ├── robyn_3seed.R                # Robyn configuration (R)
│   ├── meridian_run.py              # Meridian configuration (Python)
│   └── robyn_india.R                # Robyn India market setup
├── analysis/
│   ├── correlation_vif_analysis.py  # Correlation matrix & VIF computation
│   └── generate_figures_and_export.py  # Figure generation
├── causalmixformer/                 # CausalMixFormer (separate paper, own README)
├── README.md
└── requirements.txt
```

## Dataset

The synthetic FMCG dataset used in this study is publicly available on Kaggle. It comprises:
- **1,248 observations** (156 weeks × 8 geographic regions)
- **Target variable**: Sales_Value (weekly revenue)
- **Media variables**: TV, Digital, Print, Radio
- **Control variables**: CPI, Festival Index, Rainfall Index, Trade Spend

## Reproducibility

All experiments can be reproduced using the following random seeds:
```
{42, 123, 256, 314, 501, 628, 777, 888, 1001, 1234}
```

### Requirements

**Python** (for Google Meridian):
- Python 3.10+
- google-meridian >= 1.0.5
- JAX
- pandas, numpy

**R** (for Meta Robyn):
- R 4.3.2+
- Robyn >= 3.11.1
- Nevergrad (via reticulate)

## Key Findings

1. **Data structure vs. methodology**: The dominant factor reverses by evaluation dimension—data structure accounts for 65–72% of the predictive accuracy gap, while methodology accounts for 56% of the estimation stability gap.
2. **Estimation stability**: Meridian achieves superior stability (mean CV = 1.2%) compared to Robyn (mean CV = 42.7%).
3. **Digital channel**: Ridge regression systematically underestimates Digital channel contribution due to multicollinearity-driven coefficient suppression.
4. **Statistical significance**: All inter-framework ROI differences are significant (Wilcoxon p < 0.01, Cohen's d > 1.9).

## Citation

```bibtex
@inproceedings{vo2026mmm,
  title={Analyzing the Differences Between Google Meridian and Meta Robyn in Marketing Mix Modeling: A Three-Configuration Comparative Study},
  author={Vo, Thien T. and Phan, Tran T. B. and Nguyen, Kien C.},
  booktitle={Proceedings of the 3rd International Conference on Advanced Technology and Sustainable Development (ICATSD 2026)},
  year={2026}
}
```

## License

This project is licensed under the MIT License.
