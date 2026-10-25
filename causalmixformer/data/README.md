# Data

`dcm_190dma.csv` is `examples/data/MMM Data.csv` from DeepCausalMMM
(https://github.com/adityapt/deepcausalmmm), redistributed under its MIT licence; the full licence
text is in `LICENSE-deepcausalmmm` (Copyright (c) 2025 Aditya Puttaparthi Tirumala).

190 DMAs x 109 weeks, 13 media channels (impressions), weekly visits, 7 anonymised controls;
`cmf/realdata.py` keeps the 184 DMAs with complete data.
The simulated panels are generated on the fly by `cmf/simulate.py`; the national benchmark
datasets are in `bench/data/sim/`.
