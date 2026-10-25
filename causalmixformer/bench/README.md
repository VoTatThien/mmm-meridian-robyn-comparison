# Benchmark các phương pháp MMM có mã nguồn công khai

Mã nguồn thực nghiệm kèm báo cáo tiến độ (Võ Tất Thiên – 22652711).

## Cấu trúc
- `src/dgp.py` – bộ sinh dữ liệu mô phỏng có ROI thật (kịch bản S1, S2, S3).
- `src/run_robyn.py`, `run_meridian.py`, `run_pymc.py`, `run_lw.py`, `run_orbit.py`, `run_dcm.py` – script chạy từng phương pháp; cùng giao diện:
  `python run_X.py <data.csv> <meta.json> <output.json>`
  (`run_dcm.py` nhận thêm tham số thứ 4: `triangular` hoặc `notears`; `run_robyn.py` nhận số vòng lặp).
- `src/run_dcm_panel.py` – tái lập kết quả DeepCausalMMM trên dữ liệu bảng 190 DMA.
- `src/common.py` – chia dữ liệu theo thời gian, tính chỉ số, lưu kết quả.
- `src/aggregate.py`, `src/charts.py` – tổng hợp kết quả và vẽ hình.
- `data/sim/` – 9 bộ dữ liệu mô phỏng (.csv) kèm ROI thật (.json).
- `data/public/` – dữ liệu công khai của DeepCausalMMM gộp cấp quốc gia.
- `results/<phương pháp>/<bộ dữ liệu>.json` – kết quả từng lần chạy.
- `report/all_runs.csv` – bảng tổng hợp tất cả lần chạy.

## Môi trường (mỗi phương pháp một môi trường riêng)
| Phương pháp | Python | Gói |
|---|---|---|
| Robyn | 3.11 | robynpy 0.3.6, pandas<2.2, numpy<2; cần R + gói glmnet |
| Meridian | 3.11 | google-meridian 2.1.0 |
| PyMC-Marketing | 3.11 | pymc-marketing 0.19.2 |
| LightweightMMM | 3.10 | lightweight_mmm 0.1.9, jax/jaxlib 0.4.23, numpyro 0.13.2, numpy<2 |
| Orbit KTR | 3.11 | orbit-ml 1.1.5, numpy<2 |
| DeepCausalMMM | 3.11 | deepcausalmmm 1.0.21, torch |

Ví dụ: `uv venv -p 3.11 envs/meridian && VIRTUAL_ENV=envs/meridian uv pip install google-meridian`

## Chạy lại
```
python src/dgp.py data/sim
cd src && ../envs/meridian/bin/python run_meridian.py ../data/sim/S1_seed1.csv ../data/sim/S1_seed1.json ../results/meridian/S1_seed1.json
python aggregate.py && python charts.py
```
Lưu ý: chạy DeepCausalMMM với `OMP_NUM_THREADS=1` khi chạy song song nhiều tiến trình.
