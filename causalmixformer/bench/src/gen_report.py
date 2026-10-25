"""Sinh nội dung báo cáo (r_content.js) từ kết quả thực nghiệm."""
import json, glob, os, numpy as np, pandas as pd

R = '../report'
d = pd.read_csv(f'{R}/all_runs.csv'); sim = d[d.scenario != 'PUBLIC']; pub = d[d.scenario == 'PUBLIC']
ORDER = ['Robyn', 'Meridian', 'PyMC-Marketing', 'LightweightMMM', 'Orbit KTR', 'DeepCausalMMM', 'DeepCausalMMM (NOTEARS)']
ms = [m for m in ORDER if m in set(sim.model)]
f0 = lambda x: f'{x:.0f}'.replace('.', ',')
f1 = lambda x: f'{x:.1f}'.replace('.', ',')
f2 = lambda x: f'{x:.2f}'.replace('.', ',')
neg = lambda s: s.replace('-', '−')

def js(s):
    return "'" + s.replace('\\', '\\\\').replace("'", "\\'") + "'"

B = []  # blocks as JS source strings
def h1(t, chap): B.append(f"{{ h1: {js(t)}, chap: {chap} }}")
def h2(t): B.append(f"{{ h2: {js(t)} }}")
def h3(t): B.append(f"{{ h3: {js(t)} }}")
def p(t, keep=False): B.append(f"{{ p: {js(t)}{', keepNext: true' if keep else ''} }}")
def ul(items): B.append("{ ul: [" + ", ".join(js(i) for i in items) + "] }")
def eq(t, i): B.append(f"{{ eq: {js(t)}, id: {js(i)} }}")
def fig(i, f, cap, w=16): B.append(f"{{ fig: {{ id: {js(i)}, file: {js(f)}, caption: {js(cap)}, w: {w} }} }}")
def table(i, cap, widths, head, rows, size=21, center=(), boldCol=0, note=None):
    rs = "[" + ", ".join("[" + ", ".join(js(str(c)) for c in r) + "]" for r in rows) + "]"
    B.append(f"{{ table: {{ id: {js(i)}, caption: {js(cap)}, widths: {widths}, head: [{', '.join(js(h) for h in head)}], rows: {rs}, size: {size}, center: {list(center)}, boldCol: {boldCol}" + (f", note: {js(note)}" if note else "") + " } }")

g = sim.groupby(['scenario', 'model'])
roi_m = g['roi_mape'].mean().unstack(0); rank = sim.groupby('model')['roi_rank_corr'].mean()
tmape = sim.groupby('model')['test_mape'].mean(); tr2 = sim.groupby('model')['test_r2'].mean(); trr2 = sim.groupby('model')['train_r2'].mean()
rt = sim.groupby('model')['runtime_s'].mean()
tvb = {sc: sim[sim.scenario == sc].groupby('model').apply(lambda x: ((x.roi_tv - x.true_tv) / x.true_tv * 100).mean()) for sc in ['S1', 'S2', 'S3']}
true_avg = {sc: {c: sim[sim.scenario == sc][f'true_{c}'].mean() for c in ['tv', 'video', 'social', 'display', 'search']} for sc in ['S1', 'S3']}
# tỷ trọng đóng góp truyền thông ở S1
DIR = {'Robyn': 'robyn', 'Meridian': 'meridian', 'PyMC-Marketing': 'pymc', 'LightweightMMM': 'lw', 'Orbit KTR': 'orbit', 'DeepCausalMMM': 'dcm', 'DeepCausalMMM (NOTEARS)': 'dcm_notears'}
share = {}; share_true = []
for m in ms:
    v = []
    for sd in [1, 2, 3]:
        r = json.load(open(f'../results/{DIR[m]}/S1_seed{sd}.json')); df = pd.read_csv(f'../data/sim/S1_seed{sd}.csv')[:-16]
        v.append(sum(r['roi'][c] * df[c + '_spend'].sum() for c in r['roi']) / df.revenue.sum() * 100)
        if m == ms[0]: share_true.append(sum(r['true_roi'][c] * df[c + '_spend'].sum() for c in r['roi']) / df.revenue.sum() * 100)
    share[m] = np.mean(v)
st = np.mean(share_true)
panel = {k: json.load(open(f'../results/dcm_panel_{k}.json')) for k in ['triangular', 'notears'] if os.path.exists(f'../results/dcm_panel_{k}.json')}
NAIVE = json.load(open(f'{R}/naive_panel.json'))
best_roi = min(ms, key=lambda m: roi_m.loc[m].mean())

# ================= 1. GIỚI THIỆU =================
h1('1. GIỚI THIỆU', 1)
p('Báo cáo này trình bày kết quả thực hiện yêu cầu của Thầy ở giai đoạn hiện tại: hiện thực lại các phương pháp Marketing Mix Modeling (MMM) từ các nghiên cứu gần đây, ưu tiên các bài có mã nguồn công khai; báo cáo kết quả; và đề xuất các cải tiến dựa trên những điểm yếu quan sát được. Các cải tiến sẽ được thực nghiệm ở bước tiếp theo.')
p('Toàn bộ các phương pháp được chạy trên cùng dữ liệu, cùng cách chia dữ liệu theo thời gian và cùng bộ chỉ số. Dữ liệu gồm hai nguồn: dữ liệu mô phỏng có ROI thật, dùng để đánh giá khả năng ước lượng đúng hiệu quả kênh; và một bộ dữ liệu công khai, dùng để đánh giá khả năng dự báo. Báo cáo gồm năm phần: phần 2 mô tả các phương pháp đã hiện thực; phần 3 trình bày thiết kế thực nghiệm; phần 4 trình bày kết quả; phần 5 đề xuất cải tiến và kế hoạch tiếp theo.')

# ================= 2. CÁC PHƯƠNG PHÁP =================
h1('2. CÁC PHƯƠNG PHÁP ĐƯỢC HIỆN THỰC', 2)
h2('2.1. Khảo sát mã nguồn và lựa chọn phương pháp')
p('Em đã khảo sát mã nguồn của các phương pháp được phân tích trong đề cương. Kết quả được tổng hợp trong Bảng {r:tab_survey}. Theo yêu cầu, chỉ các phương pháp có mã nguồn công khai và cài đặt được mới được hiện thực; ba nghiên cứu không công bố mã nguồn được ghi nhận nhưng không hiện thực lại.')
table('tab_survey', 'Khảo sát mã nguồn của các phương pháp MMM gần đây', [26, 30, 26, 18],
      ['Phương pháp', 'Nguồn', 'Mã nguồn công khai', 'Hiện thực'], [
      ['Robyn (Meta)', 'Runge và Pauwels (2026); kho Robyn {c:runge2026, robyn}', 'Có (R, Python)', 'Có – robynpy 0.3.6 {c:robynpy}'],
      ['Meridian (Google)', 'Kho Meridian {c:meridian}', 'Có (Python)', 'Có – 2.1.0'],
      ['PyMC-Marketing', 'Kho PyMC-Marketing {c:pymc}', 'Có (Python)', 'Có – 0.19.2'],
      ['LightweightMMM (Google)', 'Kho LightweightMMM {c:lightweight}', 'Có (Python, ngừng phát triển)', 'Có – 0.1.9'],
      ['Orbit KTR (Uber)', 'Ng và cộng sự (2021) {c:ng2021, orbit2020}', 'Có (Python)', 'Có – orbit-ml 1.1.5'],
      ['DeepCausalMMM', 'Tirumala (2025) {c:tirumala2025, dcm_repo}', 'Có (Python, PyPI)', 'Có – 1.0.21, hai chế độ DAG'],
      ['NNN (Google)', 'Mulc và cộng sự (2025) {c:mulc2025}', 'Không', 'Không'],
      ['CausalMMM', 'Gong và cộng sự (2024) {c:gong2024}', 'Không', 'Không'],
      ['XGBoost + SHAP + LLM', 'Liu và Hu (2026) {c:liu2026}', 'Không', 'Không'],
      ], size=20)
p('Một phát hiện quan trọng trong quá trình khảo sát: phiên bản 1.0.21 của DeepCausalMMM đã bổ sung chế độ học DAG bằng NOTEARS (tham số dag_mode = \'notears\') bên cạnh ma trận tam giác trên mặc định {c:dcm_repo}, trong khi bài báo vẫn xếp NOTEARS vào kế hoạch tương lai {c:tirumala2025}. Vì vậy em chạy cả hai chế độ, tổng cộng 7 cấu hình. Phát hiện này ảnh hưởng trực tiếp đến định hướng cải tiến (mục 5.1).')

h2('2.2. Mô tả ngắn và cấu hình của từng phương pháp')
p('Bảng {r:tab_cfg} tóm tắt kỹ thuật ước lượng và cấu hình được dùng cho từng phương pháp. Nguyên tắc là dùng cấu hình mặc định hoặc khuyến nghị của gói; chỉ điều chỉnh khi cần để gói chạy được trên dữ liệu và trong giới hạn tính toán (CPU 2 nhân, không có GPU).')
table('tab_cfg', 'Kỹ thuật ước lượng và cấu hình thực nghiệm của từng phương pháp', [20, 34, 46],
      ['Phương pháp', 'Kỹ thuật chính', 'Cấu hình dùng trong thực nghiệm'], [
      ['Robyn', 'Hồi quy Ridge (glmnet) + tìm siêu tham số bằng Nevergrad, chọn mô hình trên mặt Pareto NRMSE – DECOMP.RSSD', 'Adstock hình học, Hill; 3 trial × 1.000 vòng lặp; trend và mùa vụ bằng Prophet; chọn mô hình Pareto gần gốc nhất'],
      ['Meridian', 'Bayes phân cấp, lấy mẫu NUTS', 'Tiên nghiệm mặc định; max_lag = 8; 2 chuỗi × (400 thích nghi + 300 burn-in + 300 mẫu); dùng holdout_id'],
      ['PyMC-Marketing', 'Bayes, lấy mẫu NUTS', 'Adstock hình học (l_max = 8), bão hòa logistic; 2 chuỗi × (400 tune + 300 mẫu)'],
      ['LightweightMMM', 'Bayes (NumPyro), NUTS', 'Mô hình hill_adstock; trend và mùa vụ tích hợp; 2 chuỗi × (400 warmup + 300 mẫu)'],
      ['Orbit KTR', 'Hồi quy hệ số biến thiên theo thời gian dựa trên kernel, suy luận biến phân (Pyro SVI)', 'Không có adstock tích hợp nên dùng adstock hình học cố định λ = 0,5; mùa vụ 52 tuần; 1.500 bước SVI'],
      ['DeepCausalMMM', 'GRU + DAG tam giác trên + Hill, mất mát Huber', 'Cấu hình mặc định của gói (tối đa 1.500 epoch, dừng sớm)'],
      ['DeepCausalMMM (NOTEARS)', 'Như trên, DAG học bằng NOTEARS', 'dag_mode = notears, các tham số NOTEARS mặc định'],
      ], size=20)
p('Mã nguồn chạy thực nghiệm được viết thành các script độc lập cho từng phương pháp, mỗi phương pháp một môi trường Python riêng để tránh xung đột thư viện. Với Robyn, gói Python không có hàm dự báo ngoài mẫu nên em tái dựng dự báo từ siêu tham số và hệ số của mô hình được chọn; kết quả tái dựng khớp với R² do Robyn báo cáo trên tập huấn luyện (sai khác dưới 0,01).')

# ================= 3. THIẾT KẾ THỰC NGHIỆM =================
h1('3. THIẾT KẾ THỰC NGHIỆM', 3)
h2('3.1. Dữ liệu mô phỏng có ROI thật')
p('Trên dữ liệu thực, ROI thật của từng kênh không bao giờ được biết, nên không thể kiểm tra mô hình ước lượng đúng hay sai. Vì vậy em xây dựng bộ sinh dữ liệu mô phỏng theo cấu trúc cộng tính phổ biến của MMM {c:jin2017}. Doanh thu tuần $t$ được sinh theo công thức {r:eq_dgp}:', keep=True)
eq('y_t = b_t + s_t + \\gamma\\, z_t + \\sum_{c=1}^{5}{\\beta_c\\, H\\br{\\tilde{x}_{t,c}; K_c, S_c}} + \\varepsilon_t', 'eq_dgp')
p('trong đó $b_t$ là mức nền có xu hướng tăng; $s_t$ là mùa vụ năm; $z_t$ là biến khuyến mãi; $\\tilde{x}_{t,c}$ là chi tiêu đã lưu dư của kênh $c$; $H$ là hàm Hill; và nhiễu $\\varepsilon_t$ có độ lệch chuẩn bằng 3% doanh thu trung bình. Có 5 kênh (TV, Video, Social, Display, Search), 156 tuần, trong đó 140 tuần đầu dùng để huấn luyện và 16 tuần cuối để kiểm tra. Chi tiêu các kênh có tương quan vừa phải qua một nhân tố mùa vụ chung. ROI thật được tính bằng mô phỏng phản thực tế, theo công thức {r:eq_roi}:', keep=True)
eq('{\\text{ROI}}_c = \\frac{\\sum_{t}{\\sbr{\\mathbb{E}\\br{y_t \\mid x} - \\mathbb{E}\\br{y_t \\mid x_c = 0}}}}{\\sum_{t \\in \\text{train}}{x_{t,c}}}', 'eq_roi')
p('trong đó tổng ở tử số được kéo dài thêm 12 tuần sau giai đoạn huấn luyện để tính đủ hiệu ứng lưu dư. Bảng {r:tab_scen} mô tả ba kịch bản; mỗi kịch bản sinh 3 bộ dữ liệu với hạt giống ngẫu nhiên khác nhau, tổng cộng 9 bộ.')
table('tab_scen', 'Các kịch bản dữ liệu mô phỏng', [10, 52, 38], ['Mã', 'Cơ chế sinh dữ liệu', 'Mục đích'], [
      ['S1', 'Adstock hình học, Hill tĩnh – đúng giả định của các mô hình Bayes', 'Kịch bản có lợi cho mô hình tham số'],
      ['S2', 'Như S1 nhưng TV và Video có lưu dư dạng Weibull, đạt đỉnh sau 2–4 tuần', 'Kiểm tra khi dạng hàm lưu dư bị chỉ định sai'],
      ['S3', 'Một nửa hiệu ứng của TV và Video đi qua biến trung gian "lượng tìm kiếm thương hiệu"; biến này được đưa vào mô hình như biến kiểm soát, đúng như thực hành phổ biến', 'Kiểm tra khả năng ước lượng ROI tổng khi có tác động gián tiếp'],
      ], size=21, center=[0])
p(f"ROI thật trung bình của các kênh ở kịch bản S1 lần lượt là: TV {f2(true_avg['S1']['tv'])}, Video {f2(true_avg['S1']['video'])}, Social {f2(true_avg['S1']['social'])}, Display {f2(true_avg['S1']['display'])}, Search {f2(true_avg['S1']['search'])}. Ở S3, ROI tổng của TV và Video tăng lên {f2(true_avg['S3']['tv'])} và {f2(true_avg['S3']['video'])} do có thêm phần gián tiếp. Tỷ trọng đóng góp thật của truyền thông trong doanh thu là khoảng {f0(st)}%.")
h2('3.2. Dữ liệu công khai')
p('Bộ dữ liệu công khai đi kèm DeepCausalMMM {c:dcm_repo} gồm 190 khu vực (DMA) × 109 tuần, 13 kênh truyền thông đo bằng lượt hiển thị, 7 biến kiểm soát và biến mục tiêu là lượt truy cập. Để mọi phương pháp chạy được trên cùng cấu trúc, dữ liệu được gộp về cấp quốc gia; 13 tuần cuối dùng để kiểm tra; các biến kiểm soát được chuẩn hóa z-score theo giai đoạn huấn luyện. Bộ dữ liệu không có ROI thật và không có chi tiêu, nên chỉ dùng để đánh giá dự báo. Ngoài ra, DeepCausalMMM được chạy thêm trên dữ liệu bảng 190 DMA để tái lập kết quả đã công bố.')
h2('3.3. Chỉ số đánh giá và môi trường chạy')
p('Chỉ số chính là sai số ROI tương đối trung bình trên 5 kênh, tính theo công thức {r:eq_roimape}:', keep=True)
eq('\\text{ROI-MAPE} = \\frac{100}{C}\\sum_{c=1}^{C}{\\frac{\\abs{\\hat{R}_c - R_c}}{R_c}}', 'eq_roimape')
p('trong đó $\\hat{R}_c$ và $R_c$ là ROI ước lượng và ROI thật của kênh $c$. Ngoài ra còn có: hệ số tương quan hạng Spearman giữa ROI ước lượng và ROI thật, đo khả năng xếp đúng thứ tự hiệu quả các kênh; MAPE và R² trên 16 tuần kiểm tra; và thời gian chạy. Thực nghiệm chạy trên máy ảo CPU 2 nhân, RAM 8 GB, không có GPU.')

# ================= 4. KẾT QUẢ =================
h1('4. KẾT QUẢ THỰC NGHIỆM', 4)
h2('4.1. Khả năng khôi phục ROI trên dữ liệu mô phỏng')
p('Bảng {r:tab_main} tổng hợp kết quả của 7 cấu hình trên 9 bộ dữ liệu mô phỏng; mỗi giá trị là trung bình của 3 hạt giống. Hình {r:fig_roi} biểu diễn sai số ROI theo từng kịch bản, với vạch đen là khoảng giữa giá trị nhỏ nhất và lớn nhất của 3 hạt giống.')
rows = []
for m in ms:
    rows.append([m, f0(roi_m.loc[m, 'S1']), f0(roi_m.loc[m, 'S2']), f0(roi_m.loc[m, 'S3']), neg(f2(rank[m])), f1(tmape[m]), neg(f2(tr2[m])), f0(rt[m])])
table('tab_main', 'Kết quả tổng hợp trên dữ liệu mô phỏng (trung bình 3 hạt giống)', [26, 10, 10, 10, 11, 11, 11, 11],
      ['Phương pháp', 'ROI-MAPE S1 (%)', 'ROI-MAPE S2 (%)', 'ROI-MAPE S3 (%)', 'Tương quan hạng ROI', 'MAPE kiểm tra (%)', 'R² kiểm tra', 'Thời gian (giây)'], rows,
      size=20, center=[1, 2, 3, 4, 5, 6, 7])
fig('fig_roi', '../report/' + f'fig_roi_mape.png', 'Sai số ROI tương đối của từng phương pháp theo kịch bản (thang log)', 16)
p(f"Có thể rút ra bốn nhận xét từ Bảng {{r:tab_main}} và Hình {{r:fig_roi}}.")
ul([
  f"**Không phương pháp nào khôi phục ROI tốt, kể cả ở kịch bản chỉ định đúng S1.** Sai số ROI thấp nhất vẫn ở mức {f0(roi_m.min(axis=0)['S1'])}–{f0(roi_m.loc[['Robyn','Meridian','PyMC-Marketing','Orbit KTR']].max(axis=0)['S1'])}% (Robyn, Meridian, PyMC-Marketing, Orbit KTR). Tương quan hạng giữa ROI ước lượng và ROI thật đều thấp (cao nhất {f2(rank.max())} của {rank.idxmax()}), nghĩa là các mô hình thường không xếp đúng kênh nào hiệu quả hơn. Đây là biểu hiện của vấn đề nhận dạng do đa cộng tuyến và chuỗi ngắn đã nêu trong đề cương.",
  f"**Sai số dự báo thấp không đồng nghĩa với ROI đúng.** LightweightMMM có MAPE kiểm tra tốt nhất ({f1(tmape['LightweightMMM'])}%) nhưng sai số ROI lớn nhất ({f0(roi_m.loc['LightweightMMM'].mean())}%). Meridian, PyMC-Marketing, LightweightMMM và Robyn có R² kiểm tra gần như nhau (0,8–0,87) trong khi sai số ROI chênh nhau gần 10 lần.",
  f"**Robyn có sai số ROI thấp nhất nhưng vì lý do không mong muốn.** Tiêu chí DECOMP.RSSD kéo tỷ trọng đóng góp về gần tỷ trọng chi tiêu, nên ROI của mọi kênh bị ép về gần nhau (khoảng 0,9–1,4). Kênh Search có ROI thật cao nhất ({f2(true_avg['S1']['search'])}) bị ước lượng chỉ khoảng 1,2, còn Display có ROI thật thấp nhất lại bị ước lượng cao hơn. Sai số trung bình thấp vì ROI thật của các kênh không quá chênh lệch, không phải vì Robyn nhận ra đúng kênh hiệu quả.",
  f"**Hai mô hình có xu hướng quy quá nhiều doanh thu cho truyền thông.** Với cấu hình mặc định, LightweightMMM quy {f0(share['LightweightMMM'])}% doanh thu cho truyền thông và DeepCausalMMM quy khoảng {f0(share['DeepCausalMMM'])}%, trong khi tỷ lệ thật chỉ khoảng {f0(st)}%. Meridian ({f0(share['Meridian'])}%) và PyMC-Marketing ({f0(share['PyMC-Marketing'])}%) gần với tỷ lệ thật hơn.",
])
table('tab_share', 'Tỷ trọng doanh thu được quy cho truyền thông ở kịch bản S1', [50, 25, 25], ['Phương pháp', 'Ước lượng (%)', 'Thật (%)'],
      [[m, f1(share[m]), f1(st)] for m in ms], size=21, center=[1, 2])

h2('4.2. Trường hợp có tác động gián tiếp (kịch bản S3)')
p('Kịch bản S3 mô phỏng tình huống thường gặp: quảng cáo TV và Video làm tăng lượng tìm kiếm thương hiệu, và lượng tìm kiếm này được nhà phân tích đưa vào mô hình như một biến kiểm soát. Hình {r:fig_s3} so sánh ROI ước lượng của TV và Video với hai mốc: ROI thật tổng (gồm cả phần gián tiếp) và ROI thật chỉ tính phần trực tiếp. Bảng {r:tab_tv} cho thấy độ lệch ROI của kênh TV qua ba kịch bản.')
fig('fig_s3', '../report/' + f'fig_s3_indirect.png', 'ROI ước lượng của TV và Video ở kịch bản S3 so với ROI thật', 16)
table('tab_tv', 'Độ lệch tương đối của ROI kênh TV theo kịch bản (%)', [40, 20, 20, 20], ['Phương pháp', 'S1', 'S2', 'S3'],
      [[m] + [neg(f'{tvb[sc][m]:+.0f}') for sc in ['S1', 'S2', 'S3']] for m in ms], size=21, center=[1, 2, 3],
      note='Giá trị âm là ước lượng thấp hơn ROI thật, giá trị dương là ước lượng cao hơn.')
p(f"Kết quả cho thấy rõ cơ chế mà đề cương đã dự đoán. Ở S1, Meridian ước lượng ROI của TV cao hơn thật {f0(tvb['S1']['Meridian'])}%; sang S3, khi một nửa hiệu ứng của TV đi qua biến trung gian, ROI của TV bị ước lượng thấp hơn thật {f0(-tvb['S3']['Meridian'])}%. Robyn ({neg(f0(tvb['S3']['Robyn']))}%), PyMC-Marketing ({neg(f0(tvb['S3']['PyMC-Marketing']))}%) và Orbit KTR ({neg(f0(tvb['S3']['Orbit KTR']))}%) cũng ước lượng thấp tương tự. Trên Hình {{r:fig_s3}}, ROI của TV từ các mô hình này nằm sát mốc \"chỉ phần trực tiếp\": phần hiệu ứng gián tiếp đã bị biến kiểm soát \"lượng tìm kiếm thương hiệu\" hấp thụ. Đây là hiện tượng thiên lệch do kiểm soát biến hậu can thiệp trong suy luận nhân quả {{c:pearl2009}}. Nếu dựa vào các kết quả này, doanh nghiệp sẽ cắt giảm nhầm ngân sách TV và Video.".replace('−', '−'))

h2('4.3. Khả năng dự báo trên dữ liệu công khai')
p('Bảng {r:tab_pub} trình bày kết quả trên dữ liệu công khai gộp cấp quốc gia. Do không có ROI thật, bảng chỉ gồm các chỉ số dự báo.')
pubr = pub.set_index('model')
table('tab_pub', 'Kết quả trên dữ liệu công khai cấp quốc gia (109 tuần, 13 tuần kiểm tra)', [34, 16, 16, 17, 17],
      ['Phương pháp', 'R² huấn luyện', 'MAPE kiểm tra (%)', 'R² kiểm tra', 'Thời gian (giây)'],
      [[m, neg(f2(pubr.loc[m, 'train_r2'])), f1(pubr.loc[m, 'test_mape']), neg(f1(pubr.loc[m, 'test_r2'])), f0(pubr.loc[m, 'runtime_s'])] for m in ms if m in pubr.index],
      size=21, center=[1, 2, 3, 4])
p(f"Robyn có MAPE kiểm tra thấp nhất ({f1(pubr.loc['Robyn','test_mape'])}%), tiếp theo là hai cấu hình DeepCausalMMM ({f1(pubr.loc['DeepCausalMMM (NOTEARS)','test_mape'])}% và {f1(pubr.loc['DeepCausalMMM','test_mape'])}%); các mô hình Bayes đạt khoảng 17–18%, còn Orbit KTR kém nhất. R² kiểm tra của mọi mô hình đều âm vì 13 tuần kiểm tra có phương sai nhỏ và doanh số biến động khác giai đoạn huấn luyện; với cửa sổ ngắn như vậy, MAPE là chỉ số đáng tin cậy hơn. Tuy nhiên, cả hai cấu hình DeepCausalMMM ở cấp quốc gia đều có R² huấn luyện âm, tức là mô hình không khớp được chính dữ liệu huấn luyện; MAPE kiểm tra thấp hơn các mô hình Bayes không đồng nghĩa với mô hình tốt hơn. Điều này nhất quán với kết quả trên dữ liệu mô phỏng: mô hình được thiết kế cho dữ liệu bảng nhiều vùng và không phù hợp với một chuỗi duy nhất.")

h2('4.4. Tái lập kết quả công bố của DeepCausalMMM trên dữ liệu bảng')
p('Bài báo DeepCausalMMM công bố R² holdout 0,843 trên dữ liệu bảng 190 DMA × 109 tuần {c:tirumala2025}. Em chạy lại gói với cấu hình mặc định trên đúng bộ dữ liệu này. Để đánh giá con số R² một cách khách quan, em so sánh thêm với một dự báo ngây thơ: mỗi DMA được dự báo bằng trung bình 4 tuần cuối của giai đoạn huấn luyện. Kết quả trình bày trong Bảng {r:tab_panel}.')
prow = [['Bài báo DeepCausalMMM (công bố)', '0,949', '0,843', '489 (máy của tác giả)']]
for k, lab in [('triangular', 'Tái lập – DAG tam giác trên'), ('notears', 'Tái lập – DAG NOTEARS')]:
    if k in panel: prow.append([lab, f3 if False else f"{panel[k]['train_r2']:.3f}".replace('.', ','), f"{panel[k]['holdout_r2']:.3f}".replace('.', ','), f0(panel[k]['runtime_s'])])
prow.append(['Dự báo ngây thơ: trung bình 4 tuần cuối giai đoạn huấn luyện của từng DMA', '–', f"{NAIVE['last4']:.3f}".replace('.', ','), '< 1'])
prow.append(['Dự báo ngây thơ: trung bình giai đoạn huấn luyện của từng DMA', '–', f"{NAIVE['mean']:.3f}".replace('.', ','), '< 1'])
table('tab_panel', 'Tái lập kết quả DeepCausalMMM trên dữ liệu 190 DMA × 109 tuần', [46, 16, 18, 20], ['Cấu hình', 'R² huấn luyện', 'R² holdout (gộp mọi DMA)', 'Thời gian (giây)'], prow, size=21, center=[1, 2, 3])
p('Bản tái lập đạt R² holdout cao hơn số công bố, có thể do phiên bản gói mới hơn. Tuy nhiên, chỉ một dự báo ngây thơ không dùng bất kỳ biến truyền thông nào cũng đạt R² gộp xấp xỉ hoặc cao hơn. Nguyên nhân là R² tính gộp trên 190 DMA bị chi phối bởi chênh lệch quy mô giữa các khu vực, không phản ánh khả năng mô hình nắm bắt biến động theo thời gian hay hiệu ứng truyền thông. Vì vậy, khi so sánh với các nghiên cứu khác ở bước sau, em sẽ không dùng R² gộp làm chỉ số chính.')

h2('4.5. Thời gian chạy')
p('Hình {r:fig_rt} so sánh thời gian chạy trung bình trên một bộ dữ liệu mô phỏng. Trên dữ liệu cấp quốc gia, mọi phương pháp chạy trong vài phút. Robyn chậm nhất vì 3.000 vòng tối ưu tiến hóa. Các mô hình Bayes chạy nhanh do số mẫu MCMC đã giảm để phù hợp phần cứng; đây là một hạn chế của thực nghiệm, sẽ được nêu ở mục 4.7.')
fig('fig_rt', '../report/' + f'fig_runtime.png', 'Thời gian chạy trung bình trên một bộ dữ liệu mô phỏng', 12)

h2('4.6. Tổng hợp điểm mạnh và điểm yếu')
p('Bảng {r:tab_sw} tổng hợp điểm mạnh và điểm yếu của từng phương pháp, dựa trên kết quả thực nghiệm ở các mục trên.')
table('tab_sw', 'Điểm mạnh và điểm yếu quan sát được của từng phương pháp', [20, 38, 42], ['Phương pháp', 'Điểm mạnh', 'Điểm yếu quan sát được'], [
  ['Robyn', 'Dự báo tốt trên dữ liệu công khai; sai số ROI trung bình thấp; tự động hóa cao', 'ROI các kênh bị ép về gần nhau do DECOMP.RSSD, không xếp đúng thứ tự kênh; không có dự báo ngoài mẫu trong gói Python'],
  ['Meridian', 'Tỷ trọng đóng góp hợp lý; tương quan hạng ROI tốt nhất ở S1; có hậu nghiệm đầy đủ', 'Hấp thụ hiệu ứng gián tiếp vào biến kiểm soát (TV lệch −48% ở S3); cần cấu hình biến xu hướng/mùa vụ cẩn thận'],
  ['PyMC-Marketing', 'Tỷ trọng đóng góp sát thật nhất; dự báo tốt', 'Sai số ROI dao động lớn giữa các hạt giống; cũng bị thiên lệch ở S3'],
  ['LightweightMMM', 'Dự báo tốt nhất trên dữ liệu mô phỏng', 'Quy quá nhiều doanh thu cho truyền thông với tiên nghiệm mặc định; đã ngừng phát triển, phải ghim phiên bản JAX cũ'],
  ['Orbit KTR', 'Hệ số biến thiên theo thời gian; ổn định giữa các hạt giống', 'Không có lưu dư/bão hòa tích hợp; dự báo ngoài mẫu kém; tương quan hạng ROI âm'],
  ['DeepCausalMMM', 'Mã nguồn đầy đủ; chạy được trên dữ liệu bảng lớn; có chế độ NOTEARS', 'Không phù hợp dữ liệu một chuỗi; R² kiểm tra âm; quy quá nhiều cho truyền thông; R² gộp trên dữ liệu bảng không thắng dự báo ngây thơ'],
  ['DeepCausalMMM (NOTEARS)', 'Giảm sai số ROI khoảng một phần tư so với chế độ tam giác trên', 'Vẫn còn các nhược điểm của bản gốc'],
], size=20)

h2('4.7. Hạn chế của thực nghiệm')
ul([
  'Mỗi kịch bản chỉ có 3 hạt giống (đề cương dự kiến 10) do giới hạn tính toán; các kết luận cần được xác nhận với nhiều hạt giống hơn.',
  'Số mẫu MCMC của các mô hình Bayes đã được giảm. Trên dữ liệu mô phỏng, chỉ số R̂ lớn nhất của Meridian và PyMC-Marketing nằm trong khoảng 1,01–1,04, chưa đạt hẳn khuyến nghị 1,01; trên dữ liệu công khai, PyMC-Marketing có R̂ = 1,29, tức là chưa hội tụ, nên kết quả của mô hình này trên dữ liệu công khai cần được xem là tham khảo.',
  'Các phương pháp chủ yếu được chạy với cấu hình mặc định, chưa tinh chỉnh tiên nghiệm hay siêu tham số; kết quả phản ánh trải nghiệm của người dùng mặc định, không phải giới hạn trên của từng phương pháp.',
  'Gói Robyn cho Python (robynpy) đang ở giai đoạn beta; kết quả có thể khác bản R.',
  'Dữ liệu mô phỏng chỉ ở cấp quốc gia; chưa có kịch bản dữ liệu bảng nhiều vùng, nơi Meridian và DeepCausalMMM có thể có lợi thế.',
])

# ================= 5. ĐỀ XUẤT CẢI TIẾN =================
h1('5. ĐỀ XUẤT CẢI TIẾN VÀ KẾ HOẠCH TIẾP THEO', 5)
h2('5.1. Đề xuất cải tiến')
p('Từ các điểm yếu quan sát được, em đề xuất bốn cải tiến cho mô hình CausalMixFormer. Do DeepCausalMMM đã có NOTEARS, việc dùng NOTEARS đơn thuần không còn là đóng góp mới; trọng tâm cải tiến chuyển sang những vấn đề mà thực nghiệm cho thấy chưa phương pháp nào giải quyết. Bảng {r:tab_imp} liên kết từng cải tiến với bằng chứng thực nghiệm và cách kiểm chứng.')
table('tab_imp', 'Các cải tiến đề xuất, bằng chứng thực nghiệm và cách kiểm chứng', [28, 38, 34], ['Cải tiến đề xuất', 'Bằng chứng từ thực nghiệm', 'Cách kiểm chứng ở bước tiếp theo'], [
  ['CT1. Mô hình hóa tường minh biến trung gian (ví dụ lượng tìm kiếm thương hiệu) như một nút trong đồ thị, thay vì biến kiểm soát; ROI = trực tiếp + gián tiếp', 'Ở S3, Meridian, Robyn, PyMC-Marketing và Orbit KTR ước lượng thấp ROI của TV 35–48%, sát mức "chỉ phần trực tiếp" (mục 4.2)', 'Sai số ROI TV/Video ở S3; so sánh có và không có nút trung gian'],
  ['CT2. Đầu ra cộng tính có ràng buộc tỷ trọng đóng góp và kiểm soát nền, áp dụng cho mô hình học sâu', 'LightweightMMM và DeepCausalMMM quy 40–64% doanh thu cho truyền thông so với 15% thật (mục 4.1)', 'Tỷ trọng đóng góp và ROI-MAPE ở S1–S3'],
  ['CT3. Chọn mô hình và báo cáo dựa trên khả năng khôi phục ROI và độ ổn định, kèm khoảng bất định (MC Dropout + dự báo phù hợp)', 'Sai số dự báo không phản ánh độ đúng của ROI; ROI-MAPE dao động lớn giữa hạt giống; Robyn ép ROI về gần nhau', 'Tương quan hạng ROI, độ bao phủ khoảng ROI trên dữ liệu mô phỏng'],
  ['CT4. Lưu dư phi tham số bằng chú ý thời gian, có điều chuẩn độ trơn, phù hợp chuỗi ngắn một vùng', 'Ở S2, độ lệch ROI TV của Meridian tăng từ +39% lên +62%, của Orbit KTR từ −39% lên −74%; DeepCausalMMM (GRU) không phù hợp dữ liệu một chuỗi', 'ROI-MAPE kênh TV/Video ở S2; nghiên cứu cắt bỏ thay chú ý bằng adstock hình học'],
], size=20)
p('Các cải tiến này nhất quán với đề cương: CT1 tương ứng Khối 2 và Khối 6, CT2 tương ứng nguyên tắc cộng tính theo cấu trúc, CT3 tương ứng module định lượng bất định, và CT4 tương ứng Khối 3 của CausalMixFormer. Điểm thay đổi so với đề cương là tính mới không nằm ở NOTEARS mà ở việc kết hợp biến trung gian tường minh, đầu ra cộng tính và đánh giá dưới sai chỉ định.')
h2('5.2. Kế hoạch tiếp theo')
ul([
  '**Bổ sung thực nghiệm nền:** tăng lên 10 hạt giống cho mỗi kịch bản; thêm kịch bản dữ liệu bảng nhiều vùng; thêm kịch bản hiệu ứng biến thiên theo thời gian.',
  '**Hiện thực cải tiến:** xây dựng CausalMixFormer theo bốn cải tiến trên bằng PyTorch, dùng chung bộ dữ liệu và script đánh giá của báo cáo này để bảo đảm so sánh công bằng.',
  '**Đánh giá, đối chiếu:** so sánh với 7 cấu hình đã hiện thực; thực hiện nghiên cứu cắt bỏ từng cải tiến; kiểm định thống kê (Wilcoxon, hiệu chỉnh Holm); đối chiếu với kết quả các bài báo đã công bố, có lưu ý về chỉ số R² gộp như mục 4.4.',
])
p('Mã nguồn, dữ liệu mô phỏng và toàn bộ kết quả của từng lần chạy được gửi kèm báo cáo này để Thầy có thể kiểm tra lại.')

open('../report/doc/r_content.js', 'w').write('module.exports = [\n' + ',\n'.join(B) + '\n];\n')
print('blocks', len(B))
