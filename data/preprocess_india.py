"""
preprocess_india.py
Chuẩn bị dữ liệu từ synthetic_mmm_weekly_india_SAT.csv cho MMM
Scope: BrandA × 8 Geo (Hierarchical)
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# 1. ĐỌC DỮ LIỆU GỐC
# ─────────────────────────────────────────────
RAW_PATH = r'd:\MARKETING MIX MODELING\DATASET\synthetic_mmm_weekly_india_SAT.csv'
OUT_DIR  = r'd:\MARKETING MIX MODELING\DATASET'

df = pd.read_csv(RAW_PATH, parse_dates=['Week'])
print(f"[1] Đọc dữ liệu gốc: {df.shape}")

# ─────────────────────────────────────────────
# 2. LỌC BRANDA
# ─────────────────────────────────────────────
df_a = df[df['Brand'] == 'BrandA'].copy()
print(f"[2] Sau lọc BrandA: {df_a.shape}")
print(f"    Weeks: {df_a['Week'].nunique()}, Geos: {df_a['Geo'].nunique()}, SKUs: {df_a['SKU'].nunique()}")

# ─────────────────────────────────────────────
# 3. AGGREGATE THEO WEEK × GEO
#    - Sum: sales, media impressions, trade, promo flags
#    - Mean: macro/contextual (CPI, Festival, Rainfall, GDP, Distribution)
# ─────────────────────────────────────────────
sum_cols  = ['Sales_Units', 'Sales_Value',
             'TV_Impressions', 'YouTube_Impressions',
             'Facebook_Impressions', 'Instagram_Impressions',
             'Print_Readership', 'Radio_Listenership',
             'Trade_Spend',
             'Feature_Flag', 'Display_Flag', 'TPR_Flag']

mean_cols = ['MRP', 'Net_Price',
             'FB_Banner_Content_Score', 'IG_Banner_Content_Score',
             'Weighted_Distribution', 'Numeric_Distribution',
             'TDP', 'NOS',
             'CPI', 'GDP_Growth', 'Festival_Index', 'Rainfall_Index']

agg_dict = {c: 'sum'  for c in sum_cols}
agg_dict.update({c: 'mean' for c in mean_cols})

geo_brand = df_a.groupby(['Week', 'Geo']).agg(agg_dict).reset_index()
geo_brand = geo_brand.sort_values(['Geo', 'Week']).reset_index(drop=True)

print(f"\n[3] Sau aggregate Week × Geo: {geo_brand.shape}")
print(f"    Cột: {list(geo_brand.columns)}")

# Kiểm tra tính đầy đủ (mỗi geo có đủ 156 tuần không?)
counts = geo_brand.groupby('Geo')['Week'].count()
print(f"\n    Số tuần mỗi Geo:\n{counts.to_string()}")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT (80/20 theo thời gian)
# ─────────────────────────────────────────────
all_weeks = sorted(geo_brand['Week'].unique())
n_weeks   = len(all_weeks)                    # 156
n_train   = int(n_weeks * 0.8)               # 124 tuần
cutoff    = all_weeks[n_train - 1]            # ngày cuối của train

train_df = geo_brand[geo_brand['Week'] <= cutoff].copy()
test_df  = geo_brand[geo_brand['Week'] >  cutoff].copy()

print(f"\n[4] Train/Test split:")
print(f"    Train: {train_df['Week'].nunique()} tuần ({train_df['Week'].min().date()} → {train_df['Week'].max().date()})")
print(f"    Test : {test_df['Week'].nunique()}  tuần ({test_df['Week'].min().date()} → {test_df['Week'].max().date()})")
print(f"    Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# ─────────────────────────────────────────────
# 5. NATIONAL AGGREGATE CHO ROBYN
#    (sum tất cả 8 geo → 156 hàng quốc gia)
# ─────────────────────────────────────────────
national_sum_cols  = ['Sales_Units', 'Sales_Value',
                      'TV_Impressions', 'YouTube_Impressions',
                      'Facebook_Impressions', 'Instagram_Impressions',
                      'Print_Readership', 'Radio_Listenership',
                      'Trade_Spend', 'Feature_Flag', 'Display_Flag', 'TPR_Flag']
national_mean_cols = ['CPI', 'GDP_Growth', 'Festival_Index', 'Rainfall_Index',
                      'Weighted_Distribution', 'MRP', 'Net_Price']

nat_agg = {c: 'sum'  for c in national_sum_cols}
nat_agg.update({c: 'mean' for c in national_mean_cols})

national_df = geo_brand.groupby('Week').agg(nat_agg).reset_index()
national_df = national_df.sort_values('Week').reset_index(drop=True)

# Train/test cho national
nat_train = national_df[national_df['Week'] <= cutoff].copy()
nat_test  = national_df[national_df['Week'] >  cutoff].copy()

print(f"\n[5] National aggregate (cho Robyn):")
print(f"    Total shape: {national_df.shape}")
print(f"    Train: {nat_train.shape}, Test: {nat_test.shape}")

# ─────────────────────────────────────────────
# 6. THỐNG KÊ MÔ TẢ ĐẦY ĐỦ
# ─────────────────────────────────────────────
media_cols = ['TV_Impressions', 'YouTube_Impressions',
              'Facebook_Impressions', 'Instagram_Impressions',
              'Print_Readership', 'Radio_Listenership']

print(f"\n[6] Thống kê mô tả – National train set (Robyn input):")
desc_cols = ['Sales_Value'] + media_cols + ['Trade_Spend', 'Festival_Index', 'CPI']
print(nat_train[desc_cols].describe().T[['mean','std','min','max']].round(2).to_string())

print(f"\n    Thống kê Sales_Value theo Geo (geo_brand train):")
geo_stats = train_df.groupby('Geo')['Sales_Value'].agg(['mean','std','min','max']).round(2)
print(geo_stats.to_string())

print(f"\n    Media share (% impressions tổng):")
total_imp = national_df[media_cols].sum()
pct = (total_imp / total_imp.sum() * 100).round(1)
print(pct.to_string())

# ─────────────────────────────────────────────
# 7. LƯU FILE
# ─────────────────────────────────────────────
# File cho Meridian (geo-level)
geo_brand.to_csv(os.path.join(OUT_DIR, 'brandA_geo_all.csv'), index=False)
train_df.to_csv(os.path.join(OUT_DIR, 'brandA_geo_train.csv'), index=False)
test_df.to_csv(os.path.join(OUT_DIR,  'brandA_geo_test.csv'),  index=False)

# File cho Robyn (national)
national_df.to_csv(os.path.join(OUT_DIR, 'brandA_national_all.csv'), index=False)
nat_train.to_csv(os.path.join(OUT_DIR,   'brandA_national_train.csv'), index=False)
nat_test.to_csv(os.path.join(OUT_DIR,    'brandA_national_test.csv'),  index=False)

print(f"\n[7] Đã lưu 6 file CSV vào {OUT_DIR}:")
print("    - brandA_geo_all.csv / train / test   (cho Meridian)")
print("    - brandA_national_all.csv / train / test (cho Robyn)")
print("\n✅ Preprocessing hoàn thành!")
