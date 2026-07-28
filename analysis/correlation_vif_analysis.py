"""
Correlation Matrix and VIF Analysis for MMM Comparative Study
Paper: Analyzing the Differences Between Google Meridian and Meta Robyn in MMM
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# === GEO-LEVEL CORRELATION ANALYSIS ===
print("=" * 60)
print("GEO-LEVEL PEARSON CORRELATION MATRIX (n=1,248)")
print("=" * 60)

df_geo = pd.read_csv("data/brandA_geo_all.csv")
media_control_cols = [
    'TV_Impressions', 'Digital_Impressions', 'Print_Readership',
    'Radio_Listenership', 'Trade_Spend', 'CPI',
    'Festival_Index', 'Rainfall_Index'
]
corr_geo = df_geo[media_control_cols].corr()
print(corr_geo.round(3).to_string())

# === NATIONAL-LEVEL VIF ANALYSIS ===
print("\n" + "=" * 60)
print("NATIONAL-LEVEL VIF ANALYSIS (n=156)")
print("=" * 60)

df_nat = pd.read_csv("data/brandA_national_all.csv")
X = df_nat[media_control_cols]
X_standardized = (X - X.mean()) / X.std()

vif_data = pd.DataFrame()
vif_data['Variable'] = media_control_cols
vif_data['VIF'] = [
    variance_inflation_factor(X_standardized.values, i)
    for i in range(X_standardized.shape[1])
]
vif_data = vif_data.sort_values('VIF', ascending=False)
print(vif_data.to_string(index=False))

# === KEY CORRELATIONS MENTIONED IN PAPER ===
print("\n" + "=" * 60)
print("KEY CORRELATIONS (GEO-LEVEL)")
print("=" * 60)
print(f"Digital-TV:       r = {corr_geo.loc['Digital_Impressions', 'TV_Impressions']:.3f}")
print(f"Digital-Print:    r = {corr_geo.loc['Digital_Impressions', 'Print_Readership']:.3f}")
print(f"Digital-Festival: r = {corr_geo.loc['Digital_Impressions', 'Festival_Index']:.3f}")
print(f"Digital-Rainfall: r = {corr_geo.loc['Digital_Impressions', 'Rainfall_Index']:.3f}")
print(f"TV-Print:         r = {corr_geo.loc['TV_Impressions', 'Print_Readership']:.3f}")
print(f"TV-Festival:      r = {corr_geo.loc['TV_Impressions', 'Festival_Index']:.3f}")
