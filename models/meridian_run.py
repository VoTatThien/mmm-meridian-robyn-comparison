import pandas as pd
import numpy as np
import xarray as xr
from meridian.data import input_data
from meridian.model import model as mmm_model
from meridian.analysis import analyzer
import os

# 1. Load Main Data
df = pd.read_csv('DATASET/Secondfile.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 2. Process SpecialSale.csv to get counts per month
ss = pd.read_csv('DATASET/SpecialSale.csv')
ss['Date'] = pd.to_datetime(ss['Date'], errors='coerce')
ss = ss.dropna(subset=['Date'])
ss['Month_Year'] = ss['Date'].dt.to_period('M')
sale_counts = ss.groupby('Month_Year').size().reset_index(name='Special_Sale_Days')
sale_counts['Month_Year'] = sale_counts['Month_Year'].apply(lambda x: x.to_timestamp())

# Merge
df = df.merge(sale_counts, left_on='Date', right_on='Month_Year', how='left')
df['Special_Sale_Days'] = df['Special_Sale_Days'].fillna(0)

# 3. Prepare Xarray DataArrays
time_coords = df['Date'].dt.strftime('%Y-%m-%d').values
geo_coords = ['National']
media_channels = ['TV', 'Digital', 'Sponsorship', 'Content.Marketing', 'Online.marketing', 'Affiliates', 'SEM']
control_variables = ['NPS', 'Special_Sale_Days']

# KPI
kpi_da = xr.DataArray(
    df['total_gmv'].values.reshape(-1, 1).astype(np.float32).transpose(),
    coords={'geo': geo_coords, 'time': time_coords},
    dims=('geo', 'time'),
    name='kpi'
)

# Population (Dummy for national model)
pop_da = xr.DataArray(
    np.ones((1,), dtype=np.float32),
    coords={'geo': geo_coords},
    dims=('geo'),
    name='population'
)

# Media Spend
media_spend_da = xr.DataArray(
    df[media_channels].fillna(0).values[:, np.newaxis, :].astype(np.float32).transpose(1, 0, 2),
    coords={'geo': geo_coords, 'time': time_coords, 'media_channel': media_channels},
    dims=('geo', 'time', 'media_channel'),
    name='media_spend'
)

# Media (Using spend as proxy for impressions in this demo)
media_da = xr.DataArray(
    df[media_channels].fillna(0).values[:, np.newaxis, :].astype(np.float32).transpose(1, 0, 2),
    coords={'geo': geo_coords, 'media_time': time_coords, 'media_channel': media_channels},
    dims=('geo', 'media_time', 'media_channel'),
    name='media'
)

# Controls
controls_da = xr.DataArray(
    df[control_variables].fillna(0).values[:, np.newaxis, :].astype(np.float32).transpose(1, 0, 2),
    coords={'geo': geo_coords, 'time': time_coords, 'control_variable': control_variables},
    dims=('geo', 'time', 'control_variable'),
    name='controls'
)

print("Initializing InputData...")
data = input_data.InputData(
    kpi=kpi_da,
    kpi_type='revenue',
    population=pop_da,
    controls=controls_da,
    media=media_da,
    media_spend=media_spend_da
)

print("Initializing Meridian Model...")
model = mmm_model.Meridian(input_data=data)

print("Starting Sampling (MCMC)... This may take a moment.")
# Using small numbers for demo purposes
model.sample_posterior(n_chains=2, n_adapt=100, n_burnin=100, n_keep=200)

print("\n--- Model Results Summary ---")
try:
    az = analyzer.Analyzer(model)
    # Get ROI summary
    roi_summary = az.roi()
    print("\nEstimated ROI by Channel (Mean):")
    # Convert to numpy if it's a tensor
    roi_values = roi_summary.numpy() if hasattr(roi_summary, 'numpy') else roi_summary
    # If it was an xarray, handle accordingly
    if isinstance(roi_summary, xr.DataArray):
        print(roi_summary.mean(dim=['chain', 'draw']))
    else:
        # If it's a raw numpy array (chain, draw, channel)
        # Assuming shape is (chain, draw, channel)
        mean_roi = roi_values.mean(axis=(0, 1))
        for i, chan in enumerate(media_channels):
            print(f"{chan}: {mean_roi[i]:.2f}")
    
    # Get Incremental Outcome
    inc_outcome = az.incremental_outcome()
    print("\nIncremental GMV by Channel (Total Sum):")
    
    if isinstance(inc_outcome, xr.DataArray):
        # If it's an xarray, we can sum over dimensions by name
        summed = inc_outcome.sum(dim=[d for d in inc_outcome.dims if d != 'media_channel'])
        mean_inc = summed.mean(dim=['chain', 'draw']) if 'chain' in summed.dims else summed
        for chan in media_channels:
            if chan in mean_inc.coords['media_channel'].values:
                val = float(mean_inc.sel(media_channel=chan))
                print(f"{chan}: {val:,.0f}")
    else:
        # Fallback for other formats
        inc_values = inc_outcome.numpy() if hasattr(inc_outcome, 'numpy') else np.array(inc_outcome)
        print(f"Debug - Incremental outcome shape: {inc_values.shape}")
        # If it's already collapsed to channels
        if inc_values.ndim == 1:
            for i, chan in enumerate(media_channels):
                if i < len(inc_values):
                    print(f"{chan}: {inc_values[i]:,.0f}")
        else:
            # Try to take mean over batches/chains and sum over time/geo
            # Assuming last dimension is channel
            mean_inc = inc_values.mean(axis=tuple(range(inc_values.ndim - 1)))
            for i, chan in enumerate(media_channels):
                if i < len(mean_inc):
                    print(f"{chan}: {mean_inc[i]:,.0f}")
except Exception as e:
    print(f"Error printing summary: {e}")
    import traceback
    traceback.print_exc()

print("\nMeridian run completed successfully!")
