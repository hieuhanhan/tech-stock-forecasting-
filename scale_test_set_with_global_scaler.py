import os
import pandas as pd
from joblib import load

# Load test data
test_df = pd.read_csv('data/cleaned/test_set_with_features.csv')
test_df['Date'] = pd.to_datetime(test_df['Date'])

# Load saved scalers and metadata
scaler_bundle = load('data/scalers/global_scalers.pkl')
mm = scaler_bundle['minmax']
ss = scaler_bundle['standard']
ohlcv_cols = scaler_bundle['ohlcv_cols']
ti_cols = scaler_bundle['ti_cols']

# Apply global scalers
if mm is not None and ohlcv_cols:
    test_df[ohlcv_cols] = mm.transform(test_df[ohlcv_cols])

if ss is not None and ti_cols:
    test_df[ti_cols] = ss.transform(test_df[ti_cols])

# Save scaled test set
out_path = 'data/global_scaled/test_set_scaled.csv'
test_df.to_csv(out_path, index=False)
print(f"[INFO] Scaled test set saved -> {out_path}")