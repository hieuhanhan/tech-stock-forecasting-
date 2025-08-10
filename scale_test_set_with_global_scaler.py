import os
import pandas as pd
from joblib import load
import numpy as np

GLOBAL_SCALED_PATH = 'data/scaled/global'
SCALER_OUTPUT_PATH = os.path.join(GLOBAL_SCALED_PATH, 'scalers.pkl')
TEST_IN_PATH  = 'data/cleaned/test_set_with_features.csv'
TEST_OUT_PATH = os.path.join(GLOBAL_SCALED_PATH, 'test_set_scaled.csv')

os.makedirs(GLOBAL_SCALED_PATH, exist_ok=True)

test_df = pd.read_csv(TEST_IN_PATH)
if 'Date' in test_df.columns:
    test_df['Date'] = pd.to_datetime(test_df['Date'], errors='coerce')

# Load saved scalers and metadata
scaler_bundle = load(SCALER_OUTPUT_PATH)
mm = scaler_bundle.get('minmax')
ss = scaler_bundle.get('standard')
ohlcv_cols = scaler_bundle.get('ohlcv_cols', [])
ti_cols    = scaler_bundle.get('ti_cols', [])
feature_cols = scaler_bundle.get('feature_cols', [])
non_feature_cols = scaler_bundle.get('non_feature_cols', [])
column_order = scaler_bundle.get('column_order', None)

missing_feats = [c for c in feature_cols if c not in test_df.columns]
for c in missing_feats:
    test_df[c] = np.nan

test_df[feature_cols] = test_df[feature_cols].apply(pd.to_numeric, errors='coerce')
if not np.isfinite(test_df[feature_cols].to_numpy()).all():
    raise ValueError("[ERROR] Found inf/-inf in feature columns before scaling test set. Clean them first.")

test_df[feature_cols] = (test_df[feature_cols].ffill().bfill())
test_df = test_df.astype({col: 'float64' for col in ti_cols + ohlcv_cols})

# Apply global scalers
if mm is not None and ohlcv_cols:
    for c in ohlcv_cols:
        if c not in test_df.columns: 
            test_df[c] = 0.0
    test_df[ohlcv_cols] = test_df[ohlcv_cols].astype(float)
    test_df.loc[:, ohlcv_cols] = mm.transform(test_df[ohlcv_cols].to_numpy())

if ss is not None and ti_cols:
    for c in ti_cols:
        if c not in test_df.columns: 
            test_df[c] = 0.0
    test_df[ti_cols] = test_df[ti_cols].astype(float)
    test_df.loc[:, ti_cols] = ss.transform(test_df[ti_cols].to_numpy())

if column_order is not None:
    for c in column_order:
        if c not in test_df.columns:
            test_df[c] = pd.NA
    test_df = test_df.reindex(columns=column_order)

# Save scaled test set
test_df.to_csv(TEST_OUT_PATH, index=False)
print(f"[INFO] Scaled test set saved -> {TEST_OUT_PATH}")