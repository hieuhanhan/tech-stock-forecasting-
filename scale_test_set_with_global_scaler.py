import os
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load test data
test_df = pd.read_csv('data/cleaned/test_set_with_features.csv')
test_df['Date'] = pd.to_datetime(test_df['Date'])

# Load scalers fitted on train_val
mm = load('data/scalers/global_mm.pkl')
ss = load('data/scalers/global_ss.pkl')

# Identify columns
cols_to_drop = ['Date', 'Ticker', 'target_log_returns', 'target']
feature_cols = [c for c in test_df.columns if c not in cols_to_drop]

ohlcv_cols = [c for c in feature_cols if c.startswith('Transformed_')]
ti_cols = [c for c in feature_cols if c not in ohlcv_cols]

# Apply global scalers
if ohlcv_cols:
    test_df[ohlcv_cols] = mm.transform(test_df[ohlcv_cols])
if ti_cols:
    test_df[ti_cols] = ss.transform(test_df[ti_cols])

# Save scaled test set
out_path = 'data/global_scaled/test_set_scaled.csv'
test_df.to_csv(out_path, index=False)
print(f"[INFO] Scaled test set saved -> {out_path}")