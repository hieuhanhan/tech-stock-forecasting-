import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import dump
import numpy as np
np.random.seed(42)

# CONFIG
BASE_DIR = 'data'
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'processed_folds')
GLOBAL_SCALED_PATH = os.path.join(BASE_DIR, 'scaled', 'global')
SCALER_OUTPUT_PATH = os.path.join(GLOBAL_SCALED_PATH, 'scalers.pkl')

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(GLOBAL_SCALED_PATH, exist_ok=True)

TRAIN_WINDOW_SIZE = 252
VAL_WINDOW_SIZE = 42
STEP_SIZE = 21

# GLOBAL SCALE BEFORE FOLDS
def global_scale_data(df):
    print("[INFO] Fitting global scaler on full train_val data...")
    cols_to_drop = ['Date', 'Ticker', 'target_log_returns', 'target', 'Log_Returns']
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    ohlcv_cols = [c for c in feature_cols if c.startswith('Transformed_')]
    ti_cols = [c for c in feature_cols if c not in ohlcv_cols]

    df_scaled = df.copy()

    if ohlcv_cols:
        if df_scaled[ohlcv_cols].isnull().any().any():
            raise ValueError("[ERROR] NaNs found in OHLCV columns before MinMax scaling")
        mm = MinMaxScaler()
        mm.fit(df_scaled[ohlcv_cols])
        df_scaled[ohlcv_cols] = mm.transform(df_scaled[ohlcv_cols])
    else:
        mm = None

    if ti_cols:
        if df_scaled[ti_cols].isnull().any().any():
            raise ValueError("[ERROR] NaNs found in TI columns before Standard scaling")
        ss = StandardScaler()
        ss.fit(df_scaled[ti_cols])
        df_scaled[ti_cols] = ss.transform(df_scaled[ti_cols])
    else:
        ss = None

    out_path = os.path.join(GLOBAL_SCALED_PATH, "train_val_scaled.csv")
    df_scaled.to_csv(out_path, index=False)
    print(f"[INFO] Global scaled dataset saved -> {out_path}")

    dump({
        'minmax': mm,
        'standard': ss,
        'ohlcv_cols': ohlcv_cols,
        'ti_cols': ti_cols,
        'feature_cols': feature_cols
    }, SCALER_OUTPUT_PATH)
    print(f"[INFO] Global scalers saved -> {SCALER_OUTPUT_PATH}")
    
    return df_scaled

# MAIN
if __name__ == "__main__":
    train_val_df = pd.read_csv('data/cleaned/train_val_for_wf_with_features.csv')
    train_val_df['Date'] = pd.to_datetime(train_val_df['Date'])

    train_val_df['Close_raw'] = train_val_df['Close']

    threshold_meta_path = 'data/cleaned/target_threshold.json'
    with open(threshold_meta_path, 'r') as f:
        threshold_meta = json.load(f)
    threshold = threshold_meta['threshold']
    print(f"[INFO] Loaded threshold from preprocessing: {threshold:.4f} (quantile: {threshold_meta['quantile']})")

    scaled_df = global_scale_data(train_val_df)