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
    df = df.copy()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    non_feature_cols = ['Date', 'Close_raw', 'Ticker', 'target_log_returns', 'target', 'Log_Returns']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in non_feature_cols]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    if not np.isfinite(df[feature_cols].to_numpy()).all():
        raise ValueError("[ERROR] Found inf/-inf in feature columns before scaling. Clean them first (clip or fill).")
    if df[feature_cols].isnull().any().any():
        raise ValueError("[ERROR] NaNs found in feature columns before scaling. Clean input first.")

    ohlcv_cols = [c for c in feature_cols if c.startswith('Transformed_')]
    ti_cols = [c for c in feature_cols if c not in ohlcv_cols]

    df_scaled = df.copy()

    if ohlcv_cols:
        mm = MinMaxScaler()
        ohlcv_vals = df_scaled[ohlcv_cols].astype(np.float64).to_numpy()
        ohlcv_scaled = mm.fit_transform(ohlcv_vals)
        df_scaled.loc[:, ohlcv_cols] = pd.DataFrame(ohlcv_scaled, columns=ohlcv_cols, index=df_scaled.index)

    if ti_cols:
        ss = StandardScaler(with_mean=True, with_std=True)
        ti_vals = df_scaled[ti_cols].astype(np.float64).to_numpy()
        ti_scaled = ss.fit_transform(ti_vals)
        df_scaled.loc[:, ti_cols] = pd.DataFrame(ti_scaled, columns=ti_cols, index=df_scaled.index)

    if df_scaled[feature_cols].isnull().any().any():
        raise ValueError("[ERROR] NaNs created after scaling — check your input or scaler settings.")
    if not np.isfinite(df_scaled[feature_cols].to_numpy()).all():
        raise ValueError("[ERROR] Non-finite values detected after scaling.")

    out_path = os.path.join(GLOBAL_SCALED_PATH, "train_val_scaled.csv")
    df_scaled.to_csv(out_path, index=False)
    print(f"[INFO] Global scaled dataset saved -> {out_path} (rows={len(df_scaled)}, features={len(feature_cols)})")

    dump({
        'minmax': mm,
        'standard': ss,
        'ohlcv_cols': ohlcv_cols,
        'ti_cols': ti_cols,
        'feature_cols': feature_cols,
        'non_feature_cols': non_feature_cols,
        'column_order': df_scaled.columns.tolist(),
    }, SCALER_OUTPUT_PATH)
    print(f"[INFO] Global scalers saved -> {SCALER_OUTPUT_PATH}")

    return df_scaled

# MAIN
if __name__ == "__main__":
    train_val_df = pd.read_csv('data/cleaned/train_val_for_wf_with_features.csv')
    train_val_df['Date'] = pd.to_datetime(train_val_df['Date'])

    threshold_meta_path = 'data/cleaned/target_threshold.json'
    with open(threshold_meta_path, 'r') as f:
        threshold_meta = json.load(f)
    threshold = threshold_meta['threshold']
    print(f"[INFO] Loaded threshold from preprocessing: {threshold:.4f} (quantile: {threshold_meta['quantile']})")

    scaled_df = global_scale_data(train_val_df)