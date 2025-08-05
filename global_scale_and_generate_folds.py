#!/usr/bin/env python3
import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import dump

# ===================================================================
# 1. CONFIG
# ===================================================================
BASE_DIR = 'data'
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'processed_folds')
GLOBAL_SCALED_PATH = os.path.join(BASE_DIR, 'scaled', 'global')
SCALER_OUTPUT_PATH = os.path.join(GLOBAL_SCALED_PATH, 'scalers.pkl')
SCALER_META_PATH = os.path.join(GLOBAL_SCALED_PATH, 'scaler_meta.json')

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(GLOBAL_SCALED_PATH, exist_ok=True)
os.makedirs(os.path.dirname(SCALER_OUTPUT_PATH), exist_ok=True)

TRAIN_WINDOW_SIZE = 252
VAL_WINDOW_SIZE = 42
STEP_SIZE = 21

# ===================================================================
# 2. GLOBAL SCALE BEFORE FOLDS
# ===================================================================
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

# ===================================================================
# 3. GENERATE FOLDS FROM SCALED DATA
# ===================================================================
def generate_folds(data_df_cleaned, train_window_size, val_window_size, step_size):
    all_folds_summary = []
    global_fold_counter = 0

    model_dirs = {
        'arima': {'train': os.path.join(OUTPUT_BASE_DIR, 'arima', 'train'),
                  'val': os.path.join(OUTPUT_BASE_DIR, 'arima', 'val')},
        'lstm': {'train': os.path.join(OUTPUT_BASE_DIR, 'lstm', 'train'),
                 'val': os.path.join(OUTPUT_BASE_DIR, 'lstm', 'val')},
        'shared_meta_dir_path': os.path.join(OUTPUT_BASE_DIR, 'shared_meta')
    }

    for k, v in model_dirs.items():
        if k != 'shared_meta_dir_path':
            os.makedirs(v['train'], exist_ok=True)
            os.makedirs(v['val'], exist_ok=True)
    os.makedirs(model_dirs['shared_meta_dir_path'], exist_ok=True)

    for ticker in data_df_cleaned['Ticker'].unique():
        ticker_df = data_df_cleaned[data_df_cleaned['Ticker'] == ticker].sort_values(by='Date').reset_index(drop=True)
        if len(ticker_df) < train_window_size + val_window_size:
            print(f"[WARN] Skipping {ticker} due to insufficient data")
            continue

        folds_created = 0
        max_start_idx = len(ticker_df) - (train_window_size + val_window_size)
        for fold_start_idx in range(0, max_start_idx + 1, step_size):
            train_end_idx = fold_start_idx + train_window_size
            val_start_idx = train_end_idx
            val_end_idx = val_start_idx + val_window_size

            train_data = ticker_df.iloc[fold_start_idx:train_end_idx].copy()
            val_data = ticker_df.iloc[val_start_idx:val_end_idx].copy()

            if len(train_data) < train_window_size or len(val_data) < val_window_size:
                continue

            train_data['Date'] = train_data['Date'].astype(str)
            val_data['Date'] = val_data['Date'].astype(str)

            train_file_name_prefix = f'train_fold_{global_fold_counter}'
            val_file_name_prefix = f'val_fold_{global_fold_counter}'
            meta_file_name_prefix = f'val_meta_fold_{global_fold_counter}'

            cols_for_arima = ['Date', 'Close', 'Log_Returns', 'Volume']
            train_data[cols_for_arima].to_csv(
                os.path.join(model_dirs['arima']['train'], f'{train_file_name_prefix}.csv'), index=False)
            val_data[cols_for_arima].to_csv(
                os.path.join(model_dirs['arima']['val'], f'{val_file_name_prefix}.csv'), index=False)

            cols_for_lstm = [c for c in train_data.columns if c not in ['target_log_returns']]
            train_data[cols_for_lstm].to_csv(
                os.path.join(model_dirs['lstm']['train'], f'{train_file_name_prefix}.csv'), index=False)
            val_data[cols_for_lstm].to_csv(
                os.path.join(model_dirs['lstm']['val'], f'{val_file_name_prefix}.csv'), index=False)

            val_data[['Date', 'Close', 'Ticker', 'Log_Returns', 'target_log_returns', 'target']].to_csv(
                os.path.join(model_dirs['shared_meta_dir_path'], f'{meta_file_name_prefix}.csv'), index=False)
            
            val_pos_ratio = val_data['target'].mean()
            print(f"  Fold {global_fold_counter}: {ticker} - val positive label ratio: {val_pos_ratio:.2%}")
            
            all_folds_summary.append({
                'global_fold_id': global_fold_counter,
                'ticker': ticker,
                'train_path_arima': os.path.join('arima', 'train', f'{train_file_name_prefix}.csv'),
                'val_path_arima': os.path.join('arima', 'val', f'{val_file_name_prefix}.csv'),
                'train_path_lstm': os.path.join('lstm', 'train', f'{train_file_name_prefix}.csv'),
                'val_path_lstm': os.path.join('lstm', 'val', f'{val_file_name_prefix}.csv')
            })
            global_fold_counter += 1
            folds_created += 1

        print(f"[INFO] {folds_created} folds generated for {ticker}")

    with open(os.path.join(OUTPUT_BASE_DIR, 'folds_summary_global.json'), 'w') as f:
        json.dump(all_folds_summary, f, indent=4)
    print(f"--- Total {global_fold_counter} folds generated. ---")
    return all_folds_summary, global_fold_counter

# ===================================================================
# 4. MAIN
# ===================================================================
if __name__ == "__main__":
    train_val_df = pd.read_csv('data/cleaned/train_val_for_wf_with_features.csv')
    train_val_df['Date'] = pd.to_datetime(train_val_df['Date'])

    threshold_meta_path = 'data/cleaned/target_threshold.json'
    with open(threshold_meta_path, 'r') as f:
        threshold_meta = json.load(f)
    threshold = threshold_meta['threshold']
    print(f"[INFO] Loaded threshold from preprocessing: {threshold:.4f} (quantile: {threshold_meta['quantile']})")

    scaled_df = global_scale_data(train_val_df)
    generate_folds(scaled_df, TRAIN_WINDOW_SIZE, VAL_WINDOW_SIZE, STEP_SIZE)

    

    