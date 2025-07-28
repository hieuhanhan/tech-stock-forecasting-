#!/usr/bin/env python3
import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import dump

# ===================================================================
# 1. CONFIG
# ===================================================================
OUTPUT_BASE_DIR = 'data/processed_folds'
GLOBAL_SCALED_PATH = 'data/global_scaled'
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(GLOBAL_SCALED_PATH, exist_ok=True)

TRAIN_WINDOW_SIZE = 252
VAL_WINDOW_SIZE = 42
STEP_SIZE = 21

# ===================================================================
# 2. GLOBAL SCALE BEFORE FOLDS
# ===================================================================
def global_scale_data(df):
    print("[INFO] Fitting global scaler on full train_val data...")
    cols_to_drop = ['Date', 'Ticker', 'target_log_returns', 'target']
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    ohlcv_cols = [c for c in feature_cols if c.startswith('Transformed_')]
    ti_cols = [c for c in feature_cols if c not in ohlcv_cols]

    df_scaled = df.copy()
    if ohlcv_cols:
        mm = MinMaxScaler()
        mm.fit(df_scaled[ohlcv_cols])
        df_scaled[ohlcv_cols] = mm.transform(df_scaled[ohlcv_cols])
    if ti_cols:
        ss = StandardScaler()
        ss.fit(df_scaled[ti_cols])
        df_scaled[ti_cols] = ss.transform(df_scaled[ti_cols])

    out_path = os.path.join(GLOBAL_SCALED_PATH, "train_val_scaled.csv")
    df_scaled.to_csv(out_path, index=False)
    print(f"[INFO] Global scaled dataset saved -> {out_path}")
    return df_scaled, mm, ss

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
            continue

        max_start_idx = len(ticker_df) - (train_window_size + val_window_size)
        for fold_start_idx in range(0, max_start_idx + 1, step_size):
            train_end_idx = fold_start_idx + train_window_size
            val_start_idx = train_end_idx
            val_end_idx = val_start_idx + val_window_size

            train_data = ticker_df.iloc[fold_start_idx:train_end_idx].copy()
            val_data = ticker_df.iloc[val_start_idx:val_end_idx].copy()
            if len(train_data) < train_window_size or len(val_data) < val_window_size:
                continue

            # Save ARIMA + LSTM folds
            train_file_name_prefix = f'train_fold_{global_fold_counter}'
            val_file_name_prefix = f'val_fold_{global_fold_counter}'
            meta_file_name_prefix = f'val_meta_fold_{global_fold_counter}'

            train_data.to_csv(os.path.join(model_dirs['lstm']['train'], f'{train_file_name_prefix}.csv'), index=False)
            val_data.to_csv(os.path.join(model_dirs['lstm']['val'], f'{val_file_name_prefix}.csv'), index=False)
            val_data[['Date', 'Close', 'Ticker', 'Log_Returns', 'target_log_returns', 'target']].to_csv(
                os.path.join(model_dirs['shared_meta_dir_path'], f'{meta_file_name_prefix}.csv'), index=False)

            all_folds_summary.append({
                'global_fold_id': global_fold_counter,
                'ticker': ticker,
                'train_path_lstm': os.path.join('lstm', 'train', f'{train_file_name_prefix}.csv'),
                'val_path_lstm': os.path.join('lstm', 'val', f'{val_file_name_prefix}.csv')
            })
            global_fold_counter += 1

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

    # Step 1: Global scale
    scaled_df, mm, ss = global_scale_data(train_val_df)

    # Create directory before saving scalers
    os.makedirs('data/scalers', exist_ok=True)
    dump(mm, 'data/scalers/global_mm.pkl')
    dump(ss, 'data/scalers/global_ss.pkl')
    print("[INFO] Global scalers saved -> data/scalers/global_mm.pkl & global_ss.pkl")

    # Step 2: Generate folds
    generate_folds(scaled_df, TRAIN_WINDOW_SIZE, VAL_WINDOW_SIZE, STEP_SIZE)

    

    