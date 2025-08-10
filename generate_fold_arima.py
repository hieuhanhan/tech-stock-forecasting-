#!/usr/bin/env python3
import os
import json
import pandas as pd

# CONFIG
BASE_DIR = 'data'
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'processed_folds')
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)


TRAIN_WINDOW_SIZE = 252
VAL_WINDOW_SIZE = 42
STEP_SIZE = 21

def generate_folds(data_df_scaled, train_window_size, val_window_size, step_size):
    all_folds_summary = []
    global_fold_counter = 0

    model_dirs = {
        'arima': {'train': os.path.join(OUTPUT_BASE_DIR, 'arima', 'train'),
                  'val': os.path.join(OUTPUT_BASE_DIR, 'arima', 'val')},
        'meta_dir_path': os.path.join(OUTPUT_BASE_DIR, 'arima_meta')
    }

    # Create directories
    for k, v in model_dirs.items():
        if k != 'meta_dir_path':
            os.makedirs(v['train'], exist_ok=True)
            os.makedirs(v['val'], exist_ok=True)
    os.makedirs(model_dirs['meta_dir_path'], exist_ok=True)

    # Define feature columns for meta
    non_feature_cols = ['Date', 'Ticker', 'target_log_returns', 'target', 'Log_Returns', 'Close_raw']
    feature_cols = [c for c in data_df_scaled.columns if c not in non_feature_cols]

    for ticker in data_df_scaled['Ticker'].unique():
        ticker_df = data_df_scaled[data_df_scaled['Ticker'] == ticker].sort_values(by='Date').reset_index(drop=True)
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

            train_data['Date'] = train_data['Date'].astype(str)
            val_data['Date'] = val_data['Date'].astype(str)

            cols_for_arima = ['Date', 'Close_raw', 'Log_Returns', 'Volume']
            missing = [c for c in cols_for_arima if c not in ticker_df.columns]
            if missing:
                print(f"[WARN] {ticker} missing ARIMA columns {missing}, skipping these folds.")
                continue

            train_out = os.path.join(model_dirs['arima']['train'], f'train_fold_{global_fold_counter}.csv')
            val_out   = os.path.join(model_dirs['arima']['val'],   f'val_fold_{global_fold_counter}.csv')
            meta_out  = os.path.join(model_dirs['meta_dir_path'],  f'val_meta_fold_{global_fold_counter}.csv')

            train_data[cols_for_arima].to_csv(train_out, index=False)
            val_data[cols_for_arima].to_csv(val_out, index=False)

            non_feature_cols = ['Date', 'Ticker', 'target_log_returns', 'target', 'Log_Returns', 'Close_raw']
            feature_cols = [c for c in data_df_scaled.columns if c not in non_feature_cols]
            meta_cols = ['Date', 'Ticker', 'target', 'Log_Returns', 'Close_raw'] + feature_cols

            if val_data[feature_cols].isnull().any().any():
                print(f"[WARN] NaNs in features fold {global_fold_counter}, skipping meta export.")
                continue

            val_data[meta_cols].to_csv(meta_out, index=False)

            train_rel = os.path.relpath(train_out, OUTPUT_BASE_DIR)
            val_rel   = os.path.relpath(val_out,   OUTPUT_BASE_DIR)
            meta_rel  = os.path.relpath(meta_out,  OUTPUT_BASE_DIR)

            val_pos_ratio = float(val_data['target'].mean()) if 'target' in val_data.columns else None
            val_vol = float(val_data['Log_Returns'].std()) if 'Log_Returns' in val_data.columns else None
            print(f"  Fold {global_fold_counter}: {ticker} - val positive label ratio: {val_pos_ratio:.2%}")

            all_folds_summary.append({
                'global_fold_id': global_fold_counter,
                'ticker': ticker,
                'train_path_arima': train_rel,
                'val_path_arima': val_rel,
                'val_meta_path_arima': meta_rel,
                'val_pos_ratio': val_pos_ratio,
                'val_vol': val_vol
            })

            global_fold_counter += 1
            folds_created += 1

        print(f"[INFO] {folds_created} folds generated for {ticker}")

    with open(os.path.join(OUTPUT_BASE_DIR, 'folds_summary_arima.json'), 'w') as f:
        json.dump(all_folds_summary, f, indent=4)

    print(f"--- Total {global_fold_counter} folds generated. ---")
    return all_folds_summary, global_fold_counter

if __name__ == "__main__":
    df_scaled = pd.read_csv('data/scaled/global/train_val_scaled.csv')
    df_scaled['Date'] = pd.to_datetime(df_scaled['Date'])

    threshold_meta_path = 'data/cleaned/target_threshold.json'
    with open(threshold_meta_path, 'r') as f:
        threshold_meta = json.load(f)
    threshold = threshold_meta['threshold']
    print(f"[INFO] Loaded threshold: {threshold:.4f} (quantile: {threshold_meta['quantile']})")

    generate_folds(df_scaled, TRAIN_WINDOW_SIZE, VAL_WINDOW_SIZE, STEP_SIZE)