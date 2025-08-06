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

# GENERATE FOLDS FROM SCALED DATA
def generate_folds(data_df_cleaned, train_window_size, val_window_size, step_size):
    all_folds_summary = []
    global_fold_counter = 0

    model_dirs = {
        'arima': {'train': os.path.join(OUTPUT_BASE_DIR, 'arima', 'train'),
                  'val': os.path.join(OUTPUT_BASE_DIR, 'arima', 'val')},
        'meta_dir_path': os.path.join(OUTPUT_BASE_DIR, 'arima_meta')
    }

    for k, v in model_dirs.items():
        if k != 'meta_dir_path':
            os.makedirs(v['train'], exist_ok=True)
            os.makedirs(v['val'], exist_ok=True)
    os.makedirs(model_dirs['meta_dir_path'], exist_ok=True)

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

            train_data['Date'] = train_data['Date'].astype(str)
            val_data['Date'] = val_data['Date'].astype(str)

            train_file_name_prefix = f'train_fold_{global_fold_counter}'
            val_file_name_prefix = f'val_fold_{global_fold_counter}'
            meta_file_name_prefix = f'val_meta_fold_{global_fold_counter}'

            cols_for_arima = ['Date', 'Close_raw', 'Log_Returns', 'Volume']
            missing_arima_cols = [col for col in cols_for_arima if col not in train_data.columns]
                
            if train_data[cols_for_arima].isnull().values.any() or val_data[cols_for_arima].isnull().values.any():
                print(f"[WARN] NaNs found in ARIMA data for fold {global_fold_counter}, skipping.")
                continue

            if missing_arima_cols:
                print(f"[WARN] Missing columns for ARIMA: {missing_arima_cols} – skipping ARIMA export for fold {global_fold_counter}")
            else:
                train_data[cols_for_arima].to_csv(
                    os.path.join(model_dirs['arima']['train'], f'{train_file_name_prefix}.csv'), index=False)
                val_data[cols_for_arima].to_csv(
                    os.path.join(model_dirs['arima']['val'], f'{val_file_name_prefix}.csv'), index=False)
            
            val_data[['Date', 'Close_raw', 'Ticker', 'Log_Returns', 'target_log_returns', 'target']].to_csv(
                os.path.join(model_dirs['meta_dir_path'], f'{meta_file_name_prefix}.csv'), index=False)
            
            val_pos_ratio = val_data['target'].mean()
            print(f"  Fold {global_fold_counter}: {ticker} - val positive label ratio: {val_pos_ratio:.2%}")
            
            all_folds_summary.append({
                'global_fold_id': global_fold_counter,
                'ticker': ticker,
                'train_path_arima': os.path.join('arima', 'train', f'{train_file_name_prefix}.csv'),
                'val_path_arima': os.path.join('arima', 'val', f'{val_file_name_prefix}.csv'),
                'val_meta_path_arima': os.path.join('arima_meta', f'{meta_file_name_prefix}.csv')
                })
            global_fold_counter += 1
            folds_created += 1

        print(f"[INFO] {folds_created} folds generated for {ticker}")

    with open(os.path.join(OUTPUT_BASE_DIR, 'folds_summary_arima.json'), 'w') as f:
        json.dump(all_folds_summary, f, indent=4)
    print(f"--- Total {global_fold_counter} folds generated. ---")
    return all_folds_summary, global_fold_counter

# MAIN
if __name__ == "__main__":
    df = pd.read_csv('data/cleaned/train_val_for_wf_with_features.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    threshold_meta_path = 'data/cleaned/target_threshold.json'
    with open(threshold_meta_path, 'r') as f:
        threshold_meta = json.load(f)
    threshold = threshold_meta['threshold']
    print(f"[INFO] Loaded threshold from preprocessing: {threshold:.4f} (quantile: {threshold_meta['quantile']})")

    generate_folds(df, TRAIN_WINDOW_SIZE, VAL_WINDOW_SIZE, STEP_SIZE)