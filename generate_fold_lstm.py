#!/usr/bin/env python3
import os
import json
import pandas as pd

# CONFIG
BASE_DIR = 'data'
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'processed_folds')
GLOBAL_SCALED_PATH = os.path.join(BASE_DIR, 'scaled', 'global')
PCA_SCALED_PATH = os.path.join(GLOBAL_SCALED_PATH, 'train_val_scaled_pca.csv')
SCALER_OUTPUT_PATH = os.path.join(GLOBAL_SCALED_PATH, 'scalers.pkl')
SCALER_META_PATH = os.path.join(GLOBAL_SCALED_PATH, 'scaler_meta.json')

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(GLOBAL_SCALED_PATH, exist_ok=True)
os.makedirs(os.path.dirname(SCALER_OUTPUT_PATH), exist_ok=True)

TRAIN_WINDOW_SIZE = 252
VAL_WINDOW_SIZE = 42
STEP_SIZE = 21

# GENERATE FOLDS FROM SCALED DATA
def generate_folds(data_df_cleaned, train_window_size, val_window_size, step_size):
    all_folds_summary = []
    global_fold_counter = 0

    model_dirs = {
        'lstm': {'train': os.path.join(OUTPUT_BASE_DIR, 'lstm', 'train'),
                 'val': os.path.join(OUTPUT_BASE_DIR, 'lstm', 'val')},
        'meta_dir_path': os.path.join(OUTPUT_BASE_DIR, 'lstm_meta')
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

            if len(train_data) < train_window_size or len(val_data) < val_window_size:
                continue

            train_data['Date'] = train_data['Date'].astype(str)
            val_data['Date'] = val_data['Date'].astype(str)

            train_file_name_prefix = f'train_fold_{global_fold_counter}'
            val_file_name_prefix = f'val_fold_{global_fold_counter}'
            meta_file_name_prefix = f'val_meta_fold_{global_fold_counter}'

            cols_for_lstm = [c for c in train_data.columns if c not in ['target_log_returns']]

            if train_data[cols_for_lstm].isnull().values.any() or val_data[cols_for_lstm].isnull().values.any():
                print(f"[WARN] NaNs found in LSTM data for fold {global_fold_counter}, skipping.")
                continue

            train_data[cols_for_lstm].to_csv(
                os.path.join(model_dirs['lstm']['train'], f'{train_file_name_prefix}.csv'), index=False)
            val_data[cols_for_lstm].to_csv(
                os.path.join(model_dirs['lstm']['val'], f'{val_file_name_prefix}.csv'), index=False)
            val_data.to_csv(
                os.path.join(model_dirs['meta_dir_path'], f'{meta_file_name_prefix}.csv'), index=False)
            
            val_pos_ratio = val_data['target'].mean()
            print(f"  Fold {global_fold_counter}: {ticker} - val positive label ratio: {val_pos_ratio:.2%}")
            
            all_folds_summary.append({
                'global_fold_id': global_fold_counter,
                'ticker': ticker,
                'train_path_lstm': os.path.join('lstm', 'train', f'{train_file_name_prefix}.csv'),
                'val_path_lstm': os.path.join('lstm', 'val', f'{val_file_name_prefix}.csv'),
                'val_meta_path_lstm': os.path.join('lstm_meta', f'{meta_file_name_prefix}.csv')
            })
            global_fold_counter += 1
            folds_created += 1

    with open(os.path.join(OUTPUT_BASE_DIR, 'folds_summary_lstm.json'), 'w') as f:
        json.dump(all_folds_summary, f, indent=4)
    print(f"--- Total {global_fold_counter} folds generated. ---")
    return all_folds_summary, global_fold_counter

# MAIN
if __name__ == "__main__":
    pca_scaled_df = pd.read_csv(PCA_SCALED_PATH)
    pca_scaled_df['Date'] = pd.to_datetime(pca_scaled_df['Date'])

    threshold_meta_path = 'data/cleaned/target_threshold.json'
    with open(threshold_meta_path, 'r') as f:
        threshold_meta = json.load(f)
    threshold = threshold_meta['threshold']
    print(f"[INFO] Loaded threshold from preprocessing: {threshold:.4f} (quantile: {threshold_meta['quantile']})")

    generate_folds(pca_scaled_df, TRAIN_WINDOW_SIZE, VAL_WINDOW_SIZE, STEP_SIZE)