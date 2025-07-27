import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ===================================================================
# 1. CONFIG
# ===================================================================
OUTPUT_BASE_DIR = 'data/processed_folds'
SCALED_OUTPUT_DIR = 'data/scaled_folds'
FINAL_OUTPUT_DIR = 'data/final_processed'

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(SCALED_OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

TRAIN_WINDOW_SIZE = 252
VAL_WINDOW_SIZE = 42
STEP_SIZE = 21

# ===================================================================
# 1A. GLOBAL SCALING
# ===================================================================
def global_scale_features(df):
    print("[INFO] Performing global scaling on full dataset...")

    cols_to_drop = ['Date', 'Ticker', 'target_log_returns', 'target']
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    # Split OHLCV vs TI features
    ohlcv_cols = [c for c in feature_cols if c.startswith('Transformed_')]
    ti_cols = [c for c in feature_cols if c not in ohlcv_cols]

    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    if ohlcv_cols:
        df[ohlcv_cols] = min_max_scaler.fit_transform(df[ohlcv_cols])
        print(f"[INFO] MinMax scaled {len(ohlcv_cols)} OHLCV columns.")

    if ti_cols:
        df[ti_cols] = standard_scaler.fit_transform(df[ti_cols])
        print(f"[INFO] Standard scaled {len(ti_cols)} technical indicator columns.")

    return df

# ===================================================================
# 2. GENERATE_FOLDS
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

    for model_type_name, model_type_paths in model_dirs.items():
        if model_type_name != 'shared_meta_dir_path':
            os.makedirs(model_type_paths['train'], exist_ok=True)
            os.makedirs(model_type_paths['val'], exist_ok=True)
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

            # Dates
            train_start_date = pd.to_datetime(train_data['Date']).min().strftime('%Y-%m-%d')
            train_end_date = pd.to_datetime(train_data['Date']).max().strftime('%Y-%m-%d')
            val_start_date = pd.to_datetime(val_data['Date']).min().strftime('%Y-%m-%d')
            val_end_date = pd.to_datetime(val_data['Date']).max().strftime('%Y-%m-%d')

            # Save ARIMA data
            cols_for_arima = ['Date', 'Close', 'Log_Returns', 'Volume']
            train_data['Date'] = train_data['Date'].astype(str)
            val_data['Date'] = val_data['Date'].astype(str)

            train_file_name_prefix = f'train_fold_{global_fold_counter}'
            val_file_name_prefix = f'val_fold_{global_fold_counter}'
            meta_file_name_prefix = f'val_meta_fold_{global_fold_counter}'

            train_data[cols_for_arima].to_csv(os.path.join(model_dirs['arima']['train'], f'{train_file_name_prefix}.csv'), index=False)
            val_data[cols_for_arima].to_csv(os.path.join(model_dirs['arima']['val'], f'{val_file_name_prefix}.csv'), index=False)

            # Save LSTM data
            cols_for_lstm = [c for c in data_df_cleaned.columns if c != 'target_log_returns']
            train_data[cols_for_lstm].to_csv(os.path.join(model_dirs['lstm']['train'], f'{train_file_name_prefix}.csv'), index=False)
            val_data[cols_for_lstm].to_csv(os.path.join(model_dirs['lstm']['val'], f'{val_file_name_prefix}.csv'), index=False)

            # Meta info
            val_meta_fold = val_data[['Date', 'Close', 'Ticker', 'Log_Returns', 'target_log_returns', 'target']]
            val_meta_fold.to_csv(os.path.join(model_dirs['shared_meta_dir_path'], f'{meta_file_name_prefix}.csv'), index=False)

            print(f"    Saved fold {global_fold_counter} for {ticker}: Train={len(train_data)} rows, Val={len(val_data)} rows.")

            all_folds_summary.append({
                'global_fold_id': global_fold_counter,
                'ticker': ticker,
                'train_start_date': train_start_date,
                'train_end_date': train_end_date,
                'val_start_date': val_start_date,
                'val_end_date': val_end_date,
                'num_train_rows': len(train_data),
                'num_val_rows': len(val_data),
                'train_path_arima': os.path.join('arima', 'train', f'{train_file_name_prefix}.csv'),
                'val_path_arima': os.path.join('arima', 'val', f'{val_file_name_prefix}.csv'),
                'train_path_lstm': os.path.join('lstm', 'train', f'{train_file_name_prefix}.csv'),
                'val_path_lstm': os.path.join('lstm', 'val', f'{val_file_name_prefix}.csv'),
                'meta_path': os.path.join('shared_meta_dir_path', f'{meta_file_name_prefix}.csv'),
            })
            global_fold_counter += 1

    with open(os.path.join(OUTPUT_BASE_DIR, 'folds_summary.json'), 'w') as f:
        json.dump(all_folds_summary, f, indent=4)
    print(f"\n--- Total {global_fold_counter} folds generated locally. ---")
    return all_folds_summary, global_fold_counter

# ===================================================================
# 4. SCALE TEST SET
# ===================================================================
def scale_test_set(train_val_wf_df_with_features, test_df_with_features):
    cols_to_drop = ['Date', 'Ticker', 'target_log_returns', 'target']
    feature_cols = [c for c in train_val_wf_df_with_features.columns if c not in cols_to_drop]

    ohlcv_cols = [c for c in feature_cols if c.startswith('Transformed_')]
    ti_cols = [c for c in feature_cols if c not in ohlcv_cols]

    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    if ohlcv_cols:
        min_max_scaler.fit(train_val_wf_df_with_features[ohlcv_cols])
        test_df_with_features[ohlcv_cols] = min_max_scaler.transform(test_df_with_features[ohlcv_cols])

    if ti_cols:
        standard_scaler.fit(train_val_wf_df_with_features[ti_cols])
        test_df_with_features[ti_cols] = standard_scaler.transform(test_df_with_features[ti_cols])

    test_scaled_path = os.path.join(FINAL_OUTPUT_DIR, 'test_set_scaled.csv')
    test_df_with_features.to_csv(test_scaled_path, index=False)
    print(f"[INFO] Test set scaled and saved to {test_scaled_path}")

# ===================================================================
# 5. MAIN
# ===================================================================
if __name__ == "__main__":
    train_val_wf_df_with_features = pd.read_csv('data/cleaned/train_val_for_wf_with_features.csv')
    train_val_wf_df_with_features['Date'] = pd.to_datetime(train_val_wf_df_with_features['Date'])
    test_df_with_features = pd.read_csv('data/cleaned/test_set_with_features.csv')

    # Generate folds
    all_folds_summary, total_folds_count = generate_folds(
        train_val_wf_df_with_features,
        train_window_size=TRAIN_WINDOW_SIZE,
        val_window_size=VAL_WINDOW_SIZE,
        step_size=STEP_SIZE
    )
    print(f"Overall total folds generated: {total_folds_count}")

    # Scale folds
    scale_folds(all_folds_summary)

    # Scale test set
    scale_test_set(train_val_wf_df_with_features, test_df_with_features)