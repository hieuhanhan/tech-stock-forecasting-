#!/usr/bin/env python3
import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import shutil

# ===================================================================
# 1. CONFIG
# ===================================================================
GLOBAL_FOLDS_DIR = "data/processed_folds"
REPRESENTATIVE_ARIMA = os.path.join(GLOBAL_FOLDS_DIR, "arima", "arima_tuning_folds.json")
REPRESENTATIVE_LSTM = os.path.join(GLOBAL_FOLDS_DIR, "lstm", "lstm_tuning_folds.json")
OUTPUT_RESCALED = "data/scaled_folds"

os.makedirs(OUTPUT_RESCALED, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_RESCALED, "arima", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_RESCALED, "arima", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_RESCALED, "lstm", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_RESCALED, "lstm", "val"), exist_ok=True)

# ===================================================================
# 2. FOLD-LEVEL SCALING
# ===================================================================
def fold_level_scale(train_df, val_df):
    """
    Fit scaler on train_df and transform both train & val.
    Different scalers for OHLCV-like and TI-like features.
    """
    cols_to_drop = ['Date', 'Ticker', 'target_log_returns', 'target']
    feature_cols = [c for c in train_df.columns if c not in cols_to_drop]

    ohlcv_cols = [c for c in feature_cols if c.startswith('Transformed_')]
    ti_cols = [c for c in feature_cols if c not in ohlcv_cols]

    train_scaled, val_scaled = train_df.copy(), val_df.copy()

    if ohlcv_cols:
        mm = MinMaxScaler()
        mm.fit(train_df[ohlcv_cols])
        train_scaled[ohlcv_cols] = mm.transform(train_df[ohlcv_cols])
        val_scaled[ohlcv_cols] = mm.transform(val_df[ohlcv_cols])

    if ti_cols:
        ss = StandardScaler()
        ss.fit(train_df[ti_cols])
        train_scaled[ti_cols] = ss.transform(train_df[ti_cols])
        val_scaled[ti_cols] = ss.transform(val_df[ti_cols])

    return train_scaled, val_scaled

# ===================================================================
# 3. PROCESS REPRESENTATIVE FOLDS
# ===================================================================
def process_representatives(rep_path, model_name):
    with open(rep_path, 'r') as f:
        reps = json.load(f)

    rescaled_summary = []

    for r in reps:
        fid = r['fold_id']
        fold_summary_path = os.path.join(GLOBAL_FOLDS_DIR, f"{model_name}/train/train_fold_{fid}.csv")
        fold_val_path = os.path.join(GLOBAL_FOLDS_DIR, f"{model_name}/val/val_fold_{fid}.csv")

        if not os.path.exists(fold_summary_path) or not os.path.exists(fold_val_path):
            print(f"[WARN] Missing fold {fid} for {model_name}, skipping...")
            continue

        train_df = pd.read_csv(fold_summary_path, parse_dates=['Date'])
        val_df = pd.read_csv(fold_val_path, parse_dates=['Date'])

        # Fold-level scaling
        train_scaled, val_scaled = fold_level_scale(train_df, val_df)

        # Save scaled folds
        train_out = os.path.join(OUTPUT_RESCALED, model_name, "train", f"train_fold_{fid}.csv")
        val_out = os.path.join(OUTPUT_RESCALED, model_name, "val", f"val_fold_{fid}.csv")
        train_scaled.to_csv(train_out, index=False)
        val_scaled.to_csv(val_out, index=False)

        rescaled_summary.append({
            "fold_id": fid,
            "ticker": r.get("ticker", "Unknown"),
            "train_path": os.path.relpath(train_out, OUTPUT_RESCALED),
            "val_path": os.path.relpath(val_out, OUTPUT_RESCALED)
        })

    return rescaled_summary

# ===================================================================
# 4. MAIN
# ===================================================================
if __name__ == "__main__":
    print("[INFO] Processing ARIMA representative folds...")
    arima_rescaled = process_representatives(REPRESENTATIVE_ARIMA, "arima")

    print("[INFO] Processing LSTM representative folds...")
    lstm_rescaled = process_representatives(REPRESENTATIVE_LSTM, "lstm")

    # Save combined summary
    final_summary = {
        "arima": arima_rescaled,
        "lstm": lstm_rescaled
    }
    summary_path = os.path.join(OUTPUT_RESCALED, "folds_summary_rescaled.json")
    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=4)
    # Copy representative folds JSONs
    shutil.copy('data/processed_folds/arima/arima_tuning_folds.json', 'data/scaled_folds/arima/arima_tuning_folds.json')
    shutil.copy('data/processed_folds/lstm/lstm_tuning_folds.json', 'data/scaled_folds/lstm/lstm_tuning_folds.json')

    print(f"[DONE] Rescaled fold summary saved -> {summary_path}")