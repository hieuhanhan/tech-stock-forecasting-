#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np


# CONFIG
BASE_DIR = 'data'
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'processed_folds')
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def generate_folds(data_df_scaled, train_window_size, val_window_size, step_size):
    all_folds_summary = []
    global_fold_counter = 0

    model_dirs = {
        'arima': {'train': os.path.join(OUTPUT_BASE_DIR, 'arima', 'train'),
                  'val': os.path.join(OUTPUT_BASE_DIR, 'arima', 'val')},
        'meta_dir_path': os.path.join(OUTPUT_BASE_DIR, 'arima_meta')
    }

    os.makedirs(model_dirs['arima']['train'], exist_ok=True)
    os.makedirs(model_dirs['arima']['val'],   exist_ok=True)
    os.makedirs(model_dirs['meta_dir_path'],  exist_ok=True)

    # Define feature columns for meta
    arima_cols = ['Date', 'Close_raw', 'Log_Returns', 'Volume']
    meta_cols  = ['Date', 'Ticker', 'target', 'Log_Returns', 'Close_raw']

    required_base = set(['Date', 'Ticker', 'Close_raw', 'Log_Returns', 'Volume'])
    missing_core = [c for c in required_base if c not in data_df_scaled.columns]
    if missing_core:
        raise ValueError(f"[ERROR] Missing required columns for ARIMA: {missing_core}")

    if 'target' not in data_df_scaled.columns:
        raise ValueError("[ERROR] Input dataset does not contain 'target'. Please add it upstream before generating folds.")

    for ticker in data_df_scaled['Ticker'].unique():
        ticker_df = (
            data_df_scaled[data_df_scaled['Ticker'] == ticker]
            .sort_values(by='Date')
            .reset_index(drop=True)
        )

        if len(ticker_df) < train_window_size + val_window_size:
            print(f"[WARN] Skipping {ticker} due to insufficient data")
            continue

        tkr_overall_ratio = float(pd.Series(ticker_df.get('target')).mean(skipna=True))
        print(f"[CHECK] {ticker} overall positive ratio: {tkr_overall_ratio:.2%}")

        miss_for_tkr = [c for c in arima_cols if c not in ticker_df.columns]
        if miss_for_tkr:
            print(f"[WARN] {ticker} missing ARIMA columns {miss_for_tkr}, skipping.")
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

            train_out = os.path.join(model_dirs['arima']['train'], f'train_fold_{global_fold_counter}.csv')
            val_out   = os.path.join(model_dirs['arima']['val'],   f'val_fold_{global_fold_counter}.csv')
            meta_out  = os.path.join(model_dirs['meta_dir_path'],  f'val_meta_fold_{global_fold_counter}.csv')

            train_data[arima_cols].to_csv(train_out, index=False)
            val_data[arima_cols].to_csv(val_out, index=False)

            missing_meta_cols = [c for c in meta_cols if c not in val_data.columns]
            safe_meta_cols = [c for c in meta_cols if c in val_data.columns]
            if missing_meta_cols:
                print(f"[WARN] Fold {global_fold_counter} missing meta cols {missing_meta_cols}, exporting available subset {safe_meta_cols}")
            val_data[safe_meta_cols].to_csv(meta_out, index=False)

            train_rel = os.path.relpath(train_out, OUTPUT_BASE_DIR)
            val_rel   = os.path.relpath(val_out,   OUTPUT_BASE_DIR)
            meta_rel  = os.path.relpath(meta_out,  OUTPUT_BASE_DIR)

            if 'target' in val_data.columns:
                val_pos_ratio = float(pd.to_numeric(val_data['target'], errors='coerce').mean(skipna=True))
            else:
                val_pos_ratio = None

            val_vol = float(pd.to_numeric(val_data['Log_Returns'], errors='coerce').std(skipna=True)) \
                        if 'Log_Returns' in val_data.columns else None
            if val_pos_ratio is not None and not pd.isna(val_pos_ratio):
                print(f"  Fold {global_fold_counter}: {ticker} - val positive label ratio: {val_pos_ratio:.2%}")
                if val_pos_ratio < 0.02 or val_pos_ratio > 0.98:
                    print(f"[WARN] Extreme label imbalance on fold {global_fold_counter}: {val_pos_ratio:.2%}")
            else:
                print(f"  Fold {global_fold_counter}: {ticker} - val positive label ratio: N/A (missing target in val)")

            all_folds_summary.append({
                'global_fold_id': global_fold_counter,
                'ticker': ticker,
                'train_path_arima': train_rel,
                'val_path_arima': val_rel,
                'val_meta_path_arima': meta_rel,
                'train_start': train_data['Date'].iloc[0],
                'train_end': train_data['Date'].iloc[-1],
                'val_start': val_data['Date'].iloc[0],
                'val_end': val_data['Date'].iloc[-1],
                'val_pos_ratio': val_pos_ratio,
                'val_vol': val_vol
            })

            global_fold_counter += 1
            folds_created += 1

        print(f"[INFO] {folds_created} folds generated for {ticker}")

    out_summary = os.path.join(OUTPUT_BASE_DIR, 'folds_summary_arima.json')
    with open(out_summary, 'w') as f:
        json.dump(all_folds_summary, f, indent=4)
    print(f"[INFO] Saved ARIMA folds summary -> {out_summary}")
    print(f"--- Total {global_fold_counter} folds generated. ---")
    return all_folds_summary, global_fold_counter

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ARIMA folds & meta files (use existing target)")
    parser.add_argument("--input_csv", type=str, default=os.path.join("data", "scaled", "global", "train_val_scaled.csv"),
        help="Path to the scaled train+val CSV",)
    parser.add_argument("--train_window", type=int, default=252, help="Train window size")
    parser.add_argument("--val_window",   type=int, default=42, help="Validation window size")
    parser.add_argument("--step_size",    type=int, default=21, help="Sliding step size")
    parser.add_argument("--date_col",     type=str, default="Date", help="Date column name")
    parser.add_argument("--ticker_col",   type=str, default="Ticker", help="Ticker column name")

    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"[ERROR] Input CSV not found: {args.input_csv}")

    df_scaled = pd.read_csv(args.input_csv)

    if args.date_col not in df_scaled.columns:
        raise ValueError(f"[ERROR] Missing date column '{args.date_col}' in {args.input_csv}")
    df_scaled[args.date_col] = pd.to_datetime(df_scaled[args.date_col], errors="coerce")
    if df_scaled[args.date_col].isna().all():
        raise ValueError("[ERROR] All dates failed to parse; please check date format.")
    

    if args.date_col != "Date":
        df_scaled = df_scaled.rename(columns={args.date_col: "Date"})
    if args.ticker_col != "Ticker":
        if args.ticker_col not in df_scaled.columns:
            raise ValueError(f"[ERROR] Missing ticker column '{args.ticker_col}' in {args.input_csv}")
        df_scaled = df_scaled.rename(columns={args.ticker_col: "Ticker"})

    generate_folds(
        data_df_scaled=df_scaled,
        train_window_size=args.train_window,
        val_window_size=args.val_window,
        step_size=args.step_size,
    )