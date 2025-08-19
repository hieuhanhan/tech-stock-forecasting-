#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Optional
from meta_feature_utils import MetaBuildConfig, compute_meta_features_for_fold

OUTPUT_BASE_DIR = os.path.join("data", "processed_folds")
ARIMA_META_DIR = os.path.join(OUTPUT_BASE_DIR, "arima_meta")
LSTM_META_DIR = os.path.join(OUTPUT_BASE_DIR, "lstm_meta")

os.makedirs(ARIMA_META_DIR, exist_ok=True)
os.makedirs(LSTM_META_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Build meta-features for LSTM/ARIMA folds (selection-only)")
    parser.add_argument("--model_type", type=str, required=True, choices=["lstm", "arima"], help="Model type")
    parser.add_argument("--folds_summary_path", type=str, required=True, help="Path to folds summary JSON (post-QC)")
    parser.add_argument("--selection_pca_model", type=str, default="", help="Path to selection_pca_model.pkl (LSTM only)")
    parser.add_argument("--selection_scaler_model", type=str, default="", help="Path to selection_scaler_model.pkl (LSTM only, optional)")
    parser.add_argument("--feature_columns_path", type=str, default=os.path.join(OUTPUT_BASE_DIR, "lstm_feature_columns.json"), help="Feature columns JSON for LSTM")
    parser.add_argument("--use_train_and_val", action="store_true", default=True, help="Use TRAIN+VAL for meta (overridden by --val_only)")
    parser.add_argument("--val_only", action="store_true", help="Override: use only VAL for meta")
    parser.add_argument("--out_meta_json", type=str, default="", help="Output JSON path; default derives from input")
    parser.add_argument("--out_meta_csv", type=str, default="", help="Optional CSV output path")

    args = parser.parse_args()

    with open(args.folds_summary_path, "r") as file:
        folds = json.load(file)
    print(f"[INFO] Loaded folds: {len(folds)} from {args.folds_summary_path}")

    # Load feature columns for LSTM PCA (if provided)
    feature_columns: Optional[List[str]] = None
    if args.model_type == "lstm" and args.selection_pca_model and os.path.exists(args.feature_columns_path):
        with open(args.feature_columns_path, "r") as file:
            feature_columns = json.load(file)
        if feature_columns is not None and not isinstance(feature_columns, list):
            raise ValueError("[ERROR] feature_columns_path must contain a JSON list of column names.")

    config = MetaBuildConfig(
        model_type=args.model_type,
        selection_pca_model_path=(args.selection_pca_model if args.selection_pca_model else None),
        selection_scaler_model_path=(args.selection_scaler_model if args.selection_scaler_model else None),
        feature_columns=feature_columns,
        use_train_and_val=(not args.val_only)
    )

    meta_rows: List[Dict] = []
    skipped = 0
    for fold in folds:
        meta = compute_meta_features_for_fold(fold, config)
        if meta is not None:
            meta_rows.append(meta)
        else:
            skipped += 1
    if skipped:
        print(f"[INFO] Skipped {skipped} folds due to missing CSVs or columns.")

    if not meta_rows:
        raise RuntimeError("[ERROR] No meta-features computed. Check inputs.")

    meta_dataframe = pd.DataFrame(meta_rows)

    # Derive default output paths
    if not args.out_meta_json:
        base_name = os.path.splitext(os.path.basename(args.folds_summary_path))[0]
        if args.model_type.lower() == "arima":
            os.makedirs(ARIMA_META_DIR, exist_ok=True)
            args.out_meta_json = os.path.join(ARIMA_META_DIR, f"{base_name}_meta_arima.json")
        else:
            os.makedirs(LSTM_META_DIR, exist_ok=True)
            args.out_meta_json = os.path.join(LSTM_META_DIR, f"{base_name}_meta_lstm.json")

    with open(args.out_meta_json, "w") as file:
        json.dump(meta_rows, file, indent=2)
    print(f"[INFO] Saved meta-features JSON -> {args.out_meta_json}")

    if not args.out_meta_csv:
        base_name = os.path.splitext(os.path.basename(args.folds_summary_path))[0]
        if args.model_type.lower() == "arima":
            args.out_meta_csv = os.path.join(ARIMA_META_DIR, f"{base_name}_meta_arima.csv")
        else:
            args.out_meta_csv = os.path.join(LSTM_META_DIR, f"{base_name}_meta_lstm.csv")

    meta_dataframe.to_csv(args.out_meta_csv, index=False)
    print(f"[INFO] Saved meta-features CSV -> {args.out_meta_csv}")

if __name__ == "__main__":
    main()