#!/usr/bin/env python3
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from meta_feature_utils import (
    load_and_filter_val_fold,
    compute_meta_statistics,
    compute_meta_with_pca,
    evaluate_kmeans,
    find_best_k,
    pick_representatives,
    refine_representative_folds
)

# CONFIG & LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def build_meta_arima(folds_summary, arima_val_meta_path):
    logging.info("[ARIMA] Building meta-features...")
    meta_rows, excluded_folds = [], []

    for fold in folds_summary:
        fold_id = fold['global_fold_id']
        val_path = os.path.join(arima_val_meta_path, f"val_meta_fold_{fold_id}.csv")

        df_val, reason = load_and_filter_val_fold(val_path, model_type='arima')
        if df_val is None:
            excluded_folds.append((fold_id, fold['ticker'], reason))
            continue

        meta = compute_meta_statistics(df_val)
        meta.update({'fold_id': fold_id, 'ticker': fold['ticker']})
        meta_rows.append(meta)

    excluded_df = pd.DataFrame(excluded_folds, columns=['fold_id', 'ticker', 'reason'])
    excluded_path = os.path.join(arima_val_meta_path, 'excluded_arima_folds.csv')
    excluded_df.to_csv(excluded_path, index=False)
    logging.info(f"[ARIMA] Excluded {len(excluded_folds)} folds → {excluded_path}")

    return pd.DataFrame(meta_rows)

def build_meta_lstm(folds_summary, lstm_val_meta_path):
    logging.info("[LSTM] Building meta-features...")
    meta_rows, excluded_folds = [], []

    for fold in folds_summary:
        fold_id = fold['global_fold_id']
        val_path = os.path.join(lstm_val_meta_path, f"val_meta_fold_{fold_id}.csv")
        df_val, reason = load_and_filter_val_fold(val_path, model_type='lstm')
        if df_val is None:
            excluded_folds.append((fold_id, fold['ticker'], reason))
            continue

        meta = compute_meta_with_pca(df_val)
        meta.update({'fold_id': fold_id, 'ticker': fold['ticker']})
        meta_rows.append(meta)

    excluded_df = pd.DataFrame(excluded_folds, columns=['fold_id', 'ticker', 'reason'])
    excluded_path = os.path.join(lstm_val_meta_path, 'excluded_lstm_folds.csv')
    excluded_df.to_csv(excluded_path, index=False)
    logging.info(f"[LSTM] Excluded {len(excluded_folds)} folds → {excluded_path}")
    return pd.DataFrame(meta_rows)

def main(arima_folds_path, lstm_folds_path, k_arima, k_lstm, evaluate_k, auto_k, n_per_cluster_arima, n_per_cluster_lstm, output_dir):
    arima_val_meta_path = os.path.join(output_dir, 'arima_meta')
    lstm_val_meta_path = os.path.join(output_dir, 'lstm_meta')
    os.makedirs(arima_val_meta_path, exist_ok=True)
    os.makedirs(lstm_val_meta_path, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'arima'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'lstm'), exist_ok=True)

    with open(arima_folds_path, 'r') as f:
        arima_folds_summary = json.load(f)
    with open(lstm_folds_path, 'r') as f:
        lstm_folds_summary = json.load(f)

    meta_arima = build_meta_arima(arima_folds_summary, arima_val_meta_path)
    meta_lstm = build_meta_lstm(lstm_folds_summary, lstm_val_meta_path)

    if evaluate_k:
        best_k_dict = {
            "arima": find_best_k(meta_arima, "ARIMA"),
            "lstm": find_best_k(meta_lstm, "LSTM")
        }
        best_k_path = os.path.join(output_dir, "best_k.json")
        with open(best_k_path, "w") as f:
            json.dump(best_k_dict, f, indent=4)
        logging.info(f"Saved best k values to {best_k_path}")
        return

    if auto_k:
        k_arima = find_best_k(meta_arima, "ARIMA")
        k_lstm = find_best_k(meta_lstm, "LSTM")

    arima_reps, meta_arima = pick_representatives(meta_arima, k_arima, "ARIMA", n_per_cluster_arima)
    lstm_reps, meta_lstm = pick_representatives(meta_lstm, k_lstm, "LSTM", n_per_cluster_lstm)

    arima_reps = refine_representative_folds(arima_reps, meta_arima)
    lstm_reps = refine_representative_folds(lstm_reps, meta_lstm)

    arima_path = os.path.join(output_dir, 'arima', 'arima_tuning_folds.json')
    lstm_path = os.path.join(output_dir, 'lstm', 'lstm_tuning_folds.json')
    arima_reps.to_json(arima_path, orient='records', indent=4)
    lstm_reps.to_json(lstm_path, orient='records', indent=4)

    logging.info(f"[ARIMA] {len(arima_reps)} folds selected. Tickers: {sorted(arima_reps['ticker'].unique())}")
    logging.info(f"[LSTM] {len(lstm_reps)} folds selected. Tickers: {sorted(lstm_reps['ticker'].unique())}")

    meta_arima_with_cluster = meta_arima.merge(arima_reps[['fold_id', 'cluster', 'cluster_size']], on='fold_id', how='left')
    meta_lstm_with_cluster = meta_lstm.merge(lstm_reps[['fold_id', 'cluster', 'cluster_size']], on='fold_id', how='left')

    meta_arima_with_cluster.to_csv(os.path.join(arima_val_meta_path, 'meta_arima.csv'), index=False)
    meta_lstm_with_cluster.to_csv(os.path.join(lstm_val_meta_path, 'meta_lstm.csv'), index=False)

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select representative folds for ARIMA and LSTM.")
    parser.add_argument("--output_dir", type=str, default="data/processed_folds")
    parser.add_argument("--arima_folds_path", type=str, required=True, help="Path to ARIMA fold summary JSON")
    parser.add_argument("--lstm_folds_path", type=str, required=True, help="Path to LSTM fold summary JSON")
    parser.add_argument("--k_arima", type=int, default=15, help="Number of clusters for ARIMA")
    parser.add_argument("--k_lstm", type=int, default=10, help="Number of clusters for LSTM")
    parser.add_argument("--n_per_cluster_arima", type=int, default=1, help="Number of folds to pick per ARIMA cluster")
    parser.add_argument("--n_per_cluster_lstm", type=int, default=1, help="Number of folds to pick per LSTM cluster")
    parser.add_argument("--evaluate_k", action="store_true", help="Evaluate silhouette scores for k=5..25")
    parser.add_argument("--auto_k", action="store_true", help="Automatically pick best k based on silhouette")
    args = parser.parse_args()

    main(args.arima_folds_path, args.lstm_folds_path,
         args.k_arima, args.k_lstm,
         args.evaluate_k, args.auto_k,
         args.n_per_cluster_arima, args.n_per_cluster_lstm,
         args.output_dir)