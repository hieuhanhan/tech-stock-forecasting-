#!/usr/bin/env python3
import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from meta_feature_utils import (
    load_and_filter_val_fold,
    compute_meta_statistics,
    compute_meta_with_pca,
    find_best_k,
    pick_representatives,
    refine_representative_folds,
    select_top_folds_no_cluster_overlap
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ----------------- BUILD META -----------------
def build_meta(folds_summary, val_meta_path, model_type, top_k_pcs=10):
    meta_rows, excluded = [], []
    for fold in folds_summary:
        fid = fold["global_fold_id"]
        val_path = os.path.join(val_meta_path, f"val_meta_fold_{fid}.csv")
        df_val, reason = load_and_filter_val_fold(val_path, model_type=model_type)
        if df_val is None:
            excluded.append((fid, fold["ticker"], reason))
            continue
        if model_type == "arima":
            meta = compute_meta_statistics(df_val)
        else:
            meta = compute_meta_with_pca(df_val, top_k_pcs=top_k_pcs)
        meta.update({"fold_id": fid, "ticker": fold["ticker"]})
        meta_rows.append(meta)
    if excluded:
        pd.DataFrame(excluded, columns=["fold_id", "ticker", "reason"]).to_csv(
            os.path.join(val_meta_path, f"excluded_{model_type}_folds.csv"), index=False
        )
    return pd.DataFrame(meta_rows)



def safe_float(x):
    try:
        a = np.asarray(x)
        if a.size == 1:
            return float(a.squeeze())
        return float(np.nanmean(a))
    except Exception:
        return float('nan')


# ----------------- MAIN -----------------
def main(arima_folds_path, lstm_folds_path, k_arima, k_lstm, n_per_ticker_arima,
         n_per_ticker_lstm, output_dir, evaluate_k, auto_k):

    methods = ["kmeans", "agglo", "hdbscan"]
    summary_rows = []

    # Paths
    arima_val_meta_path = os.path.join(output_dir, "arima_meta")
    lstm_val_meta_path  = os.path.join(output_dir, "lstm_meta")
    os.makedirs(arima_val_meta_path, exist_ok=True)
    os.makedirs(lstm_val_meta_path,  exist_ok=True)

    # Load folds summary
    with open(arima_folds_path) as f: arima_folds_summary = json.load(f)
    with open(lstm_folds_path)  as f: lstm_folds_summary  = json.load(f)

    # Build meta
    meta_arima = build_meta(arima_folds_summary, arima_val_meta_path, "arima")
    meta_lstm  = build_meta(lstm_folds_summary,  lstm_val_meta_path,  "lstm", top_k_pcs=10)

    # Scale
    imp = SimpleImputer(strategy="mean")
    scl = StandardScaler()
    X_arima = scl.fit_transform(imp.fit_transform(meta_arima.drop(columns=["fold_id", "ticker"])))
    X_lstm  = scl.fit_transform(imp.fit_transform(meta_lstm.drop(columns=["fold_id", "ticker"])))

    # Loop methods
    for method in methods:
        logging.info(f"\n=== Running {method.upper()} ===")
        this_k_arima, this_k_lstm = k_arima, k_lstm

        # Evaluate k for KMeans
        if evaluate_k and method == "kmeans":
            best_k_a, sil_a, _ = find_best_k(meta_arima, "ARIMA")
            best_k_l, sil_l, _ = find_best_k(meta_lstm, "LSTM")
            logging.info(
                "[EVAL_K] ARIMA best k=%s (sil=%.3f), LSTM best k=%s (sil=%.3f)",
                best_k_a, safe_float(sil_a), best_k_l, safe_float(sil_l)
            )
            if auto_k:
                this_k_arima, this_k_lstm = best_k_a, best_k_l
        elif method == "hdbscan":
            this_k_arima, this_k_lstm = None, None

        # Pick reps
        arima_reps, meta_arima_annot, sil_a, excl_a, _ = pick_representatives(
            meta_arima, X_arima, this_k_arima, "ARIMA",
            n_per_cluster=n_per_ticker_arima*2, method=method
        )
        lstm_reps, meta_lstm_annot, sil_l, excl_l, _ = pick_representatives(
            meta_lstm, X_lstm, this_k_lstm, "LSTM",
            n_per_cluster=n_per_ticker_lstm*2, method=method
        )

        # Refine & no overlap
        arima_final = select_top_folds_no_cluster_overlap(
            refine_representative_folds(arima_reps, meta_arima_annot, n_per_ticker_arima),
            n_per_ticker=n_per_ticker_arima
        )
        lstm_final = select_top_folds_no_cluster_overlap(
            refine_representative_folds(lstm_reps, meta_lstm_annot, n_per_ticker_lstm),
            n_per_ticker=n_per_ticker_lstm
        )

        # Save
        out_dir = os.path.join(output_dir, method)
        os.makedirs(os.path.join(out_dir, "arima"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "lstm"), exist_ok=True)
        arima_final.to_json(os.path.join(out_dir, "arima", "arima_tuning_folds.json"), orient="records", indent=4)
        lstm_final.to_json (os.path.join(out_dir, "lstm",  "lstm_tuning_folds.json"), orient="records", indent=4)

        # Stats
        overlap = len(set(arima_final.fold_id) & set(lstm_final.fold_id))
        summary_rows.append({
            "method": method,
            "silhouette_arima": round(sil_a, 3),
            "silhouette_lstm": round(sil_l, 3),
            "excluded_arima": len(excl_a),
            "excluded_lstm": len(excl_l),
            "overlap_folds": overlap,
            "arima_selected": len(arima_final),
            "lstm_selected": len(lstm_final)
        })

    # Summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "clustering_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"\nSummary saved to {summary_path}\n{summary_df}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/processed_folds")
    parser.add_argument("--arima_folds_path", type=str, required=True)
    parser.add_argument("--lstm_folds_path",  type=str, required=True)
    parser.add_argument("--k_arima", type=int, default=15)
    parser.add_argument("--k_lstm",  type=int, default=10)
    parser.add_argument("--n_per_ticker_arima", type=int, default=2)
    parser.add_argument("--n_per_ticker_lstm",  type=int, default=2)
    parser.add_argument("--evaluate_k", action="store_true")
    parser.add_argument("--auto_k", action="store_true")
    args = parser.parse_args()

    main(args.arima_folds_path, args.lstm_folds_path,
         args.k_arima, args.k_lstm,
         args.n_per_ticker_arima, args.n_per_ticker_lstm,
         args.output_dir, args.evaluate_k, args.auto_k)